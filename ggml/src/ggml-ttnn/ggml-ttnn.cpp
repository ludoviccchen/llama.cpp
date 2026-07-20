// Tenstorrent Blackhole backend for ggml, built on TT-Metalium / TT-NN.
//
// This backend opens a real TT-Metalium device (silicon or ttsim, selected
// the standard tt-metal way via TT_METAL_SIMULATOR - never branched on here).
// Op offload is currently limited to GGML_OP_MUL_MAT with an I2_S weight
// (Option A: dequantized to bf16 on upload, stock TT-NN ttnn::matmul) -
// everything else still returns false from supports_op().

#include "ggml-ttnn.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/types.hpp>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ttd = tt::tt_metal::distributed;

// -----------------------------------------------------------------------
// Shared TT-Metalium device
//
// One physical Blackhole chip (or its ttsim stand-in), opened lazily and
// kept alive for the life of the process - the same lifetime model CUDA and
// other real GPU backends use. Both device_init() (below) and buffer
// allocation need it, and buffers can outlive any one ggml_backend_t
// "stream" instance, so it is not owned by ggml_backend_ttnn_context.
// -----------------------------------------------------------------------

static std::shared_ptr<ttd::MeshDevice> ggml_backend_ttnn_get_mesh_device() {
    // Intentionally never destroyed. Once any real op has populated
    // tt-metal's program cache, destroying the MeshDevice crashes inside
    // GraphTracker's circular-buffer deallocation callback - reproduced with
    // a plain ttnn::matmul call, no ggml involved, and *still* crashes even
    // when close() is called first via an atexit hook ordered ahead of the
    // static's own destructor (close() apparently doesn't fully neutralize
    // the program cache before later teardown touches it). Since the
    // process is exiting either way, leaking this is safe and side-steps
    // the whole static-destruction-order question - the same pragmatic
    // choice other libraries with similar teardown fragility (e.g. CUDA
    // driver contexts) commonly make.
    static std::shared_ptr<ttd::MeshDevice> * mesh_device = new std::shared_ptr<ttd::MeshDevice>(ttd::MeshDevice::create_unit_mesh(0));
    return *mesh_device;
}

// -----------------------------------------------------------------------
// Buffer type / buffer
//
// Two upstream tt-metal bugs (both confirmed via minimal repros independent
// of this backend and of ggml, using only tt-metal's own API - see the git
// history for this file) rule out the "one big shared MeshBuffer, tensors
// placed at ggml-computed sub-offsets" design every other ggml GPU backend
// uses:
//   1. SDMeshCommandQueue::write_shard_to_device (slow dispatch, required
//      under ttsim) adds the device region offset to the *host* source
//      pointer too, reading the wrong host bytes for any non-zero offset.
//   2. Independent of (1): a BufferRegion write/read that is "interior" -
//      offset > 0 AND offset+size < the buffer's total size - lands at
//      device offset 0 instead of the requested offset. Only regions
//      anchored at the very start or the very end of a buffer are safe.
//
// So instead: every ggml tensor gets its own dedicated MeshBuffer, and every
// tt-metal-facing access is always the *entire* buffer (offset 0, full
// size), which is confirmed safe under both bugs. Partial ggml-side
// set/get/memset calls are implemented as a host-side read-modify-write over
// that whole buffer. tensor->data is a synthetic pointer into a host-only
// "shadow" allocation - never dereferenced for real data, just used by
// ggml's own tensor allocator as a unique per-tensor key (see init_tensor).
// -----------------------------------------------------------------------

struct ggml_backend_ttnn_buffer_context {
    void * host_shadow;
    std::unordered_map<const void *, std::shared_ptr<ttd::MeshBuffer>> tensor_buffers;
};

// I2_S (Option A): dequantize packed ternary weights to bf16 once, at
// upload time, so the device side can use stock TT-NN matmul before any
// custom packed-ternary kernel exists (Option B).
//
// Mirrors the *strided* bit layout verified against the real AVX2 inference
// dot product (ggml_vec_dot_i2_i8_s_1x1, same quants.c): byte
// packed[done/4+gp] holds four 2-bit codes for elements at done+gp,
// done+32+gp, done+64+gp, done+96+gp within each 128-element super-block -
// NOT quantize_i2_s's own consecutive-i/4 packing, which disagrees with the
// dot product it's supposedly paired with. See PORTING_PLAN.md sec 10.
static void ggml_backend_ttnn_dequantize_i2_s(const uint8_t * packed, int64_t n, std::vector<bfloat16> & out) {
    static const float map2bit[4] = { -1.0f, 0.0f, 1.0f, 0.0f };
    float scale;
    memcpy(&scale, packed + n / 4, sizeof(float));
    out.resize(n);
    for (int64_t done = 0; done < n; done += 128) {
        for (int gp = 0; gp < 32 && done + gp < n; gp++) {
            uint8_t byte = packed[done / 4 + gp];
            for (int lane = 0; lane < 4; lane++) {
                int64_t idx = done + lane * 32 + gp;
                if (idx >= n) {
                    continue;
                }
                uint8_t code = (byte >> (6 - 2 * lane)) & 0x03;
                out[idx] = bfloat16(scale * map2bit[code]);
            }
        }
    }
}

// I2_S tensors are stored on-device as dequantized bf16 (Option A); every
// other type keeps its normal ggml byte size.
static size_t ggml_backend_ttnn_device_alloc_size(const struct ggml_tensor * tensor) {
    if (tensor->type == GGML_TYPE_I2_S) {
        return ggml_nelements(tensor) * sizeof(bfloat16);
    }
    return ggml_nbytes(tensor);
}

static const char * ggml_backend_ttnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "TT_Metalium";
}

static void ggml_backend_ttnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_ttnn_buffer_context * ctx = (ggml_backend_ttnn_buffer_context *) buffer->context;
    free(ctx->host_shadow);
    delete ctx;
}

static void * ggml_backend_ttnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_backend_ttnn_buffer_context *) buffer->context)->host_shadow;
}

static enum ggml_status ggml_backend_ttnn_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    // A view shares its parent's device buffer at (usually non-zero,
    // non-whole-buffer) view_offs, which is exactly the unsafe access
    // pattern above - not yet supported.
    GGML_ASSERT(tensor->view_src == NULL && "ggml-ttnn: tensor views are not yet supported by the DRAM buffer type");

    ggml_backend_ttnn_buffer_context * ctx = (ggml_backend_ttnn_buffer_context *) buffer->context;
    auto mesh_device = ggml_backend_ttnn_get_mesh_device();
    size_t nbytes = ggml_backend_ttnn_device_alloc_size(tensor);

    // One page spanning the whole buffer: every access to it is whole-buffer
    // (offset 0, size == nbytes), so this is always the safe end-anchored
    // case regardless of page size.
    ttd::DeviceLocalBufferConfig device_local_config{
        /* .page_size   = */ nbytes,
        /* .buffer_type = */ tt::tt_metal::BufferType::DRAM,
    };
    auto mesh_buffer = ttd::MeshBuffer::create(ttd::ReplicatedBufferConfig{ /* .size = */ nbytes }, device_local_config, mesh_device.get());
    ctx->tensor_buffers[tensor->data] = mesh_buffer;
    return GGML_STATUS_SUCCESS;
}

static std::shared_ptr<ttd::MeshBuffer> ggml_backend_ttnn_lookup(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    ggml_backend_ttnn_buffer_context * ctx = (ggml_backend_ttnn_buffer_context *) buffer->context;
    auto it = ctx->tensor_buffers.find(tensor->data);
    GGML_ASSERT(it != ctx->tensor_buffers.end() && "ggml-ttnn: tensor has no device buffer (init_tensor not called?)");
    return it->second;
}

static void ggml_backend_ttnn_write_whole(const std::shared_ptr<ttd::MeshBuffer> & mesh_buffer, const void * data) {
    ttd::MeshCommandQueue & cq = mesh_buffer->device()->mesh_command_queue();
    ttd::ShardDataTransfer transfer(ttd::MeshCoordinate(0, 0));
    transfer.host_data(const_cast<void *>(data));
    transfer.region(tt::tt_metal::BufferRegion(0, mesh_buffer->size()));
    // Blocking: the buffer interface is expected to behave synchronously -
    // callers are free to reuse/free `data` as soon as this returns.
    cq.enqueue_write_shards(mesh_buffer, { transfer }, /*blocking=*/true);
}

static void ggml_backend_ttnn_read_whole(const std::shared_ptr<ttd::MeshBuffer> & mesh_buffer, void * data) {
    ttd::MeshCommandQueue & cq = mesh_buffer->device()->mesh_command_queue();
    ttd::ShardDataTransfer transfer(ttd::MeshCoordinate(0, 0));
    transfer.host_data(data);
    transfer.region(tt::tt_metal::BufferRegion(0, mesh_buffer->size()));
    cq.enqueue_read_shards({ transfer }, mesh_buffer, /*blocking=*/true);
}

static void ggml_backend_ttnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto mesh_buffer = ggml_backend_ttnn_lookup(buffer, tensor);
    size_t total = mesh_buffer->size();
    if (offset == 0 && size == total) {
        std::vector<uint8_t> fill(total, value);
        ggml_backend_ttnn_write_whole(mesh_buffer, fill.data());
        return;
    }
    std::vector<uint8_t> scratch(total);
    ggml_backend_ttnn_read_whole(mesh_buffer, scratch.data());
    memset(scratch.data() + offset, value, size);
    ggml_backend_ttnn_write_whole(mesh_buffer, scratch.data());
}

static void ggml_backend_ttnn_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto mesh_buffer = ggml_backend_ttnn_lookup(buffer, tensor);

    if (tensor->type == GGML_TYPE_I2_S) {
        // Always uploaded whole: `data`/`size` here are the packed ternary
        // bytes as stored in the gguf file (ggml_nbytes, not the device's
        // dequantized bf16 size) - a partial update of packed+scaled data
        // isn't a meaningful operation, so this only supports the one-shot
        // whole-tensor upload weight loading actually does.
        GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
        std::vector<bfloat16> dequantized;
        ggml_backend_ttnn_dequantize_i2_s((const uint8_t *) data, ggml_nelements(tensor), dequantized);
        ggml_backend_ttnn_write_whole(mesh_buffer, dequantized.data());
        return;
    }

    size_t total = mesh_buffer->size();
    if (offset == 0 && size == total) {
        ggml_backend_ttnn_write_whole(mesh_buffer, data);
        return;
    }
    std::vector<uint8_t> scratch(total);
    ggml_backend_ttnn_read_whole(mesh_buffer, scratch.data());
    memcpy(scratch.data() + offset, data, size);
    ggml_backend_ttnn_write_whole(mesh_buffer, scratch.data());
}

static void ggml_backend_ttnn_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto mesh_buffer = ggml_backend_ttnn_lookup(buffer, tensor);
    size_t total = mesh_buffer->size();
    if (offset == 0 && size == total) {
        ggml_backend_ttnn_read_whole(mesh_buffer, data);
        return;
    }
    std::vector<uint8_t> scratch(total);
    ggml_backend_ttnn_read_whole(mesh_buffer, scratch.data());
    memcpy(data, scratch.data() + offset, size);
}

static void ggml_backend_ttnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    // Only clears tensors already registered via init_tensor; ggml calls
    // clear() on buffers whose tensors have already been placed in the
    // usage this backend currently supports.
    ggml_backend_ttnn_buffer_context * ctx = (ggml_backend_ttnn_buffer_context *) buffer->context;
    for (auto & kv : ctx->tensor_buffers) {
        std::vector<uint8_t> fill(kv.second->size(), value);
        ggml_backend_ttnn_write_whole(kv.second, fill.data());
    }
}

static size_t ggml_backend_ttnn_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_backend_ttnn_device_alloc_size(tensor);
}

static const struct ggml_backend_buffer_i ggml_backend_ttnn_buffer_i = {
    /* .free_buffer     = */ ggml_backend_ttnn_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_ttnn_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_ttnn_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_ttnn_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_ttnn_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_ttnn_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_ttnn_buffer_clear,
    /* .reset           = */ NULL,
};

static ggml_backend_buffer_t ggml_backend_ttnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_ttnn_buffer_context * ctx = new ggml_backend_ttnn_buffer_context;
    // ggml's tallocr aligns tensor placements relative to this buffer's base
    // pointer (get_alignment() below, 32) - plain malloc() only guarantees
    // 16-byte alignment on x86_64, which silently eats into the reported
    // buffer size once ggml pads its own bookkeeping to a 32-aligned base.
    if (posix_memalign(&ctx->host_shadow, 32, size) != 0) {
        delete ctx;
        return NULL;
    }
    return ggml_backend_buffer_init(buft, ggml_backend_ttnn_buffer_i, ctx, size);
}

static size_t ggml_backend_ttnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 32;
}

static bool ggml_backend_ttnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // tensor->data is a synthetic device address, not host-readable memory.
    return false;
}

static const struct ggml_backend_buffer_type_i ggml_backend_ttnn_buffer_type_i = {
    /* .get_name       = */ ggml_backend_ttnn_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_ttnn_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_ttnn_buffer_type_get_alignment,
    /* .get_max_size   = */ NULL,
    /* .get_alloc_size = */ ggml_backend_ttnn_buffer_type_get_alloc_size,
    /* .is_host        = */ ggml_backend_ttnn_buffer_type_is_host,
};

static ggml_backend_buffer_type_t ggml_backend_ttnn_buffer_type(ggml_backend_dev_t dev) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ ggml_backend_ttnn_buffer_type_i,
        /* .device  = */ dev,
        /* .context = */ NULL,
    };
    return &buft;
}

// -----------------------------------------------------------------------
// Backend (stream)
// -----------------------------------------------------------------------

static const char * ggml_backend_ttnn_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "TT_Metalium";
}

static void ggml_backend_ttnn_free(ggml_backend_t backend) {
    // The MeshDevice is a process-lifetime singleton (see
    // ggml_backend_ttnn_get_mesh_device) shared with any still-live buffers,
    // so freeing this backend "stream" does not close it.
    delete backend;
}

// GGML_OP_MUL_MAT, I2_S weight x F32 activation -> F32 (Option A).
//
// ggml's mul_mat(src0, src1) convention: src0 (ne=[K,N]) is a row-major
// [N,K] weight matrix (N output features, K input features - ne[0] is the
// contiguous/fastest dim); src1 (ne=[K,M]) is a row-major [M,K] activation
// batch. dst (ne=[N,M]) is the row-major [M,N] result of `act @ weight^T`,
// exactly nn.Linear's convention. That maps directly onto
// ttnn::matmul(act, weight, transpose_a=false, transpose_b=true).
//
// Both operands are staged through host memory into fresh ttnn::Tensor
// objects rather than reusing the MeshBuffers this backend already manages
// (see the buffer-type comment above) - this backend doesn't yet store
// ttnn::Tensor natively, so this round-trips already-on-device weight data
// through the host on every call. Correct, not yet fast; avoiding this is
// follow-up work once the buffer type is redesigned around ttnn::Tensor.
static void ggml_backend_ttnn_compute_mul_mat(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_I2_S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const uint32_t K = (uint32_t) src0->ne[0];
    const uint32_t N = (uint32_t) src0->ne[1];
    const uint32_t M = (uint32_t) src1->ne[1];
    GGML_ASSERT((uint32_t) src1->ne[0] == K);
    GGML_ASSERT((uint32_t) dst->ne[0] == N && (uint32_t) dst->ne[1] == M);

    auto weight_buf = ggml_backend_ttnn_lookup(src0->buffer, src0);
    std::vector<bfloat16> weight_host(N * K);
    ggml_backend_ttnn_read_whole(weight_buf, weight_host.data());

    auto act_buf = ggml_backend_ttnn_lookup(src1->buffer, src1);
    std::vector<float> act_host(M * K);
    ggml_backend_ttnn_read_whole(act_buf, act_host.data());

    auto mesh_device = ggml_backend_ttnn_get_mesh_device();
    const tt::tt_metal::MemoryConfig mem_cfg{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    const tt::tt_metal::TensorLayout weight_layout(tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_cfg);
    const tt::tt_metal::TensorLayout act_layout(tt::tt_metal::DataType::FLOAT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mem_cfg);

    ttnn::Tensor weight_t = ttnn::Tensor::from_vector(weight_host, ttnn::TensorSpec(ttnn::Shape({N, K}), weight_layout))
                                 .to_device(mesh_device.get(), mem_cfg);
    ttnn::Tensor act_t = ttnn::Tensor::from_vector(act_host, ttnn::TensorSpec(ttnn::Shape({M, K}), act_layout))
                              .to_device(mesh_device.get(), mem_cfg);

    ttnn::Tensor out_t = ttnn::matmul(act_t, weight_t, /*transpose_a=*/false, /*transpose_b=*/true, mem_cfg);

    std::vector<float> out_host = out_t.to_vector<float>();

    auto dst_buf = ggml_backend_ttnn_lookup(dst->buffer, dst);
    ggml_backend_ttnn_write_whole(dst_buf, out_host.data());
}

static enum ggml_status ggml_backend_ttnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_ttnn_compute_mul_mat(node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_PERMUTE:
                break;
            default:
                GGML_ABORT("ggml-ttnn: graph_compute got op '%s', which supports_op() should not have allowed", ggml_op_name(node->op));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const struct ggml_backend_i ggml_backend_ttnn_i = {
    /* .get_name                = */ ggml_backend_ttnn_get_name,
    /* .free                    = */ ggml_backend_ttnn_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_ttnn_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_ttnn_guid(void) {
    static ggml_guid guid = { 0x9c, 0x3a, 0x1b, 0x6e, 0x4f, 0x2d, 0x4a, 0x8c,
                               0xb1, 0x5e, 0x7d, 0x0f, 0x3c, 0x9a, 0x6b, 0x21 };
    return &guid;
}

// -----------------------------------------------------------------------
// Device
// -----------------------------------------------------------------------

static const char * ggml_backend_ttnn_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "TT_METALIUM0";
}

static const char * ggml_backend_ttnn_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Tenstorrent Blackhole (TT-Metalium/TT-NN backend, registration skeleton)";
}

static void ggml_backend_ttnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    // Not yet backed by a real TT-Metalium device.
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_ttnn_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_ttnn_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_ttnn_device_get_name(dev);
    props->description = ggml_backend_ttnn_device_get_description(dev);
    ggml_backend_ttnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->type        = ggml_backend_ttnn_device_get_type(dev);
    props->device_id   = NULL;
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_ttnn_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    // Opens device 0 the standard tt-metal way: silicon vs. ttsim is chosen
    // by the driver via TT_METAL_SIMULATOR, never branched on here.
    ggml_backend_ttnn_get_mesh_device();

    ggml_backend_t backend = new ggml_backend {
        /* .guid     = */ ggml_backend_ttnn_guid(),
        /* .iface    = */ ggml_backend_ttnn_i,
        /* .device   = */ dev,
        /* .context  = */ NULL,
    };
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_ttnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_ttnn_buffer_type(dev);
}

static bool ggml_backend_ttnn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    return src0->type == GGML_TYPE_I2_S && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
}

static bool ggml_backend_ttnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_ttnn_buffer_type_i.get_name && buft->device == dev;
}

static const struct ggml_backend_device_i ggml_backend_ttnn_device_i = {
    /* .get_name             = */ ggml_backend_ttnn_device_get_name,
    /* .get_description      = */ ggml_backend_ttnn_device_get_description,
    /* .get_memory           = */ ggml_backend_ttnn_device_get_memory,
    /* .get_type             = */ ggml_backend_ttnn_device_get_type,
    /* .get_props            = */ ggml_backend_ttnn_device_get_props,
    /* .init_backend         = */ ggml_backend_ttnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_ttnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_ttnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_ttnn_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free            = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// -----------------------------------------------------------------------
// Backend registry
// -----------------------------------------------------------------------

static const char * ggml_backend_ttnn_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "TTNN";
}

static size_t ggml_backend_ttnn_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_ttnn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static struct ggml_backend_device ggml_backend_ttnn_device = {
        /* .iface   = */ ggml_backend_ttnn_device_i,
        /* .reg     = */ reg,
        /* .context = */ NULL,
    };
    return &ggml_backend_ttnn_device;
}

static const struct ggml_backend_reg_i ggml_backend_ttnn_reg_i = {
    /* .get_name         = */ ggml_backend_ttnn_reg_get_name,
    /* .get_device_count = */ ggml_backend_ttnn_reg_get_device_count,
    /* .get_device       = */ ggml_backend_ttnn_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_ttnn_reg(void) {
    static struct ggml_backend_reg ggml_backend_ttnn_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_ttnn_reg_i,
        /* .context     = */ NULL,
    };
    return &ggml_backend_ttnn_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ttnn_reg)
