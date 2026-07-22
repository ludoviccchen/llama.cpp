// Tenstorrent Blackhole backend for ggml, built on TT-Metalium.
//
// This backend opens a real TT-Metalium device (silicon or ttsim, selected
// the standard tt-metal way via TT_METAL_SIMULATOR - never branched on here).
// Op offload is currently limited to GGML_OP_MUL_MAT with an I2_S weight
// (Option B: packed 2-bit ternary weights consumed directly by custom
// reader/compute/writer kernels in kernels/ternary_matmul/, unpacked
// on-device - see that directory's README and PORTING_PLAN.md sec 13/14) -
// everything else still returns false from supports_op().

#include "ggml-ttnn.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

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
    if (tensor->view_src != NULL) {
        // ggml_gallocr wraps op outputs in a same-buffer, offset-0 "view of
        // itself" for its own bookkeeping (seen with ggml_backend_sched:
        // mul_mat's dst arrives here as a view of a GGML_OP_MUL_MAT source
        // at view_offs 0) - since offset 0 means tensor->data ==
        // view_src->data, the existing tensor_buffers entry for view_src
        // (keyed by that same pointer, registered when view_src itself was
        // init_tensor'd) already covers this tensor; nothing to do. Any
        // other view - non-zero offset, i.e. a genuine sub-region of a
        // buffer - would need interior device-buffer access, which isn't
        // supported (see the buffer-type comment above).
        GGML_ASSERT(tensor->view_offs == 0 && tensor->data == tensor->view_src->data &&
                     "ggml-ttnn: only whole-tensor views are supported by the DRAM buffer type");
        return GGML_STATUS_SUCCESS;
    }

    ggml_backend_ttnn_buffer_context * ctx = (ggml_backend_ttnn_buffer_context *) buffer->context;
    auto mesh_device = ggml_backend_ttnn_get_mesh_device();
    // I2_S tensors are stored packed (device size == host/ggml size, same as
    // every other type) - Option B's reader kernel unpacks them on-device,
    // so there is no special-cased device representation to size for here
    // any more (contrast Option A's now-removed dequant-on-upload).
    size_t nbytes = ggml_nbytes(tensor);

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
    // I2_S weights are uploaded and stored packed, byte-for-byte identical
    // to the gguf file - Option B's reader kernel does the ternary unpack
    // on-device at compute time, so there's nothing type-specific to do
    // here; this is the same generic whole-buffer-or-read-modify-write path
    // every other type uses.
    auto mesh_buffer = ggml_backend_ttnn_lookup(buffer, tensor);
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
    /* .get_alloc_size = */ NULL,  // defaults to ggml_nbytes, which is exactly right now
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

// True conditions for offloading MUL_MAT at all (shared by supports_op and
// this function, so the scheduler never routes here a shape the kernels
// can't handle - see kernels/ternary_matmul/README.md and PORTING_PLAN.md
// sec 13/14 for why these particular constraints).
static bool ggml_backend_ttnn_mul_mat_shape_ok(const struct ggml_tensor * src0, const struct ggml_tensor * src1) {
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    constexpr int64_t TILE_DIM = 32;
    // K a multiple of 128: one I2_S packing super-block never straddles a
    // row (reader kernel's row-stride assumption). N/M tile-aligned: the FPU
    // only operates on whole 32x32 tiles.
    return src1->ne[0] == K && K % 128 == 0 && N % TILE_DIM == 0 && M % TILE_DIM == 0;
}

// GGML_OP_MUL_MAT, I2_S weight x F32 activation -> F32 (Option B: packed
// ternary weights consumed directly on-device - see
// kernels/ternary_matmul/README.md and PORTING_PLAN.md sec 13/14).
//
// ggml's mul_mat(src0, src1) convention: src0 (ne=[K,N]) is a row-major
// [N,K] weight matrix (N output features, K input features - ne[0] is the
// contiguous/fastest dim); src1 (ne=[K,M]) is a row-major [M,K] activation
// batch. dst (ne=[N,M]) is the row-major **[M,N]** result of `act @
// weight^T` (M rows, N contiguous per row - N is ne[0], the fastest dim).
//
// The kernel triad's own tile grid is the opposite way round: the writer
// lays output tiles out N-tile-major (page index n*Mt+m - see the writer
// kernel/README), so untilize_nfaces(result_tiled, N, M) reconstructs an
// N-outer/M-inner matrix - the transpose of what dst's bytes need. That
// transpose has to happen explicitly below; conflating "the kernel's own
// tile-grid labeling" with "ggml's flat byte layout" here was a real bug
// caught by wiring this in (a diagnostic with one known nonzero weight
// showed row 0's value leaking into unrelated output rows once read back
// through a real ggml tensor - the raw pre-untilize tile data was already
// correct, proving the bug was in this final relayout, not the kernel).
// The reader kernel unpacks the weight straight from this backend's own
// resident MeshBuffer (no re-upload - the whole point of Option B is that
// the packed weight never needs to leave DRAM as anything but packed
// bytes). The activation, however, has to be staged through host memory to
// get transposed into [K,M] and tile-faced - matmul_tiles' operand
// convention needs it that way (see the kernel README), and neither of
// those is something this backend's generic buffer-type machinery does.
static void ggml_backend_ttnn_compute_mul_mat(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_I2_S);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_backend_ttnn_mul_mat_shape_ok(src0, src1) &&
                "ggml-ttnn: shape not offloadable - supports_op() should not have allowed this");

    constexpr uint32_t TILE_DIM = 32;
    const uint32_t K = (uint32_t) src0->ne[0];
    const uint32_t N = (uint32_t) src0->ne[1];
    const uint32_t M = (uint32_t) src1->ne[1];
    GGML_ASSERT((uint32_t) dst->ne[0] == N && (uint32_t) dst->ne[1] == M);
    const uint32_t Kt = K / TILE_DIM, Nt = N / TILE_DIM, Mt = M / TILE_DIM;

    auto weight_buf = ggml_backend_ttnn_lookup(src0->buffer, src0);  // resident packed I2_S bytes, reused as-is
    auto act_buf = ggml_backend_ttnn_lookup(src1->buffer, src1);
    auto dst_buf = ggml_backend_ttnn_lookup(dst->buffer, dst);

    // The per-tensor weight scale (quantize_i2_s / PORTING_PLAN.md sec 10)
    // sits in the last 4 of the packed blob's 32 trailing bytes; reading the
    // whole (small) blob to host is always safe here (offset 0, full size -
    // see the buffer-type comment above), unlike a sub-range read.
    std::vector<uint8_t> weight_host(weight_buf->size());
    ggml_backend_ttnn_read_whole(weight_buf, weight_host.data());
    float scale;
    memcpy(&scale, weight_host.data() + (size_t) N * K / 4, sizeof(float));

    // Activation, transposed to [K,M] (K outer) and tile-faced - see the
    // kernel README for why matmul_tiles needs it this way round.
    std::vector<float> act_host(M * K);
    ggml_backend_ttnn_read_whole(act_buf, act_host.data());
    std::vector<bfloat16> act_transposed(K * M);
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t k = 0; k < K; k++) {
            act_transposed[k * M + m] = bfloat16(act_host[m * K + k]);
        }
    }
    std::vector<bfloat16> act_tiled = tilize_nfaces(act_transposed, K, M);

    auto mesh_device = ggml_backend_ttnn_get_mesh_device();
    const uint32_t single_tile_size = sizeof(bfloat16) * tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;

    ttd::DeviceLocalBufferConfig tile_dram_cfg{
        /* .page_size   = */ single_tile_size, /* .buffer_type = */ tt::tt_metal::BufferType::DRAM};
    auto act_dram = ttd::MeshBuffer::create(
        ttd::ReplicatedBufferConfig{ /* .size = */ sizeof(bfloat16) * act_tiled.size()}, tile_dram_cfg, mesh_device.get());
    auto out_dram = ttd::MeshBuffer::create(
        ttd::ReplicatedBufferConfig{ /* .size = */ (size_t) single_tile_size * Nt * Mt}, tile_dram_cfg, mesh_device.get());

    tt::tt_metal::Program program{};
    tt::tt_metal::CoreCoord core({0, 0});
    tt::DataFormat bf16_fmt = tt::DataFormat::Float16_b;

    uint32_t cb_in0 = tt::CBIndex::c_0;
    tt::tt_metal::CreateCircularBuffer(
        program, core,
        tt::tt_metal::CircularBufferConfig(2 * single_tile_size, {{cb_in0, bf16_fmt}}).set_page_size(cb_in0, single_tile_size));
    uint32_t cb_in1 = tt::CBIndex::c_1;
    tt::tt_metal::CreateCircularBuffer(
        program, core,
        tt::tt_metal::CircularBufferConfig(2 * single_tile_size, {{cb_in1, bf16_fmt}}).set_page_size(cb_in1, single_tile_size));
    uint32_t cb_out = tt::CBIndex::c_16;
    tt::tt_metal::CreateCircularBuffer(
        program, core,
        tt::tt_metal::CircularBufferConfig(2 * single_tile_size, {{cb_out, bf16_fmt}}).set_page_size(cb_out, single_tile_size));
    uint32_t cb_scratch = tt::CBIndex::c_2;
    uint32_t weight_blob_size = (uint32_t) weight_buf->size();
    tt::tt_metal::CreateCircularBuffer(
        program, core,
        tt::tt_metal::CircularBufferConfig(weight_blob_size, {{cb_scratch, tt::DataFormat::UInt8}})
            .set_page_size(cb_scratch, weight_blob_size));

    std::vector<uint32_t> reader_compile_args;
    tt::tt_metal::TensorAccessorArgs(*weight_buf).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(*act_dram).append_to(reader_compile_args);
    auto reader_id = tt::tt_metal::CreateKernel(
        program, TERNARY_MATMUL_KERNEL_DIR "dataflow/reader_ternary_mm.cpp", core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    std::vector<uint32_t> writer_compile_args;
    tt::tt_metal::TensorAccessorArgs(*out_dram).append_to(writer_compile_args);
    auto writer_id = tt::tt_metal::CreateKernel(
        program, TERNARY_MATMUL_KERNEL_DIR "dataflow/writer_ternary_mm.cpp", core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = writer_compile_args});

    std::vector<uint32_t> compute_compile_args = {Mt, Kt, Nt};
    tt::tt_metal::CreateKernel(
        program, TERNARY_MATMUL_KERNEL_DIR "compute/mm.cpp", core,
        tt::tt_metal::ComputeConfig{.math_fidelity = tt::tt_metal::MathFidelity::HiFi4, .compile_args = compute_compile_args});

    tt::tt_metal::SetRuntimeArgs(
        program, reader_id, core, {(uint32_t) weight_buf->address(), (uint32_t) act_dram->address(), Mt, Kt, Nt, K});
    tt::tt_metal::SetRuntimeArgs(program, writer_id, core, {(uint32_t) out_dram->address(), Mt, Nt});

    ttd::MeshCommandQueue & cq = mesh_device->mesh_command_queue();
    ttd::MeshWorkload workload;
    ttd::MeshCoordinateRange device_range(mesh_device->shape());

    ttd::EnqueueWriteMeshBuffer(cq, act_dram, act_tiled, /*blocking=*/false);
    workload.add_program(device_range, std::move(program));
    ttd::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<bfloat16> result_tiled((size_t) Nt * Mt * TILE_DIM * TILE_DIM);
    ttd::EnqueueReadMeshBuffer(cq, result_tiled, out_dram, /*blocking=*/true);
    // N-outer/M-inner (the kernel's own tile-grid order, matching the
    // writer's n*Mt+m page layout) - see the comment above this function
    // for why this is NOT yet ggml's own dst byte layout.
    std::vector<bfloat16> result = untilize_nfaces(result_tiled, N, M);

    // Transpose into ggml's dst convention (M-outer/N-inner, ne0=N fastest)
    // while applying the per-tensor weight scale - device output is the
    // unscaled ternary dot product, not rescaled per-element in-kernel.
    std::vector<float> out_host(N * M);
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t m = 0; m < M; m++) {
            out_host[m * N + n] = (float) result[n * M + m] * scale;
        }
    }

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
    // GGML_OP_NONE is a real leaf (e.g. a weight tensor already sitting in
    // this backend's buffer, no view) - the scheduler queries supports_op()
    // for these too (ggml_backend_sched_backend_from_buffer), and they need
    // no compute. VIEW/RESHAPE/TRANSPOSE/PERMUTE are NOT included here even
    // though graph_compute() tolerates them: the buffer type's init_tensor
    // hard-rejects tensor->view_src != NULL (see comment there), so the
    // scheduler must never be told this device can take a real view op.
    if (op->op == GGML_OP_NONE) {
        return true;
    }
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    return src0->type == GGML_TYPE_I2_S && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           ggml_backend_ttnn_mul_mat_shape_ok(src0, src1);
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
