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
#include <tt-metalium/work_split.hpp>

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

// A view never gets its own MeshBuffer - it aliases into whichever non-view
// ancestor tensor owns the buffer, at whatever byte offset tensor->data -
// root->data works out to (walking view_src handles chained
// views/reshapes/permutes transparently, since ggml always folds the final
// pointer for us). Resolving this at *access* time rather than baking a
// separate device allocation for every view means every real tt-metal
// transfer still targets a whole buffer end-to-end (offset 0, full size) -
// set_tensor/get_tensor/memset_tensor just shift their host-side
// read-modify-write window by the view's offset - so the two upstream
// interior-access bugs documented above never come into play, no matter how
// the view slices its root tensor.
static const struct ggml_tensor * ggml_backend_ttnn_root_tensor(const struct ggml_tensor * tensor) {
    while (tensor->view_src != NULL) {
        tensor = tensor->view_src;
    }
    return tensor;
}

struct ggml_backend_ttnn_located_buffer {
    std::shared_ptr<ttd::MeshBuffer> buffer;
    size_t offset;  // tensor's byte offset within `buffer`
};

static ggml_backend_ttnn_located_buffer ggml_backend_ttnn_locate(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    const struct ggml_tensor * root = ggml_backend_ttnn_root_tensor(tensor);
    ggml_backend_ttnn_buffer_context * ctx = (ggml_backend_ttnn_buffer_context *) buffer->context;
    auto it = ctx->tensor_buffers.find(root->data);
    GGML_ASSERT(it != ctx->tensor_buffers.end() && "ggml-ttnn: tensor has no device buffer (init_tensor not called?)");
    size_t offset = (const uint8_t *) tensor->data - (const uint8_t *) root->data;
    return { it->second, offset };
}

static enum ggml_status ggml_backend_ttnn_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    if (tensor->view_src != NULL) {
        // Nothing to allocate here - see ggml_backend_ttnn_locate above.
        // Just sanity-check the view actually lands inside its root
        // tensor's buffer.
        auto located = ggml_backend_ttnn_locate(buffer, tensor);
        GGML_ASSERT(located.offset + ggml_nbytes(tensor) <= located.buffer->size() &&
                     "ggml-ttnn: view extends past its root tensor's buffer");
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
    // Callers that use this (compute_mul_mat) only ever see non-view
    // tensors, where the located offset is always 0 - the buffer alone is
    // all they need.
    return ggml_backend_ttnn_locate(buffer, tensor).buffer;
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
    auto located = ggml_backend_ttnn_locate(buffer, tensor);
    size_t total = located.buffer->size();
    size_t abs_offset = located.offset + offset;
    if (abs_offset == 0 && size == total) {
        std::vector<uint8_t> fill(total, value);
        ggml_backend_ttnn_write_whole(located.buffer, fill.data());
        return;
    }
    std::vector<uint8_t> scratch(total);
    ggml_backend_ttnn_read_whole(located.buffer, scratch.data());
    memset(scratch.data() + abs_offset, value, size);
    ggml_backend_ttnn_write_whole(located.buffer, scratch.data());
}

static void ggml_backend_ttnn_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // I2_S weights are uploaded and stored packed, byte-for-byte identical
    // to the gguf file - Option B's reader kernel does the ternary unpack
    // on-device at compute time, so there's nothing type-specific to do
    // here; this is the same generic whole-buffer-or-read-modify-write path
    // every other type uses. `located.offset` folds in any view offset on
    // top of the caller-supplied `offset`.
    auto located = ggml_backend_ttnn_locate(buffer, tensor);
    size_t total = located.buffer->size();
    size_t abs_offset = located.offset + offset;
    if (abs_offset == 0 && size == total) {
        ggml_backend_ttnn_write_whole(located.buffer, data);
        return;
    }
    std::vector<uint8_t> scratch(total);
    ggml_backend_ttnn_read_whole(located.buffer, scratch.data());
    memcpy(scratch.data() + abs_offset, data, size);
    ggml_backend_ttnn_write_whole(located.buffer, scratch.data());
}

static void ggml_backend_ttnn_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto located = ggml_backend_ttnn_locate(buffer, tensor);
    size_t total = located.buffer->size();
    size_t abs_offset = located.offset + offset;
    if (abs_offset == 0 && size == total) {
        ggml_backend_ttnn_read_whole(located.buffer, data);
        return;
    }
    std::vector<uint8_t> scratch(total);
    ggml_backend_ttnn_read_whole(located.buffer, scratch.data());
    memcpy(data, scratch.data() + abs_offset, size);
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
    //
    // compute_mul_mat below can in fact handle any M >= 1 (it pads up to the
    // next tile boundary on upload and truncates the result back down before
    // writing dst - PORTING_PLAN.md sec 22) - that padding path is real and
    // separately verified correct. The M%32==0 gate here is deliberately
    // kept anyway: relaxing it makes ordinary single-token decode (M=1)
    // reach this backend for the first time (sec 21 found it never did
    // before), which surfaces a much bigger, separate problem - the reader
    // kernel's O(Nt*Kt*1024) scalar unpack loop takes minutes per call at
    // real projection-matrix dimensions under ttsim (measured: a single
    // K=2560,N=2560 call did not finish in over 580s), because the old gate
    // had incidentally kept virtually all real per-layer decode-time matmuls
    // away from this kernel entirely. Enabling M-padding by default would
    // turn the existing, validated `-ngl 99` smoke test from ~20s-2min into
    // many minutes-to-hours, for a problem the padding fix doesn't cause and
    // can't fix (sec 22). Restoring this constraint keeps today's demo fast;
    // lifting it again is the natural next step once the reader kernel's
    // unpack loop is vectorized/optimized enough to be practical at real N/K
    // scale under ttsim.
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
    const uint32_t Kt = K / TILE_DIM, Nt = N / TILE_DIM;
    // Round up to a whole number of M-tiles. supports_op() currently still
    // requires M%32==0 (ggml_backend_ttnn_mul_mat_shape_ok, PORTING_PLAN.md
    // sec 22), so M_padded == M for every M this function is actually called
    // with today - but the logic below handles a non-tile-aligned M
    // correctly regardless (separately verified), ready for when
    // supports_op's M constraint is lifted once the reader kernel's unpack
    // loop is fast enough to make that practical. The kernel triad itself is
    // unaffected either way (it always operates on whole tiles); any padding
    // rows are zero-filled below and dropped again once the result comes
    // back, invisible to it.
    const uint32_t Mt = (M + TILE_DIM - 1) / TILE_DIM;
    const uint32_t M_padded = Mt * TILE_DIM;

    // The weight is consumed directly from its own resident device buffer
    // (the reader kernel addresses it on-device - the whole point of
    // Option B), so it must be the sole occupant of that buffer, not a
    // sub-view of something bigger; dst likewise, since the final write
    // below targets it directly. src1 has no such restriction below - it's
    // always staged through host memory anyway (transposed + tile-faced),
    // so a genuine sliced view is fine there and handled via the ordinary
    // get_tensor/set_tensor path (see ggml_backend_ttnn_locate).
    auto weight_located = ggml_backend_ttnn_locate(src0->buffer, src0);
    GGML_ASSERT(weight_located.offset == 0 &&
                "ggml-ttnn: mul_mat weight (src0) must not be a nonzero-offset view");
    auto weight_buf = weight_located.buffer;
    auto dst_located = ggml_backend_ttnn_locate(dst->buffer, dst);
    GGML_ASSERT(dst_located.offset == 0 && ggml_nbytes(dst) == dst_located.buffer->size() &&
                "ggml-ttnn: mul_mat dst must not be a view");
    auto dst_buf = dst_located.buffer;

    // The per-tensor weight scale (quantize_i2_s / PORTING_PLAN.md sec 10)
    // sits in the last 4 of the packed blob's 32 trailing bytes; reading the
    // whole (small) blob to host is always safe here (offset 0, full size -
    // see the buffer-type comment above), unlike a sub-range read.
    std::vector<uint8_t> weight_host(weight_buf->size());
    ggml_backend_ttnn_read_whole(weight_buf, weight_host.data());
    float scale;
    memcpy(&scale, weight_host.data() + (size_t) N * K / 4, sizeof(float));

    // Activation, transposed to [K,M_padded] (K outer) and tile-faced - see
    // the kernel README for why matmul_tiles needs it this way round. Goes
    // through the generic get_tensor path (not a raw whole-buffer read)
    // specifically so a sliced (nonzero-offset) view works here. Columns
    // [M, M_padded) are left zero (act_transposed's sized-constructor
    // value-initializes them) - padding rows of an all-zero activation, so
    // the kernel's extra output rows for them are simply zero times the
    // ternary weight, discarded below rather than written to dst.
    std::vector<float> act_host(M * K);
    ggml_backend_ttnn_buffer_get_tensor(src1->buffer, src1, act_host.data(), 0, (size_t) M * K * sizeof(float));
    std::vector<bfloat16> act_transposed(K * M_padded);
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t k = 0; k < K; k++) {
            act_transposed[k * M_padded + m] = bfloat16(act_host[m * K + k]);
        }
    }
    std::vector<bfloat16> act_tiled = tilize_nfaces(act_transposed, K, M_padded);

    auto mesh_device = ggml_backend_ttnn_get_mesh_device();
    const uint32_t single_tile_size = sizeof(bfloat16) * tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;

    ttd::DeviceLocalBufferConfig tile_dram_cfg{
        /* .page_size   = */ single_tile_size, /* .buffer_type = */ tt::tt_metal::BufferType::DRAM};
    auto act_dram = ttd::MeshBuffer::create(
        ttd::ReplicatedBufferConfig{ /* .size = */ sizeof(bfloat16) * act_tiled.size()}, tile_dram_cfg, mesh_device.get());
    auto out_dram = ttd::MeshBuffer::create(
        ttd::ReplicatedBufferConfig{ /* .size = */ (size_t) single_tile_size * Nt * Mt}, tile_dram_cfg, mesh_device.get());

    tt::tt_metal::Program program{};
    tt::DataFormat bf16_fmt = tt::DataFormat::Float16_b;

    // Multi-core dispatch (PORTING_PLAN.md sec 25): partition the N-tile
    // dimension across the device's available cores instead of running
    // everything on a single core. N-tile is the natural split axis for
    // this kernel - the per-N-tile chunked DRAM weight read (sec 20)
    // already loops over nt, so a core just gets a contiguous sub-range of
    // that loop instead of the whole thing, computing complete output
    // tiles independently with no cross-core communication (each core's
    // reader re-reads the full activation redundantly, same
    // "redundant but simple" tradeoff already used elsewhere in this
    // kernel - K and M are not split, every core does the full K-depth
    // accumulation for its own N-tiles). split_work_to_cores caps the
    // number of cores actually used at Nt when Nt is small (e.g. a single
    // N-tile matmul still runs on exactly one core).
    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, nt_per_core_1, nt_per_core_2] =
        tt::tt_metal::split_work_to_cores(core_grid, Nt);

    // Sized for the full Kt-tile queue: the compute kernel's Phase 1
    // (kernels/ternary_matmul/compute/mm.cpp, PORTING_PLAN.md sec 24)
    // unpacks all of this (mt, nt)'s Kt weight tiles via the SFPU before
    // Phase 2's matmul accumulation starts draining them - unlike every
    // earlier version of this kernel, cb_in0 now needs to hold Kt tiles at
    // once, not just 1-4. For K=2560 (Kt=80) that's ~160KB, still a small
    // fraction of L1 alongside everything else (contrast sec 20's old
    // whole-blob design, which needed multiple MB).
    uint32_t cb_in0 = tt::CBIndex::c_0;
    tt::tt_metal::CreateCircularBuffer(
        program, all_cores,
        tt::tt_metal::CircularBufferConfig(Kt * single_tile_size, {{cb_in0, bf16_fmt}}).set_page_size(cb_in0, single_tile_size));
    // Sized to Kt tiles, same reasoning as cb_in0 above: the reader pushes
    // all Kt activation tiles for a given (mt, nt) before the compute
    // kernel's Phase 2 (the only phase that drains cb_in1) even starts -
    // Phase 1 comes first in program order and only touches cb_raw/cb_in0.
    // A double-buffered (2-tile) cb_in1 deadlocks once Kt > 2 tiles worth of
    // reader pushes are needed before Phase 2 begins draining: the reader
    // blocks pushing this superblock's activation tiles into a full cb_in1,
    // which stalls it from ever reaching the *next* superblock's cb_raw
    // push, which Phase 1 is waiting on to proceed - a three-way circular
    // wait (reader -> cb_in1 -> Phase 2 -> Phase 1 -> cb_raw -> reader).
    uint32_t cb_in1 = tt::CBIndex::c_1;
    tt::tt_metal::CreateCircularBuffer(
        program, all_cores,
        tt::tt_metal::CircularBufferConfig(Kt * single_tile_size, {{cb_in1, bf16_fmt}}).set_page_size(cb_in1, single_tile_size));
    uint32_t cb_out = tt::CBIndex::c_16;
    tt::tt_metal::CreateCircularBuffer(
        program, all_cores,
        tt::tt_metal::CircularBufferConfig(2 * single_tile_size, {{cb_out, bf16_fmt}}).set_page_size(cb_out, single_tile_size));
    // Reader kernel streams one N-tile's packed weight row-block at a time
    // (see kernels/ternary_matmul/dataflow/reader_ternary_mm.cpp) rather than
    // keeping the whole N*K/4-byte blob resident in L1 - bounds this CB's
    // size to a single N-tile regardless of N, which is what makes real
    // model-sized weight matrices fit.
    uint32_t cb_scratch = tt::CBIndex::c_2;
    uint32_t weight_chunk_size = TILE_DIM * (K / 4);
    tt::tt_metal::CreateCircularBuffer(
        program, all_cores,
        tt::tt_metal::CircularBufferConfig(weight_chunk_size, {{cb_scratch, tt::DataFormat::UInt8}})
            .set_page_size(cb_scratch, weight_chunk_size));
    // One shared raw-byte tile per superblock (PORTING_PLAN.md sec 24): the
    // reader gathers each superblock's packed bytes into tile-face order but
    // does not decode them; the compute kernel's SFPU reads this same tile
    // once per lane (4 times) to extract each lane's 2-bit codes. UInt16
    // (not UInt8) because the bitwise/shift SFPU ops this backend uses only
    // operate on Int32/UInt32/UInt16 tiles.
    uint32_t cb_raw = tt::CBIndex::c_3;
    tt::tt_metal::CreateCircularBuffer(
        program, all_cores,
        tt::tt_metal::CircularBufferConfig(single_tile_size, {{cb_raw, tt::DataFormat::UInt16}})
            .set_page_size(cb_raw, single_tile_size));

    std::vector<uint32_t> reader_compile_args;
    tt::tt_metal::TensorAccessorArgs(*weight_buf).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(*act_dram).append_to(reader_compile_args);
    auto reader_id = tt::tt_metal::CreateKernel(
        program, TERNARY_MATMUL_KERNEL_DIR "dataflow/reader_ternary_mm.cpp", all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    std::vector<uint32_t> writer_compile_args;
    tt::tt_metal::TensorAccessorArgs(*out_dram).append_to(writer_compile_args);
    auto writer_id = tt::tt_metal::CreateKernel(
        program, TERNARY_MATMUL_KERNEL_DIR "dataflow/writer_ternary_mm.cpp", all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = writer_compile_args});

    // Nt is a runtime arg (below), not a compile-time one: each core's
    // share of the N-tile range (nt_per_core_1 vs nt_per_core_2) can differ,
    // but a single CreateKernel call compiles one binary shared by every
    // core in all_cores - only Mt/Kt are uniform across cores.
    std::vector<uint32_t> compute_compile_args = {Mt, Kt};
    auto compute_id = tt::tt_metal::CreateKernel(
        program, TERNARY_MATMUL_KERNEL_DIR "compute/mm.cpp", all_cores,
        tt::tt_metal::ComputeConfig{.math_fidelity = tt::tt_metal::MathFidelity::HiFi4, .compile_args = compute_compile_args});

    // Assign each core a contiguous, non-overlapping slice [nt_start,
    // nt_start + nt_count) of the global N-tile range - core_group_1 and
    // core_group_2 partition all_cores with no overlap (core_group_2 is
    // empty when Nt divides evenly), so iterating both unconditionally and
    // advancing nt_start by each core's nt_count covers every N-tile
    // exactly once.
    uint32_t nt_start = 0;
    for (const auto & group : {std::make_pair(core_group_1, nt_per_core_1), std::make_pair(core_group_2, nt_per_core_2)}) {
        const auto & cores = group.first;
        uint32_t nt_count = group.second;
        for (const auto & range : cores.ranges()) {
            for (const auto & c : range) {
                tt::tt_metal::SetRuntimeArgs(
                    program, reader_id, c,
                    {(uint32_t) weight_buf->address(), (uint32_t) act_dram->address(), Mt, Kt, nt_count, K, nt_start});
                tt::tt_metal::SetRuntimeArgs(
                    program, writer_id, c, {(uint32_t) out_dram->address(), Mt, nt_count, nt_start});
                tt::tt_metal::SetRuntimeArgs(program, compute_id, c, {nt_count});
                nt_start += nt_count;
            }
        }
    }

    ttd::MeshCommandQueue & cq = mesh_device->mesh_command_queue();
    ttd::MeshWorkload workload;
    ttd::MeshCoordinateRange device_range(mesh_device->shape());

    ttd::EnqueueWriteMeshBuffer(cq, act_dram, act_tiled, /*blocking=*/false);
    workload.add_program(device_range, std::move(program));
    ttd::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<bfloat16> result_tiled((size_t) Nt * Mt * TILE_DIM * TILE_DIM);
    ttd::EnqueueReadMeshBuffer(cq, result_tiled, out_dram, /*blocking=*/true);
    // N-outer/M_padded-inner (the kernel's own tile-grid order, matching the
    // writer's n*Mt+m page layout) - see the comment above this function
    // for why this is NOT yet ggml's own dst byte layout.
    std::vector<bfloat16> result = untilize_nfaces(result_tiled, N, M_padded);

    // Transpose into ggml's dst convention (M-outer/N-inner, ne0=N fastest)
    // while applying the per-tensor weight scale - device output is the
    // unscaled ternary dot product, not rescaled per-element in-kernel. Only
    // the real M rows are written; columns [M, M_padded) were all-zero
    // padding activation rows (see above) and are dropped here rather than
    // written to dst.
    std::vector<float> out_host(N * M);
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t m = 0; m < M; m++) {
            out_host[m * N + n] = (float) result[n * M_padded + m] * scale;
        }
    }

    ggml_backend_ttnn_write_whole(dst_buf, out_host.data());
}

// GGML_OP_SET_ROWS: writes rows of `src0` (F32) into `dst` at the row
// positions given by `src1` (I32/I64 indices) - the KV-cache-write op
// (`ggml_set_rows(ctx, a, b, c)` returns `view(a)`, so `dst` here is always
// a whole-tensor view of the real destination, `dst->view_src`). Mirrors
// ggml-cpu's own `ggml_compute_forward_set_rows_f32` (ggml-cpu/ops.cpp)
// element-for-element - same loop structure, same broadcast rules - just
// operating on host-side copies of this backend's buffers instead of the
// CPU's own tensor memory, and using ggml core's portable `from_float_ref`
// (no need to link ggml-cpu just for its SIMD one) to convert into
// whatever type the destination actually stores (F16 for a typical KV
// cache, but not assumed - whatever supports_op() already checked).
template <typename idx_t>
static void ggml_backend_ttnn_set_rows_apply(
        uint8_t * a_host, const struct ggml_tensor * a,
        const uint8_t * rows_host, const struct ggml_tensor * src0,
        const uint8_t * idx_host, const struct ggml_tensor * src1,
        ggml_from_float_t from_float) {
    const int64_t nc   = src0->ne[0];
    const int64_t nr   = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne1  = a->ne[1];
    GGML_ASSERT(a->ne[0] == nc && a->ne[2] == ne02 && a->ne[3] == ne03);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i = 0; i < nr; i++) {
                const int64_t i12 = i03 % ne12;
                const int64_t i11 = i02 % ne11;
                idx_t i1;
                memcpy(&i1, idx_host + i * src1->nb[0] + i11 * src1->nb[1] + i12 * src1->nb[2], sizeof(idx_t));
                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                const float * src_row = (const float *) (rows_host + i * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);
                void * dst_row = a_host + (size_t) i1 * a->nb[1] + i02 * a->nb[2] + i03 * a->nb[3];
                from_float(src_row, dst_row, nc);
            }
        }
    }
}

static void ggml_backend_ttnn_compute_set_rows(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // new row data, F32
    const struct ggml_tensor * src1 = dst->src[1]; // row indices, I32/I64
    struct ggml_tensor * a = dst->view_src;        // real destination (e.g. a KV-cache tensor)
    GGML_ASSERT(a != NULL && "ggml-ttnn: SET_ROWS dst should always be a view of its destination (ggml_set_rows)");

    // Same whole-buffer-only pattern as everywhere else in this backend
    // (sec 9/16): `a` must be the sole occupant of its buffer (true for a
    // real KV-cache tensor, never itself a sub-view) so this is a plain
    // read-modify-write over the entire thing, not an interior access.
    auto a_located = ggml_backend_ttnn_locate(a->buffer, a);
    GGML_ASSERT(a_located.offset == 0 && ggml_nbytes(a) == a_located.buffer->size() &&
                "ggml-ttnn: SET_ROWS destination must be the sole occupant of its buffer");

    const struct ggml_type_traits * type_traits = ggml_get_type_traits(a->type);
    GGML_ASSERT(type_traits->from_float_ref != NULL &&
                "ggml-ttnn: no from_float conversion for this dst type - supports_op() should not have allowed this");

    std::vector<uint8_t> a_host(a_located.buffer->size());
    ggml_backend_ttnn_read_whole(a_located.buffer, a_host.data());

    std::vector<uint8_t> rows_host(ggml_nbytes(src0));
    ggml_backend_ttnn_buffer_get_tensor(src0->buffer, src0, rows_host.data(), 0, rows_host.size());

    std::vector<uint8_t> idx_host(ggml_nbytes(src1));
    ggml_backend_ttnn_buffer_get_tensor(src1->buffer, src1, idx_host.data(), 0, idx_host.size());

    if (src1->type == GGML_TYPE_I64) {
        ggml_backend_ttnn_set_rows_apply<int64_t>(
            a_host.data(), a, rows_host.data(), src0, idx_host.data(), src1, type_traits->from_float_ref);
    } else {
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        ggml_backend_ttnn_set_rows_apply<int32_t>(
            a_host.data(), a, rows_host.data(), src0, idx_host.data(), src1, type_traits->from_float_ref);
    }

    ggml_backend_ttnn_write_whole(a_located.buffer, a_host.data());
}

static enum ggml_status ggml_backend_ttnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_ttnn_compute_mul_mat(node);
                break;
            case GGML_OP_SET_ROWS:
                ggml_backend_ttnn_compute_set_rows(node);
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
    // GGML_OP_NONE is a real leaf (e.g. a weight or KV-cache tensor already
    // sitting in this backend's buffer, no view) - the scheduler queries
    // supports_op() for these too (ggml_backend_sched_backend_from_buffer),
    // and they need no compute. VIEW/RESHAPE/TRANSPOSE/PERMUTE are real graph
    // nodes (not pre-allocated leaves) whose own dst is a view of one of
    // their sources - in ggml these are always pure metadata ops (new
    // ne[]/nb[]/view_src/view_offs over the *same* underlying data, see
    // ggml_view_tensor/ggml_reshape/ggml_permute/ggml_transpose in ggml.c),
    // never data movement, so they're safe here now that the buffer type
    // can hold a view at all (sec 16) - graph_compute() already treats all
    // four as no-ops, this just lets the scheduler actually route them to
    // this device instead of aborting on a KV-cache-derived view (found
    // when enabling SET_ROWS surfaced the very next node needing this, a
    // plain VIEW of the KV cache for attention).
    if (op->op == GGML_OP_NONE || op->op == GGML_OP_VIEW || op->op == GGML_OP_RESHAPE ||
        op->op == GGML_OP_TRANSPOSE || op->op == GGML_OP_PERMUTE) {
        return true;
    }
    if (op->op == GGML_OP_MUL_MAT) {
        const struct ggml_tensor * src0 = op->src[0];
        const struct ggml_tensor * src1 = op->src[1];
        return src0->type == GGML_TYPE_I2_S && src1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
               ggml_backend_ttnn_mul_mat_shape_ok(src0, src1);
    }
    if (op->op == GGML_OP_SET_ROWS) {
        // The KV-cache write op (ggml_set_rows(ctx, a, b, c), dst = view(a)):
        // b (op->src[0]) must be F32 and c (op->src[1]) I32/I64 - the same
        // restriction ggml-cpu's own implementation has (ggml-cpu/ops.cpp,
        // ggml_compute_forward_set_rows) - and the destination type (op->type,
        // same as a's) needs a from_float conversion to exist at all.
        const struct ggml_tensor * src0 = op->src[0];
        const struct ggml_tensor * src1 = op->src[1];
        return src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32) &&
               ggml_get_type_traits(op->type)->from_float_ref != NULL;
    }
    return false;
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
