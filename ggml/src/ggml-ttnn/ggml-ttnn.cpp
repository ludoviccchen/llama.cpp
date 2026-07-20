// Tenstorrent Blackhole backend for ggml, built on TT-Metalium / TT-NN.
//
// This backend opens a real TT-Metalium device (silicon or ttsim, selected
// the standard tt-metal way via TT_METAL_SIMULATOR - never branched on here)
// but does not offload any ops yet: supports_op() always returns false, so
// the graph scheduler never assigns this backend any op. DRAM-backed buffers
// and op offload land in follow-up increments.

#include "ggml-ttnn.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

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
    static std::shared_ptr<ttd::MeshDevice> mesh_device = ttd::MeshDevice::create_unit_mesh(0);
    return mesh_device;
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
    // A view shares its parent's device buffer at (usually non-zero,
    // non-whole-buffer) view_offs, which is exactly the unsafe access
    // pattern above - not yet supported.
    GGML_ASSERT(tensor->view_src == NULL && "ggml-ttnn: tensor views are not yet supported by the DRAM buffer type");

    ggml_backend_ttnn_buffer_context * ctx = (ggml_backend_ttnn_buffer_context *) buffer->context;
    auto mesh_device = ggml_backend_ttnn_get_mesh_device();
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
    /* .get_alloc_size = */ NULL,
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

static enum ggml_status ggml_backend_ttnn_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);
    GGML_UNUSED(cgraph);
    // supports_op() always returns false today, so the scheduler never
    // assigns this backend any op - this should be unreachable.
    GGML_ABORT("ggml-ttnn: graph_compute called with no ops offloaded yet");
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
    GGML_UNUSED(op);
    // No ops offloaded yet - see file header.
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
