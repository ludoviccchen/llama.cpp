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

// -----------------------------------------------------------------------
// Buffer type / buffer
//
// Placeholder: allocates plain host memory. Replaced with TT-Metalium
// device DRAM allocation once weight upload is implemented.
// -----------------------------------------------------------------------

static const char * ggml_backend_ttnn_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "TT_Metalium";
}

static void ggml_backend_ttnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * ggml_backend_ttnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static void ggml_backend_ttnn_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memset((char *) tensor->data + offset, value, size);
}

static void ggml_backend_ttnn_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy((char *) tensor->data + offset, data, size);
}

static void ggml_backend_ttnn_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

static void ggml_backend_ttnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_ttnn_buffer_i = {
    /* .free_buffer     = */ ggml_backend_ttnn_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_ttnn_buffer_get_base,
    /* .init_tensor     = */ NULL,
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
    void * data = malloc(size);
    if (data == NULL) {
        return NULL;
    }
    return ggml_backend_buffer_init(buft, ggml_backend_ttnn_buffer_i, data, size);
}

static size_t ggml_backend_ttnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 32;
}

static bool ggml_backend_ttnn_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
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

struct ggml_backend_ttnn_context {
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device;
};

static const char * ggml_backend_ttnn_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "TT_Metalium";
}

static void ggml_backend_ttnn_free(ggml_backend_t backend) {
    ggml_backend_ttnn_context * ctx = (ggml_backend_ttnn_context *) backend->context;
    if (ctx->mesh_device) {
        ctx->mesh_device->close();
    }
    delete ctx;
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

    ggml_backend_ttnn_context * ctx = new ggml_backend_ttnn_context;
    // Opens device 0 the standard tt-metal way: silicon vs. ttsim is chosen
    // by the driver via TT_METAL_SIMULATOR, never branched on here.
    ctx->mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);

    ggml_backend_t backend = new ggml_backend {
        /* .guid     = */ ggml_backend_ttnn_guid(),
        /* .iface    = */ ggml_backend_ttnn_i,
        /* .device   = */ dev,
        /* .context  = */ ctx,
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
