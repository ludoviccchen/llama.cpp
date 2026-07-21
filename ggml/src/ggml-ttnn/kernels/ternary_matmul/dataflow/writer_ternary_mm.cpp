// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Adapted from
// tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp.
// Writes the raw dot-product result tile(s); the combined (weight_scale *
// activation_scale) rescale is applied by the caller, not in-kernel (see
// PORTING_PLAN.md) - this writer just moves whatever the compute kernel
// packed out.
//
// One real difference from the reference: our dst is [N, M] (N outer, M
// inner - ggml's own mul_mat convention, see PORTING_PLAN.md sec 11/13),
// not the reference's [M, N]. The compute kernel still *produces* tiles in
// (mt outer, nt inner) order (matches the reader's push order), so the
// consumption loop below is unchanged - but the DRAM page index must be
// `n*Mt + m` (N-tile major) to match how the host's untilize_nfaces(N, M)
// expects tiles laid out, not the reference's `m*Nt + n`.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = 16;

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, dst_addr);

    for (uint32_t m = 0; m < Mt; ++m) {
        for (uint32_t n = 0; n < Nt; ++n) {
            cb_wait_front(cb_id_out0, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
            noc_async_write_page(n * Mt + m, s, l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out0, 1);
        }
    }
}
