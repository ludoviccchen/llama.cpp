// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Standard blocked outer-product matmul over tiles, unchanged from
// tt_metal/programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp.
// All of the ternary-specific work happens in the reader kernel, which
// unpacks I2_S weight tiles into plain bf16 before this kernel ever sees
// them - so the FPU's normal matmul_tiles op applies unmodified. Since
// weight values are exactly {-1.0, 0.0, +1.0}, matmul_tiles' per-element
// "multiply" is bit-exact to a conditional negate/pass-through/zero - the
// arithmetic already *is* an add/sub-style accumulation, just executed on
// the FPU's existing multiply-accumulate path rather than a hand-written
// SFPU one.

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "hostdevcommon/kernel_structs.h"

using std::uint32_t;

void kernel_main() {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);
    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);

                matmul_tiles(cb_in0, cb_in1, 0, 0, 0);

                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            tile_regs_release();
        }
    }
}
