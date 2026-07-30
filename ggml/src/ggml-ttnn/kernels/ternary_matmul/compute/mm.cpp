// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Two-phase compute kernel (PORTING_PLAN.md sec 24): for each (mt, nt),
// first unpacks this weight tile's Kt ternary K-tiles from raw packed bytes
// (cb_raw, one shared byte-per-element tile per superblock, produced by the
// reader - see dataflow/reader_ternary_mm.cpp) into decoded bf16 tiles
// (cb_in0) using the SFPU - a vector engine, unlike the scalar RISC-V
// data-movement core the decode used to run on (sec 22/23) - then runs the
// stock FPU matmul_tiles accumulation over cb_in0/cb_in1, unchanged from
// every earlier version of this kernel.
//
// Phase 1 (unpack, per superblock sb, 4 lanes lane=0..3, kt = sb*4+lane):
//   copy_tile(cb_raw) -> DST any lane's decode starts from the *same* raw
//   byte tile (the reader only produced it once per superblock, not once
//   per lane); right_shift_tile(6-2*lane) then bitwise_and_tile(0x3)
//   isolates that lane's 2-bit code in the low bits; typecast_tile<UInt16,
//   Float16_b> converts the integer code (0/1/2) to a float; sub_unary_tile
//   subtracts 1.0 to land on the actual ternary value (-1/0/+1) - the
//   code->value mapping (PORTING_PLAN.md sec 10) is exactly `value = code -
//   1`, so no lookup table is needed on this path, just one subtract.
//   Result is packed into cb_in0.
// Phase 2 (matmul, per nt, after all Kt tiles are in cb_in0): identical to
//   every earlier version of this kernel - accumulate Kt tile products via
//   matmul_tiles into one DST slot, then pack the (mt, nt) output tile.
//
// The two phases use the compute engine in different modes (SFPU eltwise
// unary vs. FPU matmul) and read different-format source CBs (cb_raw is
// UInt16, cb_in1 is Float16_b) - reconfig_data_format_srca /
// copy_tile_to_dst_init_short / matmul_init are called at each transition
// to reprogram the unpack/math hardware state accordingly (same pattern
// used by in-tree kernels that mix op types within one compute kernel, e.g.
// ttnn/cpp/ttnn/operations/moreh/moreh_sum's h-reduction kernel).
//
// cb_in0 must hold all Kt tiles produced by Phase 1 before Phase 2 starts
// draining it (host-side CB sized to Kt tiles, ggml-ttnn.cpp) - Phase 1 and
// Phase 2 do not interleave within one (mt, nt): matmul_tiles' accumulation
// holds an open tile_regs session across the whole Kt loop, which cannot be
// interrupted by Phase 1's own per-lane tile_regs sessions without
// corrupting the in-progress accumulation.

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/bitwise.h"
#include "api/compute/eltwise_unary/shift.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "hostdevcommon/kernel_structs.h"

using std::uint32_t;

void kernel_main() {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);
    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;  // decoded ternary weight tiles (bf16), produced by Phase 1
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;  // activation tiles (bf16), from the reader
    constexpr tt::CBIndex cb_raw = tt::CBIndex::c_3;  // one shared raw-byte tile per superblock (UInt16), from the reader
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // fp32 bit pattern for the scalar operand sub_unary_tile takes.
    constexpr uint32_t FP32_ONE = 0x3F800000;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1);

    const uint32_t Sb = Kt / 4;  // superblocks per (mt, nt); Kt % 4 == 0 always (see reader kernel header)

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            // --- Phase 1: unpack this (mt, nt)'s Kt weight tiles via SFPU ---
            reconfig_data_format_srca(cb_raw);
            copy_tile_to_dst_init_short(cb_raw);
            for (uint32_t sb = 0; sb < Sb; ++sb) {
                cb_wait_front(cb_raw, 1);
                for (uint32_t lane = 0; lane < 4; ++lane) {
                    const uint32_t shift = 6 - 2 * lane;

                    tile_regs_acquire();
                    copy_tile(cb_raw, 0, 0);

                    right_shift_tile_init();
                    right_shift_tile<DataFormat::UInt16>(0, shift);

                    bitwise_and_tile_init();
                    bitwise_and_tile<DataFormat::UInt16>(0, 0x3);

                    typecast_tile_init<(uint32_t)DataFormat::UInt16, (uint32_t)DataFormat::Float16_b>();
                    typecast_tile<(uint32_t)DataFormat::UInt16, (uint32_t)DataFormat::Float16_b>(0);

                    binop_with_scalar_tile_init();
                    sub_unary_tile(0, FP32_ONE);  // code (0/1/2) -> value (-1/0/+1)

                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_in0, 1);
                    pack_tile(0, cb_in0);
                    cb_push_back(cb_in0, 1);
                    tile_regs_release();
                }
                cb_pop_front(cb_raw, 1);
            }

            // --- Phase 2: stock FPU matmul accumulation over Kt tiles ---
            reconfig_data_format_srca(cb_in1);
            matmul_init(cb_in0, cb_in1);

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
