// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads packed I2_S ternary weight data + bf16 activation tiles from DRAM.
// The activation is a normal dense tensor, already tile-faced on the host
// (like any other ggml-ttnn tensor); only the weight is special: it stays
// packed (2 bits/element) in DRAM and is unpacked *here*, on-device, into
// the 32x32 tile-face layout the FPU expects - this is the whole point of
// Option B versus Option A's dequantize-on-upload path.
//
// Ternary packing (see BitNet.cpp PORTING_PLAN.md sec 10): weight elements
// are packed as one flat sweep over all N*K elements, in 128-element
// super-blocks. Within super-block `sb` (elements [sb*128, sb*128+128)),
// byte packed[sb*32 + gp] holds four 2-bit codes for elements at local
// positions gp, gp+32, gp+64, gp+96 (gp = 0..31), in bits
// [7:6],[5:4],[3:2],[1:0] respectively. This is the layout verified against
// the real CPU inference dot product (ggml_vec_dot_i2_i8_s_1x1), not
// quantize_i2_s's own (differently-laid-out) packer. code -> value:
// 0 -> -1, 1 -> 0, 2 -> +1, 3 unused.
//
// A 32-wide K-tile `kt` sits inside super-block `kt/4` at lane `kt%4`
// (four 32-wide K-tiles share each 128-element super-block). Row `row`'s
// packed data starts at byte offset `row*(K/4)` within the whole tensor's
// packed blob (K/4 bytes/row, assuming K is a multiple of 128 - true for
// realistic transformer dims - so no super-block ever straddles a row).
//
// Tile-face layout (tech_reports/tensor_layouts/tensor_layouts.md): each
// 32x32 tile is 4 faces of 16x16, stored face0(top-left)->face1(top-right)
// ->face2(bottom-left)->face3(bottom-right), row-major within each face.
//
// Scope: the whole packed weight blob (N*K/4 + 32 bytes) is read once, as a
// single DRAM page, into a scratch L1 CB and kept resident for the entire
// kernel (weight doesn't vary with mt). This does not yet scale to
// real-model-sized weights - the whole blob must fit in L1 alongside
// everything else, so this is still a correctness vehicle, not a
// performance-ready design; chunked/streamed reads are follow-up work.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t weight_addr = get_arg_val<uint32_t>(0);
    uint32_t act_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t K = get_arg_val<uint32_t>(5);  // needed for the bytes-per-row stride (K/4)

    constexpr uint32_t cb_id_in0 = 0;      // unpacked ternary weight tiles (bf16)
    constexpr uint32_t cb_id_in1 = 1;      // activation tiles (bf16)
    constexpr uint32_t cb_id_scratch = 2;  // whole packed weight blob, resident for the kernel's life

    constexpr uint32_t TILE_DIM = 32;
    constexpr uint32_t FACE_DIM = 16;
    constexpr uint32_t FACE_HW = FACE_DIM * FACE_DIM;          // 256
    constexpr uint32_t TILE_ROW_STRIDE = FACE_DIM * TILE_DIM;  // 512

    // bfloat16 bit patterns (top 16 bits of IEEE754 float32): +1.0 = 0x3F80,
    // -1.0 = 0xBF80, 0.0 = 0x0000.
    constexpr uint16_t BF16_POS_ONE = 0x3F80;
    constexpr uint16_t BF16_NEG_ONE = 0xBF80;
    constexpr uint16_t BF16_ZERO = 0x0000;

    const uint32_t bytes_per_row = K / 4;

    constexpr auto weight_args = TensorAccessorArgs<0>();
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr);
    constexpr auto act_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    const auto act_accessor = TensorAccessor(act_args, act_addr);

    // Read the whole packed ternary weight blob once, up front - it's the
    // same for every mt/nt/kt visited below.
    cb_reserve_back(cb_id_scratch, 1);
    uint32_t scratch_addr = get_write_ptr(cb_id_scratch);
    noc_async_read_page(0, weight_accessor, scratch_addr);
    noc_async_read_barrier();
    cb_push_back(cb_id_scratch, 1);
    volatile tt_l1_ptr uint8_t* packed = (volatile tt_l1_ptr uint8_t*)scratch_addr;

    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            for (uint32_t kt = 0; kt < Kt; kt++) {
                // Unpack this (nt, kt) 32x32 ternary weight block directly
                // into tile-face order.
                cb_reserve_back(cb_id_in0, 1);
                uint32_t w_tile_addr = get_write_ptr(cb_id_in0);
                volatile tt_l1_ptr uint16_t* w_tile = (volatile tt_l1_ptr uint16_t*)w_tile_addr;

                uint32_t superblock = kt / 4;
                uint32_t lane = kt % 4;
                uint32_t shift = 6 - 2 * lane;

                for (uint32_t row = 0; row < TILE_DIM; row++) {
                    uint32_t global_row = nt * TILE_DIM + row;
                    uint32_t row_base = global_row * bytes_per_row + superblock * TILE_DIM;
                    for (uint32_t c = 0; c < TILE_DIM; c++) {
                        uint8_t byte = packed[row_base + c];
                        uint8_t code = (byte >> shift) & 0x3;
                        uint16_t bits = (code == 0) ? BF16_NEG_ONE : (code == 2) ? BF16_POS_ONE : BF16_ZERO;

                        uint32_t face_y = row / FACE_DIM;
                        uint32_t face_x = c / FACE_DIM;
                        uint32_t local_row = row % FACE_DIM;
                        uint32_t local_col = c % FACE_DIM;
                        uint32_t idx = face_y * TILE_ROW_STRIDE + face_x * FACE_HW + local_row * FACE_DIM + local_col;
                        w_tile[idx] = bits;
                    }
                }
                cb_push_back(cb_id_in0, 1);

                // Read the matching activation tile (dense bf16, already
                // tile-faced on the host - nothing special to unpack). The
                // activation tensor is [K, M] (K outer, M inner - see the
                // transpose note in test_ternary_matmul.cpp / PORTING_PLAN.md
                // sec 13), so its tile grid is K-tile-major: index =
                // kt*Mt + mt, not mt*Kt + kt (those coincide only when
                // Mt == 1, which is why this only broke once Mt > 1).
                // Constant across nt, but the compute kernel consumes one
                // push per (mt, nt, kt) visited, so it's re-read here each
                // time - same redundant-but-simple pattern as the stock
                // matmul_multi_core reader kernel this is modeled on.
                uint32_t a_tile_index = kt * Mt + mt;
                cb_reserve_back(cb_id_in1, 1);
                uint32_t a_addr = get_write_ptr(cb_id_in1);
                noc_async_read_page(a_tile_index, act_accessor, a_addr);
                noc_async_read_barrier();
                cb_push_back(cb_id_in1, 1);
            }
        }
    }

    cb_pop_front(cb_id_scratch, 1);
}
