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
// Ternary packing (see BitNet.cpp PORTING_PLAN.md sec 10): byte
// packed[row*32+c] holds four 2-bit codes for the four K-tile "lanes" 0..3
// at that (row,c) position, in bits [7:6],[5:4],[3:2],[1:0] respectively.
// This is the layout verified against the real CPU inference dot product
// (ggml_vec_dot_i2_i8_s_1x1), not quantize_i2_s's own (differently-laid-out)
// packer. code -> value: 0 -> -1, 1 -> 0, 2 -> +1, 3 unused.
//
// Tile-face layout (tech_reports/tensor_layouts/tensor_layouts.md): each
// 32x32 tile is 4 faces of 16x16, stored face0(top-left)->face1(top-right)
// ->face2(bottom-left)->face3(bottom-right), row-major within each face.
//
// Scope: single N-tile weight (N=32). The packed I2_S format stores one
// scale for the *whole* tensor, not per N-tile, so multi-N-tile support
// needs the scale handled separately first - not attempted here. The whole
// packed blob for this N-tile (N*K/4 + 32 bytes) is a single DRAM page,
// read once per M-tile row and kept resident in a scratch L1 CB for the
// whole Kt loop (avoids re-reading it once per K-tile).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t weight_addr = get_arg_val<uint32_t>(0);
    uint32_t act_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);  // must be 1 - see scope note above

    constexpr uint32_t cb_id_in0 = 0;       // unpacked ternary weight tiles (bf16)
    constexpr uint32_t cb_id_in1 = 1;       // activation tiles (bf16)
    constexpr uint32_t cb_id_scratch = 2;   // raw packed weight bytes for this M-tile row

    constexpr uint32_t TILE_DIM = 32;
    constexpr uint32_t FACE_DIM = 16;
    constexpr uint32_t FACE_HW = FACE_DIM * FACE_DIM;          // 256
    constexpr uint32_t TILE_ROW_STRIDE = FACE_DIM * TILE_DIM;  // 512

    // bfloat16 bit patterns (top 16 bits of IEEE754 float32): +1.0 = 0x3F80,
    // -1.0 = 0xBF80, 0.0 = 0x0000.
    constexpr uint16_t BF16_POS_ONE = 0x3F80;
    constexpr uint16_t BF16_NEG_ONE = 0xBF80;
    constexpr uint16_t BF16_ZERO = 0x0000;

    constexpr auto weight_args = TensorAccessorArgs<0>();
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr);
    constexpr auto act_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    const auto act_accessor = TensorAccessor(act_args, act_addr);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        // Read this M-tile row's whole packed ternary weight blob once.
        cb_reserve_back(cb_id_scratch, 1);
        uint32_t scratch_addr = get_write_ptr(cb_id_scratch);
        noc_async_read_page(mt, weight_accessor, scratch_addr);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint8_t* packed = (volatile tt_l1_ptr uint8_t*)scratch_addr;

        for (uint32_t kt = 0; kt < Kt; kt++) {
            // Unpack this K-tile's 32x32 ternary weight block directly into
            // tile-face order.
            cb_reserve_back(cb_id_in0, 1);
            uint32_t w_tile_addr = get_write_ptr(cb_id_in0);
            volatile tt_l1_ptr uint16_t* w_tile = (volatile tt_l1_ptr uint16_t*)w_tile_addr;

            for (uint32_t row = 0; row < TILE_DIM; row++) {
                for (uint32_t c = 0; c < TILE_DIM; c++) {
                    uint8_t byte = packed[row * TILE_DIM + c];
                    uint8_t code = (byte >> (6 - 2 * kt)) & 0x3;
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
            // tile-faced on the host - nothing special to unpack).
            uint32_t a_tile_index = mt * Kt + kt;
            cb_reserve_back(cb_id_in1, 1);
            uint32_t a_addr = get_write_ptr(cb_id_in1);
            noc_async_read_page(a_tile_index, act_accessor, a_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in1, 1);
        }

        cb_push_back(cb_id_scratch, 1);
        cb_pop_front(cb_id_scratch, 1);
    }
}
