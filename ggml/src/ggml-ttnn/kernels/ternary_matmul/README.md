# Ternary matmul kernels (Option B)

Custom TT-Metalium reader/compute/writer kernel triad for I2_S ternary
matmul, consuming packed 2-bit weights directly rather than dequantizing
them to bf16 on upload (Option A, `ggml-ttnn.cpp`'s current `graph_compute`
path). See `PORTING_PLAN.md` sec 13/14 for the full design writeup and
validation results.

- `dataflow/reader_ternary_mm.cpp` - pure data movement: gathers each
  superblock's packed I2_S weight bytes on-device into the 32x32 tile-face
  layout, *undecoded* (PORTING_PLAN.md sec 24 moved the actual 2-bit decode
  to the compute kernel's SFPU), and reads activation tiles (dense bf16,
  tile-faced on the host - nothing ternary about them).
- `compute/mm.cpp` - two phases per (mt, nt) (PORTING_PLAN.md sec 24):
  Phase 1 decodes this tile's Kt ternary K-tiles from the reader's raw-byte
  tiles using SFPU bitwise/shift/typecast ops (a vector engine - the earlier
  from-sec-13 design did this decode on the reader's scalar RISC-V core
  instead, which doesn't scale to real dimensions under ttsim - see sec
  22/23); Phase 2 is the stock FPU `matmul_tiles` accumulation (identical to
  `tt_metal/programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp`),
  unchanged from every earlier version of this kernel. Since ternary weight
  values are exactly {-1, 0, +1}, the FPU's "multiply" in Phase 2 is
  bit-exact to a conditional negate/pass-through/zero of the activation -
  already an add/sub-style accumulation, just executed on the existing
  multiply-accumulate datapath.
- `dataflow/writer_ternary_mm.cpp` - stock tile writer, adapted for our
  `[N, M]` (not the stock example's `[M, N]`) output convention: DRAM page
  index is `n*Mt + m` (N-tile major), not `m*Nt + n`. Writes the *unscaled*
  dot product; the caller applies the per-tensor weight scale afterward
  (not baked in per-element here).

All three kernels now run on multiple cores at once (PORTING_PLAN.md sec
25): `ggml-ttnn.cpp` partitions the tensor's N-tile range across the
device's available cores via `split_work_to_cores`, and each core runs
this exact same triad independently on its own slice `[nt_start,
nt_start+nt_count)` - complete output tiles, full K-depth, no cross-core
communication. `Nt` in the reader/compute kernels is each core's *local*
tile count, not the whole tensor's; the reader and writer additionally
take an `nt_start` runtime arg to translate local tile indices back to
global ones when addressing the (tensor-wide, cross-core-shared) weight
and output DRAM buffers.

## Current scope

Handles arbitrary Mt/Kt/Nt (multiple M-tiles, N-tiles, and K
super-blocks/row - see PORTING_PLAN.md sec 14). The reader streams one
N-tile's packed weight row-block (32 rows x K/4 bytes) into a scratch L1 CB
at a time, refreshed per (mt, nt) pair, rather than keeping the whole
N*K/4-byte blob resident - see PORTING_PLAN.md sec 20. This bounds L1 usage
to a fixed size per weight tensor regardless of N (verified against a real
2B-param model's projection matrices: the old whole-blob approach needed up
to 4.3MB resident per tensor, well over a Blackhole core's ~1.5MB L1; the
chunked version needs 20-54KB). The tradeoff is redundant DRAM reads (the
weight is re-fetched once per M-tile, since it doesn't vary with mt) -
acceptable since hardware performance work is explicitly out of scope until
real silicon is available (PORTING_PLAN.md sec 8).

## Status

Wired into `ggml-ttnn.cpp`'s `supports_op`/`graph_compute`, replacing
Option A entirely for `GGML_OP_MUL_MAT` (see PORTING_PLAN.md sec 15) -
`compute_mul_mat` dispatches this exact kernel triad via a real TT-Metalium
`Program`/`Kernel`, reusing the weight tensor's own resident `MeshBuffer`
directly (no re-upload). TT-NN is no longer linked by this backend at all.
Re-verified under ttsim through the real `ggml_backend_sched`/
`graph_compute` path, not just a hand-rolled dispatch.

The SFPU-based decode (sec 24 above) measures ~6x faster than the prior
scalar-only version at real projection-matrix dimensions (K=N=2560: ~154s
vs ~930s under ttsim) with bit-identical output - a real architectural
win, not just a tidy-up. Multi-core dispatch (sec 25) measures another
~10-15x on top of that (K=N=2560: ~9.9s), ~94x faster than sec 23 combined.
The `M % 32 == 0` gate (sec 21/22) has since been lifted (sec 26) - the
kernel triad itself needed no changes for this (it always operated on
whole tiles; `compute_mul_mat`'s M-padding logic, sec 22, handles the
rest) and is verified correct at every M tested. A real end-to-end model
run currently hits a separate, pre-existing `ggml-ttnn.cpp`-level gap this
surfaces for the first time (`mul_mat dst must not be a view`) - not a
kernel-triad issue, see PORTING_PLAN.md sec 26 for the full story and
current status.
