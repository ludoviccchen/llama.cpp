# Ternary matmul kernels (Option B)

Custom TT-Metalium reader/compute/writer kernel triad for I2_S ternary
matmul, consuming packed 2-bit weights directly rather than dequantizing
them to bf16 on upload (Option A, `ggml-ttnn.cpp`'s current `graph_compute`
path). See `PORTING_PLAN.md` sec 13/14 for the full design writeup and
validation results.

- `dataflow/reader_ternary_mm.cpp` - unpacks packed I2_S weight bytes
  on-device into the 32x32 tile-face layout the FPU expects, and reads
  activation tiles (dense bf16, tile-faced on the host - nothing ternary
  about them).
- `compute/mm.cpp` - unmodified stock FPU `matmul_tiles` accumulation
  (identical to
  `tt_metal/programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp`).
  All the ternary-specific work is in the reader; since ternary weight
  values are exactly {-1, 0, +1}, the FPU's "multiply" is bit-exact to a
  conditional negate/pass-through/zero of the activation - already an
  add/sub-style accumulation, just executed on the existing
  multiply-accumulate datapath.
- `dataflow/writer_ternary_mm.cpp` - stock tile writer, adapted for our
  `[N, M]` (not the stock example's `[M, N]`) output convention: DRAM page
  index is `n*Mt + m` (N-tile major), not `m*Nt + n`. Writes the *unscaled*
  dot product; the caller applies the per-tensor weight scale afterward
  (not baked in per-element here).

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
