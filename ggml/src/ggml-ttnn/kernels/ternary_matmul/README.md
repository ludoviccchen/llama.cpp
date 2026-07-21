# Ternary matmul kernels (Option B)

Custom TT-Metalium reader/compute/writer kernel triad for I2_S ternary
matmul, consuming packed 2-bit weights directly rather than dequantizing
them to bf16 on upload (Option A, `ggml-ttnn.cpp`'s current `graph_compute`
path). See `PORTING_PLAN.md` sec 13 for the full design writeup and
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
- `dataflow/writer_ternary_mm.cpp` - unmodified stock tile writer. Writes
  the *unscaled* dot product; the caller applies the per-tensor weight
  scale afterward (not baked in per-element here).

## Current scope

Single N-tile weight (N=32): the packed I2_S format stores one scale for
the whole tensor, not per N-tile, so the packed blob for one N-tile is
read as a single DRAM page and multi-N-tile support isn't implemented yet
(needs the scale handled per-tensor across tiles, not per-page).

## Status

Validated under ttsim via a standalone TT-Metalium `Program`/`Kernel`
dispatch (bypassing TT-NN, unlike the Option A path) - not yet wired into
`ggml-ttnn.cpp`'s `supports_op`/`graph_compute`, which still uses Option A.
Wiring these in as a selectable/preferred path is follow-up work.
