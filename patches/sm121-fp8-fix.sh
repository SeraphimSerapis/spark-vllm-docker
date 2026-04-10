#!/bin/bash
# SM121 (DGX Spark) FP8 fixes — equivalent to PR #35568
# https://github.com/vllm-project/vllm/pull/35568
#
# Uses sed instead of git apply to survive upstream context drift.
# SM121 shares FP8 MMA capabilities with SM120 but was excluded
# by exact-match arch guards (== 120, enable_sm120_only, etc.)

set -euo pipefail

# generate_kernels.py (both marlin and moe) — arch check
sed -i 's/if arch in \[89, 120\]:/if arch == 89 or arch \/\/ 10 == 12:/' \
    csrc/quantization/marlin/generate_kernels.py \
    csrc/moe/marlin_moe_wna16/generate_kernels.py

# ops.cu — capability check
sed -i 's/major_capability \* 10 + minor_capability == 120/major_capability == 12/' \
    csrc/moe/marlin_moe_wna16/ops.cu
sed -i 's/Marlin W4A8-FP8 only support SM89 or SM120/Marlin W4A8-FP8 only support SM89 or SM12x/' \
    csrc/moe/marlin_moe_wna16/ops.cu

# scaled_mm.cuh — enable_sm120_only -> enable_sm120_family
sed -i 's/enable_sm120_only/enable_sm120_family/g' \
    csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm.cuh \
    csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh

# marlin_utils.py — is_device_capability(120) -> is_device_capability_family(120)
sed -i 's/is_device_capability(120)/is_device_capability_family(120)/' \
    vllm/model_executor/layers/quantization/utils/marlin_utils.py
sed -i 's/SM89 or SM120 device/SM89 or SM12x device/' \
    vllm/model_executor/layers/quantization/utils/marlin_utils.py

# Comment updates
sed -i 's/# only SM89 and SM120 fully support/# SM89 and the SM12x family (SM120 RTX 5090, SM121 DGX Spark GB10)/' \
    csrc/quantization/marlin/generate_kernels.py \
    csrc/moe/marlin_moe_wna16/generate_kernels.py
sed -i 's/# mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32./# fully support mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32./' \
    csrc/quantization/marlin/generate_kernels.py \
    csrc/moe/marlin_moe_wna16/generate_kernels.py

echo "Applied SM121 (DGX Spark) FP8 fixes (PR #35568 equivalent)"
