// SPDX-License-Identifier: Apache-2.0
//
// Offline AOT compile proof for a bounded BF16 down-projection + LoRA +
// residual stage and its frozen-base VJP. This file is intentionally scalar
// and reference-oriented: it proves a versioned gfx1100 ABI and bounded code
// generation, not production performance or runtime correctness.

#include <cstdint>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "down_lora_residual_gfx1100_abi.h"

namespace {

namespace abi = skyrl::rocm::down_lora_residual_v1;

using bf16 = hip_bfloat16;

static_assert(sizeof(bf16) == 2);
static_assert(abi::kThreads == 64);
static_assert(abi::kRankCapacity == 8);

__device__ __forceinline__ std::uint32_t ceil_tiles(std::uint32_t size) {
  return (size + abi::kThreads - 1) / abi::kThreads;
}

__device__ __forceinline__ std::uint32_t max_rows_or_rank_capacity(
    std::uint32_t rows) {
  return rows > abi::kRankCapacity ? rows : abi::kRankCapacity;
}

__device__ __forceinline__ bool valid_shape(std::uint32_t rows,
                                            std::uint32_t input_features,
                                            std::uint32_t output_features,
                                            std::uint32_t active_rank) {
  return rows > 0 && rows <= abi::kMaxRows && input_features > 0 &&
         input_features <= abi::kMaxInputFeatures && output_features > 0 &&
         output_features <= abi::kMaxOutputFeatures && active_rank > 0 &&
         active_rank <= abi::kRankCapacity;
}

__device__ __forceinline__ bool valid_block() {
  return blockDim.x == abi::kThreads && blockDim.y == 1 && blockDim.z == 1;
}

__device__ __forceinline__ float load_bf16(const bf16* source,
                                           std::uint64_t index) {
  return static_cast<float>(source[index]);
}

__device__ __forceinline__ void store_bf16(bf16* destination,
                                           std::uint64_t index, float value) {
  destination[index] = bf16(value);
}

}  // namespace

extern "C" __global__ __launch_bounds__(abi::kThreads) void
skyrl_down_lora_residual_fwd_bf16_gfx1100_v1(
    const bf16* __restrict__ x, const bf16* __restrict__ residual,
    const bf16* __restrict__ weight, const bf16* __restrict__ lora_a,
    const bf16* __restrict__ lora_b, bf16* __restrict__ output,
    float* __restrict__ lora_xa, std::uint32_t rows,
    std::uint32_t input_features, std::uint32_t output_features,
    std::uint32_t active_rank, float lora_scale) {
  if (!valid_block() || !valid_shape(rows, input_features, output_features,
                                     active_rank) ||
      gridDim.x != ceil_tiles(output_features) || gridDim.y != rows ||
      gridDim.z != 1 || x == nullptr || residual == nullptr ||
      weight == nullptr || lora_a == nullptr || lora_b == nullptr ||
      output == nullptr || lora_xa == nullptr) {
    return;
  }

  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  const std::uint32_t column = blockIdx.x * abi::kThreads + lane;
  const std::uint64_t x_row = static_cast<std::uint64_t>(row) * input_features;

  __shared__ float shared_lora_xa[abi::kRankCapacity];
  if (lane < abi::kRankCapacity) {
    float accumulator = 0.0F;
    if (lane < active_rank) {
      for (std::uint32_t feature = 0; feature < input_features; ++feature) {
        accumulator +=
            load_bf16(x, x_row + feature) *
            load_bf16(lora_a,
                      static_cast<std::uint64_t>(feature) * abi::kRankCapacity + lane);
      }
    }
    shared_lora_xa[lane] = accumulator;
    // Only the first output tile publishes the compact forward residual.
    if (blockIdx.x == 0) {
      lora_xa[static_cast<std::uint64_t>(row) * abi::kRankCapacity + lane] =
          accumulator;
    }
  }
  __syncthreads();

  if (column >= output_features) {
    return;
  }

  float accumulator = 0.0F;
  for (std::uint32_t feature = 0; feature < input_features; ++feature) {
    accumulator +=
        load_bf16(x, x_row + feature) *
        load_bf16(weight, static_cast<std::uint64_t>(feature) * output_features +
                              column);
  }
  float lora_accumulator = 0.0F;
  for (std::uint32_t rank = 0; rank < active_rank; ++rank) {
    lora_accumulator +=
        shared_lora_xa[rank] *
        load_bf16(lora_b, static_cast<std::uint64_t>(rank) * output_features +
                              column);
  }
  const std::uint64_t output_index =
      static_cast<std::uint64_t>(row) * output_features + column;
  accumulator += lora_scale * lora_accumulator;
  accumulator += load_bf16(residual, output_index);
  store_bf16(output, output_index, accumulator);
}

extern "C" __global__ __launch_bounds__(abi::kThreads) void
skyrl_down_lora_residual_bwd_prepare_bf16_gfx1100_v1(
    const bf16* __restrict__ output_cotangent,
    const bf16* __restrict__ lora_b, bf16* __restrict__ residual_cotangent,
    float* __restrict__ lora_dy_b, std::uint32_t rows,
    std::uint32_t output_features, std::uint32_t active_rank,
    float lora_scale) {
  if (!valid_block() ||
      !valid_shape(rows, 1, output_features, active_rank) ||
      gridDim.x != rows || gridDim.y != 1 || gridDim.z != 1 ||
      output_cotangent == nullptr || lora_b == nullptr ||
      residual_cotangent == nullptr || lora_dy_b == nullptr) {
    return;
  }

  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t row = blockIdx.x;
  const std::uint64_t row_offset =
      static_cast<std::uint64_t>(row) * output_features;

  // The residual branch has an identity VJP. Copy it while the same bounded
  // row block prepares V = scale * dY * B^T for the other two kernels.
  for (std::uint32_t column = lane; column < output_features;
       column += abi::kThreads) {
    store_bf16(residual_cotangent, row_offset + column,
               load_bf16(output_cotangent, row_offset + column));
  }

  if (lane < abi::kRankCapacity) {
    float accumulator = 0.0F;
    if (lane < active_rank) {
      for (std::uint32_t column = 0; column < output_features; ++column) {
        accumulator +=
            load_bf16(output_cotangent, row_offset + column) *
            load_bf16(lora_b,
                      static_cast<std::uint64_t>(lane) * output_features + column);
      }
      accumulator *= lora_scale;
    }
    lora_dy_b[static_cast<std::uint64_t>(row) * abi::kRankCapacity + lane] =
        accumulator;
  }
}

extern "C" __global__ __launch_bounds__(abi::kThreads) void
skyrl_down_lora_residual_bwd_dx_da_bf16_gfx1100_v1(
    const bf16* __restrict__ output_cotangent, const bf16* __restrict__ x,
    const bf16* __restrict__ weight, const bf16* __restrict__ lora_a,
    const float* __restrict__ lora_dy_b, bf16* __restrict__ x_cotangent,
    bf16* __restrict__ lora_a_cotangent, std::uint32_t rows,
    std::uint32_t input_features, std::uint32_t output_features,
    std::uint32_t active_rank) {
  if (!valid_block() || !valid_shape(rows, input_features, output_features,
                                     active_rank) ||
      gridDim.x != ceil_tiles(input_features) ||
      gridDim.y != max_rows_or_rank_capacity(rows) || gridDim.z != 1 ||
      output_cotangent == nullptr || x == nullptr || weight == nullptr ||
      lora_a == nullptr || lora_dy_b == nullptr || x_cotangent == nullptr ||
      lora_a_cotangent == nullptr) {
    return;
  }

  const std::uint32_t feature = blockIdx.x * abi::kThreads + threadIdx.x;
  if (feature >= input_features) {
    return;
  }
  const std::uint32_t row_or_rank = blockIdx.y;

  if (row_or_rank < rows) {
    const std::uint64_t dy_row =
        static_cast<std::uint64_t>(row_or_rank) * output_features;
    float accumulator = 0.0F;
    for (std::uint32_t column = 0; column < output_features; ++column) {
      accumulator +=
          load_bf16(output_cotangent, dy_row + column) *
          load_bf16(weight, static_cast<std::uint64_t>(feature) * output_features +
                                column);
    }
    const std::uint64_t workspace_row =
        static_cast<std::uint64_t>(row_or_rank) * abi::kRankCapacity;
    const std::uint64_t a_row =
        static_cast<std::uint64_t>(feature) * abi::kRankCapacity;
    for (std::uint32_t rank = 0; rank < active_rank; ++rank) {
      accumulator += lora_dy_b[workspace_row + rank] *
                     load_bf16(lora_a, a_row + rank);
    }
    store_bf16(x_cotangent,
               static_cast<std::uint64_t>(row_or_rank) * input_features + feature,
               accumulator);
  }

  // The first eight grid rows also reduce dA. Inactive padded columns are
  // explicitly zeroed so the full [K, 8] output has deterministic contents.
  if (row_or_rank < abi::kRankCapacity) {
    float accumulator = 0.0F;
    if (row_or_rank < active_rank) {
      for (std::uint32_t row = 0; row < rows; ++row) {
        accumulator +=
            load_bf16(x, static_cast<std::uint64_t>(row) * input_features + feature) *
            lora_dy_b[static_cast<std::uint64_t>(row) * abi::kRankCapacity +
                       row_or_rank];
      }
    }
    store_bf16(lora_a_cotangent,
               static_cast<std::uint64_t>(feature) * abi::kRankCapacity +
                   row_or_rank,
               accumulator);
  }
}

extern "C" __global__ __launch_bounds__(abi::kThreads) void
skyrl_down_lora_residual_bwd_db_bf16_gfx1100_v1(
    const bf16* __restrict__ output_cotangent,
    const float* __restrict__ lora_xa, bf16* __restrict__ lora_b_cotangent,
    std::uint32_t rows, std::uint32_t output_features,
    std::uint32_t active_rank, float lora_scale) {
  if (!valid_block() ||
      !valid_shape(rows, 1, output_features, active_rank) ||
      gridDim.x != ceil_tiles(output_features) ||
      gridDim.y != abi::kRankCapacity || gridDim.z != 1 ||
      output_cotangent == nullptr || lora_xa == nullptr ||
      lora_b_cotangent == nullptr) {
    return;
  }

  const std::uint32_t column = blockIdx.x * abi::kThreads + threadIdx.x;
  if (column >= output_features) {
    return;
  }
  const std::uint32_t rank = blockIdx.y;

  float accumulator = 0.0F;
  if (rank < active_rank) {
    for (std::uint32_t row = 0; row < rows; ++row) {
      accumulator +=
          lora_xa[static_cast<std::uint64_t>(row) * abi::kRankCapacity + rank] *
          load_bf16(output_cotangent,
                    static_cast<std::uint64_t>(row) * output_features + column);
    }
    accumulator *= lora_scale;
  }
  store_bf16(lora_b_cotangent,
             static_cast<std::uint64_t>(rank) * output_features + column,
             accumulator);
}
