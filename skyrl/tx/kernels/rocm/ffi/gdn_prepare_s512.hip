// SPDX-License-Identifier: Apache-2.0
//
// Experimental exact-shape Qwen3.5 GDN WY preparation for gfx1100.
// This is a prepare-only rung: it does not execute the recurrent state update,
// provide a transpose, or integrate with the model.

#include <array>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>
#include "xla/ffi/api/ffi.h"

namespace {

constexpr std::int64_t kBatch = 1;
constexpr std::int64_t kTokens = 512;
constexpr std::int64_t kChunks = 8;
constexpr std::int64_t kChunk = 64;
constexpr std::int64_t kKeyHeads = 16;
constexpr std::int64_t kValueHeads = 32;
constexpr std::int64_t kHeadsPerKey = 2;
constexpr std::int64_t kHeadDimension = 128;
constexpr std::int64_t kSolveWidths = 2 * kHeadDimension;
constexpr std::int64_t kSolveTile = 64;
constexpr std::uint32_t kThreads = 128;
constexpr std::uint32_t kWorkgroups = kChunks * kKeyHeads;

constexpr std::size_t kKeyElements =
    kBatch * kTokens * kKeyHeads * kHeadDimension;
constexpr std::size_t kValueElements =
    kBatch * kTokens * kValueHeads * kHeadDimension;
constexpr std::size_t kGateElements = kBatch * kTokens * kValueHeads;
constexpr std::size_t kKeyBytes = kKeyElements * sizeof(float);
constexpr std::size_t kValueBytes = kValueElements * sizeof(float);
constexpr std::size_t kGateBytes = kGateElements * sizeof(float);
constexpr std::size_t kLdsFloats =
    2 * kChunk * kChunk + kChunk * kSolveTile + 2 * kChunk;
constexpr std::size_t kLdsBytes = kLdsFloats * sizeof(float);

static_assert(kTokens == kChunks * kChunk);
static_assert(kValueHeads == kKeyHeads * kHeadsPerKey);
static_assert(kSolveWidths % kSolveTile == 0);
static_assert(kKeyBytes == 4 * 1024 * 1024);
static_assert(kValueBytes == 8 * 1024 * 1024);
static_assert(kGateBytes == 64 * 1024);
static_assert(kLdsBytes == 49'664);
static_assert(kLdsBytes < 64 * 1024);

__device__ __forceinline__ std::size_t KeyOffset(std::int64_t token,
                                                 std::int64_t key_head,
                                                 std::int64_t feature) {
  return (token * kKeyHeads + key_head) * kHeadDimension + feature;
}

__device__ __forceinline__ std::size_t ValueOffset(std::int64_t token,
                                                   std::int64_t value_head,
                                                   std::int64_t feature) {
  return (token * kValueHeads + value_head) * kHeadDimension + feature;
}

__device__ __forceinline__ std::size_t GateOffset(std::int64_t token,
                                                  std::int64_t value_head) {
  return token * kValueHeads + value_head;
}

// One workgroup owns one (64-token chunk, key head). The two associated value
// heads share key_gram. Per-value-head L and the forward solve live in bounded
// LDS; RHS values are generated directly from immutable global inputs.
__global__ __launch_bounds__(kThreads, 1) void GdnPrepareS512Kernel(
    const float* key, const float* value, const float* g, const float* beta,
    float* prepared_u, float* prepared_w, float* gamma) {
  __shared__ float key_gram[kChunk][kChunk];
  __shared__ float strict_lower[kChunk][kChunk];
  __shared__ float solution[kChunk][kSolveTile];
  __shared__ float prefix[kChunk];
  __shared__ float beta_chunk[kChunk];

  const std::int64_t chunk_index = blockIdx.x / kKeyHeads;
  const std::int64_t key_head = blockIdx.x % kKeyHeads;
  const std::int64_t token_base = chunk_index * kChunk;
  const std::int64_t local_thread = threadIdx.x;

  for (std::int64_t linear = local_thread; linear < kChunk * kChunk;
       linear += kThreads) {
    const std::int64_t row = linear / kChunk;
    const std::int64_t column = linear % kChunk;
    float dot = 0.0F;
    for (std::int64_t feature = 0; feature < kHeadDimension; ++feature) {
      dot += key[KeyOffset(token_base + row, key_head, feature)] *
             key[KeyOffset(token_base + column, key_head, feature)];
    }
    key_gram[row][column] = dot;
  }
  __syncthreads();

  for (std::int64_t head_in_pair = 0; head_in_pair < kHeadsPerKey;
       ++head_in_pair) {
    const std::int64_t value_head = kHeadsPerKey * key_head + head_in_pair;

    // A single lane computes the ordered cumulative sum. This is deliberately
    // scalar and reproducible for the first correctness rung; a later kernel
    // may replace it only behind a new numerical gate.
    if (local_thread == 0) {
      float running = 0.0F;
      for (std::int64_t row = 0; row < kChunk; ++row) {
        const std::size_t gate_index =
            GateOffset(token_base + row, value_head);
        running += g[gate_index];
        prefix[row] = running;
        beta_chunk[row] = beta[gate_index];
        gamma[gate_index] = expf(running);
      }
    }
    __syncthreads();

    for (std::int64_t linear = local_thread; linear < kChunk * kChunk;
         linear += kThreads) {
      const std::int64_t row = linear / kChunk;
      const std::int64_t column = linear % kChunk;
      strict_lower[row][column] =
          column < row ? beta_chunk[row] * key_gram[row][column] *
                             expf(prefix[row] - prefix[column])
                       : 0.0F;
    }
    __syncthreads();

    // Solve (I + L) [U | W] = [beta * V | beta * gamma * K] in four
    // 64-column tiles. Unit diagonal means no division is required.
    for (std::int64_t column_base = 0; column_base < kSolveWidths;
         column_base += kSolveTile) {
      for (std::int64_t row = 0; row < kChunk; ++row) {
        if (local_thread < kSolveTile) {
          const std::int64_t solve_column = column_base + local_thread;
          const std::int64_t feature =
              solve_column < kHeadDimension
                  ? solve_column
                  : solve_column - kHeadDimension;
          const std::int64_t token = token_base + row;
          float rhs = 0.0F;
          if (solve_column < kHeadDimension) {
            rhs = beta_chunk[row] *
                  value[ValueOffset(token, value_head, feature)];
          } else {
            rhs = beta_chunk[row] * expf(prefix[row]) *
                  key[KeyOffset(token, key_head, feature)];
          }

          float correction = 0.0F;
          for (std::int64_t previous = 0; previous < row; ++previous) {
            correction += strict_lower[row][previous] *
                          solution[previous][local_thread];
          }
          solution[row][local_thread] = rhs - correction;
        }
        __syncthreads();
      }

      if (local_thread < kSolveTile) {
        const std::int64_t solve_column = column_base + local_thread;
        const std::int64_t feature =
            solve_column < kHeadDimension
                ? solve_column
                : solve_column - kHeadDimension;
        for (std::int64_t row = 0; row < kChunk; ++row) {
          const std::size_t output_index =
              ValueOffset(token_base + row, value_head, feature);
          if (solve_column < kHeadDimension) {
            prepared_u[output_index] = solution[row][local_thread];
          } else {
            prepared_w[output_index] = solution[row][local_thread];
          }
        }
      }
      __syncthreads();
    }
  }
}

bool HasExactKeyShape(xla::ffi::Span<const std::int64_t> dimensions) {
  return dimensions.size() == 4 && dimensions[0] == kBatch &&
         dimensions[1] == kTokens && dimensions[2] == kKeyHeads &&
         dimensions[3] == kHeadDimension;
}

bool HasExactValueShape(xla::ffi::Span<const std::int64_t> dimensions) {
  return dimensions.size() == 4 && dimensions[0] == kBatch &&
         dimensions[1] == kTokens && dimensions[2] == kValueHeads &&
         dimensions[3] == kHeadDimension;
}

bool HasExactGateShape(xla::ffi::Span<const std::int64_t> dimensions) {
  return dimensions.size() == 3 && dimensions[0] == kBatch &&
         dimensions[1] == kTokens && dimensions[2] == kValueHeads;
}

template <std::size_t N>
bool AreDistinctNonNull(const std::array<const void*, N>& buffers) {
  for (std::size_t left = 0; left < N; ++left) {
    if (buffers[left] == nullptr) {
      return false;
    }
    for (std::size_t right = left + 1; right < N; ++right) {
      if (buffers[left] == buffers[right]) {
        return false;
      }
    }
  }
  return true;
}

xla::ffi::Error GdnPrepareS512(
    xla::ffi::BufferR4<xla::ffi::F32> key,
    xla::ffi::BufferR4<xla::ffi::F32> value,
    xla::ffi::BufferR3<xla::ffi::F32> g,
    xla::ffi::BufferR3<xla::ffi::F32> beta,
    xla::ffi::ResultBufferR4<xla::ffi::F32> prepared_u,
    xla::ffi::ResultBufferR4<xla::ffi::F32> prepared_w,
    xla::ffi::ResultBufferR3<xla::ffi::F32> gamma, hipStream_t stream) {
  if (!HasExactKeyShape(key.dimensions()) ||
      !HasExactValueShape(value.dimensions()) ||
      !HasExactGateShape(g.dimensions()) ||
      !HasExactGateShape(beta.dimensions()) ||
      !HasExactValueShape(prepared_u->dimensions()) ||
      !HasExactValueShape(prepared_w->dimensions()) ||
      !HasExactGateShape(gamma->dimensions()) || key.size_bytes() != kKeyBytes ||
      value.size_bytes() != kValueBytes || g.size_bytes() != kGateBytes ||
      beta.size_bytes() != kGateBytes ||
      prepared_u->size_bytes() != kValueBytes ||
      prepared_w->size_bytes() != kValueBytes ||
      gamma->size_bytes() != kGateBytes) {
    return xla::ffi::Error::InvalidArgument(
        "gdn_prepare_s512 requires exact FP32 K[1,512,16,128], "
        "V/U/W[1,512,32,128], and g/beta/G[1,512,32] buffers");
  }

  const auto* key_data = key.typed_data();
  const auto* value_data = value.typed_data();
  const auto* g_data = g.typed_data();
  const auto* beta_data = beta.typed_data();
  auto* u_data = prepared_u->typed_data();
  auto* w_data = prepared_w->typed_data();
  auto* gamma_data = gamma->typed_data();
  const std::array<const void*, 7> buffers = {
      static_cast<const void*>(key_data),   static_cast<const void*>(value_data),
      static_cast<const void*>(g_data),     static_cast<const void*>(beta_data),
      static_cast<const void*>(u_data),     static_cast<const void*>(w_data),
      static_cast<const void*>(gamma_data),
  };
  if (!AreDistinctNonNull(buffers)) {
    return xla::ffi::Error::InvalidArgument(
        "gdn_prepare_s512 requires seven distinct non-null buffers");
  }

  (void)hipGetLastError();
  hipLaunchKernelGGL(GdnPrepareS512Kernel, dim3(kWorkgroups), dim3(kThreads), 0,
                     stream, key_data, value_data, g_data, beta_data, u_data,
                     w_data, gamma_data);
  if (hipGetLastError() != hipSuccess) {
    return xla::ffi::Error::Internal(
        "gdn_prepare_s512 asynchronous launch submission failed");
  }

  // XLA owns buffer lifetime and dependency ordering on the supplied stream.
  return xla::ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    skyrl_gdn_prepare_s512_f32_v1, GdnPrepareS512,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::BufferR4<xla::ffi::F32>>()
        .Arg<xla::ffi::BufferR4<xla::ffi::F32>>()
        .Arg<xla::ffi::BufferR3<xla::ffi::F32>>()
        .Arg<xla::ffi::BufferR3<xla::ffi::F32>>()
        .Ret<xla::ffi::BufferR4<xla::ffi::F32>>()
        .Ret<xla::ffi::BufferR4<xla::ffi::F32>>()
        .Ret<xla::ffi::BufferR3<xla::ffi::F32>>()
        .Ctx<xla::ffi::PlatformStream<hipStream_t>>());
