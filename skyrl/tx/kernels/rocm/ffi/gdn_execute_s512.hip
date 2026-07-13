// SPDX-License-Identifier: Apache-2.0
//
// Experimental exact-shape Qwen3.5 GDN execute stage for gfx1100.
// This first correctness rung consumes prepared FP32 values, executes eight
// recurrent 64-token chunks, and returns BF16 output plus FP32 final state.
// It does not prepare U/W/gamma or provide a transpose.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <hip/hip_bfloat16.h>
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
constexpr std::uint32_t kThreads = kHeadDimension;
constexpr std::uint32_t kWorkgroups = kValueHeads;

constexpr std::size_t kQueryElements =
    kBatch * kTokens * kKeyHeads * kHeadDimension;
constexpr std::size_t kPreparedElements =
    kBatch * kTokens * kValueHeads * kHeadDimension;
constexpr std::size_t kGammaElements = kBatch * kTokens * kValueHeads;
constexpr std::size_t kStateElements =
    kBatch * kValueHeads * kHeadDimension * kHeadDimension;
constexpr std::size_t kQueryBytes = kQueryElements * sizeof(float);
constexpr std::size_t kPreparedBytes = kPreparedElements * sizeof(float);
constexpr std::size_t kGammaBytes = kGammaElements * sizeof(float);
constexpr std::size_t kStateBytes = kStateElements * sizeof(float);
constexpr std::size_t kOutputBytes =
    kPreparedElements * sizeof(std::uint16_t);
constexpr std::size_t kLdsFloats =
    kChunk * kChunk + kChunk * kHeadDimension + kChunk;
constexpr std::size_t kLdsBytes = kLdsFloats * sizeof(float);

static_assert(kTokens == kChunks * kChunk);
static_assert(kValueHeads == kKeyHeads * kHeadsPerKey);
static_assert(kThreads == 128);
static_assert(kQueryBytes == 4 * 1024 * 1024);
static_assert(kPreparedBytes == 8 * 1024 * 1024);
static_assert(kGammaBytes == 64 * 1024);
static_assert(kStateBytes == 2 * 1024 * 1024);
static_assert(kOutputBytes == 4 * 1024 * 1024);
static_assert(kLdsBytes == 49'408);
static_assert(kLdsBytes < 64 * 1024);
static_assert(sizeof(hip_bfloat16) == sizeof(std::uint16_t));

__device__ __forceinline__ std::size_t QueryOffset(std::int64_t token,
                                                   std::int64_t key_head,
                                                   std::int64_t feature) {
  return (token * kKeyHeads + key_head) * kHeadDimension + feature;
}

__device__ __forceinline__ std::size_t ValueOffset(std::int64_t token,
                                                   std::int64_t value_head,
                                                   std::int64_t feature) {
  return (token * kValueHeads + value_head) * kHeadDimension + feature;
}

__device__ __forceinline__ std::size_t GammaOffset(std::int64_t token,
                                                   std::int64_t value_head) {
  return token * kValueHeads + value_head;
}

__device__ __forceinline__ std::size_t StateOffset(std::int64_t value_head,
                                                   std::int64_t row,
                                                   std::int64_t column) {
  return (value_head * kHeadDimension + row) * kHeadDimension + column;
}

// One workgroup owns one value head for the complete eight-chunk recurrence.
// Thread c owns state/output column c. The final-state result is first seeded
// from the immutable incoming state and then serves as this workgroup's
// disjoint global state workspace. No input/output alias is permitted.
__global__ __launch_bounds__(kThreads, 1) void GdnExecuteS512Kernel(
    const float* query, const float* key, const float* prepared_u,
    const float* prepared_w, const float* gamma, const float* initial_state,
    std::uint16_t* output, float* final_state) {
  __shared__ float attention[kChunk][kChunk];
  __shared__ float corrected[kChunk][kHeadDimension];
  __shared__ float reverse_decay[kChunk];

  const std::int64_t value_head = blockIdx.x;
  const std::int64_t key_head = value_head / kHeadsPerKey;
  const std::int64_t output_column = threadIdx.x;

  for (std::int64_t row = 0; row < kHeadDimension; ++row) {
    const std::size_t state_index =
        StateOffset(value_head, row, output_column);
    final_state[state_index] = initial_state[state_index];
  }
  __syncthreads();

  for (std::int64_t chunk_index = 0; chunk_index < kChunks; ++chunk_index) {
    const std::int64_t token_base = chunk_index * kChunk;

    // Each attention value is produced once and reused by all 128 output
    // columns. The triangular gamma ratio is the oracle's decay[i,j].
    for (std::int64_t linear = threadIdx.x; linear < kChunk * kChunk;
         linear += kThreads) {
      const std::int64_t row = linear / kChunk;
      const std::int64_t column = linear % kChunk;
      float value = 0.0F;
      if (column <= row) {
        float dot = 0.0F;
        for (std::int64_t feature = 0; feature < kHeadDimension; ++feature) {
          dot += query[QueryOffset(token_base + row, key_head, feature)] *
                 key[QueryOffset(token_base + column, key_head, feature)];
        }
        const float decay = gamma[GammaOffset(token_base + row, value_head)] /
                            gamma[GammaOffset(token_base + column, value_head)];
        value = dot * decay;
      }
      attention[row][column] = value;
    }

    // Thread c computes corrected[:,c] = U[:,c] - W @ state[:,c].
    for (std::int64_t row = 0; row < kChunk; ++row) {
      const std::int64_t token = token_base + row;
      float correction = 0.0F;
      for (std::int64_t feature = 0; feature < kHeadDimension; ++feature) {
        correction += prepared_w[ValueOffset(token, value_head, feature)] *
                      final_state[
                          StateOffset(value_head, feature, output_column)];
      }
      corrected[row][output_column] =
          prepared_u[ValueOffset(token, value_head, output_column)] - correction;
    }
    __syncthreads();

    // output[:,c] = (gamma * Q) @ state[:,c] + attention @ corrected[:,c].
    for (std::int64_t row = 0; row < kChunk; ++row) {
      const std::int64_t token = token_base + row;
      const float row_gamma = gamma[GammaOffset(token, value_head)];
      float inter_chunk = 0.0F;
      for (std::int64_t feature = 0; feature < kHeadDimension; ++feature) {
        inter_chunk +=
            (query[QueryOffset(token, key_head, feature)] * row_gamma) *
            final_state[StateOffset(value_head, feature, output_column)];
      }
      float intra_chunk = 0.0F;
      for (std::int64_t previous = 0; previous < kChunk; ++previous) {
        intra_chunk += attention[row][previous] *
                       corrected[previous][output_column];
      }
      const hip_bfloat16 converted(inter_chunk + intra_chunk);
      output[ValueOffset(token, value_head, output_column)] = converted.data;
    }

    // state[r,c] = gamma[-1] * state[r,c]
    //              + sum_i K[i,r] * gamma[-1]/gamma[i] * corrected[i,c].
    const float final_gamma =
        gamma[GammaOffset(token_base + kChunk - 1, value_head)];
    if (output_column < kChunk) {
      reverse_decay[output_column] =
          final_gamma /
          gamma[GammaOffset(token_base + output_column, value_head)];
    }
    __syncthreads();
    for (std::int64_t state_row = 0; state_row < kHeadDimension; ++state_row) {
      const std::size_t state_index =
          StateOffset(value_head, state_row, output_column);
      float update = 0.0F;
      for (std::int64_t row = 0; row < kChunk; ++row) {
        const std::int64_t token = token_base + row;
        update += key[QueryOffset(token, key_head, state_row)] *
                  reverse_decay[row] * corrected[row][output_column];
      }
      final_state[state_index] =
          final_gamma * final_state[state_index] + update;
    }
    // Protect both shared arrays and the just-written state from a faster
    // wave beginning the next recurrent chunk early.
    __syncthreads();
  }
}

bool HasExactQueryShape(xla::ffi::Span<const std::int64_t> dimensions) {
  return dimensions.size() == 4 && dimensions[0] == kBatch &&
         dimensions[1] == kTokens && dimensions[2] == kKeyHeads &&
         dimensions[3] == kHeadDimension;
}

bool HasExactPreparedShape(xla::ffi::Span<const std::int64_t> dimensions) {
  return dimensions.size() == 4 && dimensions[0] == kBatch &&
         dimensions[1] == kTokens && dimensions[2] == kValueHeads &&
         dimensions[3] == kHeadDimension;
}

bool HasExactGammaShape(xla::ffi::Span<const std::int64_t> dimensions) {
  return dimensions.size() == 3 && dimensions[0] == kBatch &&
         dimensions[1] == kTokens && dimensions[2] == kValueHeads;
}

bool HasExactStateShape(xla::ffi::Span<const std::int64_t> dimensions) {
  return dimensions.size() == 4 && dimensions[0] == kBatch &&
         dimensions[1] == kValueHeads && dimensions[2] == kHeadDimension &&
         dimensions[3] == kHeadDimension;
}

struct BufferRange {
  const void* data;
  std::size_t bytes;
};

template <std::size_t N>
bool AreDisjointNonNull(const std::array<BufferRange, N>& buffers) {
  for (std::size_t left = 0; left < N; ++left) {
    if (buffers[left].data == nullptr || buffers[left].bytes == 0) {
      return false;
    }
    const auto left_begin =
        reinterpret_cast<std::uintptr_t>(buffers[left].data);
    if (left_begin > std::numeric_limits<std::uintptr_t>::max() -
                         buffers[left].bytes) {
      return false;
    }
    const auto left_end = left_begin + buffers[left].bytes;
    for (std::size_t right = left + 1; right < N; ++right) {
      if (buffers[right].data == nullptr || buffers[right].bytes == 0) {
        return false;
      }
      const auto right_begin =
          reinterpret_cast<std::uintptr_t>(buffers[right].data);
      if (right_begin > std::numeric_limits<std::uintptr_t>::max() -
                            buffers[right].bytes) {
        return false;
      }
      const auto right_end = right_begin + buffers[right].bytes;
      if (left_begin < right_end && right_begin < left_end) {
        return false;
      }
    }
  }
  return true;
}

xla::ffi::Error GdnExecuteS512(
    xla::ffi::BufferR4<xla::ffi::F32> query,
    xla::ffi::BufferR4<xla::ffi::F32> key,
    xla::ffi::BufferR4<xla::ffi::F32> prepared_u,
    xla::ffi::BufferR4<xla::ffi::F32> prepared_w,
    xla::ffi::BufferR3<xla::ffi::F32> gamma,
    xla::ffi::BufferR4<xla::ffi::F32> initial_state,
    xla::ffi::ResultBufferR4<xla::ffi::BF16> output,
    xla::ffi::ResultBufferR4<xla::ffi::F32> final_state, hipStream_t stream) {
  // Typed FFI decoding rejects dtype/rank mismatches. Exact dimensions and
  // byte counts are checked here; physical row-major strides are fixed by the
  // wrapper's mandatory operand/result layouts because XLA_FFI_Buffer does
  // not carry a separate stride vector.
  if (!HasExactQueryShape(query.dimensions()) ||
      !HasExactQueryShape(key.dimensions()) ||
      !HasExactPreparedShape(prepared_u.dimensions()) ||
      !HasExactPreparedShape(prepared_w.dimensions()) ||
      !HasExactGammaShape(gamma.dimensions()) ||
      !HasExactStateShape(initial_state.dimensions()) ||
      !HasExactPreparedShape(output->dimensions()) ||
      !HasExactStateShape(final_state->dimensions()) ||
      query.size_bytes() != kQueryBytes || key.size_bytes() != kQueryBytes ||
      prepared_u.size_bytes() != kPreparedBytes ||
      prepared_w.size_bytes() != kPreparedBytes ||
      gamma.size_bytes() != kGammaBytes ||
      initial_state.size_bytes() != kStateBytes ||
      output->size_bytes() != kOutputBytes ||
      final_state->size_bytes() != kStateBytes) {
    return xla::ffi::Error::InvalidArgument(
        "gdn_execute_s512 requires exact FP32 Q/K[1,512,16,128], "
        "U/W[1,512,32,128], gamma[1,512,32], state[1,32,128,128], "
        "BF16 output[1,512,32,128], and FP32 final state buffers");
  }

  const auto* query_data = query.typed_data();
  const auto* key_data = key.typed_data();
  const auto* u_data = prepared_u.typed_data();
  const auto* w_data = prepared_w.typed_data();
  const auto* gamma_data = gamma.typed_data();
  const auto* initial_state_data = initial_state.typed_data();
  auto* output_data = output->typed_data();
  auto* final_state_data = final_state->typed_data();
  const std::array<BufferRange, 8> buffers = {{
      {query_data, kQueryBytes},
      {key_data, kQueryBytes},
      {u_data, kPreparedBytes},
      {w_data, kPreparedBytes},
      {gamma_data, kGammaBytes},
      {initial_state_data, kStateBytes},
      {output_data, kOutputBytes},
      {final_state_data, kStateBytes},
  }};
  if (!AreDisjointNonNull(buffers)) {
    return xla::ffi::Error::InvalidArgument(
        "gdn_execute_s512 requires eight disjoint non-null buffers");
  }

  (void)hipGetLastError();
  hipLaunchKernelGGL(GdnExecuteS512Kernel, dim3(kWorkgroups), dim3(kThreads), 0,
                     stream, query_data, key_data, u_data, w_data, gamma_data,
                     initial_state_data, output_data, final_state_data);
  if (hipGetLastError() != hipSuccess) {
    return xla::ffi::Error::Internal(
        "gdn_execute_s512 asynchronous launch submission failed");
  }

  // XLA owns buffer lifetime and dependency ordering on the supplied stream.
  return xla::ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    skyrl_gdn_execute_s512_f32_bf16_v1, GdnExecuteS512,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::BufferR4<xla::ffi::F32>>()
        .Arg<xla::ffi::BufferR4<xla::ffi::F32>>()
        .Arg<xla::ffi::BufferR4<xla::ffi::F32>>()
        .Arg<xla::ffi::BufferR4<xla::ffi::F32>>()
        .Arg<xla::ffi::BufferR3<xla::ffi::F32>>()
        .Arg<xla::ffi::BufferR4<xla::ffi::F32>>()
        .Ret<xla::ffi::BufferR4<xla::ffi::BF16>>()
        .Ret<xla::ffi::BufferR4<xla::ffi::F32>>()
        .Ctx<xla::ffi::PlatformStream<hipStream_t>>());
