// SPDX-License-Identifier: Apache-2.0
//
// Compile-only proof that the installed HIP compiler can emit gfx1100's
// signed IU8 and IU4 WMMA instructions. This is deliberately not a GEMM and
// must never be called by the model. The input buffers already contain the
// lane-distributed register fragments expected by the builtins.

#include <cstdint>
#include <hip/hip_runtime.h>

using i32x2 = int32_t __attribute__((ext_vector_type(2)));
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using i32x8 = int32_t __attribute__((ext_vector_type(8)));

namespace {

constexpr uint32_t kWaveSize = 32;

template <typename T>
__device__ __forceinline__ T load_fragment(const int32_t* source, uint32_t lane) {
  return *reinterpret_cast<const T*>(source + lane * (sizeof(T) / sizeof(int32_t)));
}

template <typename T>
__device__ __forceinline__ void store_fragment(int32_t* destination, uint32_t lane, T value) {
  *reinterpret_cast<T*>(destination + lane * (sizeof(T) / sizeof(int32_t))) = value;
}

}  // namespace

extern "C" __global__ __launch_bounds__(kWaveSize) void skyrl_wmma_iu8_fragment_compile_probe(
    const int32_t* a_fragment,
    const int32_t* b_fragment,
    const int32_t* accumulator_fragment,
    int32_t* output_fragment) {
  const uint32_t lane = threadIdx.x;
  if (lane >= kWaveSize) {
    return;
  }

  const i32x4 a = load_fragment<i32x4>(a_fragment, lane);
  const i32x4 b = load_fragment<i32x4>(b_fragment, lane);
  const i32x8 accumulator = load_fragment<i32x8>(accumulator_fragment, lane);

  // The first two flags select signed inputs. The final flag disables
  // saturation/clamping, matching the INT32 semantic oracle.
  const i32x8 result = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
      true, a, true, b, accumulator, false);
  store_fragment(output_fragment, lane, result);
}

extern "C" __global__ __launch_bounds__(kWaveSize) void skyrl_wmma_iu4_fragment_compile_probe(
    const int32_t* a_fragment,
    const int32_t* b_fragment,
    const int32_t* accumulator_fragment,
    int32_t* output_fragment) {
  const uint32_t lane = threadIdx.x;
  if (lane >= kWaveSize) {
    return;
  }

  const i32x2 a = load_fragment<i32x2>(a_fragment, lane);
  const i32x2 b = load_fragment<i32x2>(b_fragment, lane);
  const i32x8 accumulator = load_fragment<i32x8>(accumulator_fragment, lane);

  const i32x8 result = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
      true, a, true, b, accumulator, false);
  store_fragment(output_fragment, lane, result);
}
