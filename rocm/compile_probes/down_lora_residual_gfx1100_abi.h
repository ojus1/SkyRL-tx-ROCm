// SPDX-License-Identifier: Apache-2.0
//
// Versioned explicit kernel-argument contract for the compile-only gfx1100 proof.
// This header does not register, load, or launch a kernel.

#ifndef SKYRL_ROCM_COMPILE_PROBES_DOWN_LORA_RESIDUAL_GFX1100_ABI_H_
#define SKYRL_ROCM_COMPILE_PROBES_DOWN_LORA_RESIDUAL_GFX1100_ABI_H_

#include <cstddef>
#include <cstdint>

namespace skyrl::rocm::down_lora_residual_v1 {

inline constexpr std::uint32_t kAbiVersion = 1;
inline constexpr std::uint32_t kThreads = 64;
inline constexpr std::uint32_t kRankCapacity = 8;
inline constexpr std::uint32_t kMaxRows = 256;
inline constexpr std::uint32_t kMaxInputFeatures = 9216;
inline constexpr std::uint32_t kMaxOutputFeatures = 2560;

// These structures document the caller-owned prefix of each kernel's argument
// buffer. The HIP compiler appends target-specific hidden launch fields after
// this prefix; a future wrapper must let a supported HIP launch API populate
// those fields. Device addresses use uint64_t so the explicit prefix does not
// depend on the host compiler's pointer spelling.
struct alignas(8) ForwardKernarg {
  std::uint64_t x;
  std::uint64_t residual;
  std::uint64_t weight;
  std::uint64_t lora_a;
  std::uint64_t lora_b;
  std::uint64_t output;
  std::uint64_t lora_xa;
  std::uint32_t rows;
  std::uint32_t input_features;
  std::uint32_t output_features;
  std::uint32_t active_rank;
  float lora_scale;
  std::uint32_t padding;
};

struct alignas(8) BackwardPrepareKernarg {
  std::uint64_t output_cotangent;
  std::uint64_t lora_b;
  std::uint64_t residual_cotangent;
  std::uint64_t lora_dy_b;
  std::uint32_t rows;
  std::uint32_t output_features;
  std::uint32_t active_rank;
  float lora_scale;
};

struct alignas(8) BackwardDxDaKernarg {
  std::uint64_t output_cotangent;
  std::uint64_t x;
  std::uint64_t weight;
  std::uint64_t lora_a;
  std::uint64_t lora_dy_b;
  std::uint64_t x_cotangent;
  std::uint64_t lora_a_cotangent;
  std::uint32_t rows;
  std::uint32_t input_features;
  std::uint32_t output_features;
  std::uint32_t active_rank;
};

struct alignas(8) BackwardDbKernarg {
  std::uint64_t output_cotangent;
  std::uint64_t lora_xa;
  std::uint64_t lora_b_cotangent;
  std::uint32_t rows;
  std::uint32_t output_features;
  std::uint32_t active_rank;
  float lora_scale;
};

static_assert(sizeof(float) == 4);
static_assert(sizeof(ForwardKernarg) == 80);
static_assert(offsetof(ForwardKernarg, lora_scale) == 72);
static_assert(sizeof(BackwardPrepareKernarg) == 48);
static_assert(offsetof(BackwardPrepareKernarg, lora_scale) == 44);
static_assert(sizeof(BackwardDxDaKernarg) == 72);
static_assert(offsetof(BackwardDxDaKernarg, active_rank) == 68);
static_assert(sizeof(BackwardDbKernarg) == 40);
static_assert(offsetof(BackwardDbKernarg, lora_scale) == 36);

}  // namespace skyrl::rocm::down_lora_residual_v1

#endif  // SKYRL_ROCM_COMPILE_PROBES_DOWN_LORA_RESIDUAL_GFX1100_ABI_H_
