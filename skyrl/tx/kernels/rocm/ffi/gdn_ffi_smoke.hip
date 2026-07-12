// SPDX-License-Identifier: Apache-2.0
//
// Typed-FFI stream/lifetime smoke for a future gfx1100 GDN superblock.
// This is an exact-shape BF16 copy, not GDN math and not a performance kernel.

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>
#include "xla/ffi/api/ffi.h"

namespace {

constexpr std::int64_t kBatch = 1;
constexpr std::int64_t kTokens = 1024;
constexpr std::int64_t kValueHeads = 32;
constexpr std::int64_t kHeadDimension = 128;
constexpr std::uint32_t kElements =
    kBatch * kTokens * kValueHeads * kHeadDimension;
constexpr std::size_t kBytes = 8 * 1024 * 1024;
constexpr std::uint32_t kThreads = 256;
constexpr std::uint32_t kBlocks = 256;

static_assert(kElements == 4'194'304);
static_assert(kElements * sizeof(std::uint16_t) == kBytes);

__global__ __launch_bounds__(kThreads) void GdnFfiSmokeCopyKernel(
    const std::uint16_t* input, std::uint16_t* output) {
  const std::uint32_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t index = start; index < kElements; index += stride) {
    output[index] = input[index];
  }
}

bool HasExactShape(xla::ffi::Span<const std::int64_t> dimensions) {
  return dimensions.size() == 4 && dimensions[0] == kBatch &&
         dimensions[1] == kTokens && dimensions[2] == kValueHeads &&
         dimensions[3] == kHeadDimension;
}

xla::ffi::Error GdnFfiSmokeCopy(
    xla::ffi::BufferR4<xla::ffi::BF16> input,
    xla::ffi::ResultBufferR4<xla::ffi::BF16> output, hipStream_t stream) {
  if (!HasExactShape(input.dimensions()) ||
      !HasExactShape(output->dimensions()) || input.size_bytes() != kBytes ||
      output->size_bytes() != kBytes) {
    return xla::ffi::Error::InvalidArgument(
        "gdn_ffi_smoke requires exact BF16 [1,1024,32,128] buffers");
  }

  const auto* input_data = input.typed_data();
  auto* output_data = output->typed_data();
  if (input_data == nullptr || output_data == nullptr ||
      input_data == output_data) {
    return xla::ffi::Error::InvalidArgument(
        "gdn_ffi_smoke requires distinct non-null input and output buffers");
  }

  // Consume the host thread's saved launch status before enqueueing so a
  // stale error cannot be reported as a failure of this kernel.
  (void)hipGetLastError();
  hipLaunchKernelGGL(GdnFfiSmokeCopyKernel, dim3(kBlocks), dim3(kThreads), 0,
                     stream, input_data, output_data);
  if (hipGetLastError() != hipSuccess) {
    return xla::ffi::Error::Internal(
        "gdn_ffi_smoke asynchronous copy launch submission failed");
  }

  // The handler intentionally does not synchronize. XLA owns both buffers and
  // orders their lifetime through the supplied HIP stream.
  return xla::ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    skyrl_gdn_ffi_smoke_bf16_copy_v1, GdnFfiSmokeCopy,
    xla::ffi::Ffi::Bind()
        .Arg<xla::ffi::BufferR4<xla::ffi::BF16>>()
        .Ret<xla::ffi::BufferR4<xla::ffi::BF16>>()
        .Ctx<xla::ffi::PlatformStream<hipStream_t>>());
