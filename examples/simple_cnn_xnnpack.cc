// A minimal XNNPACK Subgraph API example for a small CNN.
//
// The model is:
//
//   input [batch, height, width, input_channels]
//     -> 3x3 convolution + ReLU
//     -> 3x3 convolution + ReLU
//     -> flatten
//     -> fully connected
//     -> output [batch, output_features]
//
// The same network is also computed with plain C++ loops so the numerical
// result can be compared.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/xnnpack.h"

namespace {

constexpr size_t kBatch = 1;
constexpr size_t kInputHeight = 8;
constexpr size_t kInputWidth = 8;
constexpr size_t kInputChannels = 3;
constexpr size_t kConv1Channels = 4;
constexpr size_t kConv2Channels = 5;
constexpr size_t kKernelHeight = 3;
constexpr size_t kKernelWidth = 3;
constexpr size_t kPadding = 1;
constexpr size_t kFlattenFeatures = kInputHeight * kInputWidth * kConv2Channels;
constexpr size_t kOutputFeatures = 6;

struct XnnSubgraphDeleter {
  void operator()(xnn_subgraph_t subgraph) const {
    if (subgraph != nullptr) {
      xnn_delete_subgraph(subgraph);
    }
  }
};

struct XnnRuntimeDeleter {
  void operator()(xnn_runtime_t runtime) const {
    if (runtime != nullptr) {
      xnn_delete_runtime(runtime);
    }
  }
};

using UniqueSubgraph = std::unique_ptr<xnn_subgraph, XnnSubgraphDeleter>;
using UniqueRuntime = std::unique_ptr<xnn_runtime, XnnRuntimeDeleter>;

void Check(xnn_status status, const char* what) {
  if (status != xnn_status_success) {
    throw std::runtime_error(std::string(what) + " failed with xnn_status " +
                             std::to_string(static_cast<int>(status)));
  }
}

size_t WithXnnExtraBytes(size_t elements) {
  return elements + XNN_EXTRA_BYTES / sizeof(float);
}

void FillInput(std::vector<float>& input) {
  for (size_t i = 0; i < kBatch * kInputHeight * kInputWidth * kInputChannels;
       i++) {
    input[i] = static_cast<float>(static_cast<int>((i * 17) % 41) - 20) / 20.0f;
  }
}

void FillWeights(std::vector<float>& weights, size_t useful_elements,
                 float scale) {
  for (size_t i = 0; i < useful_elements; i++) {
    weights[i] =
        static_cast<float>(static_cast<int>((i * 13 + 7) % 37) - 18) * scale;
  }
}

void FillBias(std::vector<float>& bias, size_t useful_elements, float scale) {
  for (size_t i = 0; i < useful_elements; i++) {
    bias[i] =
        static_cast<float>(static_cast<int>((i * 5 + 3) % 11) - 5) * scale;
  }
}

size_t OffsetNHWC(size_t height, size_t width, size_t channels, size_t n,
                  size_t y, size_t x, size_t c) {
  return ((n * height + y) * width + x) * channels + c;
}

void Conv2DReference(const float* input, const float* weights,
                     const float* bias, size_t batch, size_t input_height,
                     size_t input_width, size_t input_channels,
                     size_t output_channels, bool relu, float* output) {
  for (size_t n = 0; n < batch; n++) {
    for (size_t oy = 0; oy < input_height; oy++) {
      for (size_t ox = 0; ox < input_width; ox++) {
        for (size_t oc = 0; oc < output_channels; oc++) {
          float acc = bias[oc];
          for (size_t ky = 0; ky < kKernelHeight; ky++) {
            const intptr_t iy =
                static_cast<intptr_t>(oy + ky) - static_cast<intptr_t>(kPadding);
            if (iy < 0 || iy >= static_cast<intptr_t>(input_height)) {
              continue;
            }
            for (size_t kx = 0; kx < kKernelWidth; kx++) {
              const intptr_t ix = static_cast<intptr_t>(ox + kx) -
                                  static_cast<intptr_t>(kPadding);
              if (ix < 0 || ix >= static_cast<intptr_t>(input_width)) {
                continue;
              }
              for (size_t ic = 0; ic < input_channels; ic++) {
                const size_t input_index = OffsetNHWC(input_height, input_width, input_channels, n, static_cast<size_t>(iy), static_cast<size_t>(ix), ic);
                const size_t weight_index = (((oc * kKernelHeight + ky) * kKernelWidth + kx) * input_channels + ic);
                acc += input[input_index] * weights[weight_index];
              }
            }
          }
          const size_t output_index = OffsetNHWC(input_height, input_width, output_channels, n, oy, ox, oc);
          output[output_index] = relu ? std::max(acc, 0.0f) : acc;
        }
      }
    }
  }
}

void FullyConnectedReference(const float* input, const float* weights,
                             const float* bias, size_t batch,
                             size_t input_features, size_t output_features,
                             float* output) {
  for (size_t b = 0; b < batch; b++) {
    for (size_t oc = 0; oc < output_features; oc++) {
      float acc = bias[oc];
      for (size_t ic = 0; ic < input_features; ic++) {
        acc += input[b * input_features + ic] * weights[oc * input_features + ic];
      }
      output[b * output_features + oc] = acc;
    }
  }
}

void RunReferenceCnn(const std::vector<float>& input,
                     const std::vector<float>& conv1_weights,
                     const std::vector<float>& conv1_bias,
                     const std::vector<float>& conv2_weights,
                     const std::vector<float>& conv2_bias,
                     const std::vector<float>& fc_weights,
                     const std::vector<float>& fc_bias,
                     std::vector<float>& conv1_output,
                     std::vector<float>& conv2_output,
                     std::vector<float>& output) {
  Conv2DReference(input.data(), conv1_weights.data(), conv1_bias.data(),
                  kBatch, kInputHeight, kInputWidth, kInputChannels,
                  kConv1Channels, /*relu=*/true, conv1_output.data());
  Conv2DReference(conv1_output.data(), conv2_weights.data(), conv2_bias.data(),
                  kBatch, kInputHeight, kInputWidth, kConv1Channels,
                  kConv2Channels, /*relu=*/true, conv2_output.data());
  FullyConnectedReference(conv2_output.data(), fc_weights.data(),
                          fc_bias.data(), kBatch, kFlattenFeatures,
                          kOutputFeatures, output.data());
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b,
                 size_t elements) {
  float max_diff = 0.0f;
  for (size_t i = 0; i < elements; i++) {
    max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
  }
  return max_diff;
}

}  // namespace

int main() {
  try {
    Check(xnn_initialize(/*allocator=*/nullptr), "xnn_initialize");

    std::vector<float> input(WithXnnExtraBytes(kBatch * kInputHeight * kInputWidth * kInputChannels));
    std::vector<float> xnn_output(WithXnnExtraBytes(kBatch * kOutputFeatures));
    std::vector<float> ref_output(kBatch * kOutputFeatures);
    std::vector<float> ref_conv1_output(kBatch * kInputHeight * kInputWidth * kConv1Channels);
    std::vector<float> ref_conv2_output(kBatch * kInputHeight * kInputWidth * kConv2Channels);

    std::vector<float> conv1_weights( WithXnnExtraBytes(kConv1Channels * kKernelHeight * kKernelWidth * kInputChannels));
    std::vector<float> conv1_bias(WithXnnExtraBytes(kConv1Channels));
    std::vector<float> conv2_weights( WithXnnExtraBytes(kConv2Channels * kKernelHeight * kKernelWidth * kConv1Channels));
    std::vector<float> conv2_bias(WithXnnExtraBytes(kConv2Channels));
    std::vector<float> fc_weights( WithXnnExtraBytes(kOutputFeatures * kFlattenFeatures));
    std::vector<float> fc_bias(WithXnnExtraBytes(kOutputFeatures));

    FillInput(input);
    FillWeights(conv1_weights, kConv1Channels * kKernelHeight * kKernelWidth * kInputChannels, 0.015f);
    FillBias(conv1_bias, kConv1Channels, 0.03f);
    FillWeights(conv2_weights, kConv2Channels * kKernelHeight * kKernelWidth * kConv1Channels, 0.010f);
    FillBias(conv2_bias, kConv2Channels, 0.02f);
    FillWeights(fc_weights, kOutputFeatures * kFlattenFeatures, 0.004f);
    FillBias(fc_bias, kOutputFeatures, 0.01f);

    xnn_subgraph_t raw_subgraph = nullptr;
    Check(xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &raw_subgraph), "xnn_create_subgraph");
    UniqueSubgraph subgraph(raw_subgraph);

    const std::array<size_t, 4> input_dims = {{kBatch, kInputHeight, kInputWidth, kInputChannels}};
    uint32_t input_id = 0;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  input_dims.size(), input_dims.data(),
                                  /*data=*/nullptr, /*external_id=*/0,
                                  XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
                                  "define input");

    const std::array<size_t, 4> conv1_output_dims = { {kBatch, kInputHeight, kInputWidth, kConv1Channels}};
    uint32_t conv1_output_id = XNN_INVALID_VALUE_ID;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  conv1_output_dims.size(),
                                  conv1_output_dims.data(), /*data=*/nullptr,
                                  XNN_INVALID_VALUE_ID, /*flags=*/0,
                                  &conv1_output_id),
                                  "define conv1 output");

    const std::array<size_t, 4> conv2_output_dims = { {kBatch, kInputHeight, kInputWidth, kConv2Channels}};
    uint32_t conv2_output_id = XNN_INVALID_VALUE_ID;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  conv2_output_dims.size(),
                                  conv2_output_dims.data(), /*data=*/nullptr,
                                  XNN_INVALID_VALUE_ID, /*flags=*/0,
                                  &conv2_output_id),
                                  "define conv2 output");

    const std::array<size_t, 2> flatten_dims = {{kBatch, kFlattenFeatures}};
    uint32_t flatten_id = XNN_INVALID_VALUE_ID;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  flatten_dims.size(), flatten_dims.data(),
                                  /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                  /*flags=*/0, &flatten_id),
                                  "define flatten");

    const std::array<size_t, 2> output_dims = {{kBatch, kOutputFeatures}};
    uint32_t output_id = 1;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  output_dims.size(), output_dims.data(),
                                  /*data=*/nullptr, /*external_id=*/1,
                                  XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
                                  "define output");

    const std::array<size_t, 4> conv1_weights_dims = { {kConv1Channels, kKernelHeight, kKernelWidth, kInputChannels}};
    uint32_t conv1_weights_id = XNN_INVALID_VALUE_ID;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  conv1_weights_dims.size(),
                                  conv1_weights_dims.data(),
                                  conv1_weights.data(), XNN_INVALID_VALUE_ID,
                                  /*flags=*/0, &conv1_weights_id),
                                  "define conv1 weights");

    const std::array<size_t, 1> conv1_bias_dims = {{kConv1Channels}};
    uint32_t conv1_bias_id = XNN_INVALID_VALUE_ID;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  conv1_bias_dims.size(),
                                  conv1_bias_dims.data(), conv1_bias.data(),
                                  XNN_INVALID_VALUE_ID, /*flags=*/0,
                                  &conv1_bias_id),
                                  "define conv1 bias");

    const std::array<size_t, 4> conv2_weights_dims = { {kConv2Channels, kKernelHeight, kKernelWidth, kConv1Channels}};
    uint32_t conv2_weights_id = XNN_INVALID_VALUE_ID;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  conv2_weights_dims.size(),
                                  conv2_weights_dims.data(),
                                  conv2_weights.data(), XNN_INVALID_VALUE_ID,
                                  /*flags=*/0, &conv2_weights_id),
                                  "define conv2 weights");

    const std::array<size_t, 1> conv2_bias_dims = {{kConv2Channels}};
    uint32_t conv2_bias_id = XNN_INVALID_VALUE_ID;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  conv2_bias_dims.size(),
                                  conv2_bias_dims.data(), conv2_bias.data(),
                                  XNN_INVALID_VALUE_ID, /*flags=*/0,
                                  &conv2_bias_id),
                                  "define conv2 bias");

    const std::array<size_t, 2> fc_weights_dims = { {kOutputFeatures, kFlattenFeatures}};
    uint32_t fc_weights_id = XNN_INVALID_VALUE_ID;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  fc_weights_dims.size(),
                                  fc_weights_dims.data(), fc_weights.data(),
                                  XNN_INVALID_VALUE_ID, /*flags=*/0,
                                  &fc_weights_id),
                                  "define fc weights");

    const std::array<size_t, 1> fc_bias_dims = {{kOutputFeatures}};
    uint32_t fc_bias_id = XNN_INVALID_VALUE_ID;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  fc_bias_dims.size(), fc_bias_dims.data(),
                                  fc_bias.data(), XNN_INVALID_VALUE_ID,
                                  /*flags=*/0, &fc_bias_id),
                                  "define fc bias");

    Check(xnn_define_convolution_2d(
              subgraph.get(),
              /*input_padding_top=*/kPadding,
              /*input_padding_right=*/kPadding,
              /*input_padding_bottom=*/kPadding,
              /*input_padding_left=*/kPadding,
              /*kernel_height=*/kKernelHeight,
              /*kernel_width=*/kKernelWidth,
              /*subsampling_height=*/1,
              /*subsampling_width=*/1,
              /*dilation_height=*/1,
              /*dilation_width=*/1,
              /*groups=*/1,
              /*group_input_channels=*/kInputChannels,
              /*group_output_channels=*/kConv1Channels,
              /*output_min=*/0.0f,
              /*output_max=*/std::numeric_limits<float>::infinity(), input_id,
              conv1_weights_id, conv1_bias_id, conv1_output_id, /*flags=*/0),
              "xnn_define_convolution_2d conv1");

    Check(xnn_define_convolution_2d(
              subgraph.get(),
              /*input_padding_top=*/kPadding,
              /*input_padding_right=*/kPadding,
              /*input_padding_bottom=*/kPadding,
              /*input_padding_left=*/kPadding,
              /*kernel_height=*/kKernelHeight,
              /*kernel_width=*/kKernelWidth,
              /*subsampling_height=*/1,
              /*subsampling_width=*/1,
              /*dilation_height=*/1,
              /*dilation_width=*/1,
              /*groups=*/1,
              /*group_input_channels=*/kConv1Channels,
              /*group_output_channels=*/kConv2Channels,
              /*output_min=*/0.0f,
              /*output_max=*/std::numeric_limits<float>::infinity(),
              conv1_output_id, conv2_weights_id, conv2_bias_id,
              conv2_output_id, /*flags=*/0),
              "xnn_define_convolution_2d conv2");

    Check(xnn_define_static_reshape(subgraph.get(), flatten_dims.size(),
                                    flatten_dims.data(), conv2_output_id,
                                    flatten_id, /*flags=*/0),
              "xnn_define_static_reshape flatten");

    Check(xnn_define_fully_connected(
              subgraph.get(),
              /*output_min=*/-std::numeric_limits<float>::infinity(),
              /*output_max=*/std::numeric_limits<float>::infinity(), flatten_id,
              fc_weights_id, fc_bias_id, output_id, /*flags=*/0),
              "xnn_define_fully_connected fc");

    xnn_runtime_t raw_runtime = nullptr;
    Check(xnn_create_runtime_v4(subgraph.get(), /*weights_cache=*/nullptr,
                                /*workspace=*/nullptr, /*threadpool=*/nullptr,
                                /*flags=*/0, &raw_runtime),
              "xnn_create_runtime_v4");
    UniqueRuntime runtime(raw_runtime);

    Check(xnn_reshape_runtime(runtime.get()), "xnn_reshape_runtime");

    const std::array<xnn_external_value, 2> external_values = {{
        xnn_external_value{input_id, input.data()},
        xnn_external_value{output_id, xnn_output.data()},
    }};
    Check(xnn_setup_runtime_v2(runtime.get(), external_values.size(), external_values.data()), "xnn_setup_runtime_v2");

    Check(xnn_invoke_runtime(runtime.get()), "xnn_invoke_runtime");
    RunReferenceCnn(input, conv1_weights, conv1_bias, conv2_weights, conv2_bias,
                    fc_weights, fc_bias, ref_conv1_output, ref_conv2_output,
                    ref_output);

    const float max_abs_diff = MaxAbsDiff(ref_output, xnn_output, kBatch * kOutputFeatures);

    std::cout << "model: [1," << kInputHeight << "," << kInputWidth << ","
              << kInputChannels << "] -> Conv(" << kConv1Channels
              << ")+ReLU -> Conv(" << kConv2Channels << ")+ReLU -> FC("
              << kOutputFeatures << ")\n";
    std::cout << "max abs diff: " << max_abs_diff << "\n";
    std::cout << "output:";
    for (size_t i = 0; i < kOutputFeatures; i++) {
      std::cout << " " << xnn_output[i];
    }
    std::cout << "\n";

    return max_abs_diff <= 1.0e-4f ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
