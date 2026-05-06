// A minimal XNNPACK Subgraph API example.
//
// The model is a tiny FP32 MLP:
//
//   input [batch, input_features]
//     -> fully connected + ReLU
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

constexpr size_t kBatch = 64;
constexpr size_t kInputFeatures = 128;
constexpr size_t kHiddenFeatures = 64;
constexpr size_t kOutputFeatures = 10;

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
    throw std::runtime_error(std::string(what) + " failed with xnn_status " + std::to_string(static_cast<int>(status)));
  }
}

size_t WithXnnExtraBytes(size_t elements) {
  return elements + XNN_EXTRA_BYTES / sizeof(float);
}

void FillInput(std::vector<float>& input) {
  for (size_t i = 0; i < kBatch * kInputFeatures; i++) {
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
    bias[i] = static_cast<float>(static_cast<int>((i * 5 + 3) % 11) - 5) * scale;
  }
}

void FullyConnectedReference(const float* input, const float* weights,
                             const float* bias, size_t batch,
                             size_t input_features, size_t output_features,
                             bool relu, float* output) {
  for (size_t b = 0; b < batch; b++) {
    for (size_t oc = 0; oc < output_features; oc++) {
      float acc = bias[oc];
      for (size_t ic = 0; ic < input_features; ic++) {
        acc += input[b * input_features + ic] * weights[oc * input_features + ic];
      }
      output[b * output_features + oc] = relu ? std::max(acc, 0.0f) : acc;
    }
  }
}

void RunReferenceMlp(const std::vector<float>& input,
                     const std::vector<float>& fc1_weights,
                     const std::vector<float>& fc1_bias,
                     const std::vector<float>& fc2_weights,
                     const std::vector<float>& fc2_bias,
                     std::vector<float>& hidden, std::vector<float>& output) {
  FullyConnectedReference(input.data(), fc1_weights.data(), fc1_bias.data(),
                          kBatch, kInputFeatures, kHiddenFeatures,
                          /*relu=*/true, hidden.data());
  FullyConnectedReference(hidden.data(), fc2_weights.data(), fc2_bias.data(),
                          kBatch, kHiddenFeatures, kOutputFeatures,
                          /*relu=*/false, output.data());
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

    std::vector<float> input(WithXnnExtraBytes(kBatch * kInputFeatures));
    std::vector<float> xnn_output(WithXnnExtraBytes(kBatch * kOutputFeatures));
    std::vector<float> ref_output(kBatch * kOutputFeatures);
    std::vector<float> ref_hidden(kBatch * kHiddenFeatures);

    std::vector<float> fc1_weights( WithXnnExtraBytes(kHiddenFeatures * kInputFeatures));
    std::vector<float> fc1_bias(WithXnnExtraBytes(kHiddenFeatures));
    std::vector<float> fc2_weights( WithXnnExtraBytes(kOutputFeatures * kHiddenFeatures));
    std::vector<float> fc2_bias(WithXnnExtraBytes(kOutputFeatures));

    FillInput(input);
    FillWeights(fc1_weights, kHiddenFeatures * kInputFeatures, 0.006f);
    FillBias(fc1_bias, kHiddenFeatures, 0.02f);
    FillWeights(fc2_weights, kOutputFeatures * kHiddenFeatures, 0.01f);
    FillBias(fc2_bias, kOutputFeatures, 0.01f);

    xnn_subgraph_t raw_subgraph = nullptr;
    Check(xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &raw_subgraph), "xnn_create_subgraph");
    UniqueSubgraph subgraph(raw_subgraph);

    const std::array<size_t, 2> input_dims = {{kBatch, kInputFeatures}};
    uint32_t input_id = 0;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  input_dims.size(), input_dims.data(),
                                  /*data=*/nullptr, /*external_id=*/0,
                                  XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
          "define input");

    const std::array<size_t, 2> hidden_dims = {{kBatch, kHiddenFeatures}};
    uint32_t hidden_id = XNN_INVALID_VALUE_ID;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  hidden_dims.size(), hidden_dims.data(),
                                  /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                  /*flags=*/0, &hidden_id),
          "define hidden");

    const std::array<size_t, 2> output_dims = {{kBatch, kOutputFeatures}};
    uint32_t output_id = 1;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  output_dims.size(), output_dims.data(),
                                  /*data=*/nullptr, /*external_id=*/1,
                                  XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
          "define output");

    const std::array<size_t, 2> fc1_weights_dims = {
        {kHiddenFeatures, kInputFeatures}};
    uint32_t fc1_weights_id = XNN_INVALID_VALUE_ID;
    Check(xnn_define_tensor_value(
              subgraph.get(), xnn_datatype_fp32, fc1_weights_dims.size(),
              fc1_weights_dims.data(), fc1_weights.data(), XNN_INVALID_VALUE_ID,
              /*flags=*/0, &fc1_weights_id),
          "define fc1 weights");

    const std::array<size_t, 1> fc1_bias_dims = {{kHiddenFeatures}};
    uint32_t fc1_bias_id = XNN_INVALID_VALUE_ID;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  fc1_bias_dims.size(), fc1_bias_dims.data(),
                                  fc1_bias.data(), XNN_INVALID_VALUE_ID,
                                  /*flags=*/0, &fc1_bias_id),
          "define fc1 bias");

    const std::array<size_t, 2> fc2_weights_dims = {{kOutputFeatures, kHiddenFeatures}};
    uint32_t fc2_weights_id = XNN_INVALID_VALUE_ID;
    Check(xnn_define_tensor_value(
              subgraph.get(), xnn_datatype_fp32, fc2_weights_dims.size(),
              fc2_weights_dims.data(), fc2_weights.data(), XNN_INVALID_VALUE_ID,
              /*flags=*/0, &fc2_weights_id),
          "define fc2 weights");

    const std::array<size_t, 1> fc2_bias_dims = {{kOutputFeatures}};
    uint32_t fc2_bias_id = XNN_INVALID_VALUE_ID;
    Check(xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                  fc2_bias_dims.size(), fc2_bias_dims.data(),
                                  fc2_bias.data(), XNN_INVALID_VALUE_ID,
                                  /*flags=*/0, &fc2_bias_id),
          "define fc2 bias");

    Check(xnn_define_fully_connected(
              subgraph.get(),
              /*output_min=*/0.0f,
              /*output_max=*/std::numeric_limits<float>::infinity(), input_id,
              fc1_weights_id, fc1_bias_id, hidden_id, /*flags=*/0),
          "xnn_define_fully_connected fc1");

    Check(xnn_define_fully_connected(
              subgraph.get(),
              /*output_min=*/-std::numeric_limits<float>::infinity(),
              /*output_max=*/std::numeric_limits<float>::infinity(), hidden_id,
              fc2_weights_id, fc2_bias_id, output_id, /*flags=*/0),
          "xnn_define_fully_connected fc2");

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
    RunReferenceMlp(input, fc1_weights, fc1_bias, fc2_weights, fc2_bias, ref_hidden, ref_output);

    const float max_abs_diff = MaxAbsDiff(ref_output, xnn_output, kBatch * kOutputFeatures);

    std::cout << "model: [" << kBatch << "," << kInputFeatures << "] -> FC("
              << kHiddenFeatures << ")+ReLU -> FC(" << kOutputFeatures << ")\n";
    std::cout << "max abs diff: " << max_abs_diff << "\n";
    std::cout << "first output row:";
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
