// Run the TensorFlow Lite MobileNet label_image demo through XNNPACK directly.
//
// This example intentionally does not use the TensorFlow Lite interpreter. It
// reads the .tflite FlatBuffer, defines the model as an XNNPACK subgraph, loads
// grace_hopper.bmp, runs inference, and prints the top labels.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/xnnpack.h"

namespace {

constexpr const char* kDefaultModelPath = "examples/mobilenet_v1_1.0_224.tflite";
constexpr const char* kDefaultImagePath = "examples/grace_hopper.bmp";
constexpr const char* kDefaultLabelPath = "examples/labels.txt";
constexpr float kInputMean = 127.5f;
constexpr float kInputStd = 127.5f;
constexpr uint32_t kExternalInputId = 0;
constexpr uint32_t kExternalOutputId = 1;

enum TfliteTensorType : int8_t {
  kTfliteFloat32 = 0,
};

enum TfliteBuiltinOptions : uint8_t {
  kTfliteConv2DOptions = 1,
  kTfliteDepthwiseConv2DOptions = 2,
  kTflitePool2DOptions = 5,
  kTfliteSoftmaxOptions = 9,
  kTfliteSqueezeOptions = 30,
};

enum TflitePadding : int8_t {
  kTfliteSamePadding = 0,
  kTfliteValidPadding = 1,
};

enum TfliteActivation : int8_t {
  kTfliteActivationNone = 0,
  kTfliteActivationRelu = 1,
  kTfliteActivationRelu6 = 3,
};

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

std::vector<uint8_t> ReadBinaryFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open " + path);
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(file),
                              std::istreambuf_iterator<char>());
}

std::vector<std::string> ReadLabels(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("failed to open " + path);
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(file, line)) {
    labels.push_back(line);
  }
  return labels;
}

template <typename T>
T ReadScalar(const std::vector<uint8_t>& data, size_t offset) {
  if (offset + sizeof(T) > data.size()) {
    throw std::runtime_error("unexpected end of FlatBuffer");
  }
  T value;
  std::memcpy(&value, data.data() + offset, sizeof(T));
  return value;
}

class FlatTable {
 public:
  FlatTable() = default;
  FlatTable(const std::vector<uint8_t>* data, size_t pos)
      : data_(data), pos_(pos) {}

  bool IsValid() const { return data_ != nullptr; }

  template <typename T>
  T Scalar(uint16_t field, T default_value) const {
    const uint16_t offset = FieldOffset(field);
    if (offset == 0) {
      return default_value;
    }
    return ReadScalar<T>(*data_, pos_ + offset);
  }

  FlatTable TableField(uint16_t field) const {
    const uint16_t offset = FieldOffset(field);
    if (offset == 0) {
      return FlatTable();
    }
    const size_t offset_position = pos_ + offset;
    return FlatTable(data_, offset_position + ReadScalar<uint32_t>(*data_, offset_position));
  }

  size_t VectorStart(uint16_t field) const {
    const uint16_t offset = FieldOffset(field);
    if (offset == 0) {
      return 0;
    }
    const size_t offset_position = pos_ + offset;
    return offset_position + ReadScalar<uint32_t>(*data_, offset_position);
  }

  uint32_t VectorLength(uint16_t field) const {
    const size_t vector = VectorStart(field);
    if (vector == 0) {
      return 0;
    }
    return ReadScalar<uint32_t>(*data_, vector);
  }

  template <typename T>
  std::vector<T> ScalarVector(uint16_t field) const {
    const size_t vector = VectorStart(field);
    if (vector == 0) {
      return {};
    }
    const uint32_t length = ReadScalar<uint32_t>(*data_, vector);
    std::vector<T> values(length);
    for (uint32_t i = 0; i < length; i++) {
      values[i] = ReadScalar<T>(*data_, vector + 4 + i * sizeof(T));
    }
    return values;
  }

  FlatTable TableVectorElement(uint16_t field, uint32_t index) const {
    const size_t vector = VectorStart(field);
    if (vector == 0) {
      throw std::runtime_error("missing table vector");
    }
    const uint32_t length = ReadScalar<uint32_t>(*data_, vector);
    if (index >= length) {
      throw std::runtime_error("table vector index out of range");
    }
    const size_t element = vector + 4 + index * sizeof(uint32_t);
    return FlatTable(data_, element + ReadScalar<uint32_t>(*data_, element));
  }

  std::vector<uint8_t> ByteVector(uint16_t field) const {
    const size_t vector = VectorStart(field);
    if (vector == 0) {
      return {};
    }
    const uint32_t length = ReadScalar<uint32_t>(*data_, vector);
    return std::vector<uint8_t>(data_->begin() + vector + 4,
                                data_->begin() + vector + 4 + length);
  }

  std::string String(uint16_t field) const {
    const size_t vector = VectorStart(field);
    if (vector == 0) {
      return {};
    }
    const uint32_t length = ReadScalar<uint32_t>(*data_, vector);
    return std::string(reinterpret_cast<const char*>(data_->data() + vector + 4),
                       length);
  }

 private:
  uint16_t FieldOffset(uint16_t field) const {
    if (!IsValid()) {
      return 0;
    }
    const int32_t vtable_distance = ReadScalar<int32_t>(*data_, pos_);
    const size_t vtable = pos_ - static_cast<size_t>(vtable_distance);
    const uint16_t vtable_size = ReadScalar<uint16_t>(*data_, vtable);
    if (field >= vtable_size) {
      return 0;
    }
    return ReadScalar<uint16_t>(*data_, vtable + field);
  }

  const std::vector<uint8_t>* data_ = nullptr;
  size_t pos_ = 0;
};

FlatTable GetTfliteModelRoot(const std::vector<uint8_t>& model_data) {
  if (model_data.size() < 8 || std::memcmp(model_data.data() + 4, "TFL3", 4) != 0) {
    throw std::runtime_error("not a TFLite FlatBuffer");
  }
  return FlatTable(&model_data, ReadScalar<uint32_t>(model_data, 0));
}

struct TensorInfo {
  std::string name;
  std::vector<size_t> shape;
  int8_t type = 0;
  uint32_t buffer = 0;
  uint32_t value_id = XNN_INVALID_VALUE_ID;
  std::vector<float> data;
};

struct ParsedTflite {
  std::vector<uint8_t> model_data;
  FlatTable model;
  FlatTable subgraph;
  std::vector<TensorInfo> tensors;
  int32_t input_tensor = -1;
  int32_t output_tensor = -1;
};

std::vector<size_t> ShapeFromTensor(const FlatTable& tensor) {
  const std::vector<int32_t> tflite_shape = tensor.ScalarVector<int32_t>(4);
  std::vector<size_t> shape;
  shape.reserve(tflite_shape.size());
  for (int32_t dim : tflite_shape) {
    if (dim < 0) {
      throw std::runtime_error("dynamic tensor shape is not supported");
    }
    shape.push_back(static_cast<size_t>(dim));
  }
  return shape;
}

size_t NumElements(const std::vector<size_t>& shape) {
  size_t elements = 1;
  for (size_t dim : shape) {
    elements *= dim;
  }
  return elements;
}

std::vector<float> BufferAsPaddedFloats(const FlatTable& buffer) {
  const std::vector<uint8_t> bytes = buffer.ByteVector(4);
  if (bytes.empty()) {
    return {};
  }
  if (bytes.size() % sizeof(float) != 0) {
    throw std::runtime_error("only float32 buffers are supported");
  }
  const size_t elements = bytes.size() / sizeof(float);
  std::vector<float> values(WithXnnExtraBytes(elements), 0.0f);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::pair<float, float> ActivationBounds(int8_t activation) {
  switch (activation) {
    case kTfliteActivationNone:
      return {-std::numeric_limits<float>::infinity(),
              std::numeric_limits<float>::infinity()};
    case kTfliteActivationRelu:
      return {0.0f, std::numeric_limits<float>::infinity()};
    case kTfliteActivationRelu6:
      return {0.0f, 6.0f};
    default:
      throw std::runtime_error("unsupported fused activation " +
                               std::to_string(static_cast<int>(activation)));
  }
}

uint32_t PaddingFlags(int8_t padding) {
  if (padding == kTfliteSamePadding) {
    return XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }
  if (padding == kTfliteValidPadding) {
    return 0;
  }
  throw std::runtime_error("unsupported padding " +
                           std::to_string(static_cast<int>(padding)));
}

void DefineTensorValues(ParsedTflite& parsed, xnn_subgraph_t subgraph) {
  const uint32_t num_tensors = parsed.subgraph.VectorLength(4);
  parsed.tensors.resize(num_tensors);

  for (uint32_t i = 0; i < num_tensors; i++) {
    const FlatTable tensor = parsed.subgraph.TableVectorElement(4, i);
    TensorInfo& info = parsed.tensors[i];
    info.shape = ShapeFromTensor(tensor);
    info.type = tensor.Scalar<int8_t>(6, 0);
    info.buffer = tensor.Scalar<uint32_t>(8, 0);
    info.name = tensor.String(10);

    if (info.type != kTfliteFloat32) {
      throw std::runtime_error("only float32 tensors are supported: " +
                               info.name);
    }

    const FlatTable buffer = parsed.model.TableVectorElement(12, info.buffer);
    info.data = BufferAsPaddedFloats(buffer);
    const float* data = info.data.empty() ? nullptr : info.data.data();

    uint32_t external_id = XNN_INVALID_VALUE_ID;
    uint32_t flags = 0;
    if (static_cast<int32_t>(i) == parsed.input_tensor) {
      external_id = kExternalInputId;
      flags = XNN_VALUE_FLAG_EXTERNAL_INPUT;
      info.value_id = kExternalInputId;
    } else if (static_cast<int32_t>(i) == parsed.output_tensor) {
      external_id = kExternalOutputId;
      flags = XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
      info.value_id = kExternalOutputId;
    } else {
      info.value_id = XNN_INVALID_VALUE_ID;
    }

    Check(xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                  info.shape.size(), info.shape.data(), data,
                                  external_id, flags, &info.value_id),
          ("define tensor " + info.name).c_str());
  }
}

void DefineOperators(const ParsedTflite& parsed, xnn_subgraph_t subgraph) {
  const uint32_t num_operators = parsed.subgraph.VectorLength(10);
  for (uint32_t i = 0; i < num_operators; i++) {
    const FlatTable op = parsed.subgraph.TableVectorElement(10, i);
    const std::vector<int32_t> inputs = op.ScalarVector<int32_t>(6);
    const std::vector<int32_t> outputs = op.ScalarVector<int32_t>(8);
    const uint8_t options_type = op.Scalar<uint8_t>(10, 0);
    const FlatTable options = op.TableField(12);

    if (outputs.empty()) {
      throw std::runtime_error("operator has no outputs");
    }

    if (options_type == kTfliteConv2DOptions) {
      if (inputs.size() != 3 || outputs.size() != 1) {
        throw std::runtime_error("Conv2D has unexpected arity");
      }
      const TensorInfo& input = parsed.tensors.at(inputs[0]);
      const TensorInfo& filter = parsed.tensors.at(inputs[1]);
      const int8_t padding = options.Scalar<int8_t>(4, 0);
      const uint32_t stride_w = static_cast<uint32_t>(options.Scalar<int32_t>(6, 0));
      const uint32_t stride_h = static_cast<uint32_t>(options.Scalar<int32_t>(8, 0));
      const int8_t activation = options.Scalar<int8_t>(10, 0);
      const uint32_t dilation_w = static_cast<uint32_t>(options.Scalar<int32_t>(12, 1));
      const uint32_t dilation_h = static_cast<uint32_t>(options.Scalar<int32_t>(14, 1));
      const auto bounds = ActivationBounds(activation);

      Check(xnn_define_convolution_2d(
                subgraph,
                /*input_padding_top=*/0,
                /*input_padding_right=*/0,
                /*input_padding_bottom=*/0,
                /*input_padding_left=*/0,
                /*kernel_height=*/static_cast<uint32_t>(filter.shape.at(1)),
                /*kernel_width=*/static_cast<uint32_t>(filter.shape.at(2)),
                /*subsampling_height=*/stride_h,
                /*subsampling_width=*/stride_w,
                /*dilation_height=*/dilation_h,
                /*dilation_width=*/dilation_w,
                /*groups=*/1,
                /*group_input_channels=*/input.shape.at(3),
                /*group_output_channels=*/filter.shape.at(0), bounds.first,
                bounds.second, input.value_id, parsed.tensors.at(inputs[1]).value_id,
                parsed.tensors.at(inputs[2]).value_id,
                parsed.tensors.at(outputs[0]).value_id, PaddingFlags(padding)),
            "xnn_define_convolution_2d");
    } else if (options_type == kTfliteDepthwiseConv2DOptions) {
      if (inputs.size() != 3 || outputs.size() != 1) {
        throw std::runtime_error("DepthwiseConv2D has unexpected arity");
      }
      const TensorInfo& input = parsed.tensors.at(inputs[0]);
      const TensorInfo& filter = parsed.tensors.at(inputs[1]);
      const int8_t padding = options.Scalar<int8_t>(4, 0);
      const uint32_t stride_w = static_cast<uint32_t>(options.Scalar<int32_t>(6, 0));
      const uint32_t stride_h = static_cast<uint32_t>(options.Scalar<int32_t>(8, 0));
      const uint32_t depth_multiplier =
          static_cast<uint32_t>(options.Scalar<int32_t>(10, 0));
      const int8_t activation = options.Scalar<int8_t>(12, 0);
      const uint32_t dilation_w = static_cast<uint32_t>(options.Scalar<int32_t>(14, 1));
      const uint32_t dilation_h = static_cast<uint32_t>(options.Scalar<int32_t>(16, 1));
      const auto bounds = ActivationBounds(activation);

      Check(xnn_define_depthwise_convolution_2d(
                subgraph,
                /*input_padding_top=*/0,
                /*input_padding_right=*/0,
                /*input_padding_bottom=*/0,
                /*input_padding_left=*/0,
                /*kernel_height=*/static_cast<uint32_t>(filter.shape.at(1)),
                /*kernel_width=*/static_cast<uint32_t>(filter.shape.at(2)),
                /*subsampling_height=*/stride_h,
                /*subsampling_width=*/stride_w,
                /*dilation_height=*/dilation_h,
                /*dilation_width=*/dilation_w,
                /*depth_multiplier=*/depth_multiplier,
                /*input_channels=*/input.shape.at(3), bounds.first, bounds.second,
                input.value_id, parsed.tensors.at(inputs[1]).value_id,
                parsed.tensors.at(inputs[2]).value_id,
                parsed.tensors.at(outputs[0]).value_id, PaddingFlags(padding)),
            "xnn_define_depthwise_convolution_2d");
    } else if (options_type == kTflitePool2DOptions) {
      if (inputs.size() != 1 || outputs.size() != 1) {
        throw std::runtime_error("AveragePool2D has unexpected arity");
      }
      const int8_t padding = options.Scalar<int8_t>(4, 0);
      const uint32_t stride_w = static_cast<uint32_t>(options.Scalar<int32_t>(6, 0));
      const uint32_t stride_h = static_cast<uint32_t>(options.Scalar<int32_t>(8, 0));
      const uint32_t filter_w = static_cast<uint32_t>(options.Scalar<int32_t>(10, 0));
      const uint32_t filter_h = static_cast<uint32_t>(options.Scalar<int32_t>(12, 0));
      const int8_t activation = options.Scalar<int8_t>(14, 0);
      const auto bounds = ActivationBounds(activation);

      Check(xnn_define_average_pooling_2d(
                subgraph,
                /*input_padding_top=*/0,
                /*input_padding_right=*/0,
                /*input_padding_bottom=*/0,
                /*input_padding_left=*/0,
                /*pooling_height=*/filter_h,
                /*pooling_width=*/filter_w,
                /*stride_height=*/stride_h,
                /*stride_width=*/stride_w, bounds.first, bounds.second,
                parsed.tensors.at(inputs[0]).value_id,
                parsed.tensors.at(outputs[0]).value_id, PaddingFlags(padding)),
            "xnn_define_average_pooling_2d");
    } else if (options_type == kTfliteSqueezeOptions) {
      Check(xnn_define_static_reshape(
                subgraph, parsed.tensors.at(outputs[0]).shape.size(),
                parsed.tensors.at(outputs[0]).shape.data(),
                parsed.tensors.at(inputs[0]).value_id,
                parsed.tensors.at(outputs[0]).value_id, /*flags=*/0),
            "xnn_define_static_reshape squeeze");
    } else if (options_type == kTfliteSoftmaxOptions) {
      const float beta = options.Scalar<float>(4, 0.0f);
      if (std::fabs(beta - 1.0f) > 1.0e-6f) {
        throw std::runtime_error("XNNPACK softmax path expects beta=1");
      }
      Check(xnn_define_softmax(subgraph, parsed.tensors.at(inputs[0]).value_id,
                               parsed.tensors.at(outputs[0]).value_id,
                               /*flags=*/0),
            "xnn_define_softmax");
    } else {
      throw std::runtime_error("unsupported builtin options type " +
                               std::to_string(static_cast<int>(options_type)));
    }
  }
}

ParsedTflite ParseAndDefineTfliteModel(const std::string& model_path,
                                       xnn_subgraph_t subgraph) {
  ParsedTflite parsed;
  parsed.model_data = ReadBinaryFile(model_path);
  parsed.model = GetTfliteModelRoot(parsed.model_data);
  parsed.subgraph = parsed.model.TableVectorElement(8, 0);

  const std::vector<int32_t> inputs = parsed.subgraph.ScalarVector<int32_t>(6);
  const std::vector<int32_t> outputs = parsed.subgraph.ScalarVector<int32_t>(8);
  if (inputs.size() != 1 || outputs.size() != 1) {
    throw std::runtime_error("only one-input/one-output models are supported");
  }
  parsed.input_tensor = inputs[0];
  parsed.output_tensor = outputs[0];

  DefineTensorValues(parsed, subgraph);
  DefineOperators(parsed, subgraph);
  return parsed;
}

struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgb;
};

struct ResizeCoefficients {
  struct Bounds {
    size_t start = 0;
    size_t length = 0;
  };

  size_t kernel_size = 0;
  std::vector<Bounds> bounds;
  std::vector<double> weights;
};

Image LoadBmp(const std::string& path) {
  const std::vector<uint8_t> bytes = ReadBinaryFile(path);
  if (bytes.size() < 54 || bytes[0] != 'B' || bytes[1] != 'M') {
    throw std::runtime_error("expected a BMP image");
  }
  const uint32_t pixel_offset = ReadScalar<uint32_t>(bytes, 10);
  const int32_t width = ReadScalar<int32_t>(bytes, 18);
  const int32_t signed_height = ReadScalar<int32_t>(bytes, 22);
  const uint16_t planes = ReadScalar<uint16_t>(bytes, 26);
  const uint16_t bits_per_pixel = ReadScalar<uint16_t>(bytes, 28);
  const uint32_t compression = ReadScalar<uint32_t>(bytes, 30);
  if (width <= 0 || signed_height == 0 || planes != 1 || bits_per_pixel != 24 ||
      compression != 0) {
    throw std::runtime_error("only uncompressed 24-bit BMP images are supported");
  }

  const int height = signed_height < 0 ? -signed_height : signed_height;
  const bool top_down = signed_height < 0;
  const size_t row_stride = ((static_cast<size_t>(width) * 3 + 3) / 4) * 4;
  if (pixel_offset + row_stride * static_cast<size_t>(height) > bytes.size()) {
    throw std::runtime_error("BMP pixel data is truncated");
  }

  Image image;
  image.width = width;
  image.height = height;
  image.rgb.resize(static_cast<size_t>(width) * height * 3);
  for (int y = 0; y < height; y++) {
    const int source_y = top_down ? y : height - 1 - y;
    const size_t source_row = pixel_offset + static_cast<size_t>(source_y) * row_stride;
    for (int x = 0; x < width; x++) {
      const size_t source = source_row + static_cast<size_t>(x) * 3;
      const size_t dest = (static_cast<size_t>(y) * width + x) * 3;
      image.rgb[dest + 0] = bytes[source + 2];
      image.rgb[dest + 1] = bytes[source + 1];
      image.rgb[dest + 2] = bytes[source + 0];
    }
  }
  return image;
}

double BicubicFilter(double x) {
  x = std::fabs(x);
  if (x < 1.0) {
    return ((1.5 * x - 2.5) * x) * x + 1.0;
  }
  if (x < 2.0) {
    return (((-0.5 * x + 2.5) * x - 4.0) * x) + 2.0;
  }
  return 0.0;
}

ResizeCoefficients PrecomputePillowBicubicCoefficients(size_t input_size,
                                                       size_t output_size) {
  constexpr double kBicubicSupport = 2.0;
  const double scale = static_cast<double>(input_size) / output_size;
  const double filter_scale = std::max(scale, 1.0);
  const double support = kBicubicSupport * filter_scale;

  ResizeCoefficients coefficients;
  coefficients.kernel_size =
      static_cast<size_t>(std::ceil(support)) * 2 + 1;
  coefficients.bounds.resize(output_size);
  coefficients.weights.assign(output_size * coefficients.kernel_size, 0.0);

  for (size_t output = 0; output < output_size; output++) {
    const double center = (static_cast<double>(output) + 0.5) * scale;
    int64_t xmin = static_cast<int64_t>(center - support + 0.5);
    int64_t xmax = static_cast<int64_t>(center + support + 0.5);
    xmin = std::max<int64_t>(xmin, 0);
    xmax = std::min<int64_t>(xmax, static_cast<int64_t>(input_size));

    const size_t start = static_cast<size_t>(xmin);
    const size_t length = static_cast<size_t>(std::max<int64_t>(xmax - xmin, 0));
    coefficients.bounds[output] = ResizeCoefficients::Bounds{start, length};

    double weights_sum = 0.0;
    double* weights =
        coefficients.weights.data() + output * coefficients.kernel_size;
    for (size_t i = 0; i < length; i++) {
      weights[i] = BicubicFilter(
          (static_cast<double>(start + i) - center + 0.5) / filter_scale);
      weights_sum += weights[i];
    }
    if (weights_sum != 0.0) {
      for (size_t i = 0; i < length; i++) {
        weights[i] /= weights_sum;
      }
    }
  }
  return coefficients;
}

uint8_t ClampToByte(double value) {
  if (value <= 0.0) {
    return 0;
  }
  if (value >= 255.0) {
    return 255;
  }
  return static_cast<uint8_t>(value + 0.5);
}

std::vector<float> ResizeAndNormalizePillowBicubic(const Image& image,
                                                   size_t width,
                                                   size_t height) {
  const ResizeCoefficients x_coefficients =
      PrecomputePillowBicubicCoefficients(image.width, width);
  const ResizeCoefficients y_coefficients =
      PrecomputePillowBicubicCoefficients(image.height, height);

  std::vector<uint8_t> horizontal(
      static_cast<size_t>(image.height) * width * 3);
  for (size_t y = 0; y < static_cast<size_t>(image.height); y++) {
    for (size_t x = 0; x < width; x++) {
      const ResizeCoefficients::Bounds bounds = x_coefficients.bounds[x];
      const double* weights =
          x_coefficients.weights.data() + x * x_coefficients.kernel_size;
      for (size_t c = 0; c < 3; c++) {
        double value = 0.0;
        for (size_t i = 0; i < bounds.length; i++) {
          const size_t source = (y * image.width + bounds.start + i) * 3 + c;
          value += weights[i] * static_cast<double>(image.rgb[source]);
        }
        horizontal[(y * width + x) * 3 + c] = ClampToByte(value);
      }
    }
  }

  std::vector<float> input(WithXnnExtraBytes(width * height * 3), 0.0f);
  for (size_t y = 0; y < height; y++) {
    const ResizeCoefficients::Bounds bounds = y_coefficients.bounds[y];
    const double* weights =
        y_coefficients.weights.data() + y * y_coefficients.kernel_size;
    for (size_t x = 0; x < width; x++) {
      for (size_t c = 0; c < 3; c++) {
        double value = 0.0;
        for (size_t i = 0; i < bounds.length; i++) {
          const size_t source = ((bounds.start + i) * width + x) * 3 + c;
          value += weights[i] * static_cast<double>(horizontal[source]);
        }
        const uint8_t pixel = ClampToByte(value);
        input[(y * width + x) * 3 + c] =
            (static_cast<float>(pixel) - kInputMean) / kInputStd;
      }
    }
  }
  return input;
}

void PrintTopK(const std::vector<float>& scores,
               const std::vector<std::string>& labels, size_t k) {
  std::vector<size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + std::min(k, order.size()),
                    order.end(), [&](size_t a, size_t b) {
                      return scores[a] > scores[b];
                    });

  for (size_t rank = 0; rank < k && rank < order.size(); rank++) {
    const size_t index = order[rank];
    const std::string label =
        index < labels.size() ? labels[index] : std::string("<missing label>");
    std::cout << scores[index] << ": " << label << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string model_path = argc > 1 ? argv[1] : kDefaultModelPath;
    const std::string image_path = argc > 2 ? argv[2] : kDefaultImagePath;
    const std::string label_path = argc > 3 ? argv[3] : kDefaultLabelPath;

    Check(xnn_initialize(/*allocator=*/nullptr), "xnn_initialize");

    xnn_subgraph_t raw_subgraph = nullptr;
    Check(xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0,
                              &raw_subgraph),
          "xnn_create_subgraph");
    UniqueSubgraph subgraph(raw_subgraph);

    ParsedTflite parsed = ParseAndDefineTfliteModel(model_path, subgraph.get());
    const TensorInfo& input_tensor = parsed.tensors.at(parsed.input_tensor);
    const TensorInfo& output_tensor = parsed.tensors.at(parsed.output_tensor);
    if (input_tensor.shape.size() != 4 || input_tensor.shape[0] != 1 ||
        input_tensor.shape[3] != 3) {
      throw std::runtime_error("expected input shape [1, height, width, 3]");
    }

    Image image = LoadBmp(image_path);
    std::vector<float> input =
        ResizeAndNormalizePillowBicubic(image, input_tensor.shape[2],
                                        input_tensor.shape[1]);
    std::vector<float> output(WithXnnExtraBytes(NumElements(output_tensor.shape)),
                              0.0f);

    xnn_runtime_t raw_runtime = nullptr;
    Check(xnn_create_runtime_v4(subgraph.get(), /*weights_cache=*/nullptr,
                                /*workspace=*/nullptr, /*threadpool=*/nullptr,
                                /*flags=*/0, &raw_runtime),
          "xnn_create_runtime_v4");
    UniqueRuntime runtime(raw_runtime);

    Check(xnn_reshape_runtime(runtime.get()), "xnn_reshape_runtime");

    const std::array<xnn_external_value, 2> external_values = {{
        xnn_external_value{kExternalInputId, input.data()},
        xnn_external_value{kExternalOutputId, output.data()},
    }};
    Check(xnn_setup_runtime_v2(runtime.get(), external_values.size(),
                               external_values.data()),
          "xnn_setup_runtime_v2");

    Check(xnn_invoke_runtime(runtime.get()), "xnn_invoke_runtime");

    const std::vector<std::string> labels = ReadLabels(label_path);
    PrintTopK(output, labels, /*k=*/5);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
