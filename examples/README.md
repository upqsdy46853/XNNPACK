# XNNPACK examples

這個目錄放了三個直接使用 XNNPACK C API 的 C++ 範例：

- `simple_mlp_xnnpack.cc`：建立一個小型 FP32 MLP，並用 C++ reference 實作比對數值誤差。
- `simple_cnn_xnnpack.cc`：建立一個小型 FP32 CNN，並用 C++ reference 實作比對數值誤差。
- `mobilenet_label_xnnpack.cc`：讀取 MobileNet v1 `.tflite`、BMP 圖片與 labels，直接建立 XNNPACK subgraph 做分類。

## Build

請從 XNNPACK repo 根目錄執行：

```sh
cmake -S . -B build/examples \
  -DCMAKE_BUILD_TYPE=Release \
  -DXNNPACK_LIBRARY_TYPE=static \
  -DXNNPACK_BUILD_TESTS=OFF \
  -DXNNPACK_BUILD_BENCHMARKS=OFF \
  -DXNNPACK_BUILD_EXAMPLES=ON

cmake --build build/examples --parallel --target \
  simple_mlp_xnnpack \
  simple_cnn_xnnpack \
  mobilenet_label_xnnpack
```

第一次 configure/build 時，CMake 可能會下載 XNNPACK 的相依套件。若你使用的是 multi-config generator，例如 Visual Studio，執行檔可能會出現在 `build/examples/Release/` 這類設定目錄中；一般 Make/Ninja build 會直接輸出在 `build/examples/`。

## Run

以下指令都從 XNNPACK repo 根目錄執行。

### simple_mlp_xnnpack

```sh
./build/examples/simple_mlp_xnnpack
```

成功時會印出模型形狀、`max abs diff` 和第一列輸出。程式會在 XNNPACK 結果與 reference 結果的最大絕對誤差小於等於 `1.0e-4` 時回傳 `0`。

### simple_cnn_xnnpack

```sh
./build/examples/simple_cnn_xnnpack
```

成功時會印出模型形狀、`max abs diff` 和輸出向量。程式會在 XNNPACK 結果與 reference 結果的最大絕對誤差小於等於 `1.0e-4` 時回傳 `0`。

### mobilenet_label_xnnpack

```sh
./build/examples/mobilenet_label_xnnpack
```

不帶參數時會使用這些預設檔案：

- model: `examples/mobilenet_v1_1.0_224.tflite`
- image: `examples/grace_hopper.bmp`
- labels: `examples/labels.txt`

也可以自行指定 model、image、labels：

```sh
./build/examples/mobilenet_label_xnnpack \
  examples/mobilenet_v1_1.0_224.tflite \
  examples/grace_hopper.bmp \
  examples/labels.txt
```

成功時會印出 top-5 分類分數與 labels。

## Optional TensorFlow Lite reference

`label_image.py` 是 TensorFlow Lite 的 Python reference script，可用來和 C++ XNNPACK 範例結果做人工比對。它需要額外安裝 `tensorflow`、`numpy` 和 `Pillow`，不是執行 `mobilenet_label_xnnpack` 的必要條件。
