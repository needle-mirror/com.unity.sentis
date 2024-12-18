# What's new in Sentis 2.1.2

### Added
- Support for accelerated inference using [DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/dml) in future Unity versions
- Support for RMSNormalization subgraph optimization

### Fixed
- Improved stability, performance and memory usage in the Unity Editor and Runtime

# What's new in Sentis 2.1.1

This is a summary of the changes from Sentis 2.1.0 to 2.1.1.

### Fixed
- Models over 2GB or with large constants (over 500MB) import and serialize correctly.
- The ordering of inputs and outputs is now guaranteed to match that in the ONNX file.
- Other small fixes for individual operators that were causing inference or import issues.
- Improved text and sample code in documentation.

# What's new in Sentis 2.1.0

This is a summary of the changes from Sentis 2.0.0 to 2.1.0.

### Updated
- Fast activation fusing for MatMul
- Batched MatMul optimization
- Faster Upload for CPUTensorData

### Fixed
- CPU Tensors were not properly disposed.
- Reshaping of empty tensor
- Exposed internal fields of TextureTensorData

# What's new in Sentis 2.0.0

This is a summary of the changes from Sentis 1.6.0-pre.1 to 2.0.0.

### Added
- Ability to copy model outputs into allocated tensors.
- Funtional API now shows the data type and shape (if known) for easy debugging.

### Updated
- Unified Tensor to allow easy definition of different data-types as well as data upload and shape manipulation
- Reworked and unified methods for setting scheduling a model.
- Improved functional API to make it easier to debug and define custom models.
- Unified Compute and CommandBuffer backend

# What's new in Sentis 1.6.0-pre.1

This is a summary of the changes from Sentis 1.5.0-pre.3 to 1.6.0-pre.1.

### Added
- Support for GridSample ONNX operator along with functional API method.
- Functional API method for numpy-style RandomChoice.
- BitonicSort backend method on GPUCompute for a fast GPU sort.
- Improved error handling when loading unsupported legacy .sentis models.

### Updated
- Fully rewritten NonMaxSuppression operator, with fast inference on CPU and GPUCompute backends.
- Reduced CPU allocation in some operators, resulting in less garbage collection.
- Fixed inference errors in for some operators such as Slice and Multinomial.
- Optimized inference for some Gather operations with a single index.
- Optimized loading of compute functions for better performance on GPU.
- Unified how tensors of zero length are handled reducing potential errors.
- New methods for downloading and cloning tensors to the CPU.

# What's new in Sentis 1.5.0-pre.3

This is a summary of the changes from Sentis 1.4.0-pre.3 to Sentis 1.5.0-pre.3.

### Updated
- Main thread cpu performance has been improved
- Unary pointwise operations are now faster on GPU/CPU
- Model serialization and import speed has been improved
- Project dependencies has been reworked to reduce build size
- Unconnected model input tensors are handled properly
- Better ONNX opset 18 support
- Multinomial randomness is coherent across frames
- Shader compilation errors on XBOX and Switch have been resolved

# What's new in Sentis 1.4.0-pre.3

This is a summary of the changes from Sentis 1.3.0-pre.3 to Sentis 1.4.0-pre.3.

## Added

- Functional API for building and editing models using PyTorch style syntax. This includes operator overloads, sliced indexing with ranges, and automatic type promotion.
- Quantization API for compressing model weights by up to a factor of 4. Quantized models take up less space on disk and use less memory during inference.
- Fast path for ScatterElements and GatherElements operations.
- All backends support the resize operator for all tensor shapes, and it includes support for the axes parameter.
- Pad operator supports integer tensors and wrap mode, and supports axes input tensor.
- Docs pages and package samples for new features.

## Updated

- Model serialization now uses FlatBuffers rather than binary serialization. This will allow far greater backwards compatibility for models exported in versions of Sentis from 1.4 onwards.
- Fixed many small importer and inference bugs for improved model compatibility.
- Renamed methods on Tensor and IWorker for clarity and consistency.
- Docs pages and package samples to be accurate to the new API.

## Removed

- Model API where inputs, constants, and layers can be edited directly on a model. See the new functional API where you can build new models as well as adapt and extend existing models.
- Ops for direct operations on tensors. See the new functional API where you can create models to perform optimized operations on tensors.
- Users can no longer define custom Open Neural Network Exchange (ONNX) layers due to their incompatibility with the new model serialization scheme. The custom import functionality will be reworked for an upcoming release.

# What's new in Sentis 1.3.0-pre.3

This is a summary of the changes from Sentis 1.3.0-pre.2 to Sentis 1.3.0-pre.3.

## Added

- Docs pages for exporting models and docs menu for project submissions.

## Updated

- Clip operator to support integer data type.
- Fixed inference and compatibility issues for Resize, Dense, Split, and NonZero operators.
- Inspector improvements including horizontal scrolling and operating while out of focus.
- Docstrings have been updated for accuracy and links to models in docs are fixed.

# What's new in Sentis 1.3.0-pre.1 and 1.3.0-pre.2

This is a summary of the changes from Sentis 1.2 to Sentis 1.3.0-pre.1 and 1.3.0-pre.2.

## Added

- Support for pre-allocated tensors in TextureConverter methods.
- Importing of models with external weight files.
- More Ops methods and ability to use generic types.
- Methods and sample for model encryption.

# What's new in Sentis 1.2

This is a summary of the changes from Sentis 1.1 to Sentis 1.2.

## Removed

- License requirement for using the Sentis package.
- Watermark on content generated with Sentis.

## Added

- Support and improvement for LayerNormalization, Gelu, Resize1D, Upsample, and OneHot operators.
- Improvements for model import and saving to serialized assets.
- Optimizations for models with ScalarMad layer and operator.

# What's new in Sentis 1.1

This is a summary of the changes from Sentis 1.0 to Sentis 1.1.

## Added

- GPUPixel backend now supports TensorInts and many more operators.
- Models can now be exported to `StreamingAssets`.
- Interface for asynchronous readback of GPU tensors with callback.
- Ops utility methods for float mathematics and getting and setting slices of tensors.

## Updated

- Improved compatibility for many operators including Conv, ConvTranspose, and BatchNormalization.
- Improved performance of many operators on GPUCompute and GPUCommandBuffer backends.
- Reduced allocation when executing worker.
- Improved import times.
- Improved shape and data type inference to optimize models better.
- Inference accuracy fixes.

# What's new in Sentis 1.0

Sentis is the new name for the [Barracuda package](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/manual/index.html).

This is a summary of the changes from Barracuda 3.0 to Sentis 1.0.

For information on how to upgrade, refer to the [Upgrade Guide](upgrade-guide.md).

## Added

- You can now override model layers in your code and add your own layers. Refer to the `Add a custom layer` example in the [sample scripts](package-samples.md) for an example.


## Updated

- All the backend types have been optimized and improved so Sentis as a whole has better performance on both the GPU and the CPU.
- The way Sentis represents tensors has changed to make Sentis compatible with more models and make it easier to convert Python preprocessing code into C#. Refer to [Tensor fundamentals](tensor-fundamentals.md) for more information.
- Sentis supports more ONNX operators. Refer to [Supported ONNX operators](supported-operators.md) for more information.
- You can now import models larger than two gigabytes.
- The [Model Asset Import Settings](onnx-model-importer-properties.md) window no longer has the options for `Force Arbitrary Batch Size` and `Treat Errors as Warnings` settings. Additionally, the `Open imported NN model as temp file` button is no longer available, simply use the `Open` button.
