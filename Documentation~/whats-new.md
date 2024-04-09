# What's new in Sentis 1.4.0-pre.3

This is a summary of the changes from Sentis 1.3.0-pre.3 to Sentis 1.4.0-pre.3.

## Added

- Functional API for building and editing models using PyTorch style syntax. This includes operator overloads, sliced indexing with ranges and automatic type promotion.
- Quantization API for compressing model weights by up to a factor of 4. Quantized models take up less space on disk and use less memory during inference.
- Fast path for ScatterElements and GatherElements operations.
- Resize operator is supported on all back ends with all tensor shapes, and supports axes parameter.
- Pad operator supports integer tensors and wrap mode, and supports axes input tensor.
- Docs pages and package samples for new features.

## Updated

- Model serialization now uses FlatBuffers rather than binary serialization. This will allow far greater backwards compatibility for models exported in versions of Sentis from 1.4 onwards.
- Fixed many small importer and inference bugs for improved model compatibility.
- Renamed methods on Tensor and IWorker for clarity and consistency.
- Docs pages and package samples to be accurate to the new API.

## Removed

- Model API where inputs, constants and layers can be edited directly on a model. See the new functional API where you can build new models as well as adapt and extend existing models. 
- Ops for direct operations on tensors. See the new functional API where you can create models to perform optimized operations on tensors.
- Custom ONNX layers can no longer be defined by users as they were not compatible with the new model serialization scheme. Custom import will be reworked for an upcoming release.

# What's new in Sentis 1.3.0-pre.3

This is a summary of the changes from Sentis 1.3.0-pre.2 to Sentis 1.3.0-pre.3.

## Added

- Docs pages for exporting models, and docs menu for project submissions.

## Updated

- Updated Clip operator to support integer data type.
- Fixed inference and compatibility issues for Resize, Dense, Split and NonZero operators.
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

- Removed the license requirement for using the Sentis package.
- Removed the watermark on content generated with Sentis.

## Added

- Support and improvement for LayerNormalization, Gelu, Resize1D, Upsample, OneHot operators.
- Improvements for model import and saving to serialized assets.
- Optimizations for models with ScalarMad layer and operator.

# What's new in Sentis 1.1

This is a summary of the changes from Sentis 1.0 to Sentis 1.1.

## Added

- GPUPixel back end now supports TensorInts and many more operators.
- Models can now be exported to `StreamingAssets`.
- Interface for asynchronous readback of GPU tensors with callback.
- Ops utility methods for float mathematics and getting and setting slices of tensors. 

## Updated

- Improved compatibility for many operators including Conv, ConvTranspose and BatchNormalization.
- Improved performance of many operators on GPUCompute and GPUCommandBuffer back ends.
- Reduced allocation when executing worker.
- Improved import times.
- Improved shape and data type inference to optimize models better.
- Inference accuracy fixes.

# What's new in Sentis 1.0

Sentis is the new name for the [Barracuda package](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/manual/index.html).

This is a summary of the changes from Barracuda 3.0 to Sentis 1.0.

For information on how to upgrade, refer to the [Upgrade Guide](upgrade-guide.md).

## Added

- You can now override model layers in your code, and add your own layers. Refer to the `Add a custom layer` example in the [sample scripts](package-samples.md) for an example.


## Updated

- All the back end types have been optimized and improved so Sentis as a whole has better performance on both the GPU and the CPU.
- The way Sentis represents tensors has changed to make Sentis compatible with more models, and make it easier to convert Python preprocessing code into C#. Refer to [Tensor fundamentals](tensor-fundamentals.md) and [Do operations on tensors](do-operations-on-tensors.md) for more information.
- Sentis supports more Open Neural Network Exchange (ONNX) operators. Refer to [Supported ONNX operators](supported-operators.md) for more information. 
- You can now import models larger than 2 gigabytes.
- The [Model Asset Import Settings](onnx-model-importer-properties.md) window no longer contain contains `Force Arbitrary Batch Size` and `Treat Errors as Warnings` settings. The window also no longer contains an `Open imported NN model as temp file` button. Use `Open` instead.
