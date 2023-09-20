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

- GPUPixel backend now supports TensorInts and many more operators.
- Models can now be exported to `StreamingAssets`.
- Interface for asynchronous readback of GPU tensors with callback.
- Ops utility methods for float mathematics and getting and setting slices of tensors. 

## Updated

- Improved compatibility for many operators including Conv, ConvTranspose and BatchNormalization.
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

- You can now override model layers in your code, and add your own layers. Refer to the `Add a custom layer` example in the [sample scripts](package-samples.md) for an example.


## Updated

- All the back end types have been optimized and improved so Sentis as a whole has better performance on both the GPU and the CPU.
- The way Sentis represents tensors has changed to make Sentis compatible with more models, and make it easier to convert Python preprocessing code into C#. Refer to [Tensor fundamentals](tensor-fundamentals.md) and [Do operations on tensors](do-complex-tensor-operations.md) for more information.
- Sentis supports more Open Neural Network Exchange (ONNX) operators. Refer to [Supported ONNX operators](supported-operators.md) for more information. 
- You can now import models larger than 2 gigabytes.
- The [Model Asset Import Settings](onnx-model-importer-properties.md) window no longer contain contains `Force Arbitrary Batch Size` and `Treat Errors as Warnings` settings. The window also no longer contains an `Open imported NN model as temp file` button. Use `Open` instead.
