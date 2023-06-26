# What's new in Sentis 1.0

Sentis is the new name for the [Barracuda package](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/manual/index.html).

This is a summary of the changes from Barracuda 3.0 to Sentis 1.0.

For information on how to upgrade, refer to the [Upgrade Guide](upgrade-guide.md).

## Added

- You can now override model layers in your code, and add your own layers. Refer to the `CustomLayer` example in the [sample scripts](package-samples.md) for a working example.


## Updated

- All the back end types have been optimized and improved so Sentis as a whole has better performance on both the GPU and the CPU.
- The way Sentis represents tensors has changed to make Sentis compatible with more models, and make it easier to convert Python preprocessing code into C#. Refer to [Tensor fundamentals](tensor-fundamentals.md) and [Do operations on tensors](do-complex-tensor-operations.md) for more information.
- Sentis supports more Open Neural Network Exchange (ONNX) operators. Refer to [Supported ONNX operators](supported-operators.md) for more information. 
- You can now import models larger than 2 gigabytes.
- The [Model Asset Import Settings](onnx-model-importer-properties.md) window no longer contain contains `Force Arbitrary Batch Size` and `Treat Errors as Warnings` settings. The window also no longer contains an `Open imported NN model as temp file` button. Use `Open` instead.
