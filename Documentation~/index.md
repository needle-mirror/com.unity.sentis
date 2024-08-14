# Sentis overview

Sentis is a neural network inference library for Unity. The package is in a "release" state and available to all Unity users via the package manager.

You can use Sentis to import trained neural network models into Unity, and then run them in real-time. Sentis utilizes the end-users device compute (GPU or CPU) and can run any supported Unity runtime platform.

To use Sentis, it helps if you have some experience in using machine learning models, for example in a framework like [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/).

|Section|Description|
|-|-|
|[Get started](get-started.md)|Install Sentis, find and use sample projects, and understand the Sentis workflow.|
|[Create a model](create-a-model.md)|Create a runtime model by importing an ONNX model file or using the Sentis model API.|
|[Run a model](run-an-imported-model.md)|Create input data for a model, create an engine to run the model, and get output.|
|[Use Tensors](use-tensors.md)|Get, set and modify input and output data.|
|[Profile a model](profile-a-model.md)|Use Unity tools to profile the speed and performance of a model.|

## Supported platforms

Sentis works on [all Unity runtime platforms](https://docs.unity3d.com/Documentation/Manual/PlatformSpecific.html).

Performance may vary and is based upon:
* Model operators and complexity.
* End-user hardware and software platform constraints.
* Type of engine used. Refer to [Models](models-concept.md) and [Create an engine](create-an-engine.md) for more information.

## Supported model types

Sentis supports most models in Open Neural Network Exchange (ONNX) format with an [opset version](https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions) between 7 and 15. Refer to [Supported models](supported-models.md) and [Supported ONNX operators](supported-operators.md) for more information.

## Places to find pre-trained models

[!include[](snippets/model-registry.md)]

## Additional resources

- [Sample scripts](package-samples.md)
- [Unity Discussions group](https://discussions.unity.com/tag/sentis)
- [Understand the Sentis workflow](understand-sentis-workflow.md)
- [Understand models in Sentis](models-concept.md)
- [Tensor fundamentals in Sentis](tensor-fundamentals.md)
- [Collaborate with us](https://create.unity.com/sentis-project-submission)

