# Sentis overview

This is an experimental package in an open beta program available to Unity users via the [package manager](https://tinyurl.com/4eun48fb). The features and documentation might change before it is verified for release, so it is not ready for production use.

Visit the [Unity Discussions Sentis topic](https://discussions.unity.com/c/10) for additional getting started information, the latest updates, and to collaborate with other beta participants.

The Sentis package is a neural network inference library for Unity.

You can use Sentis to import trained neural network models into Unity, then run them in real time locally on any platform Unity supports and in the Editor. You can run models on the GPU or the CPU.

To use Sentis, it helps if you have some experience in using machine learning models, for example in a framework like [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/).

|Section|Description|
|-|-|
|[Get started](get-started.md)|Install Sentis, find and use the sample projects, and get started.|
|[Import a model](import-a-model.md)|Import an Open Neural Network Exchange (ONNX) model into Sentis and inspect it.|
|[Run an imported model](run-an-imported-model.md)|Create input data for a model, create an engine to run the model, and get output.|
|[Use Tensors](use-tensors.md)|Get, set and modify input and output data.|
|[Profile a model](profile-a-model.md)|Use Unity tools to profile the speed and performance of a model.|

## Supported platforms

Sentis supports [all the platforms Unity supports](https://docs.unity3d.com/Documentation/Manual/PlatformSpecific.html).

How long a model takes to run depends on the complexity of the model, the platform, and the engine type you use. Refer to [Models](models-concept.md) and [Create an engine](create-an-engine.md) for more information.

## Supported model types

Sentis supports most models in ONNX format with an [opset version](https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions) between 7 and 15. Refer to [Supported ONNX operators](supported-operators.md) for more information.

## Additional resources

- [Sample scripts](package-samples.md)
- [Unity Discussions group for the Sentis beta](https://discussions.unity.com/c/10)
- [Understand the Sentis workflow](understand-sentis-workflow.md)
- [Understand models in Sentis](models-concept.md)
- [Tensor fundamentals in Sentis](tensor-fundamentals.md)

