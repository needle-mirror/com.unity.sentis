# Sentis overview

Sentis is a neural network inference library for Unity. The package is in a "pre-release" state in an open beta program available to all Unity users via the [package manager](https://tinyurl.com/4eun48fb). The features and documentation may change before it is verified for release near the end of 2024. If you would like to release a commercial project with Sentis before it is released, please reach out to [bill.cullen@unity3d.com](mailto:bill.cullen@unity3d.com).

Visit the [Unity Discussions Sentis topic](https://discussions.unity.com/c/10) for more information, communications from our team, and to collaborate with other beta users. If you have an interesting project that would like to collaborate with Unity on, please submit [this form](https://create.unity.com/sentis-project-submission) and we'll be excited to speak with you.

You can use Sentis to import trained neural network models into Unity, and then run them in real-time locally on any runtime platform that Unity supports. Models can be deployed on either the GPU or the CPU.

To use Sentis, it helps if you have some experience in using machine learning models, for example in a framework like [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/).

|Section|Description|
|-|-|
|[Get started](get-started.md)|Install Sentis, find and use the sample projects, and get started.|
|[Create a model](create-a-model.md)|Create a runtime model by importing an ONNX model file or using the Sentis model API.|
|[Run a model](run-an-imported-model.md)|Create input data for a model, create an engine to run the model, and get output.|
|[Use Tensors](use-tensors.md)|Get, set and modify input and output data.|
|[Profile a model](profile-a-model.md)|Use Unity tools to profile the speed and performance of a model.|

## Supported platforms

Sentis supports [all the platforms Unity supports](https://docs.unity3d.com/Documentation/Manual/PlatformSpecific.html).

How long a model takes to run depends on the complexity of the model, the platform, and the engine type you use. Refer to [Models](models-concept.md) and [Create an engine](create-an-engine.md) for more information.

## Supported model types

Sentis supports most models in ONNX format with an [opset version](https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions) between 7 and 15. Refer to [Supported ONNX operators](supported-operators.md) for more information.

## Places to find pre-trained models
There are many places to find pre-trained models either already in the ONNX format or in a format that you can convert. Some popular places include:
* [Huggingface 🤗](https://huggingface.co/models) A huge database. Many of the newest models are posted here.
* [Kaggle Models](https://www.kaggle.com/models) Formerly Tensorflow Hub.
* [Pytorch Hub](https://pytorch.org/hub/) A repository from PyTorch.
* [Model Zoo](https://github.com/onnx/models) Many smaller older models that work very fast. Now being updated with newer models.
* [XetData](https://github.com/xetdata/onnx-models) Model Zoo but with some newer models.
* [Meta Research](https://github.com/facebookresearch) Formerly known as Facebook Research hosts some popular larger models.

For training your own models you may want to check out:
* [Google Colab](https://colab.research.google.com/) A space to train models online
* [Kaggle](https://www.kaggle.com/) A popular machine learning commnuity

## Additional resources

- [Sample scripts](package-samples.md)
- [Unity Discussions group for the Sentis beta](https://discussions.unity.com/c/10)
- [Understand the Sentis workflow](understand-sentis-workflow.md)
- [Understand models in Sentis](models-concept.md)
- [Tensor fundamentals in Sentis](tensor-fundamentals.md)

