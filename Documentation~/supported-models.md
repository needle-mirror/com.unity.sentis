# Supported models

You can import open-source models into your Sentis project. Explore the following sections to understand the models Sentis supports, and find an appropriate model for your project.

## Pre-trained models

[!include[](snippets/model-registry.md)]

### Models from Hugging Face

You can access validated AI models for use with Sentis from [Hugging Face](https://huggingface.co/models). Models available from Hugging Face are already in the `.sentis` format to remove the need for ONNX conversion.

To find and models from Hugging Face, you can either:

- Navigate to the [Unity Hugging Face](https://huggingface.co/unity) space and select a model under the **Models** section.
- View models that are validated for Sentis, identified by the `unity-sentis` tag on the [**Models**](https://huggingface.co/models?library=unity-sentis&sort=likes) page.

To import and use the model in a Unity project, follow the instructions in the **How to Use** section on the model page .

### ONNX models

You can download ONNX model files from the [ONNX model zoo](https://github.com/onnx/models) on GitHub. Sentis supports most ONNX model files with an [opset version](https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions) between 7 and 15. While versions below 7 or above 15 may still be imported into Sentis, the results obtained may be unexpected.

## Unsupported models

Sentis doesn't support the following:

- Models that use tensors with more than eight dimensions
- Sparse input tensors or constants
- String tensors
- Complex number tensors

Sentis also converts some tensor data types like bools to floats or ints. This might increase the memory your model uses.

## Additional resources

- [Unity Hugging Face](https://huggingface.co)
- [ONNX model zoo](https://github.com/onnx/models)
- [Understand models in Sentis](models-concept.md)
- [Import a model file](import-a-model-file.md)
- [Supported ONNX operators](supported-operators.md)

