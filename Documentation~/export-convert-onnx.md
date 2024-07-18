# Export and convert a file to ONNX

Sentis currently only supports importing files in the Open Neural Network Exchange (ONNX) format, or the Sentis serialized format. If your model is not in one of these formats, you must convert it.

The following sections describe how to export a file in ONNX format and convert files from a different format to ONNX.

## Export an ONNX file from a machine learning framework

You can export a model from most machine learning frameworks in ONNX format.

Refer to the following documentation to understand how to export files in ONNX format from common machine learning frameworks:

- [Exporting a model from PyTorch to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) on the PyTorch website.
- [Convert TensorFlow, Keras, Tensorflow.js and Tflite models to ONNX](https://github.com/onnx/tensorflow-onnx) on the ONNX GitHub repository.

To make sure the exported model is compatible with Sentis, set the ONNX opset version to 15. Refer to [import a model file](import-a-model-file.md) for more information about ONNX compatibility.

## Convert TensorFlow files to ONNX

Exporting files from TensorFlow involves two key file types: SavedModel and Checkpoints. Refer to the following sections to understand each file type and how to convert them to the ONNX format.

### Model files

TensorFlow saves models in SavedModel files, which contain a complete TensorFlow program, including trained parameters and computation. SavedModels have the `.pb` file extension. Refer to [Using the SavedModel format](https://www.tensorflow.org/guide/saved_model) (TensorFlow documentation) for more information on SavedModels.

To generate an ONNX file from a SavedModel, you can use the [tf2onnx](https://github.com/onnx/tensorflow-onnx) tool. This is a command line tool that works best when you provide the full path names.

### Checkpoints

[Checkpoints](https://www.tensorflow.org/guide/checkpoint) (TensorFlow documentation) contain only the parameters of the model.

Checkpoints in TensorFlow can consist up of two file formats:

- A file to store the graph, with the extension .ckpt.meta.
- A file to store the weights, with the extension .ckpt.

If you have both the graph and weight file types, you can use the [tf2onnx](https://github.com/onnx/tensorflow-onnx) tool to create an ONNX file.

If you only have the `.ckpt` file, you will need to find the Python code that constructs the model and loads in the weights. After that, you can proceed to [export the model to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html).

## Convert PyTorch files to ONNX

Refer to the following sections to understand how to convert PyTorch files to the ONNX format.

### Model files

PyTorch model files usually have the .pt file extension.

To export a model file to ONNX, refer to the links in the following instructions:

1. [Load the model](https://pytorch.org/tutorials/beginner/saving_loading_models.html) in Python.
2. [Export the model](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) as an ONNX file. It is recommended to use Opset 15 or higher when exporting your model.

If your `.pt` file doesn't contain the model graph, you must find the Python code that constructs the model and loads in the weights. After that, you can proceed to [export the model to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html).

###  Checkpoints

You can create [Checkpoints](https://pytorch.org/docs/stable/checkpoint.html) in PyTorch to save the state of your model at any instance of time. Checkpoint files are usually denoted with the `.tar` or `.pth` extension.

To convert a checkpoint file to ONNX, you must find the Python code which constructs the model and loads in the weights. After that, you can proceed to [export the model to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html).

## Additional resources

- [Open Neural Network Exchange](https://onnx.ai/)
- [Supported ONNX operators](supported-operators.md)
- [Profile a model](profile-a-model.md)
- [Convert TensorFlow, Keras, Tensorflow.js and Tflite models to ONNX](https://github.com/onnx/tensorflow-onnx)
