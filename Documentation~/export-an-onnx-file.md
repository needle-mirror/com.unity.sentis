## Export an ONNX file from a machine learning framework

You can export a model from most machine learning frameworks in the Open Neural Network Exchange (ONNX) format Sentis needs.

Refer to the following documentation:

- [Exporting a model from PyTorch to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) on the PyTorch website.
- [Convert TensorFlow, Keras, Tensorflow.js and Tflite models to ONNX](https://github.com/onnx/tensorflow-onnx) on the ONNX GitHub repository.

To make sure the exported model is compatible with Sentis, set the ONNX opset version to 15. Refer to [import a model file](import-a-model-file.md) for more information about ONNX compatibility.

## Additional resources

- [Open Neural Network Exchange](https://onnx.ai/)
- [Supported ONNX operators](supported-operators.md)
- [Profile a model](profile-a-model.md)

