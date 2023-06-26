# Understand models in Sentis

Sentis can import and run trained machine learning model files in Open Neural Network Exchange (ONNX) format. 

To get a model that's compatible with Sentis, you can do one of the following:

- Train a model in a framework like TensorFlow, PyTorch or Keras, then [export it in ONNX format](export-an-onnx-file.md).
- Download a trained model file and convert it to ONNX format. Refer to the [ONNXXMLTools](https://github.com/onnx/onnxmltools) Python package for more information.
- Download a trained model that's already in ONNX format, for example from the [ONNX Model Zoo](https://github.com/onnx/models).

## How Sentis optimizes a model

When you import a model, each ONNX operator in the model graph becomes a Sentis layer.

Open the [Model Asset Import Settings](onnx-model-importer-properties.md) to check the list of layers in the imported model, in the order Sentis runs them. Refer to [Supported ONNX operators](supported-operators.md) for more information.

Sentis optimizes models to make them smaller and more efficient. For example, Sentis might do the following to an imported model:

- Remove a layer or subgraph and turn it into a constant.
- Replace a layer or subgraph with a simpler layer or subgraph that works the same way.
- Set a layer to run on the CPU, if the data must be read at inference time.

The optimization doesn't affect what the model inputs or outputs.

To deactivate optimization, select your [imported model asset](import-a-model-file.md) then deactivate **Optimize Model** in the [Model Asset Import Settings](onnx-model-importer-properties.md) window.

## Model performance

The performance of a model depends on the following:

- The complexity of the model.
- Whether the model uses performance-heavy operators such as Conv or MatMul.
- The features of the platform you run the model on, for example CPU memory, GPU memory, and number of cores.
- Whether Sentis downloads data to CPU memory when you access a tensor. Refer to [Get output from a model](get-the-output.md) for more information.

[Profile a model](profile-a-model.md) to understand the performance on a particular platform, and check whether your application is CPU or GPU bound. If your application is GPU bound, you can run a model on the CPU to offload work from the GPU.

## Model inputs

The input dimensions for a model are either fixed or dynamic.

When you [Inspect a model](inspect-a-model.md) to get its inputs, each dimension in the an input shape displays one of the following:

- A digit to represent a fixed dimension, for example `(1)`. This means your input tensor should use the same dimension size, because Sentis optimizes the model for that size.
- A question mark or a string to represent a dynamic dimension, for example `(?)` or `(batch)`. This means the input tensor shape is symbolic, and your input tensor can use any dimension size.

If a model uses inputs with dynamic shapes, Sentis might not be able to optimize the model as efficiently as a model that uses fixed input dimensions. This might slow down the model.

## Additional resources

- [Import a model](import-a-model.md)
- [Export an ONNX file from a machine learning framework](export-an-onnx-file.md)
- [Supported ONNX operators](supported-operators.md)
