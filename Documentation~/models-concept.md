# Understand models in Sentis

Sentis can import and run trained machine learning model files in Open Neural Network Exchange (ONNX) format.

To get a model that's compatible with Sentis, you can do one of the following:

- Train a model using a framework like TensorFlow, PyTorch, or Keras, and subsequently [export it in ONNX format](export-convert-onnx.md).
- Download a trained model file and [convert to ONNX format](export-convert-onnx.md). Refer to the [ONNXXMLTools](https://github.com/onnx/onnxmltools) Python package for more information.
- Download a trained model that's already in ONNX format, such as those available in the [ONNX Model Zoo](https://github.com/onnx/models). Refer to [supported models](supported-models.md) for more resources.

## How Sentis optimizes a model

When you import a model, each ONNX operator in the model graph becomes a Sentis layer.

Open the [Model Asset Import Settings](onnx-model-importer-properties.md) to check the list of layers in the imported model, in the order Sentis runs them. Refer to [Supported ONNX operators](supported-operators.md) for more information.

Sentis optimizes models to make them smaller and more efficient. For example, Sentis might do the following to an imported model:

- Remove a layer or subgraph and turn it into a constant.
- Replace a layer or subgraph with a simpler layer or subgraph that works the same way.
- Set a layer to run on the CPU, if the data must be read at inference time.

The optimization doesn't affect what the model inputs or outputs.

## Model inputs

You can get the shape of your model inputs in one of two ways:
- [Inspect a model](inspect-a-model.md) to use the [`inputs`](xref:Unity.Sentis.Model.inputs) property of the runtime model.
- Select your model from the **Project** window to open the [**Model Asset Import Settings**](onnx-model-importer-properties.md) and view the **Inputs** section.

The shape of a model input consists of multiple dimensions, defined as a [`DynamicTensorShape`](xref:Unity.Sentis.DynamicTensorShape).

 The dimensions of a model input are either fixed or dynamic:
- An `int` denotes a fixed dimension.
- The strings `?` and `d0`, `d1` etc. denote dynamic dimensions.

### Fixed dimensions

The value of the `int` defines the specific size of the input the model accepts.

For example, if the **Inputs** section displays **(1, 1, 28, 28)**, the model only accepts a tensor of size `1 x 1 x 28 x 28`.

### Dynamic dimensions

When the shape of a model input contains a dynamic dimension, a `?` or `string`, for example `batch_size` or `height`, the input dimension can be any size.

For example, if the input is **(d0, 1, 28, 28)**, the first dimension of the input can be any size.

When you define the input tensor for this input shape, the following input tensor shapes are valid:

```
[1, 1, 28, 28]
[2, 1, 28, 28]
[3, 1, 28, 28] ...
```

If you change the size of another dimension, however, the tensor input is not valid. For example:

```
[1, 3, 28, 28]
```

> [!NOTE]
> If a model uses inputs with dynamic shapes, Sentis might not be able to optimize the model as efficiently as a model that uses fixed input dimensions. This might slow down the model.

## Additional resources

- [Import a model](import-a-model-file.md)
- [Supported models](supported-models.md)
- [Export an ONNX file from a machine learning framework](export-convert-onnx.md)
- [Supported ONNX operators](supported-operators.md)
