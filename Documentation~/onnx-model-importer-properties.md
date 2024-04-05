# Model Asset Import Settings

The **Model Asset Import Settings** window lets you do the following:

- Serialize the model to the StreamingAssets folder.
- Check the properties of an imported and optimized model.

To access this window, select an imported model asset in the **Project** window.

## Imported Object settings

### Inputs

This section shows the total number of inputs the model has, and a list of the inputs.

|Property|Description|
|-|-|
|**name**|The name of the input.|
|**index**|The index of the input.|
|**shape**|The tensor shape of the input. For example **(1, 1, 28, 28)** means the model accepts a tensor of size 1 × 1 × 28 × 28 exactly. If a dimension in the shape is a string instead of a number, for example `?`, `d0` or `d1`, the dimension can be any size (a dynamic input). Refer to [Model inputs](models-concept.md#model-inputs) for more information about dynamic inputs. |
|**dataType**|The data type of the input.|

### Outputs

This section shows the total number of outputs the model has, and a list of the outputs.

|Property|Description|
|-|-|
|**name**|The name of the output.|
|**index**|The index of the output.|
|**shape**|The tensor shape of the output. Sentis tries to precalculate this shape from the model. If a dimension in the shape is a question mark (**?**), Sentis can't calculate the size of the dimension, or the size depends on the input (a dynamic output). If the entire tensor shape is **Unknown**, Sentis can't calculate the number of dimensions, or the number of dimensions is dynamic. |
|**dataType**|The data type of the output.|

### Layers

This section shows the total number of layers the model has, and a list of the layers in the order Sentis runs them.

|Property|Description|
|-|-|
|**type**|The type of layer. Refer to [Supported ONNX operators](supported-operators.md) for more information.|
|**index**|The index of the layer.|
|**inputs**|The names of the inputs to the layer. Possible values are a model input from the **Inputs** section, another layer, or a constant.|
|**properties**|The properties of the layer. The properties will depend on the type of layer. Refer to [Supported ONNX operators](supported-operators.md) for more information.|

### Constants

This section shows the total number of constants the model has and the total number of weights, and a list of the constants.

|Property|Description|
|-|-|
|**type**|The type is always **Constant**.|
|**index**|The index of the constant.|
|**weights**|The tensor shape of the constant. If the tensor shape is empty - **()** - the constant is a scalar (a zero-dimensional tensor).|

## Model information

This section shows the model's total size, the ONNX [opset version](https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions), and the tool used to create the model file.

## Additional resources

- [Import a model](import-a-model-file.md)
- [How Sentis optimizes a model](models-concept.md#how-sentis-optimizes-a-model)
- [Supported ONNX operators](supported-operators.md)


