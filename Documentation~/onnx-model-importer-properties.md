# Model Asset Import Settings

The **Model Asset Import Settings** window lets you do the following:

- Change settings that affect how Sentis imports a model.
- Check the properties of an imported and optimized model.

To access this window, select an imported model asset in the Project window.

## General settings

|Property|Description|
|-|-|
|**Open**|Open the model in the default application for ONNX files. Refer to [Open a model as a graph](inspect-a-model.md#open-a-model-as-a-graph) for more information.|
|**Optimize Model**|When enabled, Sentis [optimizes the model](models-concept.md#how-sentis-optimizes-a-model) when you import it. The default is enabled.|
|**Apply**|Save changes you've made to the settings.|
|**Revert**|Revert changes you've made to the settings.|

## Imported Object settings

### Metadata

This section only appears if the model file contains ONNX metadata.

### Inputs

This section shows the total number of inputs the model has, and a list of the inputs.

|Property|Description|
|-|-|
|**name**|The name of the input.|
|**shape**|The tensor shape of the input. For example **(1, 1, 28, 28)** means the model accepts a tensor of size 1 × 1 × 28 × 28 exactly. If a dimension in the shape is a string instead of a number, for example `batch_size` or `height`, the dimension can be any size (a dynamic input). Refer to [Model inputs](models-concept.md#model-inputs) for more information about dynamic inputs. |
|**dataType**|The data type of the input. The possible values are **float** and **int**.|

### Outputs

This section shows the total number of outputs the model has, and a list of the outputs.

|Property|Description|
|-|-|
|**name**|The name of the output.|
|**shape**|The tensor shape of the output. Sentis tries to precalculate this shape from the model. If a dimension in the shape is a question mark (**?**), Sentis can't calculate the size of the dimension, or the size depends on the input (a dynamic output). If the entire tensor shape is **Unknown**, Sentis can't calculate the number of dimensions, or the number of dimensions is dynamic. |
|**dataType**|The data type of the output. The possible values are `float` and `int`.|

### Layers

This section shows the total number of layers the model has, and a list of the layers in the order Sentis runs them.

|Property|Description|
|-|-|
|Type|The type of layer. Refer to [Supported ONNX operators](supported-operators.md) for more information.|
|**name**|The name of the layer.|
|**inputs**|The names of the inputs to the layer. Possible values are a model input from the **Inputs** section, another layer, or a constant.|
|Properties|The properties of the layer. The properties will depend on the type of layer. Refer to [Supported ONNX operators](supported-operators.md) for more information.|

### Constants

This section shows the total number of constants the model has and the total number of weights, and a list of the constants.

|Property|Description|
|-|-|
|**Type**|The type is always **Constant**.|
|**name**|The name of the constant.|
|**weights**|The tensor shape of the constant. If the tensor shape is empty - **()** - the constant is a scalar (a zero-dimensional tensor).|

## Model information

This section shows the total size of the model, its source, the ONNX [opset version](https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions), and the tool that was used to create the model file.

### Errors

This section only appears if there was a problem when Sentis imported and optimized the model, and the problems mean you can't run the model.

### Warnings

This section only appears if there was a problem when Sentis imported and optimized the model. You can usually still run the model, but you might get unexpected results.

You can add a custom layer to implement a missing operator. Refer to the `Add a custom layer` example in the [sample scripts](package-samples.md) for an example.

## Additional resources

- [Import a model](import-a-model.md)
- [How Sentis optimizes a model](models-concept.md#how-sentis-optimizes-a-model)
- [Supported ONNX operators](supported-operators.md)


