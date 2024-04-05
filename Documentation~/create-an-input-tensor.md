# Create input for a model

A model requires input tensors with certain shapes and data types. Refer to the following information to understand how to find model inputs, and how to create input tensors for your model.

## Understand the required input

Before you can create input tensors for a model, you must first understand the shape and data types of the model inputs. 

To understand model inputs, and the types of input shapes, refer to [Model inputs](models-concept.md#model-inputs).

The [`TensorShape`](xref:Unity.Sentis.TensorShape) of the `[`Tensor`](xref:Unity.Sentis.Tensor)` you create, must be compatible with the [`SymbolicTensorShape`](xref:Unity.Sentis.SymbolicTensorShape), which defines the shape of the model input.

## Convert an array to a tensor

To create a tensor from a one-dimensional data array, follow these steps:

1. Create a [`TensorShape`](xref:Unity.Sentis.TensorShape) object that has the length of each axis.
2. Create a [`Tensor`](xref:Unity.Sentis.Tensor) object with the `TensorShape` object and the data array.

For example, the following code creates a tensor for a model that takes an input tensor of shape 3 × 1 × 3.

```
using UnityEngine;
using Unity.Sentis;

public class ConvertArrayToTensor : MonoBehaviour
{
    void Start()
    {
        // Create a data array with 9 values
        float[] data = new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f };

        // Create a 3D tensor shape with size 3 × 1 × 3
        TensorShape shape = new TensorShape(3, 1, 3);

        // Create a new tensor from the array
        TensorFloat tensor = new TensorFloat(shape, data);
    }
}
```

## Create multiple inputs

If a model needs multiple input tensors, you can create a dictionary that contains the inputs. For example:

```
Dictionary<string, Tensor> inputTensors = new Dictionary<string, Tensor>()
{
    { "x", xTensor },
    { "y", yTensor },
};
```

## Edit a model

Use the functional API if you need to add operations to your model inputs. Refer to [Edit a model](edit-a-model.md) for more information.

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Edit a model](edit-a-model.md)
- [Convert a texture to a tensor](convert-texture-to-tensor.md)
