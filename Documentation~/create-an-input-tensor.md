# Create input for a model

To check the shape and size of the input the model needs, open the [ONNX Model Import Settings](onnx-model-importer-properties.md) and check the **Inputs** section.

## Convert an array to a tensor

To create a tensor from a one-dimensional data array, follow these steps:

1. Create a `TensorShape` object that has the length of each axis.
2. Create a `Tensor` object with the `TensorShape` object and the data array.

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

## Do operations

Use `WorkerFactory.CreateOps` if you need to do operations on a tensor. Refer to [Do operations on a tensor](do-complex-tensor-operations.md) for more information.

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Create a runtime model](import-a-model-file.md#create-a-runtime-model)
- [Convert a texture to a tensor](convert-texture-to-tensor.md)
