# Create input for a model

A model requires input tensors with certain shapes and data types. Use this information to find model inputs and create input tensors for your model.

## Understand the required input

Before you can create input tensors for a model, you must first understand the shape and data types of the model inputs. For more information, refer to [Model inputs](models-concept.md#model-inputs).

The [`TensorShape`](xref:Unity.Sentis.TensorShape) of the [`Tensor`](xref:Unity.Sentis.Tensor) you create must be compatible with the [`DynamicTensorShape`](xref:Unity.Sentis.DynamicTensorShape), which defines the shape of the model input.

## Convert an array to a tensor

To create a CPU tensor from a one-dimensional data array, follow these steps:

1. Create a [`TensorShape`](xref:Unity.Sentis.TensorShape) object that has the length of each axis.
2. Create a [`Tensor<T>`](xref:Unity.Sentis.Tensor`1) object with the shape and the data array.

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
        Tensor<float> tensor = new Tensor<float>(shape, data);
    }
}
```

## Create an empty tensor

You can create a CPU tensor with its memory 0-initialized as follows:

```
var tensor = new Tensor<int>(new TensorShape(1), clearOnInit: true);
```
The parameter `clearOnInit` determines whether the resulting tensor memory should be 0-initialized. If it is unimportant what the initial data of the tensor is, set `clearOnInit` to `false` to avoid this additional operation.

## Pass inputs to a worker

If a model needs multiple input tensors, do one of the following:

- Call [`SetInput`](xref:Unity.Sentis.Worker.SetInput*) on the worker for each input, and then call [`Schedule`](xref:Unity.Sentis.Worker.Schedule) on the worker with no arguments.
- Call [`Schedule`](xref:Unity.Sentis.Worker.Schedule(Unity.Sentis.Tensor[])) on the worker with an array of the desired inputs.

```
worker.SetInput("x", xTensor);
worker.SetInput("y", yTensor);
worker.Schedule();
```

```
worker.Schedule(xTensor, yTensor);
```
To avoid garbage collection due to the `params` array creation, you can pre-allocate the input array.
```
var inputs = new [] { xTensor, yTensor };
//...
worker.Schedule(inputs);
```

## Edit a model

Use the functional API to add operations to your model inputs. For more information, refer to [Edit a model](edit-a-model.md).

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Edit a model](edit-a-model.md)
- [Convert a texture to a tensor](convert-texture-to-tensor.md)
