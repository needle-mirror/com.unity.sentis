## Create and modify tensors

Tensor methods in Sentis are similar to methods in frameworks like numpy, TensorFlow, and PyTorch. 

## Create a basic tensor

Create a basic tensor using the methods in the [`TensorFloat`](xref:Unity.Sentis.TensorFloat) or [`TensorInt`](xref:Unity.Sentis.TensorInt) APIs. For example use [`TensorFloat.Zeros`](xref:Unity.Sentis.TensorFloat.Zeros(Unity.Sentis.TensorShape)) to create a tensor filled with `0`.

Refer to [Create input for a model](create-an-input-tensor.md) for more information.

## Reshape a tensor

You can't change the shape of a tensor, but you can use [`ShallowReshape`](xref:Unity.Sentis.TensorFloat.ShallowReshape(Unity.Sentis.TensorShape)) to create a reshaped copy of a tensor.

For example:

```
// Create a 2 × 2 × 2 tensor
TensorShape shape = new TensorShape(2, 2, 2);
int[] data = new int[] { 1, 2, 3, 4, 10, 20, 30, 40 };
TensorInt tensor = new TensorInt(shape, data);

// Create a copy of the tensor with the new shape 2 × 1 × 4
TensorShape newShape = new TensorShape(2, 1, 4);
Tensor reshapedTensor = tensor.ShallowReshape(newShape);
```

The shape of the tensor and the new shape must have the same number of elements. You can use the `length` property of a tensor to check the number of elements.

The result of `ShallowReshape` is a shallow copy that points to the same memory as the original output.

**Note:** Don't use `ShallowReshape`if you use `BackendType.GPUPixel`, because `ShallowReshape` isn't compatible with the way Sentis stores tensor data in textures.

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Do operations on a tensor](do-complex-tensor-operations.md)
- [Model inputs](models-concept.md#model-inputs)

