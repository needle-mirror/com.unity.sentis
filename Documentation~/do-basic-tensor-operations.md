## Create and modify tensors

Tensor methods in Sentis are similar to methods found in frameworks like NumPy, TensorFlow, and PyTorch. 

## Create a basic tensor

Create a basic tensor using the methods in the [`TensorFloat`](xref:Unity.Sentis.TensorFloat) or [`TensorInt`](xref:Unity.Sentis.TensorInt) APIs. For example use [`TensorFloat.AllocZeros`](xref:Unity.Sentis.TensorFloat.AllocZeros(Unity.Sentis.TensorShape)) to create a tensor filled with `0`. Or [`TensorFloat.AllocNoData`](xref:Unity.Sentis.TensorFloat.AllocNoData(Unity.Sentis.TensorShape)) for a tensor with no underlying backend data.

Refer to [Create input for a model](create-an-input-tensor.md) for more information.

## Reshape a tensor

You can reshape a tensor directly, for example:

```
TensorFloat tensor = TensorFloat.AllocZeros(new TensorShape(10));
tensor.Reshape(new TensorShape(5, 2));
```

The shape of the tensor and the new shape must have the same number of elements. You can use the `length` property of a tensor shape to check the number of elements.

> [!NOTE] 
> If you use `BackendType.GPUPixel`, tensors aren't stored in a linear format. You will not be able to reshape a tensor if the data is on the GPU.

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Model inputs](models-concept.md#model-inputs)

