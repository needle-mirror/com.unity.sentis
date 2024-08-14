## Create and modify tensors

Tensor methods in Sentis are similar to methods found in frameworks like NumPy, TensorFlow, and PyTorch.

## Create a tensor

Create a basic tensor using the methods in the [`Tensor`](xref:Unity.Sentis.Tensor) API.

Refer to [Create input for a model](create-an-input-tensor.md) for more information.

## Get and set values of a tensor

If your tensor data [`backendType`](xref:Unity.Sentis.ITensorData.backendType) is [`BackendType.CPU`](xref:Unity.Sentis.BackendType.CPU) and has finished being computed [`IsReadbackRequestDone`](xref:Unity.Sentis.Tensor.IsReadbackRequestDone*) you can directly set and get values.

```
var tensor = new Tensor<float>(new TensorShape(1, 2, 3));
tensor[0, 1, 2] = 5.2f; // set value at index 0 of dim0 = 1, index 1 of dim1 = 2 and index 2 of dim2 = 3

float value = tensor[0, 1, 2];
Assert.AreEqual(5.2f, value);
```

## Reshape a tensor

You can reshape a tensor directly, for example:

```
var tensor = new Tensor<float>(new TensorShape(10));
tensor.Reshape(new TensorShape(2, 5));
```

The new shape of the tensor must fit in the allocated data on the backend. You can use the [`length`](xref:Unity.Sentis.TensorShape.length) property of a tensor shape and the [`maxCapacity`](xref:Unity.Sentis.ITensorData.maxCapacity) property of the tensor data to check the number of elements.

```
var tensor = new Tensor<float>(new TensorShape(10));
Assert.AreEqual(10, tensor.count);
Assert.AreEqual(10, tensor.dataOnBackend.maxCapacity);

// Reshaping the tensor with a smaller shape

tensor.Reshape(new TensorShape(2, 3));
Assert.AreEqual(6, tensor.count);
Assert.AreEqual(10, tensor.dataOnBackend.maxCapacity);
// The underlying dataOnBackend still contains 10 elements

// reshape to match dataOnBackend.maxCapacity
tensor.Reshape(new TensorShape(1, 10));
```

When you reshape a tensor, Sentis does not modify the data or capacity of the underlying [`dataOnBackend`](xref:Unity.Sentis.Tensor.dataOnBackend).

> [!NOTE]
> If you use [`BackendType.GPUPixel`](xref:Unity.Sentis.BackendType.GPUPixel), tensors aren't stored in a linear format. Consequently, you will not be able to reshape a tensor if the data is on the GPU.

## Download values of a tensor

You can do a blocking download to get a copy of the data of a tensor to a `NativeArray` or `Array` as follows:

```
var nativeArray = tensor.DownloadToNativeArray();
var array = tensor.DownloadToArray();
```

Note: These methods return copies of your tensor data. Editing an array returned from one of these methods doesn't edit the tensor.

This download is a blocking call and will force a wait if [`ReadbackRequest`](xref:Unity.Sentis.Tensor.ReadbackRequest*) hasn't been called or [`IsReadbackRequestDone`](xref:Unity.Sentis.Tensor.IsReadbackRequestDone*) is false. Refer to [Read Outputs Asynchronously](read-output-async.md) for details.


## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Model inputs](models-concept.md#model-inputs)

