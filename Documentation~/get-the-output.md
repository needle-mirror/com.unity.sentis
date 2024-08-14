# Get output from a model

This section provides information on how to get the output from a model. To get intermediate tensors from layers other than the model outputs, refer to [Get output from any layer](profile-a-model.md#get-output-from-any-layer).

## Get the tensor output

To get the tensor output, you have two options: either use [`PeekOutput`](xref:Unity.Sentis.Worker.PeekOutput*) to get a reference to an output or [`CopyOutput`](xref:Unity.Sentis.Worker.CopyOutput*) to copy an output into a tensor that is owned outside of the scope of a worker.

The following sections provide information on the methods available for retrieving the tensor output, along with their respective strengths and weaknesses.

### Use PeekOutput

Use [`PeekOutput`](xref:Unity.Sentis.Worker.PeekOutput*) to get a reference to the output of the tensor. [`PeekOutput`](xref:Unity.Sentis.Worker.PeekOutput*) returns a [`Tensor`](xref:Unity.Sentis.Tensor) object so you usually need to cast it to a [`Tensor<float>`](xref:Unity.Sentis.Tensor`1) or a [`Tensor<int>`](xref:Unity.Sentis.Tensor`1).

For example:

```
worker.Schedule(inputTensor);
Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
```

Sentis worker memory allocator owns the reference returned by [`PeekOutput`](xref:Unity.Sentis.Worker.PeekOutput*). It implies the following:

- You don't need to use `Dispose` on the output.
- If you change the output or you rerun the worker, both the worker output and the [`PeekOutput`](xref:Unity.Sentis.Worker.PeekOutput*) copy change.
- Using `Dispose` on the worker disposes the [`PeekOutput`](xref:Unity.Sentis.Worker.PeekOutput*) copy.

If you call `Schedule` again, the tensor is overwritten.

> [!NOTE]
> Be careful when reading data from an output tensor, as in many instances, you might unintentionally trigger a blocking wait until the model finishes running before downloading the data from the graphics processing unit (GPU) or Burst to the central processing unit (CPU). To mitigate this overhead, consider [reading output from a model asynchronously](read-output-async.md). Additionally, [profiling a model](profile-a-model.md) can provide valuable insight into its performance.

### Download the data of the original tensor

You can do a blocking download to a read only `NativeArray` or `Array` copy of the output tensor's data.

* Use [`DownloadToNativeArray`](xref:Unity.Sentis.Tensor`1.DownloadToNativeArray*) on the tensor after you use [`PeekOutput`](xref:Unity.Sentis.Worker.PeekOutput*).
* Use [`DownloadToArray`](xref:Unity.Sentis.Tensor`1.DownloadToArray*) on the tensor after you use [`PeekOutput`](xref:Unity.Sentis.Worker.PeekOutput*).

### Wait on the data of the original tensor

You can request a async readback of the output tensor data without interrupting the worker execution.
```
Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
var result = await outputTensor.ReadbackAndCloneAsync();
```
```
Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
outputTensor.ReadbackRequest();

// when done
outputTensor.ReadbackAndClone(); // not blocking
```

Refer to [Read Outputs Asynchronously](read-output-async.md) for details.

### Use CopyOutput

You can copy a worker output into another tensor that is owned outside of the scope of a worker.
If the tensor you pass in is null, Sentis will create a new tensor which is a copy of the worker's output. If the tensor is allready created, Sentis will reshape it to the correct size and copy the worker's output.

```
Tensor myOutputTensor;
//...
void Update () {
   worker.Schedule(inputTensor);
   worker.CopyOutput("output", ref myOutputTensor);
}
```

[`CopyOutput`](xref:Unity.Sentis.Worker.CopyOutput*) reshapes the provided tensor to match calculated output shape. Ensure that the provided tensor has capacity for the output.

```
// The model outputs a tensor of shape (1, 10)

// CopyOutput works on empty tensors, i.e. tensors without a tensor data.
myOutputTensor = new Tensor<float>(new TensorShape(1, 10), data: null);
worker.CopyOutput("output", ref myOutputTensor);

// CopyOutputInto works on tensors of different shape as long as the dataOnBackend has large enough capacity
myOutputTensor = new Tensor<float>(new TensorShape(152));
worker.CopyOutput("output", ref myOutputTensor);
// myOutputTensor now has shape (1, 10) but still has dataOnBackend.maxCapacity == 152
```

You are responsible for the tensor which output is copied into:

* You must `Dispose` of the tensor after you finish using it.
* The tensor isn't overwritten if you call [`Worker.Schedule`](xref:Unity.Sentis.Worker.Schedule*) again.

## Multiple outputs

If the model has multiple outputs, you can use each output name as a parameter in [`Worker.PeekOutput`](xref:Unity.Sentis.Worker.PeekOutput(string)).

## Additional resources

- [Manage memory](manage-memory.md)
- [Tensor fundamentals](tensor-fundamentals.md)
- [Use output data](use-model-output.md)
- [Read output from a model asynchronously](read-output-async.md)
- [Get output from any layer](profile-a-model.md#get-output-from-any-layer)
