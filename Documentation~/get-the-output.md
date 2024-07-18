# Get output from a model

This section provides information on how to get the output from a model. To get intermediate tensors from layers other than the model outputs, refer to [Get output from any layer](profile-a-model.md#get-output-from-any-layer).

## Get the tensor output

To get the tensor output, you have two options: either use `PeekOutput` to get a reference to an output or `TakeOwnership` of the original tensor.

The following sections provide information on the methods available for retrieving the tensor output, along with their respective strengths and weaknesses.

### Use PeekOutput

Use [`PeekOutput`](Unity.Sentis.IWorker.PeekOutput) to get a reference to the output of the tensor. `PeekOutput` returns a `Tensor` object so you usually need to cast it to a `TensorFloat` or a `TensorInt`.

For example:

```
worker.Execute(inputTensor);
TensorFloat outputTensor = worker.PeekOutput() as TensorFloat;
```

Sentis worker memory allocator owns the reference returned by `PeekOutput`. It implies the following:

- You don't need to use `Dispose` on the output.
- If you change the output or you rerun the worker, both the worker output and the `PeekOutput` copy change.
- Using `Dispose` on the worker disposes the `PeekOutput` copy.

If you call `Execute` again, the tensor is overwritten.

> [!NOTE]
> Be careful when reading data from an output tensor, as in many instances, you might unintentionally trigger a blocking wait until the model finishes running before downloading the data from the graphics processing unit (GPU) or Burst to the central processing unit (CPU). To mitigate this overhead, consider [reading output from a model asynchronously](read-output-async.md). Additionally, [profiling a model](profile-a-model.md) can provide valuable insight into its performance.

### Take ownership of the original tensor

To take ownership of the original tensor:

* Use [`TakeOutputOwnership`](xref:Unity.Sentis.IWorker.TakeOutputOwnership) instead of `PeekOutput`. Sentis downloads the tensor from native memory.

To make a hard copy of the original tensor:

* Use [`ReadbackAndClone`](xref:Unity.Sentis.Tensor.ReadbackAndClone) on the tensor after you use `PeekOutput`.

You can also use an `Awaitable` version of this method.

* Use [`ReadbackAndCloneAsync`](xref:Unity.Sentis.Tensor.ReadbackAndCloneAsync) on the tensor after you use `PeekOutput`.

If you take ownership of the original tensor, you're responsible for the lifetime of the output:

* You must `Dispose` of the tensor once you have finished using it.
* The tensor will not be overwritten if you call `Execute` again.
* The memory allocator will need to re-allocate the output if the worker runs again.

### Downloading the data of the original tensor

You can download to a read only `NativeArray` or `Array` copy of the output tensor's data.

* Use [`ToReadOnlyNativeArray`](xref:Unity.Sentis.Tensor.ToReadOnlyArray) on the tensor after you use `PeekOutput`.
* Use [`ToReadOnlyArray`](xref:Unity.Sentis.Tensor.ToReadOnlyArray) on the tensor after you use `PeekOutput`.

## Multiple outputs

If the model has multiple outputs, you can use each output name as a parameter in `PeekOutput`.

For example, the following code sample prints the output from each layer of the model.

```
using UnityEngine;
using Unity.Sentis;

public class GetMultipleOutputs : MonoBehaviour
{
    ModelAsset modelAsset;
    Model runtimeModel;
    IWorker worker;

    void Start()
    {
        // Create an input tensor
        TensorFloat inputTensor = new TensorFloat(new TensorShape(4), new[] { 2.0f, 1.0f, 3.0f, 0.0f });

        // Create runtime model
        runtimeModel = ModelLoader.Load(modelAsset);

        // Create engine and execute
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);
        worker.Execute(inputTensor);

        // Iterate through the outputs of the model and print the output tensor from each
        foreach (var output in runtimeModel.outputs)
        {
            TensorFloat outputTensor = worker.PeekOutput(output.name) as TensorFloat;
            // Make the tensor readable by downloading it to the CPU
            using var downloadedTensorCopy = outputTensor.ReadbackAndClone();
            downloadedTensorCopy.PrintDataPart(10);
        }
    }
}
```

## Print outputs

You can use the following methods to log tensor data to the Console window:

- `Print`.
- [PrintDataPart](xref:Unity.Sentis.TensorExtensions.PrintDataPart(Unity.Sentis.Tensor,System.Int32,System.String)), which prints the first elements from the tensor data.

## Additional resources

- [Manage memory](manage-memory.md)
- [Tensor fundamentals](tensor-fundamentals.md)
- [Use output data](use-model-output.md)
- [Read output from a model asynchronously](read-output-async.md)
- [Get output from any layer](profile-a-model.md#get-output-from-any-layer)
