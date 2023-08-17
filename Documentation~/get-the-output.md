# Get output from a model

## Get the tensor output

Use `PeekOutput` to access the output of the tensor. `PeekOutput` returns a `Tensor` object, so you usually need to cast it to a `TensorFloat` or a `TensorInt`. For example:

```
worker.Execute(inputTensor);
TensorFloat outputTensor = worker.PeekOutput() as TensorFloat;
```

The result of `PeekOutput` is a shallow copy that points to the same memory as the original output, which means the following:

- You don't need to use `Dispose` on the output.
- If you change the output or you rerun the worker, both the worker output and the `PeekOutput` copy change.
- If you use `Dispose` on the worker, the `PeekOutput` copy will also be disposed.

To take ownership of the original tensor instead, do either of the following:

- Use `TakeOwnership` on the tensor after you use `PeekOutput`.
- Use `FinishExecutionAndDownloadOutput` instead of `PeekOutput`. Sentis downloads the tensor from native memory.

If you use either method, you must `Dispose` of the tensor when you're finished with it.

When you read data from the tensor that `PeekOutput` returns, there might be a performance cost because Sentis waits for the model to finish running then downloads the data from the GPU or Burst to the CPU. You can [read output from a model asynchronously](read-output-async.md) to avoid this cost. You can also [profile a model](profile-a-model.md) to understand more about the performance of a model. 

To get intermediate tensors from layers other than the model outputs, refer to [Get output from any layer](profile-a-model.md#get-output-from-any-layer).

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

        // Iterate through the output layer names of the model and print the output from each
        foreach (var outputName in runtimeModel.outputs)
        {
            TensorFloat outputTensor = worker.PeekOutput(outputName) as TensorFloat;
            // Make the tensor readable by downloading it to the CPU
            outputTensor.MakeReadable();
            outputTensor.PrintDataPart(10);
        }
    }
}
```

## Print outputs

You can use the following methods to log tensor data to the Console window:

- `Print`.
- [PrintDataPart](xref:Unity.Sentis.TensorExtensions.PrintDataPart(Unity.Sentis.Tensor,System.Int32,System.String)), which prints the first elements from the tensor data.

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Use output data](use-model-output.md)
- [Read output from a model asynchronously](read-output-async.md)
- [Get output from any layer](profile-a-model.md#get-output-from-any-layer)
