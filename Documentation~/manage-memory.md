# Manage memory

As a Sentis user, you are responsible for calling [`Dispose`](xref:Unity.Sentis.GenericWorker.Dispose) on any worker, inputs, and sometimes outputs. Specifically, you must call `Dispose` on outputs obtained via [`TakeOutputOwnership`](xref:Unity.Sentis.IWorker.TakeOutputOwnership) or if you take ownership by calling [`CompleteOperationsAndDownload`](xref:Unity.Sentis.Tensor.CompleteOperationsAndDownload). 

> [!NOTE]
> Calling `Dispose` is necessary to properly free up GPU resources.

For example:

```
public void OnDestroy()
{
    worker?.Dispose();

    // Assuming model with multiple inputs that were passed as a Dictionary
    foreach (var key in inputs.Keys)
    {
        inputs[key].Dispose();
    }
    
    inputs.Clear();
}
```

When you get a handle to a tensor using the [`worker.PeekOutput`](xref:Unity.Sentis.IWorker.PeekOutput) call, the memory allocator remains responsible for that memory, so there is no need for you to `Dispose` of it. For more information on `PeekOutput`, refer to [Get output from a model](get-the-output.md).

## Additional resources

- [Profile a model](profile-a-model.md)
- [Create an engine to run a model](create-an-engine.md)
- [Create and modify tensors](do-basic-tensor-operations.md)
