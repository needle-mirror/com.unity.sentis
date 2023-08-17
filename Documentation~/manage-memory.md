# Manage memory

As a Sentis user you are responsible for calling `Dispose` on any worker, inputs and sometimes outputs. You must call `Dispose` on outputs if you obtain them via `worker.FinishExecutionAndDownloadOutput` or if you take ownership of them by calling `tensor.TakeOwnership`.  

**Note:** Calling `Dispose` is necessary to properly free up GPU resources.

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

You don't need to call `Dispose` for the following:

- Tensors that you receive via the `worker.PeekOutput` call.
- `CPUOps`, `GPUPixelOps`, `GPUComputeOps` and `GPUCommandBufferOps` objects you create with no allocator.

## Additional resources

- [Profile a model](profile-a-model.md)
- [Create an engine to run a model](create-an-engine.md)
- [Create and modify tensors](do-basic-tensor-operations.md)
