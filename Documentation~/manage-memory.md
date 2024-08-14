# Manage memory

As a Sentis user, you're responsible for calling [`Dispose`](xref:Unity.Sentis.Worker.Dispose) on workers and tensors you instantiate. You must also call `Dispose` on cloned output tensors returned from the [`ReadbackAndClone`](Unity.Sentis.Tensor.ReadbackAndClone*) method.

> [!NOTE]
> Calling `Dispose` is necessary to free up graphics processing unit (GPU) resources.

For example:

```
void OnDestroy()
{
    worker?.Dispose();

    // Assuming model with multiple inputs that were passed as a array
    foreach (var input in inputs)
    {
        input.Dispose();
    }
}
```

When you get a handle to a tensor from a worker using the [`PeekOutput`](xref:Unity.Sentis.Worker.PeekOutput*) method, the memory allocator remains responsible for that memory. You don't need to `Dispose` of it. Refer to [Get output from a model](get-the-output.md) for more information.

## Additional resources

- [Profile a model](profile-a-model.md)
- [Create an engine to run a model](create-an-engine.md)
- [Create and modify tensors](do-basic-tensor-operations.md)
