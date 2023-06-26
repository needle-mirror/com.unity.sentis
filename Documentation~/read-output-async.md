# Read output from a model asynchronously

After you run a model, you can read the output from a model asynchronously. This avoids Sentis blocking the main thread while it waits for the model to finish running then downloads the output data to the CPU.

Follow these steps:

1. Get a reference to the tensor output data using `PeekOutput`.
2. Use the [`Tensor.PrepareCacheForAccess`](xref:Unity.Sentis.Tensor.PrepareCacheForAccess(System.Boolean)) method and set the `blocking` parameter to `false`.
3. Use `Tensor.PrepareCacheForAccess` again to check if the data is complete.
4. The method returns `true` when the data is complete. You can then access the data.

Refer to the `AsyncReadback/AsyncReadbackCompute` example in the [sample scripts](package-samples.md) for a working example.

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Use output data](use-model-output.md)
- [Get output from any layer](profile-a-model.md#get-output-from-any-layer)
