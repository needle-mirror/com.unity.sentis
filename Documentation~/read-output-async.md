# Read output from a model asynchronously

After you run a model, you can read the output from a model asynchronously. This avoids Sentis blocking the main thread while it waits for the model to finish running then downloads the output data to the CPU.

Follow these steps:

1. Get a reference to the tensor output data using `PeekOutput`.
2. Use the [`Tensor.AsyncReadbackRequest`](xref:Unity.Sentis.Tensor.AsyncReadbackRequest(Action<bool>)) method and provide a callback.
3. Sentis invokes the callback when the readback is complete. The boolean argument is `true` when readback was successful.
4. Use [`Tensor.MakeReadable`](xref:Unity.Sentis.Tensor.MakeReadable()) to put the downloaded data into a readable state.

Refer to the `Read output asynchronously` example in the [sample scripts](package-samples.md) for an example.

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Use output data](use-model-output.md)
- [Get output from any layer](profile-a-model.md#get-output-from-any-layer)
