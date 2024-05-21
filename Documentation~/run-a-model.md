# Run a model

After you [create a worker](create-an-engine.md), use [`Execute`](xref:Unity.Sentis.IWorker.Execute) to run the model.

```
worker.Execute(inputTensor);
```

The initial execution of a model within the Unity Editor may be slow as Sentis needs to compile code and shaders. Subsequent runs will be faster due to caching.

For an example, refer to the `Run a model` sample in the [sample scripts](package-samples.md).

## Additional resources

- [Run a model a layer at a time](run-a-model-a-layer-at-a-time.md)
- [Understand models in Sentis](models-concept.md)
- [Create an engine to run a model](create-an-engine.md)
- [Profile a model](profile-a-model.md)
