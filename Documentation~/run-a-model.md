# Run a model

After you [create a worker](create-an-engine.md), call [`Schedule`](xref:Unity.Sentis.Worker.Schedule*) to run the model.

```
worker.Schedule(inputTensor);
```

The first scheduling of a model within the Unity Editor may be slow as Sentis needs to compile code and shaders as well as allocating internal memory. Subsequent runs will be faster due to caching.
Including a dummy run when starting the application is a good idea.

For an example, refer to the `Run a model` sample in the [sample scripts](package-samples.md).

## Additional resources

- [Run a model a layer at a time](run-a-model-a-layer-at-a-time.md)
- [Understand models in Sentis](models-concept.md)
- [Create an engine to run a model](create-an-engine.md)
- [Profile a model](profile-a-model.md)
