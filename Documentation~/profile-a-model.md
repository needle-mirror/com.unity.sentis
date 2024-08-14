# Profile a model

The performance of a model depends on the following factors:

- The complexity of the model.
- Whether the model uses performance-heavy operators such as `Conv` or `MatMul`.
- The features of the platform you run the model on, for example CPU memory, GPU memory, and number of cores.
- Whether Sentis downloads data to CPU memory when you access a tensor. Refer to [Get output from a model](get-the-output.md) for more information.

## Profile a model in the Profiler window

To get performance information when you run a model, you can use the following:

- [The Profiler window](https://docs.unity3d.com/Documentation/Manual/Profiler.html).
- [RenderDoc](https://docs.unity3d.com/Documentation/Manual/RenderDocIntegration.html), a third-party graphics debugger.

The **Profiler** window displays each Sentis layer as a dropdown item in the **Module Details** panel. Open a layer to get a detailed timeline of the execution of the layer.

When a layer executes methods that include **Download** or **Upload**, Sentis transfers data to or from the CPU or the GPU. This might slow down the model.

If your model runs slower than you expect, refer to:

- [Understand models in Sentis](models-concept.md) for information about how the complexity of a model might affect performance.
- [Create an engine to run a model](create-an-engine.md) for information about different types of worker.

## Additional resources

- [Inspect a model](inspect-a-model.md)
- [How Sentis optimizes a model](models-concept.md#how-sentis-optimizes-a-model)
- [Supported ONNX operators](supported-operators.md)

