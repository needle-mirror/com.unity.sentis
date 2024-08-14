# Understand the Sentis workflow

To use Sentis to run a neural network in Unity, follow these steps:

1. Use the `Unity.Sentis` namespace.
2. Load a neural network model file.
3. Create input for the model.
4. Create an inference engine (a worker).
5. Run the model with the input to compute a result (inference).
6. Get the result.

> [!TIP]
> Use the [Workflow example](workflow-example.md) to understand the workflow applied to a simple example.

## Use the Unity.Sentis namespace

To use the `Unity.Sentis` namespace, add the following to the top of your script:

```
using Unity.Sentis;
```

## Load a model

Sentis can import model files in [Open Neural Network Exchange](https://onnx.ai/) (ONNX) format. To load a model, follow these steps:

1. [Export a model to ONNX format from a machine learning framework](export-convert-onnx.md) or download an ONNX model from the Internet.

2. Add the model file to the `Assets` folder of the **Project** window.

3. Create a runtime model in your script:

```
ModelAsset modelAsset = Resources.Load("model-file-in-assets-folder") as ModelAsset;
var runtimeModel = ModelLoader.Load(modelAsset);
```
You can also add a `public ModelAsset modelAsset` as a public variable in GameObjects. In this case specify the model manually.

Refer to [Import a model file](import-a-model-file.md) for more information.

## Create input for the model

Use the [Tensor](xref:Unity.Sentis.Tensor) API to create a tensor with data for the model. You can convert an array or a texture to a tensor. For example:

```
// Convert a texture to a tensor
Texture2D inputTexture = Resources.Load("image-file") as Texture2D;
Tensor<float> inputTensor = TextureConverter.ToTensor(inputTexture);
// Convert an array to a tensor
int[] array = new int[] {1,2,3,4};
Tensor<int> inputTensor = new Tensor<int>(new TensorShape(4), array);
```

Refer to [Create input for a model](create-an-input-tensor.md) for more information.

## Create an inference engine (a worker)

In Sentis, a worker is the inference engine. You create a worker to break down the model into executable tasks, run the tasks on the GPU or CPU, and retrieve the result.

For example, the following creates a worker that runs on the GPU using Sentis compute shaders:

```
Worker worker = new Worker(runtimeModel, BackendType.GPUCompute);
```

Refer to [Create an engine](create-an-engine.md) for more information.

## Schedule the model

To run the model, use the [`Schedule`](xref:Unity.Sentis.Worker.Schedule*) method of the worker object with the input tensor.

```
worker.Schedule(inputTensor);
```
Sentis schedules the model layers on the given backend. Execution is asynchronous, so after this is called, tensor operations may still be pending.

Refer to [Run a model](run-a-model.md) for more information.

## Get the output

You can use methods such as [`PeekOutput`](xref:Unity.Sentis.Worker.PeekOutput*) to get the output data from the model. For example:

```
Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
```

Refer to [Get output from a model](get-the-output.md) for more information.

## Additional resources

- [Workflow example](workflow-example.md)
- [Samples](package-samples.md)
- [Unity Discussions group for the Sentis beta](https://discussions.unity.com/c/10)
- [Understand models in Sentis](models-concept.md)
- [Tensor fundamentals in Sentis](tensor-fundamentals.md)
