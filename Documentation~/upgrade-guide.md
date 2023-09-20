# Upgrade from Sentis 1.1 to Sentis 1.2

To upgrade from Sentis 1.1 to Sentis 1.2, do the following:

- Reimport models that were previously imported in an earlier version of Sentis.

# Upgrade from Sentis 1.0 to Sentis 1.1

To upgrade from Sentis 1.0 to Sentis 1.1, do the following:

- Use `MakeReadable` on GPU output tensor before indexing (see `Use tensor indexing methods` sample).
- Remove `prepareCacheForAccess` parameter when calling `PeekOutput` and instead use `MakeReadable` on the returned tensor before reading it.
- Use AsyncReadbackRequest for async readbacks from GPU (see `Read output asynchronously` sample).
- Replace `: Layer` with `: CustomLayer` for custom layers and implement `InferOutputDataTypes` method (see `Add a custom layer` sample).
- Replace instances of `IOps` with `Ops`.
- Replace `uploadCache` with `clearOnInit` when using .Pin method.
- Replace uses of `CopyOutput` with `FinishExecutionAndDownloadOutput`.

# Upgrade from Barracuda 3.0 to Sentis 1.0

To upgrade from Barracuda 3.0 to Sentis 1.0, do the following:

- Replace references to `Barracuda` with `Sentis`.
- Update tensor operations in your project.
- Use `TensorFloat` or `TensorInt` for input and output tensors.
- Update methods that convert between tensors and textures.
- Convert the model asset type.
- Convert back end types.
- Update getting output from intermediate layers.
- Replace `asFloats` and `asInts`.

## Replace references to Barracuda with Sentis

All namespaces now use `Sentis` instead of `Barracuda`. To upgrade your project, change all references to `Barracuda`. For example, change `using Unity.Barracuda` to `using Unity.Sentis`.

## Update tensor operations in your project

The way tensors work has changed. Sentis no longer converts tensors to different layouts automatically, so you might need to update your code to make sure input and output tensors are the layout you expect. Refer to [Tensor fundamentals](tensor-fundamentals.md) for more information.

## Use TensorFloat or TensorInt to create tensors

Sentis supports tensors that contain floats or ints.

If you use the `new Tensor()` constructor in your code, you must replace it with either `new TensorFloat()` or `new TensorInt()`.

You can no longer pass dimension sizes to the constructor directly. Instead, you can use the `TensorShape` constructor to create a tensor shape, then pass the `TensorShape` to the `TensorFloat` or `TensorInt` constructor.

The following example creates a 1D tensor of length 4:

```
TensorFloat inputTensor = new TensorFloat(new TensorShape(4), new[] { 2.0f, 1.0f, 3.0f, 0.0f });
```

Refer to [Create input for a model](create-an-input-tensor.md) for more information.

## Update methods that convert between tensors and textures

You can no longer pass a texture as a parameter to a `Tensor` constructor directly. Use the `TextureConverter.ToTensor` API instead. Refer to [Convert a tensor to a texture](convert-texture-to-tensor.md) for more information.

For example:

```
TensorFloat inputTensor = TextureConvert.ToTensor(inputTexture);
```

Refer to [Use output data](use-model-output.md) for more information.

## Convert the model asset type

The `NNModel` object is now called `ModelAsset`. Update your code with the new name.

For example, when you create a runtime model:

```
using UnityEngine;
using Unity.Sentis;

public class CreateRuntimeModel : MonoBehaviour
{

    ModelAsset modelAsset;
    
    void Start()
    {
        Model runtimeModel = ModelLoader.Load(modelAsset);
    }
}
```

Refer to [Import a model](import-a-model.md) for more information.

## Convert back end types

Update your code to reflect the following changes to back end type names:

- Use `BackendType.GPUCompute` instead of `WorkerFactory.Type.Compute`, `WorkerFactory.Type.ComputeRef` or `WorkerFactory.Type.ComputeRefPrecompiled`. 
- Use `BackendType.CPU` instead of `WorkerFactory.Type.CSharpBurst`, `WorkerFactory.Type.CSharpRef` or `WorkerFactory.Type.CSharp`.

For example, use the following to create a worker that runs on the GPU with Sentis compute shaders:

```
IWorker worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);
```

## Update getting output from intermediate layers

To get output from layers other than the output layers from the model, you now need to use `AddOutput` before you create the worker and use `PeekOutput`. 
For example:

```
// Add the layer to the model outputs in the runtime model
runtimeModel.AddOutput("ConvolutionLayer");

// Create a worker
worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);

// Run the model with the input data
worker.Execute(inputTensor);

// Get the output from the model
TensorFloat outputTensor = worker.PeekOutput() as TensorFloat;

// Get the output from the ConvolutionLayer layer
TensorFloat convolutionLayerOutputTensor = worker.PeekOutput("ConvolutionLayer") as TensorFloat;
```

You should only use this for debugging. Refer to [Profile a model](profile-a-model.md) for more information.

## Replace asFloats and asInts

The `Tensor` classes no longer contain the `asFloats` and `asInts` methods. Use `ToReadOnlyArray` instead.
