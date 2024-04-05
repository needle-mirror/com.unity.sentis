# Upgrade from Sentis 1.3 to Sentis 1.4

To upgrade from Sentis 1.3 to Sentis 1.4, do the following:

- Reimport models that were previously imported in an earlier version of Sentis.
- Reexport serialized .sentis files and encrypted serialized models using Sentis 1.4.
- Replace uses of `IWorker.FinishExecutionAndDownloadOutput` with `IWorker.TakeOutputOwnership`.
- Replace uses of `Tensor.TakeOwnership` with `Tensor.CompleteOperationsAndDownload`.
- Replace uses of `Tensor.tensorOnDevice` with `Tensor.dataOnBackend`.
- Replace uses of `Tensor.Zeros` with `Tensor.AllocZeros`.
- Replace uses of the `Tensor.shape` setter with `Tensor.Reshape`.
- Remove uses of `Tensor.ShallowReshape`, use `Tensor.Reshape` to reshape a tensor in place or `Tensor.AllocNoData` and `IBackend.Reshape` to create a new reshaped tensor.
- Remove uses of `Tensor.DeepCopy`, use `Tensor.AllocNoData` and `IBackend.MemCopy` to copy a tensor.
- Remove uses of the `Model` API, use the functional API to edit models:
    - Remove `Model.AddInput`, use `InputDef` for inputs in a `Functional.Compile` call.
    - Remove `Model.AddConstant`, use `Functional.Tensor` in a `Functional.Compile` call.
    - Remove `Model.AddLayer` and `Layer` constructors, use `Functional` methods in a `Functional.Compile` call.
- Edit `Model.AddOuput` parameters to include both the output name and output index (from the inspector).
- When calling `Model.outputs`, use `Model.Output.name` to get the name of the output.
- Replace uses of `Layer.name` and `Constant.name` with `Layer.index` and `Constant.index`.
- Remove uses of the `Ops` API, use the functional API to build models to operate on tensors, or use `IBackend` to operate on allocated tensors. 
- Replace uses of `ArrayTensorData` and `SharedArrayTensorData` with `BurstTensorData`.
- Remove `CustomLayer` custom ONNX layer importers as these are not compatible with Sentis 1.4 serialization. These will be reimplemented in an upcoming release.
- Replace uses of `IBackend.deviceType` with `IBackend.backendType` to get the back end type.
- Remove allocation of tensors using `IBackend` methods, either allocate tensors with `Tensor` or `IModelStorage`.
- Remove offset from constructors of `BurstTensorData`, if the offset needs to be greater than zero use a `NativeTensorArrayFromManagedArray` in the `BurstTensorData` constructor.
- Replace uses of `IWorker.StartManualSchedule` with `IWorker.ExecuteLayerByLayer`.
- Replace uses of `ITensorData.shape` with `Tensor.shape`.
- Remove uses of `ITensorData.AsyncReadbackRequest` and `ITensorData.IsAsyncReadbackRequestDone`, use `Tensor.ReadbackRequest`, `Tensor.ReadbackRequestAsync`, and `Tensor.IsReadbackRequestDone` for async readback of tensor data.
- Remove uses of `Model.Metadata`, this is not compatible with Sentis 1.4 serialization.
- Remove references to `Model.Warnings`, these are not compatible with Sentis 1.4 serialization, use the console to view importer errors and warnings.
- Remove uses of `Model.CreateWorker`, use `WorkerFactory.CreateWorker` to create a worker.
- Remove uses of `Model.ShallowCopy`, use the functional API to copy a model.
- Replace constructors of `SymbolicTensorDim` with `SymbolicTensorDim.Int`, `SymbolicTensorDim.Param` and `SymbolicTensorDim.Unknown` static methods.


# Upgrade from Sentis 1.1 or 1.2 to Sentis 1.3

To upgrade from Sentis 1.1 or 1.2 to Sentis 1.3, do the following:

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

Refer to [Import a model](import-a-model-file.md) for more information.

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
