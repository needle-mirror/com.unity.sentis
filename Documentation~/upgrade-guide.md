# Upgrade to Sentis 2.1.2

You do not need to take any actions to upgrade your project when upgrading from Sentis 2.0.0 or later. If you are upgrading from Sentis 1.6 or earlier please follow the instructions below.

# Upgrade from Sentis 1.6 to Sentis 2

To upgrade from Sentis 1.6 to Sentis 2, do the following:

- Replace uses of `TensorFloat` with `Tensor<float>` and uses of `TensorInt` with `Tensor<int>`.
- Replace uses of `TensorFloat.AllocZeros(shape)` with `new Tensor<float>(shape)` and uses of `TensorInt.AllocZeros(shape)` with `new Tensor<int>(shape)`.
- Replace uses of `TensorFloat.AllocEmpty(shape)` with `new Tensor<float>(shape, null)` and uses of `TensorInt.AllocEmpty(shape)` with `new Tensor<int>(shape, null)`.
- Replace uses of `new TensorFloat(float)` with `new Tensor<float>(new TensorShape(), new[] { float })` and uses of `new TensorInt(int)` with `new Tensor<int>(new TensorShape(), new[] { int })` for scalars.
- Replace uses of `tensor.ToReadOnlyArray()` with `tensor.DownloadToArray()`.
- Replace uses of `SymbolicTensorShape` with `DynamicTensorShape`, and `SymbolicTensorDim` with integers, where -1 represents an unknown dimension.
- Replace uses of `BurstTensorData` with `CPUTensorData`.
- Replace uses of `IWorker` or `GenericWorker` with `Worker`.
- Replace uses of `WorkerFactory.CreateWorker(backendType, model)` with `new Worker(model, backendType)`.
- Replace uses of `worker.Execute` with `worker.Schedule`.
- Remove uses of `worker.Execute` or `worker.SetInputs` with a dictionary, instead use an array or set the inputs one at a time by name.
- Replace uses of `worker.ExecuteLayerByLayer` with `worker.ScheduleIterable`.
- Replace uses of `BackendType.GPUCommandBuffer` with `BackendType.GPUCompute`.
- Replace uses of `commandBuffer.ExecuteWorker` with `commandBuffer.ScheduleWorker`.
- Rewrite uses of `worker.TakeOutputOwnership(...)` using `tensor.ReadbackAndClone(...)`,`tensor.ReadbackAndCloneAsync(...)` or `worker.CopyOutput(...)`.
- Replace uses of the `IBackend` methods with functional graph calls to build models.
- Replace uses of `Functional.Compile(Func)` with `new FunctionalGraph()` then `graph.Compile(outputs)` where the outputs are calculated from the inputs and constants.
- Replace uses of `InputDefs` with `graph.AddInput<T>(shape)`.
- Replace uses of `Functional.Tensor(shape, values)` with `Functional.Constant(shape, values)`.

# Upgrade from Sentis 1.5 to Sentis 1.6

To upgrade from Sentis 1.5 to Sentis 1.6, do the following:

- Replace uses of `IBackend.Concat` with calls to `IBackend.SliceSet`.
- Rewrite calls to `IBackend.Min` and `IBackend.Max` to not use tensor arrays as inputs.
- Rewrite calls to `IBackend.Sum` and `IBackend.Mean` to use `IBackend.Add` and `IBackend.ScalarMad` instead.
- Remove `keepdim` argument from calls to backend reduction methods such as `IBackend.ReduceMin`.
- Replace strings with ints for tensor indexing internal to models, e.g. when referencing `Output.index`.
- Replace uses of `CompleteOperationsAndDownload` with `ReadbackAndClone`, this will create a new tensor object which you are responsible for disposing.
- Rewrite any code that assumes that a zero-length tensor will have a null `Tensor.dataOnBackend`.

# Upgrade from Sentis 1.3 or Sentis 1.4 to Sentis 1.5

To upgrade from Sentis 1.3 or Sentis 1.4 to Sentis 1.5, do the following:

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
- Replace uses of `ArrayTensorData` and `SharedArrayTensorData` with `CPUTensorData`.
- Remove `CustomLayer` custom ONNX layer importers as these are not compatible with Sentis 1.4 serialization. These will be reimplemented in an upcoming release.
- Replace uses of `IBackend.deviceType` with `IBackend.backendType` to get the backend type.
- Remove allocation of tensors using `IBackend` methods, either allocate tensors with `Tensor` or `IModelStorage`.
- Remove offset from constructors of `CPUTensorData`, if the offset needs to be greater than zero use a `NativeTensorArrayFromManagedArray` in the `CPUTensorData` constructor.
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
- Use `Tensor<float>` or `Tensor<int>` for input and output tensors.
- Update methods that convert between tensors and textures.
- Convert the model asset type.
- Convert backend types.
- Update getting output from intermediate layers.
- Replace `asFloats` and `asInts`.

## Replace references to Barracuda with Sentis

All namespaces now use `Sentis` instead of `Barracuda`. To upgrade your project, change all references to `Barracuda`. For example, change `using Unity.Barracuda` to `using Unity.Sentis`.

## Update tensor operations in your project

The way tensors work has changed. Sentis no longer converts tensors to different layouts automatically, so you might need to update your code to make sure input and output tensors are the layout you expect. Refer to [Tensor fundamentals](tensor-fundamentals.md) for more information.

## Use Tensor<float> or Tensor<int> to create tensors

Sentis supports tensors that contain floats or ints.

If you use the `new Tensor()` constructor in your code, you must replace it with either `new Tensor<float>()` or `new Tensor<int>()`.

You can no longer pass dimension sizes to the constructor directly. Instead, you can use the `TensorShape` constructor to create a tensor shape, then pass the `TensorShape` to the `Tensor<float>` or `Tensor<int>` constructor.

The following example creates a 1D tensor of length 4:

```
Tensor<float> inputTensor = new Tensor<float>(new TensorShape(4), new[] { 2.0f, 1.0f, 3.0f, 0.0f });
```

Refer to [Create input for a model](create-an-input-tensor.md) for more information.

## Update methods that convert between tensors and textures

You can no longer pass a texture as a parameter to a `Tensor` constructor directly. Use the `TextureConverter.ToTensor` API instead. Refer to [Convert a tensor to a texture](convert-texture-to-tensor.md) for more information.

For example:

```
Tensor<float> inputTensor = TextureConvert.ToTensor(inputTexture);
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

## Convert backend types

Update your code to reflect the following changes to backend type names:

- Use `BackendType.GPUCompute` instead of `WorkerFactory.Type.Compute`, `WorkerFactory.Type.ComputeRef` or `WorkerFactory.Type.ComputeRefPrecompiled`.
- Use `BackendType.CPU` instead of `WorkerFactory.Type.CSharpBurst`, `WorkerFactory.Type.CSharpRef` or `WorkerFactory.Type.CSharp`.

For example, use the following to create a worker that runs on the GPU with Sentis compute shaders:

```
IWorker worker = new Worker(runtimeModel, BackendType.GPUCompute);
```

## Update getting output from intermediate layers

To get output from layers other than the output layers from the model, you now need to use `AddOutput` before you create the worker and use `PeekOutput`.
For example:

```
// Add the layer to the model outputs in the runtime model
runtimeModel.AddOutput("ConvolutionLayer");

// Create a worker
worker = new Worker(runtimeModel, BackendType.GPUCompute);

// Run the model with the input data
worker.Schedule(inputTensor);

// Get the output from the model
Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;

// Get the output from the ConvolutionLayer layer
Tensor<float> convolutionLayerOutputTensor = worker.PeekOutput("ConvolutionLayer") as Tensor<float>;
```

You should only use this for debugging. Refer to [Profile a model](profile-a-model.md) for more information.

## Replace asFloats and asInts

The `Tensor` classes no longer contain the `asFloats` and `asInts` methods. Use `ToReadOnlyArray` instead.
