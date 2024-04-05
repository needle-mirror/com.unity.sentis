# How Sentis runs a model

Sentis performs optimized tensor operations by scheduling the work across multiple threaded jobs on the CPU or executing the task in parallel on the GPU using pixel and compute shaders.

When the [engine](create-an-engine.md) executes a model, it steps through each layer of the model, and executes the layer operation on the input tensors to calculate one or more output tensors.
The [`BackendType`](xref:Unity.Sentis.BackendType) you choose determines how and when the worker performs each operation.

The following tables defines the types of back end available:

|`BackendType`|Runs on|Description| 
|-|-|-|
|`CPU`|CPU, using [Burst](https://docs.unity3d.com/Packages/com.unity.burst@latest/)|Sentis creates, sets up and [schedules](https://docs.unity3d.com/Manual/JobSystemCreatingJobs.html) a Burst job for the operation. If the input tensors are output from other jobs the engine creates a [job dependency](https://docs.unity3d.com/Manual/JobSystemJobDependencies.html) to ensure correct inference without blocking.|
|`GPUCompute`|GPU, using Sentis compute shaders|Sentis creates, sets up and [dispatches](https://docs.unity3d.com/ScriptReference/ComputeShader.Dispatch.html) a compute shader for the operation. The GPU queues the work to be executed in the correct order.|
|`GPUCommandBuffer`|GPU, using Sentis compute shaders with [command buffers](https://docs.unity3d.com/ScriptReference/Rendering.CommandBuffer.html)|Sentis creates, sets up and adds a compute shader the command buffer. You have to execute the command buffer manually to perform the operations.|
|`GPUPixel`|GPU, using Sentis pixel shaders|Sentis creates, sets up and executes a pixel shader by blitting.|

## Tensor outputs

Sentis schedules all tensor operations on the main thread, and returns output tensors synchronously. The tensor data is stored as a handle to a native memory location on the CPU or GPU, depending on the back end.

The values of the tensor may not be calculated at the point the tensor object is returned, with work still scheduled to be done. This allows you to schedule further tensor operations without waiting for or interrupting scheduled operations.

To complete the execution of the work on the back end, move the tensor data to the CPU.

You can call [`CompleteOperationsAndDownload`](xref:Unity.Sentis.Tensor.CompleteOperationsAndDownload) to convert the tensor to a CPU tensor. This is a blocking call that waits synchronously for both the execution to complete and the data to be read back from the back end. Note that this process can be slow, especially when reading back from the GPU.

To move the tensor data to the CPU with a non-blocking, non-destructive download: 

1. Do a [`ReadbackRequest`](xref:Unity.Sentis.Tensor.ReadbackRequest) on your tensor.
2. Use `tensor.dataOnBackend.Download`.

## CPU fallback

Sentis does not support every operator on every back end type. Refer to [Supported ONNX operators](supported-operators.md) for more information.

If Sentis supports an operator on the CPU but not the GPU, Sentis might automatically fall back to running on the CPU. This requires Sentis to sync with the GPU and read back the input tensors to the CPU. If the output tensor is used in a GPU operation, Sentis completes the operation and uploads the tensor to the GPU.

If a model contains a lot of layers that use CPU fallback, Sentis might spend a lot of time uploading and reading back from the CPU. This can have a big effect on the performance of your model. To reduce CPU fallback, build the model in a way that Sentis can run more fully on your chosen back end type, or use the CPU back end.

For some operations, Sentis needs to read the data contents of a tensor in the main thread so it can schedule the operation. For example, the `shape` input tensor for an `Expand`, or the `axes` input for a `Reduce`. These tensors might be the result of other operations. Sentis optimizes the model when you input it, to calculate which of the tensors need to be calculated on the CPU regardless of the back end type for the model.


## Additional resources

- [Use output data](use-model-output.md)
- [Read output from a model asynchronously](read-output-async.md)
