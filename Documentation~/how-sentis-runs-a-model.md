# How Sentis runs a model

Sentis performs optimized tensor operations by scheduling the work across multiple threaded jobs on the central processing unit (CPU) or executing the task in parallel on the graphics processing unit (GPU) using pixel and compute shaders.

When the [engine](create-an-engine.md) executes a model, it steps through each layer of the model. It executes the layer operation on the input tensors to calculate one or more output tensors.

The [`BackendType`](xref:Unity.Sentis.BackendType) you choose determines how and when the worker performs each operation.

The following tables defines the types of backend available:

|`BackendType`|Runs on|Description|
|-|-|-|
|`CPU`|CPU, using [Burst](https://docs.unity3d.com/Packages/com.unity.burst@latest/)|Sentis creates, sets up, and [schedules](https://docs.unity3d.com/Manual/JobSystemCreatingJobs.html) a Burst job for the operation. If the input tensors are output from other jobs, the engine creates a [job dependency](https://docs.unity3d.com/Manual/JobSystemJobDependencies.html) to ensure correct inference without blocking.|
|`GPUCompute`|GPU, using Sentis compute shaders|Sentis creates, sets up, and [dispatches](https://docs.unity3d.com/ScriptReference/ComputeShader.Dispatch.html) a compute shader for the operation. The GPU queues the work to be executed in the correct order.|
|`GPUCommandBuffer`|GPU, using Sentis compute shaders with [command buffers](https://docs.unity3d.com/ScriptReference/Rendering.CommandBuffer.html)|Sentis creates, sets up, and adds a compute shader the command buffer. You have to execute the command buffer manually to perform the operations.|
|`GPUPixel`|GPU, using Sentis pixel shaders|Sentis creates, sets up, and executes a pixel shader by blitting.|

## Tensor outputs

Sentis schedules all tensor operations on the main thread and returns output tensors synchronously. Depending on the backend, a native memory location on the CPU or GPU stores the tensor data.

The tensor values might not be fully calculated when the tensor object is returned, as there might still be scheduled work pending. This allows you to schedule further tensor operations without waiting for or interrupting scheduled operations.

To complete the execution of the work on the backend, move the tensor data to the CPU.

Call [`ReadbackAndClone`](xref:Unity.Sentis.Tensor.ReadbackAndClone) to get a CPU copy of the tensor. This is a blocking call that waits synchronously for both the execution to complete and the data to be read back from the backend. Note that this process can be slow, especially when reading back from the GPU.
Alternatively, use [`ReadbackAndCloneAsync`](xref:Unity.Sentis.Tensor.ReadbackAndCloneAsync) for an `Awaitable` version of this method.

To move the tensor data to the CPU with a non-blocking, non-destructive download:

1. Do a [`ReadbackRequest`](xref:Unity.Sentis.Tensor.ReadbackRequest) on your tensor.
2. Use `ReadbackAndCloneAsync` on your tensor.
3. Use `tensor.dataOnBackend.Download`.

## CPU fallback

Sentis doesn't support every operator on every backend type. For more information, refer to [Supported ONNX operators](supported-operators.md).

If Sentis supports an operator on the CPU but not the GPU, Sentis might automatically fall back to running on the CPU. This requires Sentis to sync with the GPU and read back the input tensors to the CPU. If a GPU operation uses the output tensor, Sentis completes the operation and uploads the tensor to the GPU.

If a model has many layers that use CPU fallback, Sentis might spend significant time to upload and read back from the CPU. This can impact the performance of your model. To reduce CPU fallback, build the model so that Sentis runs effectively on your chosen backend type or use the CPU backend.

For some operations, Sentis needs to read the data contents of a tensor in the main thread so it can schedule the operation. For example, the `shape` input tensor for an `Expand` or the `axes` input for a `Reduce`. These tensors might be the result of other operations. Sentis optimizes the model during input to determine which tensors need to be calculated on the CPU, irrespective of the model's backend type.


## Additional resources

- [Use output data](use-model-output.md)
- [Read output from a model asynchronously](read-output-async.md)
