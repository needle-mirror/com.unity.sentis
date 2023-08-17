# Tensor fundamentals in Sentis

In Sentis, you input and output data in multi-dimensional arrays called tensors. Tensors in Sentis work similarly to tensors in TensorFlow, PyTorch, and other machine learning frameworks.

Tensors in Sentis can have up to 8 dimensions. If a tensor has zero dimensions, the tensor contains a single value and is called a scalar.

You can create the following types of tensor:

- `TensorFloat`, which stores the data as floats.
- `TensorInt`, which stores the data as ints.

Refer to [Create and modify tensors](do-basic-tensor-operations.md) for more information.

## Memory layout

Sentis stores tensors in memory in row-major order. This means the values of the last dimension of a tensor are adjacent in memory.

For example, for a 2 × 2 × 3 tensor with the values 0 to 11:

![](images/tensor-memory-layout.svg)
A 2 × 2 × 3 tensor with the values 0 to 11, and how Sentis stores the tensor in memory.

## Format

A model usually needs an input tensor in a certain format. For example, a model that processes images might need a 3-channel 240 × 240 image in one of the following formats:

- 1 × 240 × 240 × 3, where the order of the dimensions is batch size, height, width, channels (NHWC)
- 1 × 3 × 240 × 240, where the order of the dimensions is batch size, channels, height, width (NCHW)

If your tensor doesn't match the format the model needs, you might get unexpected results. 

You can use [Ops.Transpose](xref:Unity.Sentis.Ops.Transpose(Unity.Sentis.Tensor)) to convert a tensor to a different format. Refer to [Create and modify tensors](do-basic-tensor-operations.md) for more information.

Refer to [Create input for a model](create-an-input-tensor.md) for more information if you need to convert a texture to a tensor in a specific format.

## Memory location

Sentis stores tensor data in GPU memory or CPU memory.

Sentis usually stores tensors in the memory that matches the [back end type](create-an-engine.md#back-end-types) you use. For example if you use the `BackendType.GPUCompute` back end type, Sentis usually stores tensors in GPU memory.

You can only read from and write to the elements of a tensor directly if the tensor is on the CPU, and it can be slow. It's faster if you use the optimized methods in the `Ops` API.

If you need to read from and write to the elements of a tensor directly, use `Tensor.MakeReadable()`. Sentis performs a blocking readback of the tensor to the CPU. The next time you use this tensor in a model or operation on the GPU there will be an automatic blocking upload.   

To avoid Sentis performing a blocking readback and upload, you can also use a compute shader, Burst or a native array to read from and write to the tensor data directly in memory. Refer to [Access tensor data directly](access-tensor-data-directly.md) for more information.

When you need to read an output tensor, you can also do an asynchronous readback, so Sentis doesn't block the main code thread while it waits for the model to finish then downloads the whole tensor. Refer to [Read output from a model asynchronously](read-output-async.md) for more information.

## Additional resources

- [Understand the Sentis workflow](understand-sentis-workflow.md)
- [Understand models in Sentis](models-concept.md)
- [Create and modify tensors](do-basic-tensor-operations.md)

