# Access tensor data directly

To avoid a slow readback of a tensor from a device when accessing or passing it between multiple models, opt to directly read from and write to the tensor's underlying native data.

Refer to [Tensor fundamentals in Sentis](tensor-fundamentals.md#memory-location) for more information about how Sentis stores tensor data.

## Check where the data for a tensor is stored

Use the [`dataOnBackend.backendType`](xref:Unity.Sentis.ITensorData.backendType) property of a tensor to check where the tensor data is stored. The property is either [`BackendType.CPU`](xref:Unity.Sentis.BackendType.CPU), [`BackendType.GPUCompute`](xref:Unity.Sentis.BackendType.GPUCompute), or [`BackendType.GPUPixel`](xref:Unity.Sentis.BackendType.GPUPixel).

For example:

```
using UnityEngine;
using Unity.Sentis;

public class CheckTensorLocation : MonoBehaviour
{

    public Texture2D inputTexture;

    void Start()
    {
        // Create input data as a tensor
        Tensor inputTensor = TextureConverter.ToTensor(inputTexture);

        // Check if the tensor is stored in CPU or GPU memory, and write to the Console window.
        Debug.Log(inputTensor.dataOnBackend.backendType);
    }
}
```

If you want to force a tensor to the other device, use the following:

- [`ComputeTensorData.Pin`](xref:Unity.Sentis.ComputeTensorData.Pin*) to force a tensor into GPU compute shader memory in a [`ComputeBuffer`](xref:UnityEngine.ComputeBuffer).
- [`CPUTensorData.Pin`](xref:Unity.Sentis.CPUTensorData.Pin*) to force a tensor into CPU memory.

For example:

```
// Create a tensor
Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 3, 2, 2));

// Force the tensor into GPU memory
ComputeTensorData computeTensorData = ComputeTensorData.Pin(inputTensor);
```

!!! note "Note"

    * If the tensor data is already on the device you force it to, the method is a passthrough.
    * If not, the previous data will be disposed and new memory will be allocated on the target backend.

## Access CPU data directly

When the tensor is on the CPU and all operations that depend on it have finished, the tensor becomes readable and writable.

You can use indexers to manipulate the tensor data.

```
var tensor = new Tensor<float>(new TensorShape(1, 2, 3));
//...
if (tensor.backendType == BackendType.CPU && tensor.IsReadbackRequestDone()) {
    // tensor is read-writable directly
    tensor[0, 1, 0] = 1f;
    tensor[0, 1, 1] = 2f;
    tensor[0, 1, 2] = 3f;
    float val = tensor[0, 0, 2];
}
```

You can also get a readable flattened-version of the tensor as a span or NativeArray. The data will be row major flattened memory layout of the Tensor.

```
var tensor = new Tensor<float>(new TensorShape(1, 2, 3));
//...
if (tensor.backendType == BackendType.CPU && tensor.IsReadbackRequestDone()) {
    // tensor is readable
    var nativeArray = tensor.AsReadOnlyNativeArray();
    float val010 = nativeArray[3 + 0];
    float val011 = nativeArray[3 + 1];
    float val012 = nativeArray[3 + 2];

    var span = tensor.AsReadOnlySpan();
    float val002 = span[2];
}
```

## Upload data directly to backend memory

You can use [`Upload`](xref:Unity.Sentis.Tensor`1.Upload*) to upload data directly to the tensor.

```
var tensor = new Tensor<float>(new TensorShape(1,2,3), new [] { 0f, 1f, 2f, 3f, 4f, 5f });
tensor.Upload(new [] { 6f, 7f, 8f });
// tensor dataOnBackend now contains {6,7,8,3,4,5}
```
This method works for all tensor data backends but may be a blocking call. If the tensor data is on the CPU, Sentis will block until the tensor's pending jobs are complete. If the tensor data is on the GPU Sentis will perform a GPU upload.

## Access a tensor in GPU memory

To access a tensor in GPU-compute memory, first get the tensor data as a [`ComputeTensorData`](xref:Unity.Sentis.ComputeTensorData) by using [`ComputeTensorData.Pin`](xref:Unity.Sentis.ComputeTensorData.Pin*).

You can then use the [`buffer`](xref:Unity.Sentis.ComputeTensorData.buffer) property to directly access the tensor data in the compute buffer. Refer to [`ComputeBuffer`](xref:UnityEngine.ComputeBuffer) in the Unity API reference for more information about how to access a compute buffer.

Refer to the `Read output asynchronously` example in the [sample scripts](package-samples.md) for an example.

## Access a tensor in CPU memory

To access a tensor in CPU memory, first get the tensor data as a [`CPUTensorData`](xref:Unity.Sentis.CPUTensorData) object by using [`CPUTensorData.Pin`](xref:Unity.Sentis.CPUTensorData.Pin*).

You can then use the object in a Burst function like [`IJobParallelFor`](xref:Unity.Jobs.IJobParallelFor) to read from and write to the tensor data. You can also use the read and write fence ([`CPUTensorData.fence`](xref:Unity.Sentis.CPUTensorData.fence) and [`CPUTensorData.reuse`](xref:Unity.Sentis.CPUTensorData.reuse) respectively) properties of the object to handle Burst job dependencies.

Refer to the following:

- `Use the job system to write data` example in the [sample scripts](package-samples.md) for an example
- [Job System](https://docs.unity3d.com/Manual/JobSystem.html)

You can then use methods in the [`NativeTensorArray`](xref:Unity.Sentis.NativeTensorArray) class to read from and write to the tensor data as a native array.

## Additional resources

- [Tensor fundamentals in Sentis](tensor-fundamentals.md)
