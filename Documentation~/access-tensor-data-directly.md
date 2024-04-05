# Access tensor data directly

To avoid having to do a slow readback of a tensor from a device when you want to access a tensor, or when you need to pass a tensor between multiple models, you can read from and write to the tensor underlying native data directly instead.

Refer to [Tensor fundamentals in Sentis](tensor-fundamentals.md#memory-location) for more information about how Sentis stores tensor data.

## Check where the data for a tensor is stored

Use the `dataOnBackend.backendType` property of a tensor to check where the tensor data is stored. The property is either `CPU`, `GPUPixel` or `GPUCompute`.

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

- [`ComputeTensorData.Pin`](xref:Unity.Sentis.ComputeTensorData.Pin(Unity.Sentis.Tensor,System.Boolean)) to force a tensor into GPU compute shader memory in a [ComputeBuffer](https://docs.unity3d.com/ScriptReference/ComputeBuffer.html).
- [`BurstTensorData.Pin`](xref:Unity.Sentis.ComputeTensorData.Pin(Unity.Sentis.Tensor,System.Boolean)) to force a tensor into CPU memory.

For example:

```
// Create a tensor
TensorFloat inputTensor = TensorFloat.AllocZeros(new TensorShape(1, 3, 2, 2));

// Force the tensor into GPU memory
ComputeTensorData computeTensorData = ComputeTensorData.Pin(inputTensor);
```

Note:
* If the tensor data is already on the device you force it to, the method is a passthrough.
* If not the previous data will be disposed and new memory will be allocated on the target backend.
## Access a tensor in GPU memory

To access a tensor in GPU-compute memory, first get the tensor data as a [`ComputeTensorData`](xref:Unity.Sentis.ComputeTensorData) by using [`ComputeTensorData.Pin`](xref:Unity.Sentis.ComputeTensorData.Pin(Unity.Sentis.Tensor,System.Boolean)).

You can then use the `buffer` property of the `ComputeTensorData` object to access the tensor data in the compute buffer directly. Refer to [`ComputeBuffer`](https://docs.unity3d.com/ScriptReference/ComputeBuffer.html) in the Unity API reference for more information about how to access a compute buffer.

Refer to the `Read output asynchronously` example in the [sample scripts](package-samples.md) for an example.

## Access a tensor in CPU memory

To access a tensor in CPU memory, first get the tensor data as a [`BurstTensorData`](xref:Unity.Sentis.BurstTensorData) object by using [`BurstTensorData.Pin`](xref:Unity.Sentis.BurstTensorData.Pin(Unity.Sentis.Tensor,System.Boolean)).

You can then use the object in a Burst function like [`IJobParallelFor`](https://docs.unity3d.com/ScriptReference/Unity.Jobs.IJobParallelFor.html) to read from and write to the tensor data. You can also use the read and write fence ([`fence`](xref:Unity.Sentis.BurstTensorData.fence) and [`reuse`](xref:Unity.Sentis.BurstTensorData.reuse) respectively) properties of the object to handle Burst job depedencies.

Refer to the following:

- The `Use Burst to write data` example in the [sample scripts](package-samples.md) for an example.
- The [Burst documentation](https://docs.unity3d.com/Packages/com.unity.burst@latest).

You can then use methods in the [`NativeTensorArray`](xref:Unity.Sentis.NativeTensorArray) class to read from and write to the tensor data as a native array.

## Additional resources

- [Tensor fundamentals in Sentis](tensor-fundamentals.md)
