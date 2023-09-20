# Access tensor data directly

To avoid having to do a slow readback of a tensor from a device when you want to access a tensor, or when you need to pass a tensor between multiple models, you can read from and write to the tensor data directly in memory instead.

Refer to [Tensor fundamentals in Sentis](tensor-fundamentals.md#memory-location) for more information about how Sentis stores tensor data.

You can also use `Ops` methods to do complicated tensor operations with a Sentis back end type that uses compute shaders or Burst. For example, you can use `Ops` to do matrix multiplication. Refer to [Do operations on tensors](do-complex-tensor-operations.md) for more information.

## Check where the data for a tensor is stored

Use the `tensorOnDevice.deviceType` property of a tensor to check where the tensor data is stored. The property is either `CPU` or `GPU`.

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
        Debug.Log(inputTensor.tensorOnDevice.deviceType);
    }
}
```

If you want to force a tensor to the other device, use the following:

- [`ComputeTensorData.Pin`](xref:Unity.Sentis.ComputeTensorData.Pin(Unity.Sentis.Tensor,System.Boolean)) to force a tensor into GPU compute shader memory in a [ComputeBuffer](https://docs.unity3d.com/ScriptReference/ComputeBuffer.html).
- [`BurstTensorData.Pin`](xref:Unity.Sentis.ComputeTensorData.Pin(Unity.Sentis.Tensor,System.Boolean)) to force a tensor into CPU memory for a Burst job.
- [`ArrayTensorData.Pin`](xref:Unity.Sentis.ArrayTensorData.Pin(Unity.Sentis.Tensor,System.Boolean)) to force a tensor into CPU memory as a [NativeArray](https://docs.unity3d.com/ScriptReference/Unity.Collections.NativeArray_1.html).

For example:

```
// Create a tensor
TensorFloat inputTensor = TensorFloat.Zeros(new TensorShape(1, 3, 2, 2));

// Force the tensor into GPU memory
ComputeTensorData gpuTensor = ComputeTensorData.Pin(inputTensor);
```

If the tensor data is already on the device you force it to, the method is a passthrough.

## Access a tensor in GPU memory

To access a tensor in GPU memory, first get the tensor data as a `ComputeTensorData` object. You can do either of the following to get a `ComputeTensorData` object:

- Use the return value of `ComputeTensorData.Pin`.
- Use `myTensor.tensorOnDevice as ComputeTensorData`.

    For example:

    ```
    // Create a tensor
    TensorFloat inputTensor = TensorFloat.Zeros(new TensorShape(1, 3, 2, 2));

    // Get the tensor data as a ComputeTensorData object
    ComputeTensorData gpuTensor = inputTensor.tensorOnDevice as ComputeTensorData;
    ```

You can then use the `buffer` property of the `ComputeTensorData` object and a compute shader to access the tensor data in the compute buffer directly. Refer to [`ComputeBuffer`](https://docs.unity3d.com/ScriptReference/ComputeBuffer.html) in the Unity API reference for more information about how to access a compute buffer.

Refer to the `Read output asynchronously` example in the [sample scripts](package-samples.md) for an example.

## Access a tensor in CPU memory

To access a tensor in CPU memory, first get the tensor data as a `BurstTensorData` or `ArrayTensorData` object.

### Get a BurstTensorData object

You can do either of the following to get a `BurstTensorData` object:

- Use the return value of `BurstTensorData.Pin`.
- Use `myTensor.tensorOnDevice as BurstTensorData`.

You can then use the object in a Burst function like `IJobParallelFor` to read from and write to the tensor data. You can also use the `fence` and `reuse` properties of the object to force your code to wait for a Burst job to finish.

Refer to the following:

- The `Use Burst to write data` example in the [sample scripts](package-samples.md) for an example.
- The [Burst documentation](https://docs.unity3d.com/Packages/com.unity.burst@latest).

### Get an ArrayTensorData object

You can do either of the following to get an `ArrayTensorData` object:

- Use the return value of `ArrayTensorData.Pin`.
- Use `myTensor.tensorOnDevice as ArrayTensorData`.

You can then use methods in the [`NativeTensorArray`](xref:Unity.Sentis.NativeTensorArray) class to read from and write to the tensor data as a native array.

Refer to [NativeArray](https://docs.unity3d.com/ScriptReference/Unity.Collections.NativeArray_1.html) in the Unity API reference for more information about how to use a native array.

## Additional resources

- [Tensor fundamentals in Sentis](tensor-fundamentals.md)
- [Do operations on tensors](do-complex-tensor-operations.md)
