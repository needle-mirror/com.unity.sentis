# Read output from a model asynchronously

If you want the output tensor data from your model execution in a readable format you can use the `Tensor.MakeReadable` method directly.

However the following might be true when Sentis returns the tensor from `PeekOutput`:
- Sentis might not have finished calculating the final tensor data, so there's scheduled work remaining.
- If you using a GPU back end, the calculated tensor data might be on the GPU. This requires a read back to copy the data to the CPU in a readable format.

If either of the above are true, `Tensor.MakeReadable` blocks the main thread until the steps complete. To avoid this, you can follow these steps to use asynchronous readback:

1. Get a reference to the tensor output data using `PeekOutput`. You don't need to dispose of the output unless you call `TakeOwnership`.
2. Use the [`Tensor.AsyncReadbackRequest`](xref:Unity.Sentis.Tensor.AsyncReadbackRequest(Action{System.Boolean})) method and provide a callback.
3. Sentis invokes the callback when the readback is complete. The boolean argument is `true` when readback was successful.
4. Use [`Tensor.MakeReadable`](xref:Unity.Sentis.Tensor.MakeReadable) to put the downloaded data into a readable state.

```
using Unity.Sentis;
using UnityEngine;

public class AsyncReadbackCompute : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;

    Tensor m_Input;
    IWorker m_Engine;
    TensorFloat m_OutputTensor;

    void ReadbackCallback(bool completed)
    {
        // The call to `MakeReadable` will no longer block with a readback as the data is already on the CPU
        m_OutputTensor.MakeReadable();
        // The output tensor is now in a readable state on the CPU
    }

    void OnEnable()
    {
        var model = ModelLoader.Load(modelAsset);
        m_Input = new TensorFloat(new TensorShape(1, 1), new[] { 43.0f });
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        m_Engine.Execute(m_Input);

        // Peek the value from Sentis, without taking ownership of the tensor
        m_OutputTensor = m_Engine.PeekOutput() as TensorFloat;
        m_OutputTensor.AsyncReadbackRequest(ReadbackCallback);
        
        // Continue to run code on the main thread while waiting for the tensor readback
    }

    void OnDisable()
    {
        m_Input.Dispose();
        m_Engine.Dispose();
    }
}
```

Refer to the `Read output asynchronously` example in the [sample scripts](package-samples.md) for an example.

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Use output data](use-model-output.md)
- [Get output from any layer](profile-a-model.md#get-output-from-any-layer)
