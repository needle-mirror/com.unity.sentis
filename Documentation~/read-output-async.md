# Read output from a model asynchronously

If you want the output tensor data from your model execution in a readable format you can use the [`CompleteOperationsAndDownload`](xref:Unity.Sentis.Tensor.CompleteOperationsAndDownload) method directly.

However the following might be true when Sentis returns the tensor from `PeekOutput`:
- Sentis might not have finished calculating the final tensor data, so there's scheduled work remaining.
- If you using a GPU back end, the calculated tensor data might be on the GPU. This requires a read back to copy the data to the CPU in a readable format.

If either of the above are true, `CompleteOperationsAndDownload` blocks the main thread until the steps complete. To avoid this, you can follow these steps to use asynchronous readback:

1. Use the [`Tensor.ReadbackRequest`](xref:Unity.Sentis.Tensor.ReadbackRequest(Action{System.Boolean})) method and provide a callback.
2. Sentis invokes the callback when the readback is complete. The boolean argument is `true` when readback was successful.
3. Use [`CompleteOperationsAndDownload`](xref:Unity.Sentis.Tensor.CompleteOperationsAndDownload) to put the downloaded data into a readable state.

You can also use `ReadbackRequestAsync` to await the completion of readback request.
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
        // The call to `CompleteOperationsAndDownload` will no longer block with a readback as the data is already on the CPU
        m_OutputTensor.CompleteOperationsAndDownload();
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
        m_OutputTensor.ReadbackRequest(ReadbackCallback);

        // Continue to run code on the main thread while waiting for the tensor readback
    }

    void OnDisable()
    {
        m_Input.Dispose();
        m_Engine.Dispose();
    }
}

```

Note:
You can also avoid a Tensor data mutation to a CPU tensor that `CompleteOperationsAndDownload` does.
For that, simply call `tensor.dataOnBackend.Download<T>()` to get the data directly. This will keep the `tensor.dataOnDevice` on the given backend but you will have a CPU copy of it.
Be careful with synchronization issues if you re-run a worker, you will need to issue a new download request.

Refer to the `Read output asynchronously` example in the [sample scripts](package-samples.md) for an example.

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Use output data](use-model-output.md)
- [Get output from any layer](profile-a-model.md#get-output-from-any-layer)
