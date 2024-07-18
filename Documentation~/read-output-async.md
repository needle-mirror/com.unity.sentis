# Read output from a model asynchronously

After you execute a model and access an output tensor from `PeekOutput`, the following are true:
- Sentis might not have finished calculating the final tensor data, so there's pending scheduled work.
- If you use a graphics processing unit (GPU) backend, the calculated tensor data might be on the GPU. This requires a read back to copy the data to the central processing unit (CPU) in a readable format.

If either of these conditions is true, `ReadbackAndClone` or `CompleteAllPendingOperations` blocks the main thread until the steps complete.

To avoid this, follow these two methods to use asynchronous readback:

1. Use the awaitable [`Tensor.ReadbackAndCloneAsync`](xref:Unity.Sentis.Tensor.ReadbackAndCloneAsync()) method. Sentis returns a CPU copy of the input tensor in a non blocking way.

```
using Unity.Sentis;
using UnityEngine;

public class AsyncReadbackCompute : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;

    TensorFloat m_Input;
    IWorker m_Engine;

    async void OnEnable()
    {
        var model = ModelLoader.Load(modelAsset);
        m_Input = new TensorFloat(new TensorShape(1, 1), new[] { 43.0f });
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        m_Engine.Execute(m_Input);

        // Peek the value from Sentis, without taking ownership of the tensor
        var outputTensor = m_Engine.PeekOutput() as TensorFloat;
        var cpuCopyTensor = await outputTensor.ReadbackAndCloneAsync();

        Debug.Assert(m_Output[0] == 42);
        Debug.Log($"Output tensor value {m_Output[0]}");
        cpuCopyTensor.Dispose();
    }

    void OnDisable()
    {
        m_Input.Dispose();
        m_Engine.Dispose();
    }
}
```

2. Use a polling mechanism with [`Tensor.ReadbackRequest`](xref:Unity.Sentis.Tensor.ReadbackRequest()) and [`Tensor.IsReadbackRequestDone`](xref:Unity.Sentis.Tensor.IsReadbackRequestDone()) methods.

```
using Unity.Sentis;
using UnityEngine;

public class AsyncReadbackCompute : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;

    TensorFloat m_Input, m_Output;
    IWorker m_Engine;

    void OnEnable()
    {
        var model = ModelLoader.Load(modelAsset);
        m_Input = new TensorFloat(new TensorShape(1, 1), new[] { 43.0f });
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
    }

    bool inferencePending = false;

    void OnUpdate()
    {
        if (!inferencePending)
        {
            m_Engine.Execute(m_Input);

            // Peek the value from Sentis, without taking ownership of the tensor
            m_Output = m_Engine.PeekOutput() as TensorFloat;

            // Trigger a non blocking readback request
            m_Output.ReadbackRequest();
            inferencePending = true;
        }
        else if (inferencePending && m_Output.IsReadbackRequestDone())
        {
            // m_Output is now downloaded to the cpu. Using ReadbackAndClone or ToReadOnlyArray will not be blocking
            var array = outputTensor.ToReadOnlyArray();
            Debug.Assert(array[0] == 42);
            Debug.Log($"Output tensor value {m_Output[0]}");
            inferencePending = false;
        }
    }

    void OnDisable()
    {
        m_Input.Dispose();
        TensorFloat.Dispose();
        m_Engine.Dispose();
    }
}
```

> [!NOTE]
> To avoid a Tensor data mutation to a CPU tensor that `ReadbackAndClone` does, call `tensor.dataOnBackend.Download<T>()` to get the data directly. This keeps the `tensor.dataOnDevice` on the given backend while providing a CPU copy. Be cautious with synchronization issues: if you re-run a worker, issue a new download request.

For an example, refer to the `Read output asynchronously` example in the [sample scripts](package-samples.md).

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Use output data](use-model-output.md)
- [Get output from any layer](profile-a-model.md#get-output-from-any-layer)
