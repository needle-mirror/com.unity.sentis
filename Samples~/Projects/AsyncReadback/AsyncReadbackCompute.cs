using Unity.Sentis;
using UnityEngine;
using System.Threading.Tasks;

public class AsyncReadbackCompute : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;

    Tensor m_Input;
    IWorker m_Engine;
    Task<TensorFloat> m_AsyncRead;

    void OnEnable()
    {
        // Everything that can be statically assigned is setup during Start to avoid memory churn.
        var model = ModelLoader.Load(modelAsset);
        m_Input = new TensorFloat(43.0f);
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
    }

    void OnDisable()
    {
        m_Input.Dispose();
        m_Engine.Dispose();
    }

    async Task<TensorFloat> ReadDataAsync(TensorFloat output)
    {
        // An async function is not strictly necessary, because PrepareCacheForAccess can be used in the
        // same way, however this async function wrapper is nice for keeping the local state encapsulated
        // in a single concept.
        while (!output.PrepareCacheForAccess(blocking: false))
        {
            await Task.Yield();
        }

        return output;
    }

    void Update()
    {
        if (m_AsyncRead != null && !m_AsyncRead.IsCompleted) return;

        if (m_AsyncRead != null)
        {
            Debug.Assert(m_AsyncRead.Result[0] == 42);
            Debug.Log($"Compute {m_AsyncRead.Result[0]}");
        }

        m_Engine.Execute(m_Input);

        // Peek the value from Sentis, without taking ownership of the Tensor (see PeekOutput docs for details).
        var output = m_Engine.PeekOutput() as TensorFloat;
        output.PrepareCacheForAccess(blocking: false);
        m_AsyncRead = ReadDataAsync(output);
    }
}
