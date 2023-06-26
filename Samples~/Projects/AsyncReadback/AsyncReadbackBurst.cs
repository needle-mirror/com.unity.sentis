using Unity.Sentis;
using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using System.Collections.Generic;
using System.Threading.Tasks;

public class AsyncReadbackBurst : MonoBehaviour
{
    [SerializeField] ModelAsset m_ModelAsset;

    Tensor m_Input;
    IWorker m_Engine;
    Task<TensorFloat> m_AsyncRead = null;

    void OnEnable()
    {
        // Everything that can be statically assigned is setup during Start to avoid memory churn.
        var model = ModelLoader.Load(m_ModelAsset);
        m_Input = new TensorFloat(43.0f);
        m_Engine = WorkerFactory.CreateWorker(BackendType.CPU, model);
    }

    void OnDisable()
    {
        m_Input.Dispose();
        m_Engine.Dispose();
    }

    async Task<TensorFloat> ReadDataAsync(TensorFloat output)
    {
        // An async function is not strictly necessary, because the burstData.fence.IsCompleted property
        // can be used in the same way, however this async function wrapper is nice for keeping the local
        // state encapsulated in a single concept.
        var burstData = BurstTensorData.Pin(output, uploadCache: false);
        burstData.ScheduleAsyncDownload();
        while (!burstData.fence.IsCompleted)
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
            // Use -1 index to read the last value in the tensor.
            Debug.Assert(m_AsyncRead.Result[0] == 42);
            Debug.Log($"Burst {m_AsyncRead.Result[0]}");
        }

        m_Engine.Execute(m_Input);

        // Peek the value from Sentis, without taking ownership of the Tensor (see PeekOutput docs for details).
        var outputTensor = m_Engine.PeekOutput() as TensorFloat;
        m_AsyncRead = ReadDataAsync(outputTensor);
    }
}

