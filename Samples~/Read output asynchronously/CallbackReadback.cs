using Unity.Sentis;
using UnityEngine;

public class CallbackReadback : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;

    TensorFloat m_Input, m_Output;
    IWorker m_Engine;

    bool isRunning = false;

    public void DownloadAction()
    {
        Debug.Assert(m_Output[0] == 42);
        Debug.Log($"Output tensor value {m_Output[0]}");
        isRunning = false;
    }

    void OnEnable()
    {
        if (isRunning)
            return;

        // Everything that can be statically assigned is setup during Start to avoid memory churn.
        var model = ModelLoader.Load(modelAsset);
        m_Input = new TensorFloat(new TensorShape(1, 1), new[] { 43.0f });
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);

        m_Engine.Execute(m_Input);
        // Peek the value from Sentis, without taking ownership of the Tensor (see PeekOutput docs for details).
        var result = m_Engine.PeekOutput() as TensorFloat;

        #if UNITY_2023_2_OR_NEWER
        // awaitable.awaiter calls callback when completed
        var awaiter = result.ReadbackAndCloneAsync().GetAwaiter();
        awaiter.OnCompleted(() =>
        {
            m_Output = awaiter.GetResult() as TensorFloat;
            DownloadAction();
        });
        #else
        m_Output = result.ReadbackAndClone();
        DownloadAction();
        #endif

        Debug.Log($"This code is called before the callback.");
    }

    void OnDisable()
    {
        m_Input.Dispose();
        m_Output.Dispose();
        m_Engine.Dispose();
    }
}
