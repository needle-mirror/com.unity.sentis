using Unity.Sentis;
using UnityEngine;

public class PollingReadback : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;

    TensorFloat m_Input, m_Output;
    IWorker m_Engine;

    bool isRunning = false;

    void Start()
    {
        // Everything that can be statically assigned is setup during Start to avoid memory churn.
        var model = ModelLoader.Load(modelAsset);
        m_Input = new TensorFloat(new TensorShape(1, 1), new[] { 43.0f });
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
    }

    void Update()
    {
        if (!isRunning)
        {
            m_Engine.Execute(m_Input);
            // Peek the value from Sentis, without taking ownership of the Tensor (see PeekOutput docs for details).
            m_Output = m_Engine.PeekOutput() as TensorFloat;
            // start a readback request. tensor's internal data is scheduled for download once all execution has finished
            m_Output.ReadbackRequest();
            isRunning = true;
        }

        if (m_Output.IsReadbackRequestDone())
        {
            // computations are finished, can convert to cpu without hard download
            var result = m_Output.ReadbackAndClone();
            Debug.Assert(result[0] == 42);
            Debug.Log($"Output tensor value {result[0]}");
            result.Dispose();
            isRunning = false;
        }
    }

    void OnDisable()
    {
        m_Input.Dispose();
        // m_Output is still owned by the worker, no need to dispose of it
        m_Engine.Dispose();
    }
}
