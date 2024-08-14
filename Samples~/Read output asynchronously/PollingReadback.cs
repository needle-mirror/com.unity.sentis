using Unity.Sentis;
using UnityEngine;

public class PollingReadback : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;

    Tensor<float> m_Input, m_Output;
    Worker m_Worker;

    bool isRunning = false;

    void Start()
    {
        // Everything that can be statically assigned is setup during Start to avoid memory churn.
        var model = ModelLoader.Load(modelAsset);
        m_Input = new Tensor<float>(new TensorShape(1, 1), new[] { 43.0f });
        m_Worker = new Worker(model, BackendType.GPUCompute);
    }

    void Update()
    {
        if (!isRunning)
        {
            m_Worker.Schedule(m_Input);
            // Peek the value from Sentis, without taking ownership of the Tensor (see PeekOutput docs for details).
            m_Output = m_Worker.PeekOutput() as Tensor<float>;
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
        m_Worker.Dispose();
    }
}
