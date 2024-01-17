using UnityEngine;
using Unity.Sentis;
using System;
using System.Collections;

public class ModelExecutionInParts : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;
    IWorker m_Engine;
    Tensor m_Input;
    const int k_LayersPerFrame = 5;

    IEnumerator m_Schedule;
    bool m_Started = false;

    void OnEnable()
    {
        var model = ModelLoader.Load(modelAsset);
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        m_Input = TensorFloat.Zeros(new TensorShape(1024));
    }

    void Update()
    {
        if (!m_Started)
        {
            // StartManualSchedule starts the scheduling of the model
            // it returns a IEnumerator to iterate over the model layers, scheduling each layer sequentially
            m_Schedule = m_Engine.StartManualSchedule(m_Input);
            m_Started = true;
        }

        int it = 0;
        while (m_Schedule.MoveNext())
        {
            if (++it % k_LayersPerFrame == 0)
                return;
        }

        var outputTensor = m_Engine.PeekOutput() as TensorFloat;
        outputTensor.MakeReadable();

        // Data is now ready to read.
        // See async examples for non-blocking readback.

        // To run the network again just set:
        // m_Started = false;
    }

    void OnDisable()
    {
        // Clean up Sentis resources.
        m_Engine.Dispose();
        m_Input.Dispose();
    }
}
