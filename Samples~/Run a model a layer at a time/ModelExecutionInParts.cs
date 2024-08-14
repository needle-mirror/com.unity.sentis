using UnityEngine;
using Unity.Sentis;
using System;
using System.Collections;

public class ModelExecutionInParts : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;
    Worker m_Worker;
    Tensor m_Input;
    const int k_LayersPerFrame = 5;

    IEnumerator m_Schedule;
    bool m_Started = false;

    void OnEnable()
    {
        var model = ModelLoader.Load(modelAsset);
        m_Worker = new Worker(model, BackendType.GPUCompute);
        m_Input = new Tensor<float>(new TensorShape(1024));
    }

    void Update()
    {
        if (!m_Started)
        {
            // ScheduleIterable starts the scheduling of the model
            // it returns a IEnumerator to iterate over the model layers, scheduling each layer sequentially
            m_Schedule = m_Worker.ScheduleIterable(m_Input);
            m_Started = true;
        }

        int it = 0;
        while (m_Schedule.MoveNext())
        {
            if (++it % k_LayersPerFrame == 0)
                return;
        }

        var outputTensor = m_Worker.PeekOutput() as Tensor<float>;

        // If you wish to read from the tensor, download it to cpu.
        var cpuTensor = outputTensor.ReadbackAndClone();
        // See async examples for non-blocking readback.

        cpuTensor.Dispose();

        // To run the network again just set:
        m_Started = false;
    }

    void OnDisable()
    {
        // Clean up Sentis resources.
        m_Worker.Dispose();
        m_Input.Dispose();
    }
}
