using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;

public class ModelExecutionMultipleInputsOutputs : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;
    Worker m_Worker;
    Tensor[] m_Inputs;

    void OnEnable()
    {
        var model = ModelLoader.Load(modelAsset);
        m_Worker = new Worker(model, BackendType.GPUCompute);

        // The MultipleInputMultipleOuput model takes two inputs, "input0" and "input1"
        // since it has multiple inputs, we need to use a dictionary tensorName -> tensor
        m_Inputs = new Tensor[]
        {
            new Tensor<float>(new TensorShape(1024)),
            new Tensor<float>(new TensorShape(1))
        };
    }

    void Update()
    {
        m_Worker.Schedule(m_Inputs);

        // model has multiple output, so to know which output to get we need to specify which one we are referring to
        var outputTensor0 = m_Worker.PeekOutput("output0") as Tensor<float>;
        var outputTensor1 = m_Worker.PeekOutput("output1") as Tensor<float>;

        // If you wish to read from the tensor, download it to cpu.
        // See async examples for non-blocking readback.
        var cpuTensor0 = outputTensor0.ReadbackAndClone();
        var cpuTensor1 = outputTensor1.ReadbackAndClone();
    }

    void OnDisable()
    {
        // Clean up Sentis resources.
        m_Worker.Dispose();
        foreach (var tensor in m_Inputs) { tensor.Dispose(); }
    }
}
