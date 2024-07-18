using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;

public class ModelExecutionMultipleInputsOutputs : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;
    IWorker m_Engine;
    Dictionary<string, Tensor> m_Inputs;

    void OnEnable()
    {
        var model = ModelLoader.Load(modelAsset);
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);

        // The MultipleInputMultipleOuput model takes two inputs, "input0" and "input1"
        // since it has multiple inputs, we need to use a dictionary tensorName -> tensor
        m_Inputs = new Dictionary<string, Tensor>
        {
            { "input0", TensorFloat.AllocZeros(new TensorShape(1024)) },
            { "input1", TensorFloat.AllocZeros(new TensorShape(1)) },
        };
    }

    void Update()
    {
        m_Engine.Execute(m_Inputs);

        // model has multiple output, so to know which output to get we need to specify which one we are referring to
        var outputTensor0 = m_Engine.PeekOutput("output0") as TensorFloat;
        var outputTensor1 = m_Engine.PeekOutput("output1") as TensorFloat;

        // If you wish to read from the tensor, download it to cpu.
        // See async examples for non-blocking readback.
        var cpuTensor0 = outputTensor0.ReadbackAndClone();
        var cpuTensor1 = outputTensor1.ReadbackAndClone();
    }

    void OnDisable()
    {
        // Clean up Sentis resources.
        m_Engine.Dispose();
        foreach (var tensor in m_Inputs.Values) { tensor.Dispose(); }
    }
}
