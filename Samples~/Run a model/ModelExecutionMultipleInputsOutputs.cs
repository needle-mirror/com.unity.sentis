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
        var outputTensor0 = m_Engine.PeekOutput("output0");
        var outputTensor1 = m_Engine.PeekOutput("output1");
        outputTensor0.CompleteOperationsAndDownload();
        outputTensor1.CompleteOperationsAndDownload();

        // Data is now ready to read.
        // See async examples for non-blocking readback.
    }

    void OnDisable()
    {
        // Clean up Sentis resources.
        m_Engine.Dispose();
        foreach (var tensor in m_Inputs.Values) { tensor.Dispose(); }
    }
}
