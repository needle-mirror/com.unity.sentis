using UnityEngine;
using Unity.Sentis;

public class ModelExecution : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;
    IWorker m_Engine;
    Tensor m_Input;

    void OnEnable()
    {
        var model = ModelLoader.Load(modelAsset);
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);

        // The SingleInputSingleOutput model takes one input and runs a Relu activation
        m_Input = TensorFloat.AllocZeros(new TensorShape(1024));
    }

    void Update()
    {
        // model has a single input, so no ambiguity due to its name
        m_Engine.Execute(m_Input);

        // model has a single output, so no ambiguity due to its name
        var outputTensor = m_Engine.PeekOutput() as TensorFloat;

        // If you wish to read from the tensor, download it to cpu.
        var cpuTensor = outputTensor.ReadbackAndClone();
        // See async examples for non-blocking readback.
        cpuTensor.Dispose();
    }

    void OnDisable()
    {
        // Clean up Sentis resources.
        m_Engine.Dispose();
        m_Input.Dispose();
    }
}
