using UnityEngine;
using Unity.Sentis;

public class ModelExecution : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;
    Worker m_Worker;
    Tensor m_Input;

    void OnEnable()
    {
        var model = ModelLoader.Load(modelAsset);
        m_Worker = new Worker(model, BackendType.GPUCompute);

        // The SingleInputSingleOutput model takes one input and runs a Relu activation
        m_Input = new Tensor<float>(new TensorShape(1024));
    }

    void Update()
    {
        // model has a single input, so no ambiguity due to its name
        m_Worker.Schedule(m_Input);

        // model has a single output, so no ambiguity due to its name
        var outputTensor = m_Worker.PeekOutput() as Tensor<float>;

        // If you wish to read from the tensor, download it to cpu.
        var cpuTensor = outputTensor.ReadbackAndClone();
        // See async examples for non-blocking readback.
        cpuTensor.Dispose();
    }

    void OnDisable()
    {
        // Clean up Sentis resources.
        m_Worker.Dispose();
        m_Input.Dispose();
    }
}
