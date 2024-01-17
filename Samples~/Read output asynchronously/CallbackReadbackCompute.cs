using Unity.Sentis;
using UnityEngine;
using System.Threading.Tasks;

public class CallbackReadbackCompute : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;

    Tensor m_Input;
    IWorker m_Engine;
    TensorFloat m_OutputTensor;

    void ReadbackCallback(bool completed)
    {
        // Put the downloaded tensor data into a readable tensor before indexing.
        m_OutputTensor.MakeReadable();
        Debug.Assert(m_OutputTensor[0] == 42);
        Debug.Log($"Output tensor value {m_OutputTensor[0]}");
    }

    void OnEnable()
    {
        // Everything that can be statically assigned is setup during Start to avoid memory churn.
        var model = ModelLoader.Load(modelAsset);
        m_Input = new TensorFloat(new TensorShape(1, 1), new[] { 43.0f });
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        m_Engine.Execute(m_Input);

        // Peek the value from Sentis, without taking ownership of the Tensor (see PeekOutput docs for details).
        m_OutputTensor = m_Engine.PeekOutput() as TensorFloat;
        m_OutputTensor.ReadbackRequest(ReadbackCallback);
        Debug.Log($"This code is called before the callback.");
    }

    void OnDisable()
    {
        m_Input.Dispose();
        m_Engine.Dispose();
    }
}
