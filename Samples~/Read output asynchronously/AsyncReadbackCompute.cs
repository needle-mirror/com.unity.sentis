using Unity.Sentis;
using UnityEngine;

public class AsyncReadbackCompute : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;

    Tensor m_Input;
    IWorker m_Engine;

    async void OnEnable()
    {
        var model = ModelLoader.Load(modelAsset);
        m_Input = new TensorFloat(new TensorShape(1, 1), new[] { 43.0f });
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);

        m_Engine.Execute(m_Input);
        var outputTensor = m_Engine.PeekOutput() as TensorFloat;

        await outputTensor.ReadbackRequestAsync();

        // Put the downloaded tensor data into a readable tensor before indexing.
        outputTensor.MakeReadable();
        Debug.Assert(outputTensor[0] == 42);
        Debug.Log($"Output tensor value {outputTensor[0]}");
    }

    void OnDisable()
    {
        m_Input.Dispose();
        m_Engine.Dispose();
    }
}
