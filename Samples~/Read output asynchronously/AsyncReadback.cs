using Unity.Sentis;
using UnityEngine;

#if UNITY_2023_2_OR_NEWER
public class AsyncReadback : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;

    Tensor<float> m_Input, m_Output;
    Worker m_Worker;

    async void OnEnable()
    {
        // Everything that can be statically assigned is setup during Start to avoid memory churn.
        var model = ModelLoader.Load(modelAsset);
        m_Input = new Tensor<float>(new TensorShape(1, 1), new[] { 43.0f });
        m_Worker = new Worker(model, BackendType.GPUCompute);

        m_Worker.Schedule(m_Input);
        // Peek the value from Sentis, without taking ownership of the Tensor (see PeekOutput docs for details).
        var outputTensor = m_Worker.PeekOutput() as Tensor<float>;

        m_Output = await outputTensor.ReadbackAndCloneAsync();
        // m_Output is assigned as a cpu copy once download is finished

        Debug.Assert(m_Output[0] == 42);
        Debug.Log($"Output tensor value {m_Output[0]}");
    }

    void OnDisable()
    {
        m_Input.Dispose();
        m_Output.Dispose();
        m_Worker.Dispose();
    }
}
#endif
