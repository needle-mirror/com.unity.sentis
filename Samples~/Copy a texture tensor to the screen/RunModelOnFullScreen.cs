using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;

public class RunModelOnFullScreen : MonoBehaviour
{
    Model m_Model;
    Worker m_Worker;
    Tensor<float> m_TensorY;
    Tensor[] m_Inputs;

    void OnEnable()
    {
        Debug.Log("When running this example, the game vew should fade from black to white to black");
        var graph = new FunctionalGraph();
        var x = graph.AddInput<float>(new DynamicTensorShape(1, 4, -1, -1));
        var y = graph.AddInput<float>(new TensorShape());
        m_Model = graph.Compile(x + y);
        m_Worker = new Worker(m_Model, BackendType.GPUCompute);

        // The value of Y will be added to all pixel values.
        m_TensorY = new Tensor<float>(new TensorShape(), new[] { 0.1f });

        m_Inputs = new Tensor[] { null, m_TensorY };
    }

    void OnDisable()
    {
        foreach (var tensor in m_Inputs) { tensor?.Dispose(); }

        m_Worker.Dispose();
    }

    void Update()
    {
        // readback tensor from GPU to CPU to use tensor indexing
        CPUTensorData.Pin(m_TensorY);
        m_TensorY[0] = Mathf.Sin(Time.timeSinceLevelLoad);
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        // TextureToTensor takes optional parameters for resampling, channel swizzles etc
        using Tensor<float> frameInputTensor = TextureConverter.ToTensor(source);

        m_Inputs[0] = frameInputTensor;

        m_Worker.Schedule(m_Inputs);
        Tensor<float> output = m_Worker.PeekOutput() as Tensor<float>;

        // TensorToTextureUtils also has methods for rendering to a render texture with optional parameters
        TextureConverter.RenderToScreen(output);
    }
}
