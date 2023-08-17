using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using Unity.Sentis.Layers;
using UnityEngine.Serialization;

public class RunModelOnFullScreen : MonoBehaviour
{
    IWorker m_Engine;
    GPUComputeOps m_Ops = new GPUComputeOps();
    TensorFloat m_TensorY;
    Dictionary<string, Tensor> m_Inputs;

    void OnEnable()
    {
        Debug.Log("When running this example, the game vew should fade from black to white to black");
        var model = new Model();
        model.AddInput("x", DataType.Float, new SymbolicTensorShape(new SymbolicTensorDim(1), new SymbolicTensorDim(4), new SymbolicTensorDim('w'), new SymbolicTensorDim('h')));
        model.AddInput("y", DataType.Float, new SymbolicTensorShape());
        model.AddLayer(new Add("z", "x", "y"));
        model.AddOutput("z");
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);

        // The value of Y will be added to all pixel values.
        m_TensorY = new TensorFloat(0.1f);

        m_Inputs = new Dictionary<string, Tensor>
        {
            { "x", null },
            { "y", m_TensorY },
        };
    }

    void OnDisable()
    {
        foreach (var tensor in m_Inputs.Values) { tensor?.Dispose(); }

        m_Engine.Dispose();
        m_Ops.Dispose();
    }

    void Update()
    {
        // readback tensor from GPU to CPU to use tensor indexing
        m_TensorY.MakeReadable();
        m_TensorY[0] = Mathf.Sin(Time.timeSinceLevelLoad);
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        // TextureToTensor takes optional parameters for resampling, channel swizzles etc
        using TensorFloat frameInputTensor = TextureConverter.ToTensor(source);

        m_Inputs["x"] = frameInputTensor;

        m_Engine.Execute(m_Inputs);
        TensorFloat output = m_Engine.PeekOutput() as TensorFloat;

        // TensorToTextureUtils also has methods for rendering to a render texture with optional parameters
        TextureConverter.RenderToScreen(output);
    }
}
