using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;

public class RunModelOnFullScreen : MonoBehaviour
{
    Model m_Model;
    IWorker m_Engine;
    TensorFloat m_TensorY;
    Dictionary<string, Tensor> m_Inputs;

    void OnEnable()
    {
        Debug.Log("When running this example, the game vew should fade from black to white to black");
        m_Model = Functional.Compile(
            forward: (x, y) => x + y,
            inputDefs: (
                new InputDef(DataType.Float, new SymbolicTensorShape(SymbolicTensorDim.Int(1), SymbolicTensorDim.Int(4), SymbolicTensorDim.Param(0), SymbolicTensorDim.Param(1))),
                new InputDef(DataType.Float, new SymbolicTensorShape())
            )
        );
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, m_Model);

        // The value of Y will be added to all pixel values.
        m_TensorY = new TensorFloat(0.1f);

        m_Inputs = new Dictionary<string, Tensor>
        {
            { m_Model.inputs[0].name, null },
            { m_Model.inputs[1].name, m_TensorY },
        };
    }

    void OnDisable()
    {
        foreach (var tensor in m_Inputs.Values) { tensor?.Dispose(); }

        m_Engine.Dispose();
    }

    void Update()
    {
        // readback tensor from GPU to CPU to use tensor indexing
        m_TensorY.CompleteOperationsAndDownload();
        m_TensorY[0] = Mathf.Sin(Time.timeSinceLevelLoad);
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        // TextureToTensor takes optional parameters for resampling, channel swizzles etc
        using TensorFloat frameInputTensor = TextureConverter.ToTensor(source);

        m_Inputs[m_Model.inputs[0].name] = frameInputTensor;

        m_Engine.Execute(m_Inputs);
        TensorFloat output = m_Engine.PeekOutput() as TensorFloat;

        // TensorToTextureUtils also has methods for rendering to a render texture with optional parameters
        TextureConverter.RenderToScreen(output);
    }
}
