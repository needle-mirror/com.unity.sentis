using UnityEngine;
using Unity.Sentis;

public class RunQuantizedModel : MonoBehaviour
{
    // 1. Download a .sentis or .onnx model you want to quantize, e.g. tinystories from https://huggingface.co/unity/sentis-tiny-stories/tree/main and bring into your Unity Project.
    // 2. Open the Editor window 'Sentis > Sample > Quantize and Save Model' and reference your model as the source model.
    // 3. Select the desired quantization type and click 'Quantize and Save'.

    // Reference your quantized tiny stories here in the RunQuantizedModel scene.
    [SerializeField]
    ModelAsset modelAsset;
    IWorker m_Engine;
    Tensor m_Input;

    const int maxTokens = 100;

    void OnEnable()
    {
        // Load the quantized model as any other Sentis model.
        var model = ModelLoader.Load(modelAsset);
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);

        // Initialize input for tiny stories, see https://huggingface.co/unity/sentis-tiny-stories/tree/main for full tinystories example.
        m_Input = TensorInt.AllocZeros(new TensorShape(1, maxTokens));
    }

    void Update()
    {
        // Execute worker and peek output as with any other Sentis model.
        m_Engine.Execute(m_Input);
        var output = m_Engine.PeekOutput() as TensorFloat;
    }

    void OnDisable()
    {
        // Clean up Sentis resources.
        m_Engine.Dispose();
        m_Input.Dispose();
    }
}
