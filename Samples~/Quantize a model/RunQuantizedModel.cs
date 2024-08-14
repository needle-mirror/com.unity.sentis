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
    Worker m_Worker;
    Tensor m_Input;

    const int maxTokens = 100;

    void OnEnable()
    {
        // Load the quantized model as any other Sentis model.
        var model = ModelLoader.Load(modelAsset);
        m_Worker = new Worker(model, BackendType.GPUCompute);

        // Initialize input for tiny stories, see https://huggingface.co/unity/sentis-tiny-stories/tree/main for full tinystories example.
        m_Input = new Tensor<int>(new TensorShape(1, maxTokens));
    }

    void Update()
    {
        // Execute worker and peek output as with any other Sentis model.
        m_Worker.Schedule(m_Input);
        var output = m_Worker.PeekOutput() as Tensor<float>;
    }

    void OnDisable()
    {
        // Clean up Sentis resources.
        m_Worker.Dispose();
        m_Input.Dispose();
    }
}
