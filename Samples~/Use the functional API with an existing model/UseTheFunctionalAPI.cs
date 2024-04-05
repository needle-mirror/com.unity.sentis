using UnityEngine;
using Unity.Sentis;

public class UseTheFunctionalAPI : MonoBehaviour
{
    [SerializeField]
    BackendType backendType = BackendType.GPUCompute;

    [SerializeField]
    Texture2D inputTexture;

    [SerializeField]
    ModelAsset sourceModelAsset;

    Model m_RuntimeModel;
    IWorker m_Worker;

    void OnEnable()
    {
        // Load the source model.
        // It expects input as sRGB values in the range[-1..1], however the input texture is an RGB texture in the range[0..1].
        var sourceModel = ModelLoader.Load(sourceModelAsset);

        // Compile a new model from the source model using the functional API to transform the inputs.
        m_RuntimeModel = Functional.Compile(
            // The forward method of the new model that transforms a functional input to a functional output.
            RGB =>
            {
                // Apply f(x) = x^(1/2.2) element-wise to transform from RGB to sRGB.
                var sRGB = Functional.Pow(RGB, Functional.Tensor(1 / 2.2f));
                // Apply f(x) = x * 2 - 1 element-wise to transform values from the range [0, 1] to the range [-1, 1].
                var sRGB_normalised = sRGB * 2 - 1;
                // Apply the forward method of the source model to the transformed functional input and return the output.
                return sourceModel.Forward(sRGB_normalised)[0];
            },
            // The inputDefs must be a single item or a tuple of items which give the types of the inputs.
            // Since the types are the same size and type (float) as last time we can get them from the previous model.
            sourceModel.inputs[0]
        );

        // Create worker to run the model.
        m_Worker = WorkerFactory.CreateWorker(backendType, m_RuntimeModel);

        // Create the input from our input texture.
        using var input = TextureConverter.ToTensor(inputTexture, 640, 64, 3);
        Debug.Log($"input shape = {input.shape}");

        // Execute the model.
        m_Worker.Execute(input);

        // Get the output and download onto the CPU to view it.
        // The outputs are automatically named as output_0, output_1, etc.
        var output = m_Worker.PeekOutput("output_0") as TensorFloat;
        output.CompleteOperationsAndDownload();

        Debug.Log($"output shape = {output.shape}");
        Debug.Log($"average values = {output[0, 0]}, {output[0, 1]}, {output[0, 2]}");
    }

    void OnDisable()
    {
        // Clean up Sentis resources
        m_Worker?.Dispose();
    }
}

