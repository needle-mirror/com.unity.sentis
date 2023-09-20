using System;
using Unity.Sentis;
using UnityEngine;

public class TestBinarizer : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;
    [SerializeField]
    BackendType backendType;

    // These parameters come from the binarizer model.
    TensorShape m_InputShape = new TensorShape(16, 16);
    const float k_Threshold = 0.5f;

    void Start()
    {
        // Load model for inference.
        var model = ModelLoader.Load(modelAsset);

        // Setup input tensor and expected output tensor.
        using var ops = new CPUOps();
        using var input = ops.RandomUniform(m_InputShape, low: 0, high: 1, seed: 0);

        // Setup engine of given worker type and model.
        using var engine = WorkerFactory.CreateWorker(backendType, model);
        engine.SetInput(input);
        engine.Execute();

        // Get output and cast to TensorFloat.
        var output = engine.PeekOutput() as TensorFloat;

        // Check the output has the correct data type and shape.
        Debug.Assert(output != null && output.shape == input.shape);

        // Check each value in the output against the input. This pattern of iterating through tensors is to be
        // avoided in runtime code for performance reasons. Use burst and compute optimized code wherever possible.
        for (var i = 0; i < m_InputShape[0]; i++)
        {
            for (var j = 0; j < m_InputShape[1]; j++)
            {
                Debug.Assert(Math.Abs(output[i, j] - (input[i, j] > k_Threshold ? 1f : 0f)) < 1e-5);
            }
        }

        Debug.Log("Success!");
    }
}
