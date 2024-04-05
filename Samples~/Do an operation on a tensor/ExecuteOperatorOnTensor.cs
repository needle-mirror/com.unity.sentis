using UnityEngine;
using Unity.Sentis;

public class ExecuteOperatorOnTensor : MonoBehaviour
{
    IBackend m_Backend;
    TensorFloat m_InputTensor = new TensorFloat(new TensorShape(4), new[] { 2.0f, 1.0f, 3.0f, 0.0f });
    [SerializeField]
    BackendType backendType;

    void Start()
    {
        // CreateOps allows direct operations on tensors.
        m_Backend = WorkerFactory.CreateBackend(backendType);

        // Execute an operator on the input tensor.
        TensorInt outputTensor = TensorInt.AllocNoData(new TensorShape(1));
        m_Backend.ArgMax(m_InputTensor, outputTensor, axis: 0, keepdim: true, selectLastIndex: false);

        // Ensure tensor is on CPU before indexing.
        outputTensor.CompleteOperationsAndDownload();
        Debug.Log(outputTensor[0]);

        outputTensor.Dispose();
    }

    void OnDisable()
    {
        // Clean up Sentis resources.
        m_Backend.Dispose();

        // Input tensor needs to be disposed manually, since it is not owned by an operator or allocator
        m_InputTensor.Dispose();
    }
}
