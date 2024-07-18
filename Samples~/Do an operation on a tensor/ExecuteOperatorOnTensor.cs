using UnityEngine;
using Unity.Sentis;

public class ExecuteOperatorOnTensor : MonoBehaviour
{
    [SerializeField]
    BackendType backendType;

    IBackend m_Backend;
    TensorFloat m_InputTensor;

    void OnEnable()
    {
        m_InputTensor = new TensorFloat(new TensorShape(4), new[] { 2.0f, 1.0f, 3.0f, 0.0f });
        // CreateOps allows direct operations on tensors.
        m_Backend = WorkerFactory.CreateBackend(backendType);
    }

    void Update()
    {
        // Execute an operator on the input tensor.
        TensorInt outputTensor = TensorInt.AllocNoData(new TensorShape(1));
        m_Backend.ArgMax(m_InputTensor, outputTensor, axis: 0, selectLastIndex: false);

        // Ensure tensor is on CPU before indexing.
        var cpuTensor = outputTensor.ReadbackAndClone();
        Debug.Log(cpuTensor[0]);

        // Temporary tensor allocations needs to be disposed
        outputTensor.Dispose();
        cpuTensor.Dispose();
    }

    void OnDisable()
    {
        // Input tensor needs to be disposed manually, since it is not owned by an operator or allocator
        m_InputTensor.Dispose();
        // Clean up Sentis resources.
        m_Backend.Dispose();
    }
}
