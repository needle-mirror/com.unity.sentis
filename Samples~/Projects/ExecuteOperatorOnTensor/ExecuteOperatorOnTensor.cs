using UnityEngine;
using Unity.Sentis;

public class ExecuteOperatorOnTensor : MonoBehaviour
{
    static Ops s_Ops;
    ITensorAllocator m_Allocator;
    TensorFloat m_InputTensor = new TensorFloat(new TensorShape(4), new[] { 2.0f, 1.0f, 3.0f, 0.0f });
    [SerializeField]
    BackendType backendType;

    void Start()
    {
        m_Allocator = new TensorCachingAllocator();

        // CreateOps allows direct operations on tensors.
        s_Ops = WorkerFactory.CreateOps(backendType, m_Allocator);

        // Execute an operator on the input tensor.
        var outputTensor = s_Ops.ArgMax(m_InputTensor, axis: 0, keepdim: true);
        // Ensure tensor is on CPU before indexing.
        outputTensor.MakeReadable();
        Debug.Log(outputTensor[0]);
    }

    void OnDisable()
    {
        // Clean up Sentis resources.
        s_Ops.Dispose();
        m_Allocator.Dispose();

        // Input tensor needs to be disposed manually, since it is not owned by an operator or allocator
        m_InputTensor.Dispose();
    }
}
