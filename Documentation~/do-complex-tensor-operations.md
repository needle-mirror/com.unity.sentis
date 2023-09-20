## Do operations on tensors

To do operations on tensors, follow these steps:

1. Create an `ITensorAllocator` allocator, which manages a pool of allocated tensors.
2. Use [`WorkerFactory.CreateOps`](xref:Unity.Sentis.WorkerFactory.CreateOps(Unity.Sentis.BackendType,Unity.Sentis.ITensorAllocator)) to create an `Ops` object with the allocator.
3. Use a method of the `Ops` object to do an operation on a tensor.

The following example uses the `ArgMax` operation. `ArgMax` returns the index of the maximum value in a tensor, for example the prediction with the highest probability in a classification model.

``` 
using UnityEngine;
using Unity.Sentis;

public class RunOperatorOnTensor : MonoBehaviour
{
    TensorFloat inputTensor;
    ITensorAllocator allocator;
    Ops ops;

    void Start()
    {
        // Create a one-dimensional input tensor with four values.
        inputTensor = new TensorFloat(new TensorShape(4), new[] { 2.0f, 1.0f, 3.0f, 0.0f });

        // Create an allocator.
        allocator = new TensorCachingAllocator();

        // Create an Ops object. The object uses Sentis compute shaders to do operations on the GPU.
        ops = WorkerFactory.CreateOps(BackendType.GPUCompute, allocator);

        // Run the ArgMax operator on the input tensor.
        TensorInt result = ops.ArgMax(inputTensor, axis: 0, keepdim: true);

        // Make tensor readable before indexing.
        result.MakeReadable();

        // Log the first item of the result.
        Debug.Log(result[0]);
    }

    void OnDisable()
    {
        // Tell the GPU we're finished with the memory the input tensor, allocator and Ops object used.
        inputTensor.Dispose();
        allocator.Dispose();
        ops.Dispose();
    }
}
```

Refer to the `Do an operation on a tensor` example in the [sample scripts](package-samples.md) for an example.

Refer to [Create an engine to run a model](create-an-engine.md) for more information about back end types.

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Create and modify tensors](do-basic-tensor-operations.md)
- [Model inputs](models-concept.md#model-inputs)
