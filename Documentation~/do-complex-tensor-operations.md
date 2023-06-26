## Do operations on tensors

To do operations on tensors, follow these steps:

1. Create an `ITensorAllocator` allocator, which manages a pool of allocated tensors.
3. Use [`WorkerFactory.CreateOps`](xref:Unity.Sentis.WorkerFactory.CreateOps(Unity.Sentis.BackendType,Unity.Sentis.ITensorAllocator)) to create an `IOps` object with the allocator.
4. Use a method of the `IOps` object to do an operation on a tensor.

The following example uses the `ArgMax` operation. `ArgMax` returns the index of the maximum value in a tensor, for example the prediction with the highest probability in a classification model.

``` 
using UnityEngine;
using Unity.Sentis;

public class RunOperatorOnTensor : MonoBehaviour
{
    TensorFloat inputTensor;
    ITensorAllocator allocator;
    IOps ops;

    void Start()
    {
        // Create a one-dimensional input tensor with four values.
        inputTensor = new TensorFloat(new TensorShape(4), new[] { 2.0f, 1.0f, 3.0f, 0.0f });

        // Create an allocator.
        allocator = new TensorCachingAllocator();

        // Create an IOps object. The object uses Sentis compute shaders to do operations on the GPU.
        IOps ops = WorkerFactory.CreateOps(BackendType.GPUCompute, allocator);

        // Run the ArgMax operator on the input tensor.
        TensorInt result = ops.ArgMax(inputTensor, axis: 0, keepdim: true);

        // Log the first item of the result.
        Debug.Log(result[0]);
    }

    void OnDisable()
    {
        // Tell the GPU we're finished with the memory the input tensor, allocator and IOps object used.
        inputTensor.Dispose();
        allocator.Dispose();
        ops.Dispose();
    }
}
```

Refer to the `ExecuteOperatorOnTensor` example in the [sample scripts](package-samples.md) for a working example.

Refer to [Create an engine to run a model](create-an-engine.md) for more information about back end types.

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Create and modify tensors](do-basic-tensor-operations.md)
- [Model inputs](models-concept.md#model-inputs)