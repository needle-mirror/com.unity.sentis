# Do operations on tensors

You can use the `Ops` object to do individual tensor operations on a back end. This can be easier to set up than writing a full model, and you can use it to dynamically handle input and output tensors at runtime.

If you have a fixed long list of tensor operations, you should execute them with a `Model` and `IWorker` so that Sentis can optimize performance.

## Using Ops

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

## Supported operations

Almost all of the [supported ONNX operators](supported-operators.md) have corresponding `Ops` methods. Sentis also provides convenience methods such as tensor arithmetic with scalar values.

In the following example a series of tensors are calculated by applying `Ops` methods.

``` 
using UnityEngine;
using Unity.Sentis;
using Unity.Sentis.Layers;

public class OpsMethods : MonoBehaviour
{
    void Start()
    {
        using ITensorAllocator allocator = new TensorCachingAllocator();
        using Ops ops = WorkerFactory.CreateOps(BackendType.GPUCompute, allocator);

        // Create a random uniform tensor of shape [1, 3, 32, 32]
        using TensorFloat randomTensor = ops.RandomUniform(new TensorShape(1, 3, 32, 32), 0, 1, 0.5f);

        // Multiply `randomTensor` by a scalar float
        using TensorFloat mulTensor = ops.Mul(randomTensor, 255f);

        // Resize `mulTensor` by a factor of 3/2
        using TensorFloat resizedTensor = ops.Resize(mulTensor, new[] { 1.5f, 1.5f }, InterpolationMode.Linear);

        // Tile `resizedTensor` on the x axis twice
        using TensorFloat tiledTensor = ops.Tile(resizedTensor, new[] { 1, 1, 1, 2 });

        // Pad `tiledTensor` with a constant value of 1 at the end of axis 1
        using TensorFloat paddedTensor = ops.Pad(tiledTensor, new[] { 0, 0, 0, 0, 0, 1, 0, 0 }, constant: 1f);
    }
}
```

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Create and modify tensors](do-basic-tensor-operations.md)
- [Model inputs](models-concept.md#model-inputs)
