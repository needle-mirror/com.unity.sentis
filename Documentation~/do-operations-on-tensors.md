# Do operations on tensors

You can use the `IBackend` object to do individual tensor operations on a given back end. This can be easier to set up than writing a full model, and you can use it to dynamically handle input and output tensors at runtime.

## Using IBackend

To do operations on tensors, follow these steps:

1. Use [`BackendFactory.CreateBackend`](xref:Unity.Sentis.BackendFactory.CreateBackend(Unity.Sentis.BackendType)) to create an `IBackend` object.
2. Use a method of the `IBackend` object to do an operation on a tensor.

The following example uses the `ArgMax` operation. `ArgMax` returns the index of the maximum value in a tensor, for example the prediction with the highest probability in a classification model.

``` 
using UnityEngine;
using Unity.Sentis;

public class RunOperatorOnTensor : MonoBehaviour
{
    IBackend backend;

    void Start()
    {
        // Create an GPUComputeBackend object. The object uses Sentis compute shaders to do operations on the GPU.
        backend = WorkerFactory.CreateBackend(BackendType.GPUCompute);
    }
    
    void Update()
    {
        // Create a one-dimensional input tensor with four values.
        using var inputTensor = new TensorFloat(new TensorShape(4), new[] { 2.0f, 1.0f, 3.0f, 0.0f });
        using var outputTensor = TensorInt.AllocNoData(new TensorShape(1));
        // Run the ArgMax operator on the input tensor.
        backend.ArgMax(inputTensor, outputTensor, axis: 0, keepdim: true);

        // Make tensor readable before indexing.
        result.MakeReadable();

        // Log the first item of the result.
        Debug.Log(result[0]);
    }

    void OnDisable()
    {
        inputTensor.Dispose();
        backend.Dispose();
    }
}
```

Refer to the `Do an operation on a tensor` example in the [sample scripts](package-samples.md) for an example.

Refer to [Create an engine to run a model](create-an-engine.md) for more information about back end types.

## Supported operations

Almost all of the [supported ONNX operators](supported-operators.md) have corresponding `IBackend` methods. Sentis also provides convenience methods such as tensor arithmetic with scalar values.

In the following example a series of tensors are calculated by applying `IBackend` methods.

``` 
using UnityEngine;
using Unity.Sentis;
using Unity.Sentis.Layers;

public class BackendMethods : MonoBehaviour
{
    void Start()
    {
        using IBackend backend = new GPUComputeBackend();

        // Create a random uniform tensor of shape [1, 3, 32, 32]
        using var randomTensor = TensorFloat.AllocNoData(new TensorShape(1, 3, 32, 32));
        backend.RandomUniform(randomTensor, 0, 1, 0.5f);

        // Multiply `randomTensor` by a scalar float
        using var mulTensor = TensorFloat.AllocNoData(randomTensor.shape);
        backend.Mul(randomTensor, mulTensor, 255f);

        // Resize `mulTensor` by a factor of 3/2
        var scales = new[] { 1.5f, 1.5f };
        using var resizedTensor = TensorFloat.AllocNoData(ShapeInference.Resize(mulTensor.shape, scales));
        backend.Resize(mulTensor, resizedTensor, scales, InterpolationMode.Linear);

        // Tile `resizedTensor` on the x axis twice
        var repeats = new[] { 1, 1, 1, 2 };
        using var tiledTensor = TensorFloat.AllocNoData(mulTensor.shape.Tile(repeats));
        backend.Tile(resizedTensor, tiledTensor, t);

        // Pad `tiledTensor` with a constant value of 1 at the end of axis 1
        var pads = new[] { 0, 0, 0, 0, 0, 1, 0, 0 };
        using var paddedTensor = TensorFloat.AllocNoData(tiledTensor.shape.Pad(pads));
        backend.Pad(tiledTensor, paddedTensor, pads, constant: 1f);
    }
}
```

## Additional resources

- [Tensor fundamentals](tensor-fundamentals.md)
- [Create and modify tensors](do-basic-tensor-operations.md)
- [Model inputs](models-concept.md#model-inputs)
