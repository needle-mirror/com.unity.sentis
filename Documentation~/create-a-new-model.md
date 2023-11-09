# Create a new model

You can create a new runtime model without an ONNX file using Sentis. For example if you want to do a series of tensor operations without weights, or if you want to build your own model serialization from another model format.

## Instantiate the model object

1. Create an instance of `Model` using the default constructor.
2. Add inputs to the model.
3. Add constants to the model.
4. Add layers to the model. Sentis executes layers in the order you add them.
5. Add outputs to the model.

In this example a simple model is created to calculate the dot product of two vectors.

```
using Unity.Sentis;
using Unity.Sentis.Layers;

public class CreateNewModel : MonoBehaviour
{
    Model runtimeModel;

    void Start()
    {
        // Instantiate model
        runtimeModel = new Model();

        // Add two vector float inputs of the same length
        runtimeModel.AddInput("a", DataType.Float, new SymbolicTensorShape(new SymbolicTensorDim('d')));
        runtimeModel.AddInput("b", DataType.Float, new SymbolicTensorShape(new SymbolicTensorDim('d')));

        // Add constant int tensor for the axes of the reduction
        runtimeModel.AddConstant(new Constant("reduce_axes", new TensorInt(new TensorShape(1), new[] { 0 })));

        // Add the required layers in order for the dot product
        runtimeModel.AddLayer(new Mul("mul", "a", "b"));
        runtimeModel.AddLayer(new ReduceSum("output", new[] { "mul", "reduce_axes" }, keepdims: false));

        // Add the output of the ReduceSum as an output of the model
        runtimeModel.AddOutput("output");
    }
}
```

You can then [create an engine to run a model](create-an-engine.md).

Sentis can't run [model optimization](models-concept.md#how-sentis-optimizes-a-model) on models you create using the `Model` API.
