# Create a new model

You can create a new runtime model without an ONNX file using Sentis. For example if you want to do a series of tensor operations without weights, or if you want to build your own model serialization from another model format.

## Using the functional API

1. Define the forward method of the model which returns the outputs as `FunctionalTensor` from the inputs.
2. Define an `InputDef` for each input to the model with the data type and shape.
3. Compile the model using the `Compile` method.

In this example a simple model is created to calculate the dot product of two vectors.

```
using System;
using Unity.Sentis;
using UnityEngine;

public class CreateNewModel : MonoBehaviour
{
    Model model;

    void Start()
    {
        // Define the forward method of the model.
        // In this example a single `FunctionalTensor` output is calculated from two `FunctionalTensor` inputs.
        FunctionalTensor DotProduct(FunctionalTensor x, FunctionalTensor y)
        {
            // Calculate the elementwise product of the input `FunctionalTensor`s using an operator.
            FunctionalTensor prod = x * y;

            // Sum the product along the first axis flattening the summed dimension.
            return Functional.ReduceSum(prod, dim: 0, keepdim: false);
        }

        // Set up the input data types and shapes for the model as `InputDef`s.
        // Our dot product operates on two vector tensors of the same size `6`.
        (InputDef, InputDef) inputDefs =
        (
            new InputDef(DataType.Float, new TensorShape(6)),
            new InputDef(DataType.Float, new TensorShape(6))
        );

        // Build the model using the `Compile` method.
        model = Functional.Compile(DotProduct, inputDefs);
    }
}
```

You can then [create an engine to run a model](create-an-engine.md).

## Additional resources

- [Supported functional methods](supported-functional-methods.md)
