# Create a new model

You can create a new runtime model without an ONNX file using Sentis. For example if you want to do a series of tensor operations without weights, or if you want to build your own model serialization from another model format.

## Using the functional API

1. Create a [`FunctionalGraph`](xref:Unity.Sentis.FunctionalGraph) object.
2. Add the inputs to the graph with their data type and shape getting the respective [`FunctionalTensor`](xref:Unity.Sentis.FunctionalTensor) for each input.
3. Apply a series of functional API methods to the functional tensors objects to create the desired output functional tensors.
4. Compile the model using the [`Compile`](xref:Unity.Sentis.FunctionalGraph.Compile*) method with the output functional tensors.

The following example shows the creation of a simple model to calculate the dot product of two vectors.

```
using System;
using Unity.Sentis;
using UnityEngine;

public class CreateNewModel : MonoBehaviour
{
    Model model;

    void Start()
    {
        // Create the functional graph.
        FunctionalGraph graph = new FunctionalGraph();

        // Add two inputs to the graph with data types and shapes.
        // Our dot product operates on two vector tensors of the same size `6`.
        FunctionalTensor x = graph.AddInput<float>(new TensorShape(6));
        FunctionalTensor y = graph.AddInput<float>(new TensorShape(6));

        // Calculate the elementwise product of the input `FunctionalTensor`s using an operator.
        FunctionalTensor prod = x * y;

        // Sum the product along the first axis flattening the summed dimension.
        FunctionalTensor reduce = Functional.ReduceSum(prod, dim: 0, keepdim: false);

        // Build the model using the `Compile` method.
        model = graph.Compile(reduce, prod);

        // The final model has two inputs `x` and `y`, which are denoted as `input_0` and `input_1` respectively
        // and two outputs `reduce` and `prod`, which are denoted as `output_0` and `output_1` respectively.
    }
}
```

To debug your code, use [`ToString`](xref:Unity.Sentis.FunctionalTensor.ToString) to retrieve information, such as its shape and datatype.

You can then [create an engine to run a model](create-an-engine.md).

### Model inputs and outputs

[`Compile`](xref:Unity.Sentis.FunctionalGraph.Compile*) returns a model with the number of inputs created with [`AddInput`](xref:Unity.Sentis.FunctionalGraph.AddInput*). In our example `x` is the first input and `y` the second.
The inputs have the names `input_0`, `input_1` ... in the compiled model.

[`Compile`](xref:Unity.Sentis.FunctionalGraph.Compile*) returns a model with the number of outputs that are provided to the [`Compile`](xref:Unity.Sentis.FunctionalGraph.Compile*) method. In our example `reduce` is the first output and `prod` the second.
The outputs have the names `output_0`, `output_1` ... in the compiled model.

## Additional resources

- [Supported functional methods](supported-functional-methods.md)
