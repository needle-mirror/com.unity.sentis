# Edit a model

Use the Sentis [`Functional`](xref:Unity.Sentis.Functional) API to edit a model after you create or load it.

## Preprocess inputs or postprocess outputs

Sometimes your model expects inputs or returns outputs in a format that doesn't match your tensor data. Use the [`Functional`](xref:Unity.Sentis.Functional) API to prepend operations to inputs, append operations to outputs, or to easily add or remove inputs or outputs.

In the following example, the [mnist-8](https://github.com/onnx/models/blob/main/validated/vision/classification/mnist/model/mnist-8.onnx) model is adjusted to return the softmax of the output.

```
using UnityEngine;
using Unity.Sentis;

public class AddOutput : MonoBehaviour
{
    ModelAsset modelAsset;

    void Start()
    {
        // Load the source model from the model asset
        Model model = ModelLoader.Load(modelAsset);

        // Define the functional graph of the model.
        var graph = new FunctionalGraph();

        // Set the inputs of the graph from the original model inputs and return an array of functional tensors
        var inputs = graph.AddInputs(model);

        // Apply the model forward function to the inputs to get the source model functional outputs.
        // Sentis will destructively change the loaded source model. To avoid this at the expense of
        // higher memory usage and compile time, use the Functional.ForwardWithCopy method.
        FunctionalTensor[] outputs = Functional.Forward(model, inputs);

        // Calculate the softmax of the first output with the functional API.
        FunctionalTensor softmaxOutput = Functional.Softmax(outputs[0]);

        // Build the model from the graph using the `Compile` method with the desired outputs.
        var modelWithSoftmax = graph.Compile(softmaxOutput);
    }
}

```

Sentis runs [model optimization](models-concept.md#how-sentis-optimizes-a-model) on models you create using the [`Functional`](xref:Unity.Sentis.Functional) API. The Sentis operations used may, therefore, look different than expected.

Note that [`Compile`](xref:Unity.Sentis.FunctionalGraph.Compile*) is a slow operation that uses a lot of memory. It is recommended to run this offline and to serialize the computed model.
See [Serialize A Model](serialize-a-model.md) for details.

## Additional resources

- [Supported functional methods](supported-functional-methods.md)
