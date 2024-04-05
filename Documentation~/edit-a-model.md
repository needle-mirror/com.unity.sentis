# Edit a model

You can edit a model after you create or load it, using the Sentis [`Functional`](xref:Unity.Sentis.Functional) API.

## Preprocess inputs or postprocess outputs

Sometimes your model expects inputs or returns outputs in a format that doesn't match your tensor data. Use the `Functional` API to prepend operations to inputs, append operations to outputs, or to easily add or remove inputs or outputs. 

In the following example the [mnist-8](https://github.com/onnx/models/blob/main/validated/vision/classification/mnist/model/mnist-8.onnx) model is adjusted to return the softmax of the output.

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

        // Define the forward method of the model.
        // In this example an array of `FunctionalTensor` outputs is calculated from an array of `FunctionalTensor` inputs.
        FunctionalTensor[] ForwardWithSoftmax(FunctionalTensor[] inputs)
        {
            // Apply the model forward function to the inputs to get the source model functional outputs.
            // Sentis will destructively change the loaded source model. To avoid this at the expense of
            // higher memory usage and compile time, use the model.ForwardWithCopy method.
            FunctionalTensor[] outputs = model.Forward(inputs);

            // Calculate the softmax of the first output with the functional API.
            FunctionalTensor softmaxOutput = Functional.Softmax(outputs[0]);

            // Return an array containing the output.
            return new[] { softmaxOutput };
        }

        // Build the model from the using the `Compile` method. The input defs are taken from the original model.
        var modelWithSoftmax = Functional.Compile(ForwardWithSoftmax, InputDef.FromModel(model));
    }
}

```

Sentis runs [model optimization](models-concept.md#how-sentis-optimizes-a-model) on models you create using the `Functional` API. The Sentis operations used may, therefore, look different than expected.

## Additional resources

- [Supported functional methods](supported-functional-methods.md)
