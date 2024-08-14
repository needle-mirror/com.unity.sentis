# Inspect a model

## Get model inputs

Use the [`inputs`](xref:Unity.Sentis.Model.inputs) property of the runtime model to get the inputs of the model with and the name and shape of each input.

For example:

```
using UnityEngine;
using System.Collections.Generic;
using Unity.Sentis;

public class GetModelInputs : MonoBehaviour
{
    public ModelAsset modelAsset;

    void Start()
    {
        Model runtimeModel = ModelLoader.Load(modelAsset);

        List<Model.Input> inputs = runtimeModel.inputs;

        // Loop through each input
        foreach (var input in inputs)
        {
            // Log the name of the input, for example Input3
            Debug.Log(input.name);

            // Log the tensor shape of the input, for example (1, 1, 28, 28)
            Debug.Log(input.shape);
        }
    }
}
```

Input dimensions can be fixed or dynamic. Refer to [Model inputs](models-concept.md#model-inputs) for more information.

## Get model outputs

Use the [`outputs`](xref:Unity.Sentis.Model.outputs) property of the runtime model to get the names of the output layers of the model.

For example:

```
List<Model.Output> outputs = runtimeModel.outputs;

// Loop through each output
foreach (var output in outputs)
{
    // Log the name of the output
    Debug.Log(output.name);
}
```

## Get layers and layer properties

Use the [`layers`](xref:Unity.Sentis.Model.layers) property of the runtime model to get the neural network layers in the model, and the index, inputs or outputs of each layer.

## Open a model as a graph

To open an ONNX model as a graph, follow these steps:

1. Install [Netron](https://github.com/lutzroeder/netron), a third-party viewer for neural networks.
2. Double-click a model asset in the Project window, or select the model asset then select **Open** in the **Model Asset Import Settings** window.

## Additional resources

- [Profile a model](profile-a-model.md)
- [Tensor fundamentals](tensor-fundamentals.md)
- [Supported ONNX operators](supported-operators.md)
