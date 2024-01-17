# Edit a model

You can edit a model after you create or load it, using the Sentis `Model` API.

## Preprocess inputs or postprocess outputs

Sometimes your model expects inputs or returns outputs in a format that doesn't match your tensor data. Using the model API you can add, remove and edit inputs, layers, constants and outputs to adjust the model.

In the following example the [mnist-8](https://github.com/onnx/models/blob/main/validated/vision/classification/mnist/model/mnist-8.onnx) model is adjusted to return the softmax of the outputs.

```
using UnityEngine;
using Unity.Sentis;

public class AddOutput : MonoBehaviour
{
    ModelAsset modelAsset;

    void Start()
    {
        // Load the runtime model from the model asset
        Model runtimeModel = ModelLoader.Load(modelAsset);

        // Define a new output name for the softmax output
        string softmaxOutputName = "Softmax_Output";

        // Append a Softmax layer to the end of the model layers using the previous model output as the layer input
        runtimeModel.AddLayer(new Softmax(softmaxOutputName, runtimeModel.outputs[0]));

        // Replace the previous model output with the sofmax output
        runtimeModel.outputs[0] = softmaxOutputName;
    }
}
```

Sentis executes the layers in the order they appear in `Model.layers`. If you preprocess a tensor, make sure you insert new layers before any dependent layers.

Sentis can't run [model optimization](models-concept.md#how-sentis-optimizes-a-model) on models you create using the `Model` API.

Correcting the same link as above.
