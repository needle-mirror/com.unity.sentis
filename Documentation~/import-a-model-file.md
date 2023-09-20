# Import a model file

To import an ONNX model file, drag the file from your computer into the **Assets** folder of the Project window.

## Supported models

You can import most ONNX model files with an [opset version](https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions) between 7 and 15. Versions below 7 or above 15 might still import into Sentis, but you might get unexpected results. 

Sentis doesn't support the following:

- Models that use tensors with more than 8 dimensions.
- Sparse input tensors or constants.
- String tensors.
- Complex number tensors.

Sentis also converts some tensor data types like bools to floats or ints. This might increase the memory your model uses.

When you import a model file, Sentis optimizes the model. Refer to [Understand models in Sentis](models-concept.md) for more information.

## Create a runtime model

To use an imported model, you must use `ModelLoader.Load` to create a runtime `Model` object.

```
using UnityEngine;
using Unity.Sentis;

public class CreateRuntimeModel : MonoBehaviour
{
    public ModelAsset modelAsset;
    Model runtimeModel;

    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
    }
}
```

You can then [create an engine to run a model](create-an-engine.md).

## Import errors

If the [Model Asset Import Settings](onnx-model-importer-properties.md) window displays a warning that your model contains unsupported operators, you can add a custom layer to implement the missing operator. Refer to the `Add a custom layer` example in the [sample scripts](package-samples.md) for an example.

## Additional resources

- [How Sentis optimizes a model](models-concept.md#how-sentis-optimizes-a-model)
- [Export an ONNX file from a machine learning framework](export-an-onnx-file.md)|Export a model from a machine learning framework in the ONNX format Sentis needs.
- [Model Asset Import Settings](onnx-model-importer-properties.md)
