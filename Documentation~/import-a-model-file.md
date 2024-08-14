# Import a model file

To import an ONNX model file, drag the file from your computer into the `Assets` folder of the **Project** window.

If your model has external weights files, put them in the same directory as the model file so that Sentis imports them automatically.

To learn about the models Sentis supports, refer to [Supported models](supported-models.md).

## Create a runtime model

To use an imported model, you must use [`ModelLoader.Load`](xref:Unity.Sentis.ModelLoader.Load*) to create a runtime [`Model`](xref:Unity.Sentis.Model) object.

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

## Additional resources

- [How Sentis optimizes a model](models-concept.md#how-sentis-optimizes-a-model)
- [Export an ONNX file from a machine learning framework](export-convert-onnx.md)
- [Model Asset Import Settings](onnx-model-importer-properties.md)
- [Supported models](supported-models.md)
