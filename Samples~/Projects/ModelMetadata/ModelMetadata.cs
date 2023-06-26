using UnityEngine;
using Unity.Sentis;
using UnityEngine.Assertions;

public class ModelMetadata : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;

    void Start()
    {
        Model model = ModelLoader.Load(modelAsset);

        // on the onnx side, you can set a model metadata as follows
        // import onnx
        // model = onnx.load("my_model")
        // metadata = model_onnx.metadata_props.add()
        // metadata.key = "key"
        // metadata.value = "value"
        // print(model.metadata_props[0])
        // onnx.save(model, "my_model_with_metadata")

        // Model.Metadata holds all of the onnx metadata
        //  it is a Dictionary<string, string>
        Assert.AreEqual(1, model.Metadata.Count);
        Assert.AreEqual("This is a custom value saved in the .onnx file, written while exporting the model from python", model.Metadata["custom_metadata_from_onnx"]);

        Assert.AreEqual("ONNX", model.IrSource);
        Assert.AreEqual(9, model.DefaultOpsetVersion);
        Assert.AreEqual("pytorch v1.10", model.ProducerName);
    }
}
