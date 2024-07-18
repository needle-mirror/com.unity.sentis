using UnityEngine;
using UnityEditor.AssetImporters;
using System.IO;
using System.Runtime.CompilerServices;
using System.Collections.Generic;

[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.RuntimeTests")]

namespace Unity.Sentis
{
    /// <summary>
    /// Represents an importer for serialized Sentis model files.
    /// </summary>
    [ScriptedImporter(2, new[] { "sentis" })]
    class SentisModelImporter : ScriptedImporter
    {
        /// <summary>
        /// Callback that Sentis calls when the ONNX model has finished importing.
        /// </summary>
        /// <param name="ctx">Asset import context</param>
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var model = ModelLoader.Load(ctx.assetPath);
            if (model == null)
                return;

            ModelAsset asset = ScriptableObject.CreateInstance<ModelAsset>();

            ModelAssetData modelAssetData = ScriptableObject.CreateInstance<ModelAssetData>();
            modelAssetData.value = ModelWriter.SaveModelDescription(model);
            modelAssetData.name = "Data";
            modelAssetData.hideFlags = HideFlags.HideInHierarchy;
            asset.modelAssetData = modelAssetData;

            var serializedWeights = ModelWriter.SaveModelWeights(model);
            asset.modelWeightsChunks = new ModelAssetWeightsData[serializedWeights.Length];
            for (int i = 0; i < serializedWeights.Length; i++)
            {
                asset.modelWeightsChunks[i] = ScriptableObject.CreateInstance<ModelAssetWeightsData>();
                asset.modelWeightsChunks[i].value = serializedWeights[i];
                asset.modelWeightsChunks[i].name = "Data";
                asset.modelWeightsChunks[i].hideFlags = HideFlags.HideInHierarchy;

                ctx.AddObjectToAsset($"model data weights {i}", asset.modelWeightsChunks[i]);
            }

            ctx.AddObjectToAsset("main obj", asset);
            ctx.AddObjectToAsset("model data", modelAssetData);

            ctx.SetMainObject(asset);
            model.DisposeWeights();
        }
    }
}
