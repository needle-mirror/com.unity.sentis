using System;
using UnityEngine;
using UnityEditor.AssetImporters;
using System.Runtime.CompilerServices;
using Unity.Sentis.ONNX;
using System.Collections.Generic;
using UnityEditor;

[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis
{
    /// <summary>
    /// Represents an importer for Open Neural Network Exchange (ONNX) files.
    /// </summary>
    [ScriptedImporter(65, new[] { "onnx" })]
    [HelpURL("https://docs.unity3d.com/Packages/com.unity.sentis@latest/index.html")]
    class ONNXModelImporter : ScriptedImporter
    {
        static readonly List<IONNXMetadataImportCallbackReceiver> k_MetadataImportCallbackReceivers;

        static ONNXModelImporter()
        {
            k_MetadataImportCallbackReceivers = new List<IONNXMetadataImportCallbackReceiver>();

            foreach (var type in TypeCache.GetTypesDerivedFrom<IONNXMetadataImportCallbackReceiver>())
            {
                if (type.IsInterface || type.IsAbstract)
                    continue;

                if (Attribute.IsDefined(type, typeof(DisableAutoRegisterAttribute)))
                    continue;

                var receiver = (IONNXMetadataImportCallbackReceiver)Activator.CreateInstance(type);
                RegisterMetadataReceiver(receiver);
            }
        }

        internal static void RegisterMetadataReceiver(IONNXMetadataImportCallbackReceiver receiver)
        {
            k_MetadataImportCallbackReceivers.Add(receiver);
        }

        internal static void UnregisterMetadataReceiver(IONNXMetadataImportCallbackReceiver receiver)
        {
            k_MetadataImportCallbackReceivers.Remove(receiver);
        }

        /// <summary>
        /// Callback that Sentis calls when the ONNX model has finished importing.
        /// </summary>
        /// <param name="ctx">Asset import context</param>
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var converter = new ONNXModelConverter(ctx.assetPath);
            converter.MetadataLoaded += metadata => InvokeMetadataHandlers(ctx, metadata);
            var model = converter.Convert();

            ModelAsset asset = ScriptableObject.CreateInstance<ModelAsset>();
            ModelWriter.SaveModel(model, out var modelDescriptionBytes, out var modelWeightsBytes);

            ModelAssetData modelAssetData = ScriptableObject.CreateInstance<ModelAssetData>();
            modelAssetData.value = modelDescriptionBytes;
            modelAssetData.name = "Data";
            modelAssetData.hideFlags = HideFlags.HideInHierarchy;
            asset.modelAssetData = modelAssetData;

            asset.modelWeightsChunks = new ModelAssetWeightsData[modelWeightsBytes.Length];
            for (int i = 0; i < modelWeightsBytes.Length; i++)
            {
                asset.modelWeightsChunks[i] = ScriptableObject.CreateInstance<ModelAssetWeightsData>();
                asset.modelWeightsChunks[i].value = modelWeightsBytes[i];
                asset.modelWeightsChunks[i].name = "Data";
                asset.modelWeightsChunks[i].hideFlags = HideFlags.HideInHierarchy;

                ctx.AddObjectToAsset($"model data weights {i}", asset.modelWeightsChunks[i]);
            }

            ctx.AddObjectToAsset("main obj", asset);
            ctx.AddObjectToAsset("model data", modelAssetData);

            ctx.SetMainObject(asset);
            model.DisposeWeights();
        }

        static void InvokeMetadataHandlers(AssetImportContext ctx, ONNXModelMetadata onnxModelMetadata)
        {
            if (k_MetadataImportCallbackReceivers == null)
                return;

            foreach (var receiver in k_MetadataImportCallbackReceivers)
            {
                receiver.OnMetadataImported(ctx, onnxModelMetadata);
            }
        }

        /// <summary>
        /// Attribute to disable automatic registration of <see cref="IONNXMetadataImportCallbackReceiver"/>
        /// implementations. Recommended for testing purposes.
        /// </summary>
        internal class DisableAutoRegisterAttribute : Attribute
        {
        }
    }
}
