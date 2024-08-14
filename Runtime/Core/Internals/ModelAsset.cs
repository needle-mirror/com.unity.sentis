using System;
using UnityEngine;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents a Sentis model asset.
    /// </summary>
    [PreferBinarySerialization]
    public class ModelAsset : ScriptableObject
    {
        /// <summary>
        /// The serialized binary data for the input descriptions, constant descriptions, layers, outputs, and metadata of the model.
        /// </summary>
        [SerializeField]
        internal ModelAssetData modelAssetData;

        /// <summary>
        /// The serialized binary data for the constant weights of the model, split into chunks.
        /// </summary>
        [SerializeField]
        internal ModelAssetWeightsData[] modelWeightsChunks;
    }
}
