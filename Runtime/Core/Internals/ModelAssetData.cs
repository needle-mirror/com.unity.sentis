using UnityEngine;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents data storage for a Sentis model asset.
    /// </summary>
    [PreferBinarySerialization]
    class ModelAssetData : ScriptableObject
    {
        /// <summary>
        /// The serialized byte array of the data.
        /// </summary>
        [HideInInspector]
        public byte[] value;
    }
}
