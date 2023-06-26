using System;
using UnityEngine;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents a Sentis model asset.
    /// </summary>
    public class ModelAsset : ScriptableObject
    {
        /// <summary>
        /// The serialized binary data for the input descriptions, constant descriptions, layers, outputs, and metadata of the model.
        /// </summary>
        [HideInInspector]
        public ModelAssetData modelAssetData;

        /// <summary>
        /// The serialized binary data for the constant weights of the model, split into chunks.
        /// </summary>
        [HideInInspector]
        public ModelAssetWeightsData[] modelWeightsChunks;

        [NonSerialized]
        Model m_Model;

        [NonSerialized]
        float m_LastLoaded;

        internal Model GetDeserializedModel()
        {
            if (m_Model == null)
            {
                m_Model = ModelLoader.Load(this);
                m_LastLoaded = Time.realtimeSinceStartup;
            }

            return m_Model;
        }

        void OnEnable()
        {
            // Used for detecting re-serialized models (e.g. adjusting import settings in the editor)
            // Force a reload on next access
            if (Time.realtimeSinceStartup >= m_LastLoaded)
                m_Model = null;
        }
    }
}
