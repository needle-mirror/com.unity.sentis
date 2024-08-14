using System;
using System.Collections.Generic;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace Unity.Sentis
{
    /// <summary>
    /// Contains additional metadata about the ONNX model, stored in the ONNX file.
    /// </summary>
    [Serializable]
    public struct ONNXModelMetadata : ISerializationCallbackReceiver
    {
        /// <summary>
        /// Human-readable documentation for this model.
        /// </summary>
        public string DocString;
        /// <summary>
        /// A reverse-DNS name to indicate the model namespace or domain.
        /// </summary>
        public string Domain;
        /// <summary>
        /// Version number of the ONNX Intermediate Representation (IR) used in this model.
        /// </summary>
        public long IRVersion;
        /// <summary>
        /// Named metadata as dictionary.
        /// </summary>
        [NonSerialized]
        public Dictionary<string, string> MetadataProps;
        /// <summary>
        /// The name of the tool used to generate the model.
        /// </summary>
        public string ProducerName;
        /// <summary>
        /// The version of the generating tool.
        /// </summary>
        public string ProducerVersion;
        /// <summary>
        /// The version of the model itself, encoded in an integer.
        /// </summary>
        public long ModelVersion;

        [SerializeField]
        List<string> m_MetadataKeys;
        [SerializeField]
        List<string> m_MetadataValues;

        /// <inheritdoc/>
        public void OnBeforeSerialize()
        {
            if (MetadataProps == null)
                return;

            m_MetadataKeys = new List<string>(MetadataProps.Keys);
            m_MetadataValues = new List<string>(MetadataProps.Values);
        }

        /// <inheritdoc/>
        public void OnAfterDeserialize()
        {
            MetadataProps = new Dictionary<string, string>();
            for (int i = 0; i < m_MetadataKeys.Count; i++)
            {
                MetadataProps[m_MetadataKeys[i]] = m_MetadataValues[i];
            }
        }
    }

    /// <summary>
    /// Implement this interface to receive model metadata during ONNX import.
    /// </summary>
    public interface IONNXMetadataImportCallbackReceiver
    {
        /// <summary>
        /// This method is called when metadata is loaded during ONNX import, before the model is serialized.
        /// </summary>
        /// <param name="ctx">The context of the current import process.</param>
        /// <param name="metadata">The metadata fields of the imported ONNX file.</param>
        /// <remarks>
        /// This method is only called at import time. It is the responsibility of the implementer to store the metadata
        /// in a way that it can be accessed later. The <see cref="AssetImportContext"/> is provided to so that
        /// additional assets can be created and added to the import context, if necessary.
        /// Note that the model itself is not available in the <see cref="AssetImportContext"/> at this point.
        /// </remarks>
        void OnMetadataImported(AssetImportContext ctx, ONNXModelMetadata metadata);
    }
}
