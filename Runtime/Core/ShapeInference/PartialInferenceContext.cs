using System;
using System.Collections.Generic;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents a dictionary of partial tensors during partial tensor inference.
    /// </summary>
    class PartialInferenceContext
    {
        Dictionary<string, PartialTensor> m_PartialTensors;

        /// <summary>
        /// Instantiates and returns an empty partial inference context.
        /// </summary>
        public PartialInferenceContext()
        {
            m_PartialTensors = new Dictionary<string, PartialTensor>();
        }

        /// <summary>
        /// Add partial tensor with a given index to context.
        /// </summary>
        public void AddPartialTensor(string index, PartialTensor partialTensor)
        {
            if (string.IsNullOrEmpty(index))
                return;
            if (m_PartialTensors.TryGetValue(index, out var prevTensor))
                partialTensor = PartialTensor.MaxDefinedPartialTensor(partialTensor, prevTensor);
            m_PartialTensors[index] = partialTensor;
        }

        /// <summary>
        /// Returns array of partial tensors from array of indexes.
        /// </summary>
        public PartialTensor[] GetPartialTensors(string[] indexes)
        {
            var partialTensors = new PartialTensor[indexes.Length];

            for (var i = 0; i < indexes.Length; i++)
            {
                partialTensors[i] = GetPartialTensor(indexes[i]);
            }

            return partialTensors;
        }

        /// <summary>
        /// Returns partial tensor from index.
        /// </summary>
        public PartialTensor GetPartialTensor(string index)
        {
            if (string.IsNullOrEmpty(index))
                return null;

            return m_PartialTensors[index];
        }
    }
}
