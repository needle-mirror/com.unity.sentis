using System;
using Unity.Collections;
using UnityEngine;

namespace Unity.Sentis
{
    /// <inheritdoc/>
    public class TensorShort : Tensor
    {
        int m_LengthPacked32Bit;

        /// <inheritdoc/>
        public override DataType dataType => DataType.Short;

        /// <inheritdoc/>
        public override int count { get { return m_LengthPacked32Bit; } }

        /// <summary>
        /// Instantiates and returns a Tensor with the specified `shape`, an `ITensorData` `data`.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="data">The optional tensor data.</param>
        internal TensorShort(TensorShape shape, ITensorData data)
        {
            this.shape = shape;
            this.m_DataOnBackend = data;
            this.m_LengthPacked32Bit = data.maxCapacity;
        }

        /// <summary>
        /// Initializes and returns a tensor with the specified `shape` and filled with `0`.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The instantiated zero tensor.</returns>
        public static TensorShort AllocZeros(TensorShape shape)
        {
            int lengthPacked32Bit = ((shape.length * sizeof(ushort) + sizeof(int) - 1) / sizeof(int));
            var burstTensorData = new BurstTensorData(lengthPacked32Bit, clearOnInit: true);
            return new TensorShort(shape, data: burstTensorData);
        }
    }
}
