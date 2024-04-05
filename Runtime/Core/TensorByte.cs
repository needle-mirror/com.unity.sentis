using System;
using Unity.Collections;
using UnityEngine.Assertions;

namespace Unity.Sentis
{
    /// <inheritdoc/>
    public class TensorByte : Tensor
    {
        int m_CountPacked32Bit;

        /// <inheritdoc/>
        public override DataType dataType => DataType.Byte;

        /// <inheritdoc/>
        public override int count { get { return m_CountPacked32Bit; } }

        /// <summary>
        /// Instantiates and returns a Tensor with the specified `shape`, an `ITensorData` `data`.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="data">The optional tensor data.</param>
        internal TensorByte(TensorShape shape, ITensorData data)
        {
            this.shape = shape;
            this.m_DataOnBackend = data;
            this.m_CountPacked32Bit = data.maxCapacity;
        }

        /// <summary>
        /// Initializes and returns a tensor with the specified `shape` and filled with `0`.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>The instantiated zero tensor.</returns>
        public static TensorByte AllocZeros(TensorShape shape)
        {
            int countPacked32Bit = ((shape.length * sizeof(byte) + sizeof(int) - 1) / sizeof(int));
            var burstTensorData = new BurstTensorData(countPacked32Bit, clearOnInit: true);
            return new TensorByte(shape, data: burstTensorData);
        }

        /// <inheritdoc/>
        public override void UploadToDevice(ITensorData destination)
        {
            var data = m_DataOnBackend.Download<int>(count);
            destination.Upload(data, count); data.Dispose();
            PinToDevice(destination, disposeUnpinned: true);
        }
    }
} // namespace Unity.Sentis
