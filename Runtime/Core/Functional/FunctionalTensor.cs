using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents a tensor that is a result of tensor operations.
    /// </summary>
    public partial class FunctionalTensor
    {
        DataType m_DataType;
        TensorShape m_Shape;
        bool m_IsShapeKnown;

        Node m_Source;
        int m_OutputIndex;

        internal DataType dataType => m_DataType;
        internal TensorShape shape => m_Shape;
        internal bool isShapeKnown => m_IsShapeKnown;
        internal Node source => m_Source;
        internal int outputIndex => m_OutputIndex;
        internal int index => m_Source.OutputIndices[m_OutputIndex];

        internal FunctionalTensor(DataType dataType, Node source, int outputIndex)
        {
            m_DataType = dataType;
            m_Source = source;
            m_OutputIndex = outputIndex;
            m_IsShapeKnown = false;
        }

        internal FunctionalTensor(DataType dataType, TensorShape shape, Node source, int outputIndex)
        {
            m_DataType = dataType;
            m_Source = source;
            m_OutputIndex = outputIndex;
            m_IsShapeKnown = true;
            m_Shape = shape;
        }

        internal void SetShape(TensorShape shape)
        {
            m_IsShapeKnown = true;
            m_Shape = shape;
        }

        internal static FunctionalTensor FromTensor(Tensor tensor)
        {
            Constant constant;
            switch (tensor.dataType)
            {
                case DataType.Float:
                {
                    constant = new Constant(-1, tensor.shape, (tensor as Tensor<float>).DownloadToNativeArray().ToArray());
                    break;
                }
                case DataType.Int:
                {
                    constant = new Constant(-1, tensor.shape, (tensor as Tensor<int>).DownloadToNativeArray().ToArray());
                    break;
                }
                default:
                    throw new NotImplementedException();
            }

            var constantNode = new ConstantNode(constant);
            return new FunctionalTensor(constant.dataType, tensor.shape, constantNode, 0);
        }

        internal static FunctionalTensor FromConstant(Constant constant)
        {
            var constantNode = new ConstantNode(constant);
            return new FunctionalTensor(constant.dataType, constant.shape, constantNode, 0);
        }

        /// <summary>
        /// Returns a string that represents the `FunctionalTensor` with the data type and shape if known.
        /// </summary>
        /// <returns>The string representation of the `FunctionalTensor`.</returns>
        public override string ToString()
        {
            if (isShapeKnown)
                return $"{m_DataType}{m_Shape}";
            else
                return $"{m_DataType}(?)";
        }
    }
}
