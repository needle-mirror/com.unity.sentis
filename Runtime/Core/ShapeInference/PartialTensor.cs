using System;
using System.Runtime.CompilerServices;
using UnityEngine;

[assembly: InternalsVisibleTo("Unity.Sentis.Tests")]
[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.BurstCPUOps")]

namespace Unity.Sentis
{
    struct PartialTensor
    {
        bool m_IsPartiallyKnown;
        TensorShape m_Shape;
        PartialTensorElement[] m_Elements;

        public PartialTensor(TensorShape shape)
        {
            m_IsPartiallyKnown = true;
            m_Shape = shape;
            m_Elements = new PartialTensorElement[shape.length];
        }

        public static PartialTensor ConstantOfShape(TensorShape shape, int intValue)
        {
            var tensorOut = new PartialTensor(shape);
            for (var i = 0; i < tensorOut.shape.length; i++)
            {
                tensorOut[i] = new PartialTensorElement(intValue);
            }

            return tensorOut;
        }

        public static PartialTensor Range(int start, int end)
        {
            var tensorOut = new PartialTensor(new TensorShape(end - start));
            for (var i = 0; i < tensorOut.shape.length; i++)
            {
                tensorOut[i] = new PartialTensorElement(start + i);
            }

            return tensorOut;
        }

        public static PartialTensor FromTensor(Tensor tensor)
        {
            if (!(tensor is TensorInt tensorInt) || tensorInt.shape.rank > 1 || tensorInt.shape.length > 8)
                return Unknown;
            var tensorOut = new PartialTensor(tensorInt.shape);
            for (var i = 0; i < tensorOut.shape.length; i++)
            {
                tensorOut[i] = new PartialTensorElement(tensorInt[i]);
            }

            return tensorOut;
        }

        public static PartialTensor Unknown => new PartialTensor { m_Elements = Array.Empty<PartialTensorElement>(), m_IsPartiallyKnown = false, m_Shape = default };

        public TensorShape shape => m_Shape;

        public SymbolicTensorShape symbolicShape => isPartiallyKnown ? new SymbolicTensorShape(shape) : SymbolicTensorShape.UnknownShape;

        public bool isPartiallyKnown => m_IsPartiallyKnown;

        public PartialTensorElement this[int d0]
        {
            get
            {
                if (!isPartiallyKnown)
                    return PartialTensorElement.Unknown;
                Logger.AssertIsTrue(d0 < shape.length, "InputError: index out of bounds");
                return m_Elements[d0];
            }
            set
            {
                Logger.AssertIsTrue(isPartiallyKnown, "InputError: partial tensor is unknown");
                m_Elements[d0] = value;
            }
        }

        public bool IsFullyKnown()
        {
            if (!isPartiallyKnown)
                return false;

            for (var i = 0; i < m_Elements.Length; i++)
            {
                if (!m_Elements[i].isValue)
                    return false;
            }

            return true;
        }

        public PartialTensor Reshape(SymbolicTensorShape newShape)
        {
            if (!newShape.IsFullyKnown())
                return Unknown;
            var reshapedTensor = new PartialTensor(newShape.ToTensorShape());

            for (var i = 0; i < shape.length; i++)
            {
                reshapedTensor.m_Elements[i] = m_Elements[i];
            }

            return reshapedTensor;
        }

        public TensorInt ToTensorInt()
        {
            Logger.AssertIsTrue(isPartiallyKnown, "InputError: partial tensor is unknown");

            var intElements = new int[m_Elements.Length];

            for (var i = 0; i < m_Elements.Length; i++)
            {
                intElements[i] = m_Elements[i].value;
            }

            return new TensorInt(m_Shape, intElements);
        }

        /// <summary>
        /// Returns a symbolic tensor shape if the `PartialTensor` represents a shape (it's a 1D tensor with no negative values).
        /// </summary>
        public SymbolicTensorShape ToSymbolicTensorShape()
        {
            if (!isPartiallyKnown)
                return SymbolicTensorShape.UnknownShape;
            AssertRank(1);
            var shapeOut = SymbolicTensorShape.UnknownOfRank(shape.length);
            for (var i = 0; i < shape.length; i++)
            {
                shapeOut[i] = this[i].ToSymbolicTensorDim();
            }

            return shapeOut;
        }

        public int[] ToIntArray()
        {
            var arrayOut = new int[shape.length];
            for (var i = 0; i < shape.length; i++)
            {
                arrayOut[i] = this[i].value;
            }

            return arrayOut;
        }

        public void AssertRank(int rank)
        {
            Logger.AssertIsTrue(isPartiallyKnown, "RankError: partial tensor is unknown");
            Logger.AssertIsTrue(shape.rank == rank, "RankError: incorrect rank, expecting {0}, got {1}", rank, shape.rank);
        }

        public static PartialTensor BroadcastWithOp(PartialTensor a, PartialTensor b, Func<PartialTensorElement, PartialTensorElement, PartialTensorElement> op)
        {
            if (!a.isPartiallyKnown || !b.isPartiallyKnown)
                return Unknown;
            if (a.shape.rank > 1 || b.shape.rank > 1)
                return Unknown;
            var outShape = a.shape.Broadcast(b.shape);

            var outTensor = new PartialTensor(outShape);
            for (var i = 0; i < outTensor.shape.length; i++)
            {
                outTensor[i] = op(a[i % a.shape.length], b[i % b.shape.length]);
            }

            return outTensor;
        }
    }
}
