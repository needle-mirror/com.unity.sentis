using System;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Sentis
{
    class CPUOps : Ops
    {
        public CPUOps()
            : base(BackendType.CPU) { }
    }

    abstract class Ops : IDisposable
    {
        IBackend m_Backend;

        protected Ops(BackendType backendType)
        {
            m_Backend = BackendFactory.CreateBackend(backendType);
        }

        public void Dispose()
        {
            m_Backend?.Dispose();
        }

        internal TensorFloat ScalarMad(TensorFloat X, float s, float b)
        {
            var O = TensorFloat.AllocNoData(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScalarMad(X, O, s, b);
            return O;
        }

        internal TensorInt ScalarMad(TensorInt X, int s, int b)
        {
            var O = TensorInt.AllocNoData(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScalarMad(X, O, s, b);
            return O;
        }

        public TensorFloat MatMul2D(TensorFloat X, TensorFloat Y, bool xTranspose, bool yTranspose)
        {
            var O = TensorFloat.AllocNoData(ShapeInference.Gemm(X.shape, Y.shape, xTranspose, yTranspose));
            if (O.shape.HasZeroDims())
                return O;
            if (X.shape.HasZeroDims() || Y.shape.HasZeroDims())
                m_Backend.MemSet(O, 0.0f);
            else
                m_Backend.MatMul2D(X, Y, O, xTranspose, yTranspose);
            return O;
        }

        public TensorFloat Dense(TensorFloat X, TensorFloat W, TensorFloat B)
        {
            var O = TensorFloat.AllocNoData(X.shape.MatMul(W.shape));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Dense(X, W, B, O, Layers.FusableActivation.None);
            return O;
        }

        public TensorFloat Add(TensorFloat A, TensorFloat B)
        {
            var O = TensorFloat.AllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Add(A, B, O);
            return O;
        }

        public TensorInt Add(TensorInt A, TensorInt B)
        {
            var O = TensorInt.AllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Add(A, B, O);
            return O;
        }

        public TensorFloat Sub(TensorFloat A, TensorFloat B)
        {
            var O = TensorFloat.AllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sub(A, B, O);
            return O;
        }

        public TensorInt Sub(TensorInt A, TensorInt B)
        {
            var O = TensorInt.AllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sub(A, B, O);
            return O;
        }

        public TensorFloat Mul(TensorFloat A, TensorFloat B)
        {
            var O = TensorFloat.AllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Mul(A, B, O);
            return O;
        }

        public TensorInt Mul(TensorInt A, TensorInt B)
        {
            var O = TensorInt.AllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Mul(A, B, O);
            return O;
        }

        public TensorFloat Div(TensorFloat A, TensorFloat B)
        {
            var O = TensorFloat.AllocNoData(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Div(A, B, O);
            return O;
        }

        public TensorFloat Sqrt(TensorFloat X)
        {
            var O = TensorFloat.AllocNoData(X.shape);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sqrt(X, O);
            return O;
        }

        public T Transpose<T>(T X) where T : Tensor
        {
            var O = AllocatorUtils.AllocTensor(X.dataType, X.shape.Transpose(), null) as T;
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Transpose(X, O);
            return O;
        }

        public TensorFloat ConstantOfShape(TensorShape X, float value)
        {
            var O = TensorFloat.AllocNoData(X);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.MemSet(O, value);
            return O;
        }
    }
}
