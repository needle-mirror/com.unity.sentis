using System;
using UnityEngine;

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

        internal Tensor<float> ScalarMad(Tensor<float> X, float s, float b)
        {
            var O = new Tensor<float>(X.shape, data: null);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScalarMad(X, O, s, b);
            return O;
        }

        internal Tensor<int> ScalarMad(Tensor<int> X, int s, int b)
        {
            var O = new Tensor<int>(X.shape, data: null);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.ScalarMad(X, O, s, b);
            return O;
        }

        public Tensor<float> MatMul2D(Tensor<float> X, Tensor<float> Y, bool xTranspose, bool yTranspose)
        {
            var O = new Tensor<float>(ShapeInference.Gemm(X.shape, Y.shape, xTranspose, yTranspose), data: null);
            if (O.shape.HasZeroDims())
                return O;
            if (X.shape.HasZeroDims() || Y.shape.HasZeroDims())
                m_Backend.MemSet(O, 0.0f);
            else
                m_Backend.MatMul2D(X, Y, O, xTranspose, yTranspose);
            return O;
        }

        public Tensor<float> Dense(Tensor<float> X, Tensor<float> W, Tensor<float> B)
        {
            var O = new Tensor<float>(X.shape.MatMul(W.shape), data: null);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Dense(X, W, B, O, Layers.FusableActivation.None);
            return O;
        }

        public Tensor<float> Add(Tensor<float> A, Tensor<float> B)
        {
            var O = new Tensor<float>(TensorShapeHelper.BroadcastShape(A, B), data: null);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Add(A, B, O);
            return O;
        }

        public Tensor<int> Add(Tensor<int> A, Tensor<int> B)
        {
            var O = new Tensor<int>(TensorShapeHelper.BroadcastShape(A, B), data: null);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Add(A, B, O);
            return O;
        }

        public Tensor<float> Sub(Tensor<float> A, Tensor<float> B)
        {
            var O = new Tensor<float>(TensorShapeHelper.BroadcastShape(A, B), data: null);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sub(A, B, O);
            return O;
        }

        public Tensor<int> Sub(Tensor<int> A, Tensor<int> B)
        {
            var O = new Tensor<int>(TensorShapeHelper.BroadcastShape(A, B), data: null);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sub(A, B, O);
            return O;
        }

        public Tensor<float> Mul(Tensor<float> A, Tensor<float> B)
        {
            var O = new Tensor<float>(TensorShapeHelper.BroadcastShape(A, B), data: null);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Mul(A, B, O);
            return O;
        }

        public Tensor<int> Mul(Tensor<int> A, Tensor<int> B)
        {
            var O = new Tensor<int>(TensorShapeHelper.BroadcastShape(A, B), data: null);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Mul(A, B, O);
            return O;
        }

        public Tensor<float> Div(Tensor<float> A, Tensor<float> B)
        {
            var O = new Tensor<float>(TensorShapeHelper.BroadcastShape(A, B), data: null);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Div(A, B, O);
            return O;
        }

        public Tensor<float> Sqrt(Tensor<float> X)
        {
            var O = new Tensor<float>(X.shape, data: null);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Sqrt(X, O);
            return O;
        }

        public Tensor<float> Transpose(Tensor<float> X)
        {
            var O = new Tensor<float>(X.shape.Transpose(), data: null);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.Transpose(X, O);
            return O;
        }

        public Tensor<float> ConstantOfShape(TensorShape X, float value)
        {
            var O = new Tensor<float>(X, data: null);
            if (O.shape.HasZeroDims())
                return O;
            m_Backend.MemSet(O, value);
            return O;
        }
    }
}
