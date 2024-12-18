#if UNITY_6000_1_OR_NEWER

using UnityEngine;
using System;
using Unity.Collections.LowLevel.Unsafe;
using static Unity.Sentis.ComputeTensorData;
using UnityEngine.Rendering;
using Unity.Sentis.Layers;

namespace Unity.Sentis
{
    class GfxDeviceBackend : GPUComputeBackend
    {
        MachineLearningContext m_GfxContext;

        public GfxDeviceBackend()
        {
            m_GfxContext = new MachineLearningContext();
        }

        public override void Dispose()
        {
            m_GfxContext.Dispose();
            base.Dispose();
        }

        static MachineLearningDataType ConvertToMachineLearningDataType(DataType dataType)
        {
            return dataType == DataType.Float ? MachineLearningDataType.Float32 : MachineLearningDataType.Int32;
        }

        static MachineLearningTensorShape ConvertToMachineLearningTensorShape(TensorShape tensorShape)
        {
            var shape = new MachineLearningTensorShape { rank = (uint)tensorShape.rank };
            unsafe
            {
                var srcPtr = (byte*)&tensorShape + sizeof(int) * (TensorShape.maxRank - tensorShape.rank);
                var destPtr = (byte*)&shape + sizeof(uint);
                UnsafeUtility.MemCpy(destPtr, srcPtr, (uint)(sizeof(int) * tensorShape.rank));
            }

            return shape;
        }

        static MachineLearningTensorDescriptor CreateDescriptor(Tensor T)
        {
            if (T == null)
                return MachineLearningTensorDescriptor.NullTensor();
            return new MachineLearningTensorDescriptor { dataType = ConvertToMachineLearningDataType(T.dataType), shape = ConvertToMachineLearningTensorShape(T.shape) };
        }

        /// <inheritdoc/>
        public override void MatMul2D(Tensor<float> X, Tensor<float> Y, Tensor<float> O, bool xTranspose, bool yTranspose)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxY = CreateDescriptor(Y);
            var descGfxO = CreateDescriptor(O);
            var opDesc = new MachineLearningOperatorFactory.GemmDescriptor()
            {
                X = descGfxX,
                Y = descGfxY,
                Z = MachineLearningTensorDescriptor.NullTensor(),
                O = descGfxO,
                transposeX = xTranspose,
                transposeY = yTranspose,
                alpha = 1.0f,
                beta = 1.0f,
                fusedActivation = MachineLearningOperatorType.None,
            };

            var op = MachineLearningOperatorFactory.Gemm(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Gemm(cb, op, Pin(X).buffer, Pin(Y).buffer, null, Pin(O).buffer);
            else
                base.MatMul2D(X, Y, O, xTranspose, yTranspose);
        }

        /// <inheritdoc/>
        public override void MatMul(Tensor<float> X, Tensor<float> Y, Tensor<float> O)
        {
            if (X.shape.rank > 4)
            {
                base.MatMul(X, Y, O);
                return;
            }

            var descGfxX = CreateDescriptor(X);
            var descGfxY = CreateDescriptor(Y);
            var descGfxO = CreateDescriptor(O);
            var opDesc = new MachineLearningOperatorFactory.GemmDescriptor()
            {
                X = descGfxX,
                Y = descGfxY,
                Z = MachineLearningTensorDescriptor.NullTensor(),
                O = descGfxO,
                transposeX = false,
                transposeY = false,
                alpha = 1.0f,
                beta = 1.0f,
                fusedActivation = MachineLearningOperatorType.None,
            };

            var op = MachineLearningOperatorFactory.Gemm(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Gemm(cb, op, Pin(X).buffer, Pin(Y).buffer, null, Pin(O).buffer);
            else
                base.MatMul(X, Y, O);
        }

        /// <inheritdoc/>
        public override void Dense(Tensor<float> X, Tensor<float> W, Tensor<float> B, Tensor<float> O, FusableActivation fusedActivation)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxY = CreateDescriptor(W);
            var descGfxZ = CreateDescriptor(B);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.GemmDescriptor()
            {
                X = descGfxX,
                Y = descGfxY,
                Z = descGfxZ,
                O = descGfxO,
                transposeX = false,
                transposeY = false,
                alpha = 1.0f,
                beta = 1.0f,
                fusedActivation = fusedActivation == FusableActivation.Relu ? MachineLearningOperatorType.ReLU : MachineLearningOperatorType.None,
            };

            var op = MachineLearningOperatorFactory.Gemm(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Gemm(cb, op, Pin(X).buffer, Pin(W).buffer, B != null ? Pin(B).buffer : null, Pin(O).buffer);
            else
                base.Dense(X, W, B, O, fusedActivation);
        }

        /// <inheritdoc/>
        public override void DenseBatched(Tensor<float> X, Tensor<float> W, Tensor<float> B, Tensor<float> O, FusableActivation fusedActivation)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxY = CreateDescriptor(W);
            var descGfxZ = CreateDescriptor(B);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.GemmDescriptor()
            {
                X = descGfxX,
                Y = descGfxY,
                Z = descGfxZ,
                O = descGfxO,
                transposeX = false,
                transposeY = false,
                alpha = 1.0f,
                beta = 1.0f,
                fusedActivation = fusedActivation == FusableActivation.Relu ? MachineLearningOperatorType.ReLU : MachineLearningOperatorType.None,
            };

            var op = MachineLearningOperatorFactory.Gemm(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Gemm(cb, op, Pin(X).buffer, Pin(W).buffer, B != null ? Pin(B).buffer : null, Pin(O).buffer);
            else
                base.DenseBatched(X, W, B, O, fusedActivation);
        }

        /// <inheritdoc/>
        public override void Conv(Tensor<float> X, Tensor<float> K, Tensor<float> B, Tensor<float> O, int groups, Span<int> strides, Span<int> pads, Span<int> dilations, FusableActivation fusedActivation)
        {
            if (dilations[0] != 1 || fusedActivation != FusableActivation.None)
            {
                base.Conv(X, K, B, O, groups, strides, pads, dilations, fusedActivation);
                return;
            }

            var descGfxX = CreateDescriptor(X);
            var descGfxK = CreateDescriptor(K);
            var descGfxB = CreateDescriptor(B);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ConvDescriptor
            {
                X = descGfxX,
                K = descGfxK,
                B = descGfxB,
                O = descGfxO,
                groups = groups,
                strides = strides,
                pads = pads,
                dilations = dilations,
                fusedActivation = fusedActivation == FusableActivation.Relu ? MachineLearningOperatorType.ReLU : MachineLearningOperatorType.None,
            };

            var op = MachineLearningOperatorFactory.Conv(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Conv(cb, op, Pin(X).buffer, Pin(K).buffer, B != null ? Pin(B).buffer : null, Pin(O).buffer);
            else
                base.Conv(X, K, B, O, groups, strides, pads, dilations, fusedActivation);
        }

        /// <inheritdoc/>
        public override void ReduceMax(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ReduceDescriptor
            {
                X = descGfxX,
                O = descGfxO,
                reduceFunc = MachineLearningOperatorType.ReduceMax,
                axes = axes
            };

            var op = MachineLearningOperatorFactory.Reduce(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Reduce(cb, op, Pin(X).buffer, Pin(O).buffer);
            else
                base.ReduceMax(X, O, axes);
        }

        /// <inheritdoc/>
        public override void ReduceMax(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ReduceDescriptor
            {
                X = descGfxX,
                O = descGfxO,
                reduceFunc = MachineLearningOperatorType.ReduceMax,
                axes = axes
            };

            var op = MachineLearningOperatorFactory.Reduce(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Reduce(cb, op, Pin(X).buffer, Pin(O).buffer);
            else
                base.ReduceMax(X, O, axes);
        }

        /// <inheritdoc/>
        public override void ReduceMean(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ReduceDescriptor
            {
                X = descGfxX,
                O = descGfxO,
                reduceFunc = MachineLearningOperatorType.ReduceMean,
                axes = axes
            };

            var op = MachineLearningOperatorFactory.Reduce(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Reduce(cb, op, Pin(X).buffer, Pin(O).buffer);
            else
                base.ReduceMean(X, O, axes);
        }

        /// <inheritdoc/>
        public override void ReduceMin(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ReduceDescriptor
            {
                X = descGfxX,
                O = descGfxO,
                reduceFunc = MachineLearningOperatorType.ReduceMin,
                axes = axes
            };

            var op = MachineLearningOperatorFactory.Reduce(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Reduce(cb, op, Pin(X).buffer, Pin(O).buffer);
            else
                base.ReduceMin(X, O, axes);
        }

        /// <inheritdoc/>
        public override void ReduceMin(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ReduceDescriptor
            {
                X = descGfxX,
                O = descGfxO,
                reduceFunc = MachineLearningOperatorType.ReduceMin,
                axes = axes
            };

            var op = MachineLearningOperatorFactory.Reduce(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Reduce(cb, op, Pin(X).buffer, Pin(O).buffer);
            else
                base.ReduceMin(X, O, axes);
        }

        /// <inheritdoc/>
        public override void ReduceProd(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ReduceDescriptor
            {
                X = descGfxX,
                O = descGfxO,
                reduceFunc = MachineLearningOperatorType.ReduceProd,
                axes = axes
            };

            var op = MachineLearningOperatorFactory.Reduce(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Reduce(cb, op, Pin(X).buffer, Pin(O).buffer);
            else
                base.ReduceProd(X, O, axes);
        }

        /// <inheritdoc/>
        public override void ReduceProd(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ReduceDescriptor
            {
                X = descGfxX,
                O = descGfxO,
                reduceFunc = MachineLearningOperatorType.ReduceProd,
                axes = axes
            };

            var op = MachineLearningOperatorFactory.Reduce(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Reduce(cb, op, Pin(X).buffer, Pin(O).buffer);
            else
                base.ReduceProd(X, O, axes);
        }

        /// <inheritdoc/>
        public override void ReduceSum(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ReduceDescriptor
            {
                X = descGfxX,
                O = descGfxO,
                reduceFunc = MachineLearningOperatorType.ReduceSum,
                axes = axes
            };

            var op = MachineLearningOperatorFactory.Reduce(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Reduce(cb, op, Pin(X).buffer, Pin(O).buffer);
            else
                base.ReduceSum(X, O, axes);
        }

        /// <inheritdoc/>
        public override void ReduceSum(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ReduceDescriptor
            {
                X = descGfxX,
                O = descGfxO,
                reduceFunc = MachineLearningOperatorType.ReduceSum,
                axes = axes
            };

            var op = MachineLearningOperatorFactory.Reduce(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Reduce(cb, op, Pin(X).buffer, Pin(O).buffer);
            else
                base.ReduceSum(X, O, axes);
        }

        /// <inheritdoc/>
        public override void ReduceSumSquare(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ReduceDescriptor
            {
                X = descGfxX,
                O = descGfxO,
                reduceFunc = MachineLearningOperatorType.ReduceSumSquare,
                axes = axes
            };

            var op = MachineLearningOperatorFactory.Reduce(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Reduce(cb, op, Pin(X).buffer, Pin(O).buffer);
            else
                base.ReduceSumSquare(X, O, axes);
        }

        /// <inheritdoc/>
        public override void ReduceSumSquare(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ReduceDescriptor
            {
                X = descGfxX,
                O = descGfxO,
                reduceFunc = MachineLearningOperatorType.ReduceSumSquare,
                axes = axes
            };

            var op = MachineLearningOperatorFactory.Reduce(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Reduce(cb, op, Pin(X).buffer, Pin(O).buffer);
            else
                base.ReduceSumSquare(X, O, axes);
        }

        /// <inheritdoc/>
        public override void ReduceL1(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ReduceDescriptor
            {
                X = descGfxX,
                O = descGfxO,
                reduceFunc = MachineLearningOperatorType.ReduceL1,
                axes = axes
            };

            var op = MachineLearningOperatorFactory.Reduce(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Reduce(cb, op, Pin(X).buffer, Pin(O).buffer);
            else
                base.ReduceL1(X, O, axes);
        }

        /// <inheritdoc/>
        public override void ReduceL2(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ReduceDescriptor
            {
                X = descGfxX,
                O = descGfxO,
                reduceFunc = MachineLearningOperatorType.ReduceL2,
                axes = axes
            };

            var op = MachineLearningOperatorFactory.Reduce(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Reduce(cb, op, Pin(X).buffer, Pin(O).buffer);
            else
                base.ReduceL2(X, O, axes);
        }

        /// <inheritdoc/>
        public override void ReduceLogSum(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
        {
            var descGfxX = CreateDescriptor(X);
            var descGfxO = CreateDescriptor(O);

            var opDesc = new MachineLearningOperatorFactory.ReduceDescriptor
            {
                X = descGfxX,
                O = descGfxO,
                reduceFunc = MachineLearningOperatorType.ReduceLogSum,
                axes = axes
            };

            var op = MachineLearningOperatorFactory.Reduce(m_GfxContext, opDesc);
            if (op.IsValid)
                MachineLearningOperatorDispatcher.Reduce(cb, op, Pin(X).buffer, Pin(O).buffer);
            else
                base.ReduceLogSum(X, O, axes);
        }
    }
}

#endif
