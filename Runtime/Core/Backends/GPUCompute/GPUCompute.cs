using UnityEngine;
using UnityEngine.Assertions;
using System;
using System.Runtime.CompilerServices;
using Unity.Mathematics;
using UnityEngine.Rendering;
using static Unity.Sentis.ComputeTensorData;
using static Unity.Sentis.ShaderPropertyID;

[assembly: InternalsVisibleTo("Unity.Sentis.RuntimeTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis
{
    /// <summary>
    /// Represents a GPUCompute backend ops.
    /// </summary>
    partial class GPUComputeBackend : IBackend
    {
        /// <summary>
        /// The command buffer to use for scheduling.
        /// </summary>
        protected CommandBuffer cb;
        bool m_InternalCommandBuffer;

        /// <summary>
        /// Initializes and returns an instance of `GPUComputeOps`.
        /// </summary>
        public GPUComputeBackend()
        {
            cb = new CommandBuffer();
            m_InternalCommandBuffer = true;
        }

        public void SetCommandBuffer(CommandBuffer commandBuffer)
        {
            cb = commandBuffer;
            m_InternalCommandBuffer = false;
        }

        public bool InternalCommandBuffer() => m_InternalCommandBuffer;

        public void ExecuteCommandBufferAndClear()
        {
            Graphics.ExecuteCommandBuffer(cb);
            cb.Clear();
        }

        // Do we need this class or operate on ComputeTensorData instead?
        TensorClassPool<Tensor<float>> m_TensorFloatPool = new TensorClassPool<Tensor<float>>();
        TensorClassPool<Tensor<int>> m_TensorIntPool = new TensorClassPool<Tensor<int>>();
        TensorDataPool<ComputeTensorData> m_MemoryPool = new TensorDataPool<ComputeTensorData>();

        Tensor<float> AllocTensorFloat(TensorShape shape)
        {
            ComputeTensorData data = m_MemoryPool.AdoptFromPool(shape.length);
            if (data == null)
                data = new ComputeTensorData(shape.length);
            var tensor = m_TensorFloatPool.AdoptFromPool();
            if (tensor == null)
                tensor = new Tensor<float>(shape, data: null);

            tensor.shape = shape;
            tensor.count = shape.length;
            tensor.dataOnBackend = data;
            return tensor;
        }

        Tensor<int> AllocTensorInt(TensorShape shape)
        {
            ComputeTensorData data = m_MemoryPool.AdoptFromPool(shape.length);
            if (data == null)
                data = new ComputeTensorData(shape.length);
            var tensor = m_TensorIntPool.AdoptFromPool();
            if (tensor == null)
                tensor = new Tensor<int>(shape, data: null);

            tensor.shape = shape;
            tensor.count = shape.length;
            tensor.dataOnBackend = data;
            return tensor;
        }

        void ReleaseTensorFloat(Tensor<float> tensor)
        {
            if (tensor == null)
                return;
            m_MemoryPool.ReleaseToPool(tensor.dataOnBackend as ComputeTensorData);
            tensor.dataOnBackend = null;
            m_TensorFloatPool.ReleaseToPool(tensor as Tensor<float>);
        }

        void ReleaseTensorInt(Tensor<int> tensor)
        {
            if (tensor == null)
                return;
            m_MemoryPool.ReleaseToPool(tensor.dataOnBackend as ComputeTensorData);
            tensor.dataOnBackend = null;
            m_TensorIntPool.ReleaseToPool(tensor as Tensor<int>);
        }

        /// <summary>
        /// Disposes of the ops and any associated memory.
        /// </summary>
        public virtual void Dispose()
        {
            m_MemoryPool?.Dispose();
            m_MemoryPool = null;
        }

        /// <inheritdoc/>
        public virtual BackendType backendType => BackendType.GPUCompute;

        /// <inheritdoc/>
        public virtual void MatMul2D(Tensor<float> X, Tensor<float> Y, Tensor<float> O, bool xTranspose, bool yTranspose)
        {
            Gemm(X, Y, O, O.shape[0], xTranspose ? X.shape[0] : X.shape[1], O.shape[1], xTranspose, yTranspose);
        }

        /// <inheritdoc/>
        public virtual void MatMul(Tensor<float> X, Tensor<float> Y, Tensor<float> O)
        {
            var xShape = X.shape.rank == 1 ? new TensorShape(1, X.shape[0]) : X.shape;
            var yShape = Y.shape.rank == 1 ? new TensorShape(Y.shape[0], 1) : Y.shape;
            var oShape = X.shape.rank > 1 && Y.shape.rank > 1 ? O.shape : xShape.MatMul(yShape);

            var M = xShape[-2];
            var K = xShape[-1];
            var N = yShape[-1];
            var batch = oShape.Length(0, -2);

            if (batch == 1)
            {
                Gemm(X, Y, O, M, K, N);
                return;
            }

            if (xShape.Length(0, -2) == batch && yShape.Length(0, -2) == batch)
            {
                BatchedGemm(X, Y, O, batch, M, K, N);
                return;
            }

            var fn = ComputeFunctions.k_MatMul;

            unsafe
            {
                var shapeA = stackalloc int[6];
                var stridesA = stackalloc int[6];
                var shapeB = stackalloc int[6];
                var stridesB = stackalloc int[6];
                var shapeO = stackalloc int[6];
                var stridesO = stackalloc int[6];
                OpsUtils.PinMatMulTensorShapeStrides(xShape, yShape, oShape, shapeA, stridesA, shapeB, stridesB, shapeO, stridesO);

                cb.SetInt6(fn, k_ID_shapeA, shapeA);
                cb.SetInt6(fn, k_ID_stridesA, stridesA);
                cb.SetInt6(fn, k_ID_shapeB, shapeB);
                cb.SetInt6(fn, k_ID_stridesB, stridesB);
                cb.SetInt6(fn, k_ID_shapeO, shapeO);
                cb.SetInt6(fn, k_ID_stridesO, stridesO);
            }

            cb.SetComputeIntParam(fn.shader, k_ID_AM, M);
            cb.SetComputeIntParam(fn.shader, k_ID_AN, K);
            cb.SetComputeIntParam(fn.shader, k_ID_BM, K);
            cb.SetComputeIntParam(fn.shader, k_ID_BN, N);
            cb.SetComputeIntParam(fn.shader, k_ID_CB, batch);
            cb.SetComputeIntParam(fn.shader, k_ID_CM, M);
            cb.SetComputeIntParam(fn.shader, k_ID_CN, N);
            cb.SetComputeIntParam(fn.shader, k_ID_rank, oShape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Y));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.Dispatch(fn, batch, M, N);
        }

        void BatchedGemm(Tensor<float> X, Tensor<float> Y, Tensor<float> O, int batch, int M, int K, int N)
        {
            ComputeFunction fn;
            int workItemsX, workItemsY, workItemsZ;
            if (M == 1)
            {
                fn = ComputeFunctions.k_GemmBatched_V_L1Cached64;
                workItemsX = ComputeHelper.IDivC(N, 4);
                workItemsY = batch;
                workItemsZ = 1;
            }
            else if (N % 64 == 0 && K % 16 == 0)
            {
                fn = ComputeFunctions.k_GemmBatched_T16x16_R4x4;
                workItemsX = ComputeHelper.IDivC(N, 4);
                workItemsY = ComputeHelper.IDivC(M, 4);
                workItemsZ = batch;
            }
            else
            {
                fn = ComputeFunctions.k_GemmBatched_T8x8_R4x4;
                workItemsX = ComputeHelper.IDivC(N, 4);
                workItemsY = ComputeHelper.IDivC(M, 4);
                workItemsZ = batch;
            }

            cb.SetComputeIntParam(fn.shader, k_ID_maxXIndex, X.shape.length - 1);
            cb.SetComputeIntParam(fn.shader, k_ID_maxWIndex, Y.shape.length - 1);
            cb.SetComputeIntParam(fn.shader, k_ID_X_width, K);
            cb.SetComputeIntParam(fn.shader, k_ID_W_width, N);
            cb.SetComputeIntParam(fn.shader, k_ID_O_width, N);
            cb.SetComputeIntParam(fn.shader, k_ID_O_height, M);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(Y));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetComputeFloatParam(fn.shader, k_ID__MinValue, float.MinValue);

            cb.Dispatch(fn, workItemsX, workItemsY, batch);
        }

        /// <inheritdoc/>
        public virtual void Dense(Tensor<float> X, Tensor<float> W, Tensor<float> B, Tensor<float> O, Layers.FusableActivation fusedActivation)
        {
            var M = O.shape.Length(0, -1);
            var K = X.shape[-1];
            var N = O.shape[-1];

            int workItemsX, workItemsY;
            ComputeFunction fn;
            if (M == 1)
            {
                fn = ComputeFunctions.k_Dense_V_L1Cached64;
                workItemsX = ComputeHelper.IDivC(N, 4);
                workItemsY = 1;
            }
            else if (N % 64 == 0 && K % 16 == 0)
            {
                fn = ComputeFunctions.k_Dense_T16x16_R4x4;
                workItemsX = ComputeHelper.IDivC(N, 4);
                workItemsY = ComputeHelper.IDivC(M, 4);
            }
            else
            {
                fn = ComputeFunctions.k_Dense_T8x8_R4x4;
                workItemsX = ComputeHelper.IDivC(N, 4);
                workItemsY = ComputeHelper.IDivC(M, 4);
            }

            cb.SetComputeIntParam(fn.shader, k_ID_X_width, K);
            cb.SetComputeIntParam(fn.shader, k_ID_W_width, N);
            cb.SetComputeIntParam(fn.shader, k_ID_O_height, M);
            cb.SetComputeIntParam(fn.shader, k_ID_O_width, N);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(W));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetComputeIntParam(fn.shader, k_ID_maxXIndex, X.shape.length - 1);
            cb.SetComputeIntParam(fn.shader, k_ID_maxWIndex, W.shape.length - 1);
            cb.SetComputeIntParam(fn.shader, k_ID_maxBIndex, B.shape.length - 1);

            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetComputeIntParam(fn.shader, k_ID_maxBIndex, N - 1);
            cb.SetComputeFloatParam(fn.shader, k_ID__MinValue, fusedActivation == Layers.FusableActivation.Relu ? 0.0f : float.MinValue);

            cb.Dispatch(fn, workItemsX, workItemsY, 1);
        }

        /// <inheritdoc/>
        public virtual void DenseBatched(Tensor<float> X, Tensor<float> W, Tensor<float> B, Tensor<float> O, Layers.FusableActivation fusedActivation)
        {
            var batch = O.shape.Length(0, -1);
            var M = X.shape[-2];
            var K = X.shape[-1];
            var N = O.shape[-1];

            ComputeFunction fn;
            int workItemsX, workItemsY, workItemsZ;
            if (M == 1)
            {
                fn = ComputeFunctions.k_DenseBatched_V_L1Cached64;
                workItemsX = ComputeHelper.IDivC(N, 4);
                workItemsY = batch;
                workItemsZ = 1;
            }
            else if (N % 64 == 0 && K % 16 == 0)
            {
                fn = ComputeFunctions.k_DenseBatched_T16x16_R4x4;
                workItemsX = ComputeHelper.IDivC(N, 4);
                workItemsY = ComputeHelper.IDivC(M, 4);
                workItemsZ = batch;
            }
            else
            {
                fn = ComputeFunctions.k_DenseBatched_T8x8_R4x4;
                workItemsX = ComputeHelper.IDivC(N, 4);
                workItemsY = ComputeHelper.IDivC(M, 4);
                workItemsZ = batch;
            }

            cb.SetComputeIntParam(fn.shader, k_ID_maxXIndex, X.shape.length - 1);
            cb.SetComputeIntParam(fn.shader, k_ID_maxWIndex, W.shape.length - 1);
            cb.SetComputeIntParam(fn.shader, k_ID_maxBIndex, B.shape.length - 1);
            cb.SetComputeIntParam(fn.shader, k_ID_X_width, K);
            cb.SetComputeIntParam(fn.shader, k_ID_W_width, N);
            cb.SetComputeIntParam(fn.shader, k_ID_O_width, N);
            cb.SetComputeIntParam(fn.shader, k_ID_O_height, M);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(W));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetComputeFloatParam(fn.shader, k_ID__MinValue, fusedActivation == Layers.FusableActivation.Relu ? 0.0f : float.MinValue);

            cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);
        }

        /// <inheritdoc/>
        public void Tril(Tensor X, Tensor O, int k)
        {
            // Warning, for some reason shared mem implementation on intel gpu is x2 faster than regular one
            var fn = ComputeFunctions.k_Tril;
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetComputeIntParam(fn.shader, k_ID_X_width, X.shape[-1]);
            cb.SetComputeIntParam(fn.shader, k_ID_X_height, X.shape[-2]);
            cb.SetComputeIntParam(fn.shader, k_ID_X_length, X.shape.length);
            cb.SetComputeIntParam(fn.shader, k_ID_diagonalK, k);

            cb.Dispatch(fn, ComputeHelper.IDivC(X.shape.length, 4), 1, 1);
        }

        /// <inheritdoc/>
        public void Triu(Tensor X, Tensor O, int k)
        {
            // Warning, for some reason shared mem implementation on intel gpu is x2 faster than regular one
            var fn = ComputeFunctions.k_Triu;
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetComputeIntParam(fn.shader, k_ID_X_width, X.shape[-1]);
            cb.SetComputeIntParam(fn.shader, k_ID_X_height, X.shape[-2]);
            cb.SetComputeIntParam(fn.shader, k_ID_X_length, X.shape.length);
            cb.SetComputeIntParam(fn.shader, k_ID_diagonalK, k);

            cb.Dispatch(fn, ComputeHelper.IDivC(X.shape.length, 4), 1, 1);
        }

        void ApplyFusedActivation(Tensor<float> X, Tensor<float> O, Layers.FusableActivation fusedActivation)
        {
            switch (fusedActivation)
            {
                case Layers.FusableActivation.None:
                    return;
                case Layers.FusableActivation.Relu:
                    Relu(X, O);
                    return;
                default:
                    throw new NotImplementedException();
            }
        }

        /// <inheritdoc/>
        public virtual void Conv(Tensor<float> X, Tensor<float> K, Tensor<float> B, Tensor<float> O, int groups, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
        {
            if (X.shape.rank > 5)
            {
                throw new NotImplementedException();
            }

            if (X.shape.rank == 4 && K.shape[0] == groups && K.shape[1] == 1)
            {
                DepthwiseConv2D(X, K, B, O, groups, strides, pads, dilations, fusedActivation);
                return;
            }

            if (groups != 1)
            {
                GroupedConv(X, K, B, O, groups, strides, pads, dilations, fusedActivation);
                return;
            }

            if (ComputeInfo.IsMobileGPU())
            {
                ConvMobile(X, K, B, O, strides, pads, dilations, fusedActivation);
                return;
            }

            int workItemsX, workItemsY, workItemsZ;

            ComputeFunction fn;
            if (X.shape.rank == 5)
            {
                var n = O.shape[0];
                var k = O.shape[1];
                var d = O.shape[2];
                var h = O.shape[3];
                var w = O.shape[4];

                fn = K.shape.Length(2) == 1 ? ComputeFunctions.k_Conv3D_1x1_T16x16_R4x4 : ComputeFunctions.k_Conv3D_T16x16_R4x4;
                cb.SetComputeIntParam(fn.shader, k_ID_O_depth, O.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_O_height, O.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_O_width, O.shape[4]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_depth, X.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_height, X.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_width, X.shape[4]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_depth, K.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_height, K.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_width, K.shape[4]);
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Kptr, Pin(K));
                if (B != null)
                {
                    cb.EnableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
                    cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                }
                else
                {
                    cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
                }
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetComputeIntParam(fn.shader, k_ID_O_batch, O.shape[0]); cb.SetComputeIntParam(fn.shader, k_ID_O_channels, O.shape[1]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_channels, X.shape[1]);
                cb.SetInt4(fn, k_ID__Stride, strides);
                cb.SetInt4(fn, k_ID__Pad, pads);
                cb.SetInt4(fn, k_ID__Dilation, dilations);
                workItemsX = ComputeHelper.IDivC(k, 4);
                workItemsY = ComputeHelper.IDivC(d * h * w, 4);
                workItemsZ = n;
            }
            // TODO multiplte dispatch + reduce for thin conv
            else if (X.shape.rank == 4)
            {
                var n = O.shape[0];
                var k = O.shape[1];
                var h = O.shape[2];
                var w = O.shape[3];

                workItemsX = ComputeHelper.IDivC(h * w, 4);
                workItemsY = ComputeHelper.IDivC(k, 8);
                workItemsZ = n;

                fn = K.shape.Length(2) == 1 ? ComputeFunctions.k_Conv2D_1x1 : ComputeFunctions.k_Conv2D_KxK;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(K));
                if (B != null)
                {
                    cb.EnableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
                    cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                }
                else
                {
                    cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
                }
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetComputeIntParam(fn.shader, k_ID_inputChannels, X.shape[1]);
                cb.SetComputeIntParam(fn.shader, k_ID_inputHeight, X.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_inputWidth, X.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_kernelHeight, K.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_kernelWidth, K.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_outputChannels, O.shape[1]);
                cb.SetComputeIntParam(fn.shader, k_ID_outputHeight, O.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_outputWidth, O.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_strideHeight, strides[0]);
                cb.SetComputeIntParam(fn.shader, k_ID_strideWidth, strides[1]);
                cb.SetComputeIntParam(fn.shader, k_ID_padHeight, pads[0]);
                cb.SetComputeIntParam(fn.shader, k_ID_padWidth, pads[1]);
                cb.SetComputeIntParam(fn.shader, k_ID_dilationHeight, dilations != null ? dilations[0] : 1);
                cb.SetComputeIntParam(fn.shader, k_ID_dilationWidth, dilations != null ? dilations[1] : 1);
                cb.SetComputeIntParam(fn.shader, k_ID_inputChannelsSize, X.shape[1] * X.shape[2] * X.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_outputChannelsSize, O.shape[1] * O.shape[2] * O.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_kernelChannelSize, K.shape[1] * K.shape[2] * K.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_inputSize, X.shape[2] * X.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_outputSize, O.shape[2] * O.shape[3]);
            }
            else //if (X.shape.rank == 3)
            {
                var n = O.shape[0];
                var k = O.shape[1];
                var h = O.shape[2];

                workItemsX = ComputeHelper.IDivC(h, 4);
                workItemsY = ComputeHelper.IDivC(k, 8);
                workItemsZ = n;

                fn = K.shape.Length(2) == 1 ? ComputeFunctions.k_Conv1D_1x1 : ComputeFunctions.k_Conv1D_KxK;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(K));
                if (B != null)
                {
                    cb.EnableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
                    cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                }
                else
                {
                    cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
                }
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetComputeIntParam(fn.shader, k_ID_inputChannels, X.shape[1]);
                cb.SetComputeIntParam(fn.shader, k_ID_inputHeight, X.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_kernelHeight, K.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_outputChannels, O.shape[1]);
                cb.SetComputeIntParam(fn.shader, k_ID_outputHeight, O.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_strideHeight, strides[0]);
                cb.SetComputeIntParam(fn.shader, k_ID_padHeight, pads[0]);
                cb.SetComputeIntParam(fn.shader, k_ID_dilationHeight, dilations[0]);
                cb.SetComputeIntParam(fn.shader, k_ID_inputChannelsSize, X.shape[1] * X.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_outputChannelsSize, O.shape[1] * O.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_kernelChannelSize, K.shape[1] * K.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_inputSize, X.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_outputSize, O.shape[2]);
            }

            cb.SetComputeIntParam(fn.shader, k_ID_kernelLength, K.shape.length);
            cb.SetComputeFloatParam(fn.shader, k_ID__MinValue, fusedActivation == Layers.FusableActivation.Relu ? 0.0f : float.MinValue);

            cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);
        }

        void ConvMobile(Tensor<float> X, Tensor<float> K, Tensor<float> B, Tensor<float> O, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
        {
            int workItemsX, workItemsY, workItemsZ;

            ComputeFunction fn;
            // TODO regular conv faster for small spatial/channels size, figure good rule of thumb
            // TODO see when to call T8x8
            if (X.shape.rank == 5)
            {
                var n = O.shape[0];
                var k = O.shape[1];
                var d = O.shape[2];
                var h = O.shape[3];
                var w = O.shape[4];

                fn = ComputeFunctions.k_Conv3D_T16x16_R4x4;
                if (K.shape.Length(2) == 1)
                    fn = ComputeFunctions.k_Conv3D_1x1_T16x16_R4x4;
                cb.SetComputeIntParam(fn.shader, k_ID_O_depth, O.shape[2]); cb.SetComputeIntParam(fn.shader, k_ID_O_height, O.shape[3]); cb.SetComputeIntParam(fn.shader, k_ID_O_width, O.shape[4]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_depth, X.shape[2]); cb.SetComputeIntParam(fn.shader, k_ID_X_height, X.shape[3]); cb.SetComputeIntParam(fn.shader, k_ID_X_width, X.shape[4]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_depth, K.shape[2]); cb.SetComputeIntParam(fn.shader, k_ID_K_height, K.shape[3]); cb.SetComputeIntParam(fn.shader, k_ID_K_width, K.shape[4]);
                workItemsX = ComputeHelper.IDivC(k, 4); workItemsY = ComputeHelper.IDivC(d * h * w, 4); workItemsZ = n;
            }
            else if (X.shape.rank == 4)
            {
                var n = O.shape[0];
                var k = O.shape[1];
                var h = O.shape[2];
                var w = O.shape[3];

                fn = ComputeFunctions.k_Conv2D_T16x16_R4x4;
                if (K.shape.Length(2) == 1)
                    fn = ComputeFunctions.k_Conv2D_1x1_T16x16_R4x4;
                cb.SetComputeIntParam(fn.shader, k_ID_O_height, O.shape[2]); cb.SetComputeIntParam(fn.shader, k_ID_O_width, O.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_height, X.shape[2]); cb.SetComputeIntParam(fn.shader, k_ID_X_width, X.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_height, K.shape[2]); cb.SetComputeIntParam(fn.shader, k_ID_K_width, K.shape[3]);
                workItemsX = ComputeHelper.IDivC(k, 4); workItemsY = ComputeHelper.IDivC(h * w, 4); workItemsZ = n;
            }
            else //if (X.shape.rank == 3)
            {
                var n = O.shape[0];
                var k = O.shape[1];
                var w = O.shape[2];

                fn = ComputeFunctions.k_Conv1D_T16x16_R4x4;
                if (K.shape.Length(2) == 1)
                    fn = ComputeFunctions.k_Conv1D_1x1_T16x16_R4x4;
                cb.SetComputeIntParam(fn.shader, k_ID_O_width, O.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_width, X.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_width, K.shape[2]);
                workItemsX = ComputeHelper.IDivC(k, 4);
                workItemsY = ComputeHelper.IDivC(w, 4);
                workItemsZ = n;
            }

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Kptr, Pin(K));
            if (B != null)
            {
                cb.EnableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            }
            else
            {
                cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
            }
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetComputeIntParam(fn.shader, k_ID_O_batch, O.shape[0]);
            cb.SetComputeIntParam(fn.shader, k_ID_O_channels, O.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_X_channels, X.shape[1]);
            cb.SetInt4(fn, k_ID__Stride, strides);
            cb.SetInt4(fn, k_ID__Pad, pads);
            cb.SetInt4(fn, k_ID__Dilation, dilations);

            cb.SetComputeFloatParam(fn.shader, k_ID__MinValue, fusedActivation == Layers.FusableActivation.Relu ? 0.0f : float.MinValue);

            cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);
        }

        /// <inheritdoc/>
        public void ConvTranspose(Tensor<float> X, Tensor<float> W, Tensor<float> B, Tensor<float> O, Span<int> strides, Span<int> pads, Span<int> outputPadding, Layers.FusableActivation fusedActivation)
        {
            if (X.shape.rank > 5)
            {
                throw new NotImplementedException();
            }

            if (ComputeInfo.IsMobileGPU() || X.shape.rank > 4)
            {
                ConvTransposeMobile(X, W, B, O, strides, pads, outputPadding, fusedActivation);
                return;
            }

            ComputeFunction fn;

            var numSpatialDims = X.shape.rank - 2;

            if (numSpatialDims == 1)
                fn = ComputeFunctions.k_ConvTranspose1D_KxK;
            else
                fn = ComputeFunctions.k_ConvTranspose2D_KxK;

            var workItemsX = ComputeHelper.IDivC(O.shape.Length(2), 4);
            var workItemsY = ComputeHelper.IDivC(O.shape[1], 8);
            var workItemsZ = O.shape[0];

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(W));
            if (B != null)
            {
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.EnableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
            }
            else
            {
                cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
            }
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetComputeIntParam(fn.shader, k_ID_inputChannels, X.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_outputChannels, O.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_dilationHeight, 1);
            cb.SetComputeIntParam(fn.shader, k_ID_dilationWidth, 1);

            var kernelSize = W.shape.Length(2);
            var inputSize = X.shape.Length(2);
            var outputSize = O.shape.Length(2);
            cb.SetComputeIntParam(fn.shader, k_ID_kernelLength, W.shape.length);
            cb.SetComputeIntParam(fn.shader, k_ID_kernelSize, kernelSize);
            cb.SetComputeIntParam(fn.shader, k_ID_inputSize, inputSize);
            cb.SetComputeIntParam(fn.shader, k_ID_outputSize, outputSize);
            cb.SetComputeIntParam(fn.shader, k_ID_inputChannelsSize, X.shape[1] * inputSize);
            cb.SetComputeIntParam(fn.shader, k_ID_outputChannelsSize, O.shape[1] * outputSize);
            cb.SetComputeIntParam(fn.shader, k_ID_kernelChannelSize, W.shape[0] * kernelSize);
            cb.SetComputeIntParam(fn.shader, k_ID_inputWidth, X.shape[-1]);
            cb.SetComputeIntParam(fn.shader, k_ID_kernelWidth, W.shape[-1]);
            cb.SetComputeIntParam(fn.shader, k_ID_outputWidth, O.shape[-1]);
            cb.SetComputeIntParam(fn.shader, k_ID_padWidth, W.shape[-1] - pads[numSpatialDims - 1] - 1);
            cb.SetComputeIntParam(fn.shader, k_ID_strideWidth, strides[numSpatialDims - 1]);
            if (numSpatialDims > 1)
            {
                cb.SetComputeIntParam(fn.shader, k_ID_inputHeight, X.shape[-2]);
                cb.SetComputeIntParam(fn.shader, k_ID_kernelHeight, W.shape[-2]);
                cb.SetComputeIntParam(fn.shader, k_ID_outputHeight, O.shape[-2]);
                cb.SetComputeIntParam(fn.shader, k_ID_padHeight, W.shape[-2] - pads[numSpatialDims - 2] - 1);
                cb.SetComputeIntParam(fn.shader, k_ID_strideHeight, strides[numSpatialDims - 2]);
            }

            cb.SetComputeFloatParam(fn.shader, k_ID__MinValue, fusedActivation == Layers.FusableActivation.Relu ? 0.0f : float.MinValue);

            cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);
        }

        void ConvTransposeMobile(Tensor<float> X, Tensor<float> W, Tensor<float> B, Tensor<float> O, Span<int> stride, Span<int> pad, Span<int> outputAdjustment, Layers.FusableActivation fusedActivation)
        {
            ComputeFunction fn;

            var numSpatialDims = X.shape.rank - 2;

            if (numSpatialDims == 1)
                fn = ComputeFunctions.k_ConvTranspose1D_T16x16_R4x4;
            else if (numSpatialDims == 2)
                fn = ComputeFunctions.k_ConvTranspose2D_T16x16_R4x4;
            else
                fn = ComputeFunctions.k_ConvTranspose3D_T16x16_R4x4;

            cb.SetComputeIntParam(fn.shader, k_ID_O_channels, O.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_X_channels, X.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_maxXIndex, X.shape.length - 1);
            cb.SetComputeIntParam(fn.shader, k_ID_maxKIndex, W.shape.length - 1);
            cb.SetInt4(fn, k_ID__Pad, pad);
            cb.SetInt4(fn, k_ID__Stride, stride);

            cb.SetComputeIntParam(fn.shader, k_ID_O_width, O.shape[-1]);
            cb.SetComputeIntParam(fn.shader, k_ID_X_width, X.shape[-1]);
            cb.SetComputeIntParam(fn.shader, k_ID_K_width, W.shape[-1]);

            if (numSpatialDims > 1)
            {
                cb.SetComputeIntParam(fn.shader, k_ID_O_height, O.shape[-2]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_height, X.shape[-2]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_height, W.shape[-2]);
            }

            if (numSpatialDims > 2)
            {
                cb.SetComputeIntParam(fn.shader, k_ID_O_depth, O.shape[-3]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_depth, X.shape[-3]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_depth, W.shape[-3]);
            }

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Kptr, Pin(W));
            if (B != null)
            {
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetComputeIntParam(fn.shader, k_ID_maxBIndex, B.shape.length - 1);
                cb.EnableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
            }
            else
            {
                cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
            }
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            var workItemsX = ComputeHelper.IDivC(O.shape[1], 4);
            var workItemsY = ComputeHelper.IDivC(O.shape.Length(2), 4);
            var workItemsZ = O.shape[0];
            if (fusedActivation == Layers.FusableActivation.Relu)
                cb.SetComputeFloatParam(fn.shader, k_ID__MinValue, 0.0f);
            else
                cb.SetComputeFloatParam(fn.shader, k_ID__MinValue, float.MinValue);

            cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);
        }

        /// <inheritdoc/>
        public void Resize(Tensor<float> X, Tensor<float> O, ReadOnlySpan<float> scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode, Layers.CoordTransformMode coordTransformMode)
        {
            if (X.shape.rank > 5 || scale[0] != 1f || scale[1] != 1f)
            {
                ResizeND(X, O, scale, interpolationMode, nearestMode, coordTransformMode);
                return;
            }

            switch (X.shape.rank)
            {
                case 3:
                    Upsample1D(X, O, scale, nearestMode, interpolationMode, coordTransformMode);
                    break;
                case 4:
                    Upsample2D(X, O, scale, nearestMode, interpolationMode, coordTransformMode);
                    break;
                case 5:
                    Upsample3D(X, O, scale, nearestMode, interpolationMode, coordTransformMode);
                    break;
            }
        }

        /// <inheritdoc/>
        public void GridSample(Tensor<float> X, Tensor<float> grid, Tensor<float> O, Layers.InterpolationMode mode, Layers.PaddingMode paddingMode, bool alignCorners)
        {
            int n = O.shape[0]; int c = O.shape[1];
            int oH = O.shape[-2]; int oW = O.shape[-1];
            int xH = X.shape[-2]; int xW = X.shape[-1];
            int oSpatialDim = oH * oW;
            int xSpatialDim = xH * xW;

            ComputeFunction fn = null;
            switch (X.shape.rank)
            {
                case 4:
                    fn = ComputeFunctions.k_GridSample2D;
                    break;
                case 5:
                    fn = ComputeFunctions.k_GridSample3D;
                    int oD = O.shape[2];
                    int xD = X.shape[2];
                    oSpatialDim *= oD;
                    xSpatialDim *= xD;

                    cb.SetComputeIntParam(fn.shader, k_ID_inDepth, xD);
                    cb.SetComputeIntParam(fn.shader, k_ID_outDepth, oD);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            cb.SetComputeIntParam(fn.shader, k_ID_outBatch, n);
            cb.SetComputeIntParam(fn.shader, k_ID_outChannels, c);
            cb.SetComputeIntParam(fn.shader, k_ID_inHeight, xH);
            cb.SetComputeIntParam(fn.shader, k_ID_inWidth, xW);
            cb.SetComputeIntParam(fn.shader, k_ID_outHeight, oH);
            cb.SetComputeIntParam(fn.shader, k_ID_outWidth, oW);
            cb.SetComputeIntParam(fn.shader, k_ID_inSpatialSize, xSpatialDim);
            cb.SetComputeIntParam(fn.shader, k_ID_outSpatialSize, oSpatialDim);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Sptr, Pin(grid));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            switch (mode)
            {
                case Layers.InterpolationMode.Nearest:
                    cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "LINEAR"));
                    break;
                case Layers.InterpolationMode.Linear:
                    cb.EnableKeyword(fn.shader, new LocalKeyword(fn.shader, "LINEAR"));
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(mode), mode, null);
            }

            switch (paddingMode)
            {
                case Layers.PaddingMode.Zeros:
                    cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "BORDER"));
                    cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "REFLECTION"));
                    break;
                case Layers.PaddingMode.Border:
                    cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "REFLECTION"));
                    cb.EnableKeyword(fn.shader, new LocalKeyword(fn.shader, "BORDER"));
                    break;
                case Layers.PaddingMode.Reflection:
                    cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "BORDER"));
                    cb.EnableKeyword(fn.shader, new LocalKeyword(fn.shader, "REFLECTION"));
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(paddingMode), paddingMode, null);
            }

            if (alignCorners)
                cb.EnableKeyword(fn.shader, new LocalKeyword(fn.shader, "ALIGN_CORNERS"));
            else
                cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "ALIGN_CORNERS"));

            cb.Dispatch(fn, oSpatialDim, c, n);
        }

        void ResizeND(Tensor<float> X, Tensor<float> O, ReadOnlySpan<float> scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode, Layers.CoordTransformMode coordTransformMode)
        {
            // calculate first and last axes with scaling
            var firstScaleAxis = scale.Length;
            var lastScaleAxis = 0;
            for (var i = 0; i < scale.Length; i++)
            {
                if (scale[i] != 1f)
                {
                    firstScaleAxis = Mathf.Min(firstScaleAxis, i);
                    lastScaleAxis = Mathf.Max(lastScaleAxis, i);
                }
            }

            if (firstScaleAxis > lastScaleAxis)
            {
                // no scale
                MemCopy(X, O);
                return;
            }

            for (var i = firstScaleAxis; i <= lastScaleAxis; i++)
            {
                if (scale[i] == 1f)
                    continue;
                var oCurr = i == lastScaleAxis ? O : AllocTensorFloat(ShapeInference.Resize(X.shape, i, scale[i]));
                Resize1D(X, oCurr, i, scale[i], interpolationMode, nearestMode, coordTransformMode);
                if (i != firstScaleAxis)
                    ReleaseTensorFloat(X);
                X = oCurr;
            }
        }

        void Resize1D(Tensor<float> X, Tensor<float> O, int axis, float scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode, Layers.CoordTransformMode coordTransformMode)
        {
            OpsUtils.GetScaleAndBias(X.shape[axis], O.shape[axis], scale, coordTransformMode, interpolationMode, nearestMode, out float outputScale, out float outputBias);

            ComputeFunction fn;
            if (interpolationMode == Layers.InterpolationMode.Nearest)
            {
                switch (nearestMode)
                {
                    case Layers.NearestMode.RoundPreferFloor:
                    case Layers.NearestMode.Ceil:
                        fn = ComputeFunctions.k_Resize1D_Nearest_Ceil;
                        break;
                    case Layers.NearestMode.RoundPreferCeil:
                    case Layers.NearestMode.Floor:
                        fn = ComputeFunctions.k_Resize1D_Nearest_Floor;
                        break;
                    default:
                        throw new NotImplementedException();
                }
            }
            else //if (interpolationMode == Layers.InterpolationMode.Linear)
            {
                fn = ComputeFunctions.k_Resize1D_Linear_None;
            }

            int innerLength = O.shape.Strides(axis);
            int outerLength = O.shape.Length(0, axis);

            cb.SetComputeFloatParam(fn.shader, k_ID_scale1D, outputScale);
            cb.SetComputeFloatParam(fn.shader, k_ID_bias1D, outputBias);
            cb.SetComputeIntParam(fn.shader, k_ID_innerLength, innerLength);
            cb.SetComputeIntParam(fn.shader, k_ID_outerLength, outerLength);
            cb.SetComputeIntParam(fn.shader, k_ID_inWidth, X.shape[axis]);
            cb.SetComputeIntParam(fn.shader, k_ID_outWidth, O.shape[axis]);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, outerLength, O.shape[axis], innerLength);
        }

        void Upsample1D(Tensor<float> X, Tensor<float> O, ReadOnlySpan<float> scale, Layers.NearestMode nearestMode, Layers.InterpolationMode interpolationMode, Layers.CoordTransformMode coordTransformMode)
        {
            OpsUtils.GetScaleAndBias(X.shape[2], O.shape[2], scale[2], coordTransformMode, interpolationMode, nearestMode, out float outputScale, out float outputBias);

            ComputeFunction fn;
            if (interpolationMode == Layers.InterpolationMode.Nearest)
            {
                switch (nearestMode)
                {
                    case Layers.NearestMode.RoundPreferFloor:
                    case Layers.NearestMode.Ceil:
                        fn = ComputeFunctions.k_Upsample1D_Nearest_Ceil;
                        break;
                    case Layers.NearestMode.RoundPreferCeil:
                    case Layers.NearestMode.Floor:
                        fn = ComputeFunctions.k_Upsample1D_Nearest_Floor;
                        break;
                    default:
                        throw new NotImplementedException();
                }
            }
            else //if (interpolationMode == Layers.InterpolationMode.Linear)
            {
                fn = ComputeFunctions.k_Upsample1D_Linear_None;
            }

            cb.SetComputeFloatParam(fn.shader, k_ID_scale1D, outputScale);
            cb.SetComputeFloatParam(fn.shader, k_ID_bias1D, outputBias);
            cb.SetComputeIntParam(fn.shader, k_ID_inWidth, X.shape[2]);
            cb.SetComputeIntParam(fn.shader, k_ID_outWidth, O.shape[2]);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, O.shape[0] * O.shape[1], O.shape[2], 1);
        }

        void Upsample2D(Tensor<float> X, Tensor<float> O, ReadOnlySpan<float> scale, Layers.NearestMode nearestMode, Layers.InterpolationMode interpolationMode, Layers.CoordTransformMode coordTransformMode)
        {
            Vector4 scaleXY = Vector4.zero;
            Vector4 biasXY = Vector4.zero;
            for (int i = 0; i < 2; i++)
            {
                OpsUtils.GetScaleAndBias(X.shape[2 + i], O.shape[2 + i], scale[2 + i], coordTransformMode, interpolationMode, nearestMode, out float outputScale, out float outputBias);
                scaleXY[i] = outputScale;
                biasXY[i] = outputBias;
            }

            ComputeFunction fn;
            if (interpolationMode == Layers.InterpolationMode.Nearest)
            {
                switch (nearestMode)
                {
                    case Layers.NearestMode.RoundPreferFloor:
                    case Layers.NearestMode.Ceil:
                        fn = ComputeFunctions.k_Upsample2D_Nearest_Ceil;
                        break;
                    case Layers.NearestMode.RoundPreferCeil:
                    case Layers.NearestMode.Floor:
                        fn = ComputeFunctions.k_Upsample2D_Nearest_Floor;
                        break;
                    default:
                        throw new NotImplementedException();
                }
            }
            else //if (interpolationMode == Layers.InterpolationMode.Linear)
            {
                fn = ComputeFunctions.k_Upsample2D_Linear_None;
            }

            cb.SetComputeVectorParam(fn.shader, k_ID_scale, scaleXY);
            cb.SetComputeVectorParam(fn.shader, k_ID_bias, biasXY);
            cb.SetComputeIntParam(fn.shader, k_ID_inHeight, X.shape[2]);
            cb.SetComputeIntParam(fn.shader, k_ID_inWidth, X.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_outHeight, O.shape[2]);
            cb.SetComputeIntParam(fn.shader, k_ID_outWidth, O.shape[3]);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, O.shape[0] * O.shape[1], O.shape[2], O.shape[3]);
        }

        void Upsample3D(Tensor<float> X, Tensor<float> O, ReadOnlySpan<float> scale, Layers.NearestMode nearestMode, Layers.InterpolationMode interpolationMode, Layers.CoordTransformMode coordTransformMode)
        {
            Vector4 scaleXYD = Vector4.zero;
            Vector4 biasXYD = Vector4.zero;
            for (int i = 0; i < 3; i++)
            {
                OpsUtils.GetScaleAndBias(X.shape[2 + i], O.shape[2 + i], scale[2 + i], coordTransformMode, interpolationMode, nearestMode, out float outputScale, out float outputBias);
                scaleXYD[i] = outputScale;
                biasXYD[i] = outputBias;
            }

            ComputeFunction fn;
            if (interpolationMode == Layers.InterpolationMode.Nearest)
            {
                switch (nearestMode)
                {
                    case Layers.NearestMode.RoundPreferFloor:
                    case Layers.NearestMode.Ceil:
                        fn = ComputeFunctions.k_Upsample3D_Nearest_Ceil;
                        break;
                    case Layers.NearestMode.RoundPreferCeil:
                    case Layers.NearestMode.Floor:
                        fn = ComputeFunctions.k_Upsample3D_Nearest_Floor;
                        break;
                    default:
                        throw new NotImplementedException();
                }
            }
            else //if (interpolationMode == Layers.InterpolationMode.Linear)
            {
                fn = ComputeFunctions.k_Upsample3D_Linear_None;
            }

            cb.SetComputeVectorParam(fn.shader, k_ID_scale, scaleXYD);
            cb.SetComputeVectorParam(fn.shader, k_ID_bias, biasXYD);
            cb.SetComputeIntParam(fn.shader, k_ID_inDepth, X.shape[2]);
            cb.SetComputeIntParam(fn.shader, k_ID_inHeight, X.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_inWidth, X.shape[4]);
            cb.SetComputeIntParam(fn.shader, k_ID_outBatch, O.shape[0]);
            cb.SetComputeIntParam(fn.shader, k_ID_outChannels, O.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_outDepth, O.shape[2]);
            cb.SetComputeIntParam(fn.shader, k_ID_outHeight, O.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_outWidth, O.shape[4]);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, O.shape[2], O.shape[3], O.shape[4]);
        }

        /// <inheritdoc/>
        public void DepthToSpace(Tensor<float> X, Tensor<float> O, int blocksize, Layers.DepthToSpaceMode mode)
        {
            var fn = (mode == Layers.DepthToSpaceMode.DepthColumnRow) ? ComputeFunctions.k_DepthToSpaceDepthColumnRow : ComputeFunctions.k_DepthToSpaceColumnRowDepth;
            cb.SetComputeIntParam(fn.shader, k_ID_blocksize, blocksize);
            cb.SetComputeIntParam(fn.shader, k_ID_inputChannels, X.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_inputHeight, X.shape[2]);
            cb.SetComputeIntParam(fn.shader, k_ID_inputWidth, X.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_inputSpatialSize, X.shape[2] * X.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_inputBatchOffset, X.shape[1] * X.shape[2] * X.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_outputChannels, O.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_outputHeight, O.shape[2]);
            cb.SetComputeIntParam(fn.shader, k_ID_outputWidth, O.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_outputSpatialSize, O.shape[2] * O.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_outputBatch, O.shape[0]);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, O.shape[0] * O.shape[1], O.shape[2] * O.shape[3], 1);
        }

        /// <inheritdoc/>
        public void SpaceToDepth(Tensor<float> X, Tensor<float> O, int blocksize)
        {
            var fn = ComputeFunctions.k_SpaceToDepth;
            cb.SetComputeIntParam(fn.shader, k_ID_blocksize, blocksize);
            cb.SetComputeIntParam(fn.shader, k_ID_inputChannels, X.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_inputHeight, X.shape[2]);
            cb.SetComputeIntParam(fn.shader, k_ID_inputWidth, X.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_inputSpatialSize, X.shape[2] * X.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_inputBatchOffset, X.shape[1] * X.shape[2] * X.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_outputChannels, O.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_outputHeight, O.shape[2]);
            cb.SetComputeIntParam(fn.shader, k_ID_outputWidth, O.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_outputSpatialSize, O.shape[2] * O.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_outputBatch, O.shape[0]);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, O.shape[0] * O.shape[1], O.shape[2] * O.shape[3], 1);
        }

        /// <inheritdoc/>
        public void GlobalAverageVariancePool(Tensor<float> X, Tensor<float> O, int axis)
        {
            int globalNonSpatialLength = X.shape.Length(0, axis);
            int globalSpatialDims = X.shape.length / globalNonSpatialLength;

            int localSpatialLength = globalSpatialDims;

            var Oshape = new TensorShape(globalNonSpatialLength, localSpatialLength);

            Tensor<float> X2 = X; // save a X^2 and do it in the first dispatch
            bool isFirstDispatch = true;

            // downsample with pyramid approach
            while (localSpatialLength > 64 * 4)
            {
                int spatialLengthO = ComputeHelper.IDivC(localSpatialLength, 64 * 4);
                Oshape[-1] = spatialLengthO;
                var Otemp = AllocTensorFloat(Oshape);
                var O2temp = AllocTensorFloat(Oshape);

                var fnPool = ComputeFunctions.k_AverageVariancePoolReduce;
                cb.SetTensorAsBuffer(fnPool, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fnPool, k_ID_X2ptr, Pin(X2));
                cb.SetTensorAsBuffer(fnPool, k_ID_Optr, Pin(Otemp));
                cb.SetTensorAsBuffer(fnPool, k_ID_O2ptr, Pin(O2temp));
                cb.SetComputeIntParam(fnPool.shader, k_ID_SpatialDims, localSpatialLength);
                cb.SetComputeIntParam(fnPool.shader, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetComputeIntParam(fnPool.shader, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(fnPool, globalNonSpatialLength, ComputeHelper.IDivC(localSpatialLength, 4), 1);

                if (!isFirstDispatch)
                {
                    ReleaseTensorFloat(X);
                    ReleaseTensorFloat(X2);
                }
                X = Otemp;
                X2 = O2temp;
                localSpatialLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var fn = ComputeFunctions.k_GlobalAverageVariancePool;
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_X2ptr, Pin(X2));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetComputeIntParam(fn.shader, k_ID_SpatialDims, localSpatialLength);
            cb.SetComputeIntParam(fn.shader, k_ID_GlobalSpatialDims, globalSpatialDims);
            cb.SetComputeIntParam(fn.shader, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

            cb.Dispatch(fn, globalNonSpatialLength, 1, 1);

            if (!isFirstDispatch)
            {
                ReleaseTensorFloat(X);
                ReleaseTensorFloat(X2);
            }
        }

        void GroupedConv(Tensor<float> X, Tensor<float> K, Tensor<float> B, Tensor<float> O, int groups, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
        {
            var Otmp = (fusedActivation != Layers.FusableActivation.None) ? AllocTensorFloat(O.shape) : O;

            int outputGroupedChannels = Otmp.shape[1] / groups;

            ComputeFunction fn;

            if (X.shape.rank == 5)
            {
                fn = (outputGroupedChannels < 64) ? ComputeFunctions.k_GroupedConv3D : ComputeFunctions.k_GroupedConv3D_GroupLower64;
                cb.SetComputeIntParam(fn.shader, k_ID_O_depth, Otmp.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_O_height, Otmp.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_O_width, Otmp.shape[4]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_depth, X.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_height, X.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_width, X.shape[4]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_depth, K.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_height, K.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_width, K.shape[4]);
            }
            else if (X.shape.rank == 4)
            {
                fn = (outputGroupedChannels < 64) ? ComputeFunctions.k_GroupedConv2D : ComputeFunctions.k_GroupedConv2D_GroupLower64;
                cb.SetComputeIntParam(fn.shader, k_ID_O_height, Otmp.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_O_width, Otmp.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_height, X.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_width, X.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_height, K.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_width, K.shape[3]);
            }
            else
            {
                fn = (outputGroupedChannels < 64) ? ComputeFunctions.k_GroupedConv1D : ComputeFunctions.k_GroupedConv1D_GroupLower64;
                cb.SetComputeIntParam(fn.shader, k_ID_O_width, Otmp.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_X_width, X.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_width, K.shape[2]);
            }

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Kptr, Pin(K));
            if (B != null)
            {
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.EnableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
            }
            else
            {
                cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
            }
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(Otmp));
            cb.SetComputeIntParam(fn.shader, k_ID_O_channels, O.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_X_channels, X.shape[1]);
            cb.SetInt4(fn, k_ID__Stride, strides);
            cb.SetInt4(fn, k_ID__Pad, pads);
            cb.SetInt4(fn, k_ID__Dilation, dilations);
            cb.SetComputeIntParam(fn.shader, k_ID__Groups, groups);
            cb.SetComputeIntParam(fn.shader, k_ID_strideX, X.shape.Length(2));
            cb.SetComputeIntParam(fn.shader, k_ID_strideO, Otmp.shape.Length(2));
            cb.SetComputeIntParam(fn.shader, k_ID_strideK, K.shape.Length(2));
            cb.SetComputeIntParam(fn.shader, k_ID_inputGroupedChannels, X.shape[1] / groups);
            cb.SetComputeIntParam(fn.shader, k_ID_outputGroupedChannels, Otmp.shape[1] / groups);

            cb.Dispatch(fn, ComputeHelper.IDivC(Otmp.shape[1], 4), ComputeHelper.IDivC(Otmp.shape.Length(2), 4), Otmp.shape[0]);

            if (fusedActivation != Layers.FusableActivation.None)
            {
                ApplyFusedActivation(Otmp, O, fusedActivation);
                ReleaseTensorFloat(Otmp);
            }
        }

        void DepthwiseConv2D(Tensor<float> X, Tensor<float> K, Tensor<float> B, Tensor<float> O, int group, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
        {
            var Otmp = (fusedActivation != Layers.FusableActivation.None) ? AllocTensorFloat(O.shape) : O;

            ComputeFunction fn;
            int workItemsX, workItemsY, workItemsZ;

            Tensor<float> KWE = null;
            if (K.shape[2] == 3 && K.shape[3] == 3 && strides[0] == 1 && strides[1] == 1 && dilations[0] == 1 && dilations[1] == 1)
            {
                KWE = AllocTensorFloat(new TensorShape(Otmp.shape[1], 4, 4));

                ComputeFunction fnKE = ComputeFunctions.k_KernelWinoExpand;
                cb.SetTensorAsBuffer(fnKE, k_ID_Kptr, Pin(K));
                cb.SetTensorAsBuffer(fnKE, k_ID_Optr, Pin(KWE));
                cb.SetComputeIntParam(fnKE.shader, k_ID_O_channels, O.shape[1]);
                cb.Dispatch(fnKE, O.shape[1], 1, 1);

                fn = ComputeFunctions.k_DepthwiseConv2DWinograd;

                cb.SetTensorAsBuffer(fn, k_ID_KWEptr, Pin(KWE));

                workItemsX = ComputeHelper.IDivC(Otmp.shape[3], 2);
                workItemsY = ComputeHelper.IDivC(Otmp.shape[2], 2);
                workItemsZ = Otmp.shape[0] * Otmp.shape[1];
            }
            else
            {
                fn = ComputeFunctions.k_DepthwiseConv2DDirect;

                cb.SetTensorAsBuffer(fn, k_ID_Kptr, Pin(K));

                cb.SetComputeIntParam(fn.shader, k_ID_K_heightDiv4, ComputeHelper.IDivC(K.shape[2], 4));
                cb.SetComputeIntParam(fn.shader, k_ID_K_widthDiv4, ComputeHelper.IDivC(K.shape[3], 4));
                cb.SetComputeIntParam(fn.shader, k_ID_K_height, K.shape[2]);
                cb.SetComputeIntParam(fn.shader, k_ID_K_width, K.shape[3]);
                cb.SetComputeIntParam(fn.shader, k_ID_StrideK, K.shape[2] * K.shape[3]);

                workItemsX = Otmp.shape[3];
                workItemsY = Otmp.shape[2];
                workItemsZ = Otmp.shape[0] * Otmp.shape[1];
            }

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            if (B != null)
            {
                cb.EnableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            }
            else
            {
                cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "USEBIAS"));
            }
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(Otmp));
            cb.SetComputeIntParam(fn.shader, k_ID_X_channels, X.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_X_height, X.shape[2]);
            cb.SetComputeIntParam(fn.shader, k_ID_X_width, X.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_O_batch, O.shape[0]);
            cb.SetComputeIntParam(fn.shader, k_ID_O_channels, O.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_O_height, O.shape[2]);
            cb.SetComputeIntParam(fn.shader, k_ID_O_width, O.shape[3]);
            cb.SetInt4(fn, k_ID_Stride, strides);
            cb.SetInt4(fn, k_ID_Pad, pads);
            cb.SetInt4(fn, k_ID_Dilation, dilations);
            cb.SetComputeIntParam(fn.shader, k_ID_StrideX, X.shape[2] * X.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_MaxLengthX, X.shape.length - 1);
            cb.SetComputeIntParam(fn.shader, k_ID_MaxLengthK, K.shape.length - 1);
            cb.SetComputeIntParam(fn.shader, k_ID_StrideO, Otmp.shape[2] * Otmp.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_StrideFeaturesO, Otmp.shape[0] * Otmp.shape[1]);

            cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);
            ReleaseTensorFloat(KWE);

            if (fusedActivation != Layers.FusableActivation.None)
            {
                ApplyFusedActivation(Otmp, O, fusedActivation);
                ReleaseTensorFloat(Otmp);
            }
        }

        /// <inheritdoc/>
        public void ScaleBias(Tensor<float> X, Tensor<float> S, Tensor<float> B, Tensor<float> O)
        {
            int batch = X.shape[0];
            int channels = X.shape[1];
            int spatialDims = X.shape.Length(2);

            var fn = ComputeFunctions.k_ScaleBias;

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Sptr, Pin(S));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetComputeIntParam(fn.shader, k_ID_LengthO, O.shape.length);
            cb.SetComputeIntParam(fn.shader, k_ID_batch, batch);
            cb.SetComputeIntParam(fn.shader, k_ID_channels, channels);
            cb.SetComputeIntParam(fn.shader, k_ID_spatialDims, spatialDims);
            cb.Dispatch(fn, spatialDims, ComputeHelper.IDivC(channels, 4), batch);
        }

        /// <inheritdoc/>
        public void InstanceNormalization(Tensor<float> X, Tensor<float> S, Tensor<float> B, Tensor<float> O, float epsilon)
        {
            var reduceOpShape = ShapeInference.GlobalAverageVariancePool(X.shape);
            var meanVariance = AllocTensorFloat(reduceOpShape);
            GlobalAverageVariancePool(X, meanVariance, 2);

            var fn = ComputeFunctions.k_InstanceNormalizationTail;

            cb.SetComputeIntParam(fn.shader, k_ID_channels, X.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_spatialDims, X.shape.length / (X.shape[0] * X.shape[1]));
            cb.SetComputeFloatParam(fn.shader, k_ID_epsilon, epsilon);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Sptr, Pin(S));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(meanVariance));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.UnrolledDispatch(fn, O.shape.length);
            ReleaseTensorFloat(meanVariance);
        }

        /// <inheritdoc/>
        public void LayerNormalization(Tensor<float> X, Tensor<float> S, Tensor<float> B, Tensor<float> O, float epsilon)
        {
            int axis = X.shape.Axis(-1);

            var reducedShape = X.shape.Reduce(axis);
            reducedShape[axis] = 2;

            int axisDim = X.shape[axis];
            int outerLength = X.shape.Length(0, -1);

            var meanVariance = AllocTensorFloat(reducedShape);
            GlobalAverageVariancePool(X, meanVariance, -1);

            var fn = ComputeFunctions.k_LayerNormalizationTail;
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(meanVariance));
            cb.SetTensorAsBuffer(fn, k_ID_Sptr, Pin(S));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetComputeIntParam(fn.shader, k_ID_axisDim, axisDim);
            cb.SetComputeIntParam(fn.shader, k_ID_outerLength, outerLength);
            cb.SetComputeFloatParam(fn.shader, k_ID_epsilon, epsilon);
            cb.Dispatch(fn, axisDim, outerLength, 1);

            ReleaseTensorFloat(meanVariance);
        }

        /// <inheritdoc/>
        public void RMSNormalization(Tensor<float> X, Tensor<float> S, Tensor<float> O, float epsilon)
        {
            int axis = X.shape.Axis(-1);

            var reducedShape = X.shape.Reduce(axis);
            int axisDim = X.shape[axis];
            int outerLength = X.shape.Length(0, -1);

            var meanSquared = AllocTensorFloat(reducedShape);
            ReduceMeanSquare(X, meanSquared, outerLength, axisDim, 1);

            var fn = ComputeFunctions.k_RMSNormalizationTail;
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(meanSquared));
            cb.SetTensorAsBuffer(fn, k_ID_Sptr, Pin(S));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetComputeIntParam(fn.shader, k_ID_axisDim, axisDim);
            cb.SetComputeIntParam(fn.shader, k_ID_outerLength, outerLength);
            cb.SetComputeFloatParam(fn.shader, k_ID_epsilon, epsilon);
            cb.Dispatch(fn, axisDim, outerLength, 1);

            ReleaseTensorFloat(meanSquared);
        }

        /// <inheritdoc/>
        public void BatchNormalization(Tensor<float> X, Tensor<float> S, Tensor<float> B, Tensor<float> mean, Tensor<float> variance, Tensor<float> O, float epsilon)
        {
            var batch = X.shape[0];
            var channels = X.shape.rank == 1 ? 1 : X.shape[1];
            var spatialDims = X.shape.Length(2);

            var fn = ComputeFunctions.k_BatchNormalization;

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(mean));
            cb.SetTensorAsBuffer(fn, k_ID_Zptr, Pin(variance));
            cb.SetTensorAsBuffer(fn, k_ID_Sptr, Pin(S));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetComputeIntParam(fn.shader, k_ID_LengthO, O.shape.length);
            cb.SetComputeIntParam(fn.shader, k_ID_batch, batch);
            cb.SetComputeIntParam(fn.shader, k_ID_channels, channels);
            cb.SetComputeIntParam(fn.shader, k_ID_spatialDims, spatialDims);
            cb.SetComputeFloatParam(fn.shader, k_ID_epsilon, epsilon);
            cb.Dispatch(fn, spatialDims, ComputeHelper.IDivC(channels, 4), batch);
        }

        /// <inheritdoc/>
        public void Range(Tensor<float> O, float start, float delta)
        {
            var fn = ComputeFunctions.k_RangeFloat;
            cb.SetComputeFloatParam(fn.shader, k_ID_alpha, start);
            cb.SetComputeFloatParam(fn.shader, k_ID_beta, delta);
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Range(Tensor<int> O, int start, int delta)
        {
            var fn = ComputeFunctions.k_RangeInt;
            cb.SetComputeIntParam(fn.shader, k_ID_alphai, start);
            cb.SetComputeIntParam(fn.shader, k_ID_betai, delta);
            cb.SetTensorAsBuffer(fn, k_ID_O_int_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Relu(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Relu;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Relu6(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Relu6;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void LeakyRelu(Tensor<float> X, Tensor<float> O, float alpha)
        {
            var fn = ComputeFunctions.k_LeakyRelu;
            cb.SetComputeFloatParam(fn.shader, k_ID_alpha, 0.5f * (1f + alpha));
            cb.SetComputeFloatParam(fn.shader, k_ID_beta, 0.5f * (1f - alpha));
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Tanh(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Tanh;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Softplus(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Softplus;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Sigmoid(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Sigmoid;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void HardSigmoid(Tensor<float> X, Tensor<float> O, float alpha, float beta)
        {
            var fn = ComputeFunctions.k_HardSigmoid;
            cb.SetComputeFloatParam(fn.shader, k_ID_alpha, alpha);
            cb.SetComputeFloatParam(fn.shader, k_ID_beta, beta);
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Elu(Tensor<float> X, Tensor<float> O, float alpha)
        {
            var fn = ComputeFunctions.k_Elu;
            cb.SetComputeFloatParam(fn.shader, k_ID_alpha, alpha);
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Gelu(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Gelu;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void GeluFast(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_GeluFast;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Selu(Tensor<float> X, Tensor<float> O, float alpha, float gamma)
        {
            var fn = ComputeFunctions.k_Selu;
            cb.SetComputeFloatParam(fn.shader, k_ID_alpha, alpha);
            cb.SetComputeFloatParam(fn.shader, k_ID_gamma, gamma);
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Swish(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Swish;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Abs(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_AbsFloat;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Abs(Tensor<int> X, Tensor<int> O)
        {
            var fn = ComputeFunctions.k_AbsInt;
            cb.SetTensorAsBuffer(fn, k_ID_X_int_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_int_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Neg(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_NegFloat;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Neg(Tensor<int> X, Tensor<int> O)
        {
            var fn = ComputeFunctions.k_NegInt;
            cb.SetTensorAsBuffer(fn, k_ID_X_int_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_int_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Ceil(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Ceil;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Clip(Tensor<float> X, Tensor<float> O, float min, float max)
        {
            var fn = ComputeFunctions.k_ClipFloat;
            cb.SetComputeFloatParam(fn.shader, k_ID_alpha, min);
            cb.SetComputeFloatParam(fn.shader, k_ID_beta, max);
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Clip(Tensor<int> X, Tensor<int> O, int min, int max)
        {
            var fn = ComputeFunctions.k_ClipInt;
            cb.SetComputeIntParam(fn.shader, k_ID_alphai, min);
            cb.SetComputeIntParam(fn.shader, k_ID_betai, max);
            cb.SetTensorAsBuffer(fn, k_ID_X_int_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_int_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Floor(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Floor;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Round(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Round;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Reciprocal(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Reciprocal;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Square(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_SquareFloat;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Square(Tensor<int> X, Tensor<int> O)
        {
            var fn = ComputeFunctions.k_SquareInt;
            cb.SetTensorAsBuffer(fn, k_ID_X_int_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_int_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Exp(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Exp;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Log(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Log;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Sqrt(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Sqrt;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Acos(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Acos;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Acosh(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Acosh;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Asin(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Asin;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Asinh(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Asinh;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Atan(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Atan;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Atanh(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Atanh;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Cos(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Cos;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Cosh(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Cosh;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Sin(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Sin;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Sinh(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Sinh;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Tan(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Tan;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Erf(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Erf;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Celu(Tensor<float> X, Tensor<float> O, float alpha)
        {
            var fn = ComputeFunctions.k_Celu;
            cb.SetComputeFloatParam(fn.shader, k_ID_alpha, alpha);
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Shrink(Tensor<float> X, Tensor<float> O, float bias, float lambd)
        {
            var fn = ComputeFunctions.k_Shrink;
            cb.SetComputeFloatParam(fn.shader, k_ID_alpha, bias);
            cb.SetComputeFloatParam(fn.shader, k_ID_beta, lambd);
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Softsign(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_Softsign;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void ThresholdedRelu(Tensor<float> X, Tensor<float> O, float alpha)
        {
            var fn = ComputeFunctions.k_ThresholdedRelu;
            cb.SetComputeFloatParam(fn.shader, k_ID_alpha, alpha);
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Softmax(Tensor<float> X, Tensor<float> O, int axis)
        {
            // Allocate temp tensors
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int reduceLength = X.shape[axis];

            var Xmax = AllocTensorFloat(new TensorShape(outerLength * innerLength));
            var XexpSums = AllocTensorFloat(Xmax.shape);

            // x_max = X.max(axis=1)
            // e_x_sum = Sum[exp(x[:,c] - x_max[:]), c]
            ReduceMax(X, Xmax, outerLength, reduceLength, innerLength);
            ReduceSumExp(X, Xmax, XexpSums, outerLength, reduceLength, innerLength);

            // exp(x[n,c] - x_max[n]) / e_x_sum[n]
            var fn = ComputeFunctions.k_SoftmaxEnd;
            cb.SetComputeIntParam(fn.shader, k_ID_innerLength, innerLength);
            cb.SetComputeIntParam(fn.shader, k_ID_reduceLength, reduceLength);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Sptr, Pin(XexpSums));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Xmax));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.UnrolledDispatch(fn, O.shape.length);

            ReleaseTensorFloat(Xmax);
            ReleaseTensorFloat(XexpSums);
        }

        /// <inheritdoc/>
        public void LogSoftmax(Tensor<float> X, Tensor<float> O, int axis)
        {
            // Allocate temp tensors
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int reduceLength = X.shape[axis];

            var Xmax = AllocTensorFloat(new TensorShape(outerLength * innerLength));
            var XexpSums = AllocTensorFloat(Xmax.shape);

            // x_max = X.max(axis=1)
            // logexp_sum = log(Sum[exp(x[:,c] - x_max[:]), c]) - x_max[:]
            ReduceMax(X, Xmax, outerLength, reduceLength, innerLength);
            ReduceLogSumExp(X, Xmax, XexpSums, outerLength, reduceLength, innerLength);

            // x[n,c] - logexp_sum
            var fn = ComputeFunctions.k_LogSoftmaxEnd;
            cb.SetComputeIntParam(fn.shader, k_ID_innerLength, innerLength);
            cb.SetComputeIntParam(fn.shader, k_ID_reduceLength, reduceLength);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(XexpSums));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.UnrolledDispatch(fn, O.shape.length);

            ReleaseTensorFloat(Xmax);
            ReleaseTensorFloat(XexpSums);
        }

        /// <inheritdoc/>
        public void Hardmax(Tensor<float> X, Tensor<float> O, int axis)
        {
            //Allocate temp tensors
            var reduceOpShape = X.shape.Reduce(axis);
            var argMax = AllocTensorFloat(reduceOpShape);

            int offsetReduce = X.shape.Strides(axis);

            // argmax
            {
                var fn = ComputeFunctions.k_ArgMaxFloatFirst;
                cb.SetComputeIntParam(fn.shader, k_ID_innerLength, offsetReduce);
                cb.SetComputeIntParam(fn.shader, k_ID_reduceLength, X.shape[axis]);
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(argMax));
                cb.UnrolledDispatch(fn, reduceOpShape.length);
            }
            // one hot from argmax
            {
                var fn = ComputeFunctions.k_HardmaxEnd;
                cb.SetComputeIntParam(fn.shader, k_ID_innerLength, offsetReduce);
                cb.SetComputeIntParam(fn.shader, k_ID_reduceLength, X.shape[axis]);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(argMax));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }

            ReleaseTensorFloat(argMax);
        }

        /// <inheritdoc/>
        public void ScalarMad(Tensor<float> X, Tensor<float> O, float s, float b)
        {
            var fn = ComputeFunctions.k_ScalarMadFloat;
            cb.SetComputeFloatParam(fn.shader, k_ID_alpha, s);
            cb.SetComputeFloatParam(fn.shader, k_ID_beta, b);
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void ScalarMad(Tensor<int> X, Tensor<int> O, int s, int b)
        {
            var fn = ComputeFunctions.k_ScalarMadInt;
            cb.SetComputeIntParam(fn.shader, k_ID_alphai, s);
            cb.SetComputeIntParam(fn.shader, k_ID_betai, b);
            cb.SetTensorAsBuffer(fn, k_ID_X_int_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_int_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void CumSum(Tensor<float> X, Tensor<float> O, int axis, bool reverse, bool exclusive)
        {
            var reduceOpShape = X.shape.Reduce(axis);
            var offsetReduce = X.shape.Strides(axis);

            var fn = (reverse ? (exclusive ? ComputeFunctions.k_CumSumFloatReverseExclusive : ComputeFunctions.k_CumSumFloatReverseInclusive) : (exclusive ? ComputeFunctions.k_CumSumFloatForwardExclusive : ComputeFunctions.k_CumSumFloatForwardInclusive));
            cb.SetComputeIntParam(fn.shader, k_ID_innerLength, offsetReduce);
            cb.SetComputeIntParam(fn.shader, k_ID_reduceLength, X.shape[axis]);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, reduceOpShape.length);
        }

        /// <inheritdoc/>
        public void CumSum(Tensor<int> X, Tensor<int> O, int axis, bool reverse, bool exclusive)
        {
            var reduceOpShape = X.shape.Reduce(axis);
            var offsetReduce = X.shape.Strides(axis);

            var fn = (reverse ? (exclusive ? ComputeFunctions.k_CumSumIntReverseExclusive : ComputeFunctions.k_CumSumIntReverseInclusive) : (exclusive ? ComputeFunctions.k_CumSumIntForwardExclusive : ComputeFunctions.k_CumSumIntForwardInclusive));
            cb.SetComputeIntParam(fn.shader, k_ID_innerLength, offsetReduce);
            cb.SetComputeIntParam(fn.shader, k_ID_reduceLength, X.shape[axis]);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, reduceOpShape.length);
        }

        /// <inheritdoc/>
        public void Einsum(Tensor<float>[] inputTensors, Tensor<float> O, TensorIndex[] operandIndices, TensorIndex outputIndices, TensorIndex sumIndices, TensorShape sumShape)
        {
            switch (inputTensors.Length)
            {
                case 1:
                {
                    var fn = ComputeFunctions.k_EinsumOne;

                    unsafe
                    {
                        var outStridesA = stackalloc int[TensorShape.maxRank];
                        var sumStridesA = stackalloc int[TensorShape.maxRank];
                        EinsumHelper.PinOperandStrides(inputTensors[0].shape, operandIndices[0], outputIndices, sumIndices, outStridesA, sumStridesA);
                        cb.SetInt8(fn, k_ID_outStridesA, outStridesA);
                        cb.SetInt8(fn, k_ID_sumStridesA, sumStridesA);

                        cb.SetTensorShapeStrides(fn, k_ID_outLengths, k_ID_outStrides, O.shape);
                        cb.SetTensorShapeStrides(fn, k_ID_sumLengths, k_ID_sumStrides, sumShape);
                    }

                    cb.SetComputeIntParam(fn.shader, k_ID_sumSize, sumShape.length);
                    cb.SetComputeIntParam(fn.shader, k_ID_sumRank, sumShape.rank);
                    cb.SetComputeIntParam(fn.shader, k_ID_outRank, O.shape.rank);

                    cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(inputTensors[0]));
                    cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                    cb.UnrolledDispatch(fn, O.shape.length);
                    return;
                }
                case 2:
                {
                    var fn = ComputeFunctions.k_EinsumTwo;

                    unsafe
                    {
                        var outStridesA = stackalloc int[TensorShape.maxRank];
                        var sumStridesA = stackalloc int[TensorShape.maxRank];
                        EinsumHelper.PinOperandStrides(inputTensors[0].shape, operandIndices[0], outputIndices, sumIndices, outStridesA, sumStridesA);
                        cb.SetInt8(fn, k_ID_outStridesA, outStridesA);
                        cb.SetInt8(fn, k_ID_sumStridesA, sumStridesA);

                        var outStridesB = stackalloc int[TensorShape.maxRank];
                        var sumStridesB = stackalloc int[TensorShape.maxRank];
                        EinsumHelper.PinOperandStrides(inputTensors[1].shape, operandIndices[1], outputIndices, sumIndices, outStridesB, sumStridesB);
                        cb.SetInt8(fn, k_ID_outStridesB, outStridesB);
                        cb.SetInt8(fn, k_ID_sumStridesB, sumStridesB);

                        cb.SetTensorShapeStrides(fn, k_ID_outLengths, k_ID_outStrides, O.shape);
                        cb.SetTensorShapeStrides(fn, k_ID_sumLengths, k_ID_sumStrides, sumShape);
                    }

                    cb.SetComputeIntParam(fn.shader, k_ID_sumSize, sumShape.length);
                    cb.SetComputeIntParam(fn.shader, k_ID_sumRank, sumShape.rank);
                    cb.SetComputeIntParam(fn.shader, k_ID_outRank, O.shape.rank);

                    cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(inputTensors[0]));
                    cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(inputTensors[1]));
                    cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                    cb.UnrolledDispatch(fn, O.shape.length);
                    return;
                }
            }
        }

        /// <summary>
        /// Performs `NonMaxSuppression` on boxes with scores.
        /// </summary>
        /// <param name="boxes">The boxes input tensor containing the position and dimensions of the boxes for each batch.</param>
        /// <param name="scores">The scores input tensor containing the score for each box per class per batch.</param>
        /// <param name="O">The output tensor to be computed and filled (and truncated) with the batch, class and index of the boxes in decreasing order of score.</param>
        /// <param name="maxOutputBoxesPerClass">The maximum number of output boxes per class.</param>
        /// <param name="iouThreshold">Boxes with intersect-over-union with a selected box above this threshold are discarded.</param>
        /// <param name="scoreThreshold">Boxes with a score below this threshold are discarded.</param>
        /// <param name="centerPointBox">The types of the box coordinates, either [x1, y1, x2, y2] or [x, y, w, h].</param>
        public void NonMaxSuppression(Tensor<float> boxes, Tensor<float> scores, Tensor<int> O, int maxOutputBoxesPerClass, float iouThreshold, float scoreThreshold, Layers.CenterPointBox centerPointBox)
        {
            // based on https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
            // extended to onnx multiple class multiple batch inputs as here
            // https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/object_detection/non_max_suppression.cc

            var numBatches = scores.shape[0];
            var numClasses = scores.shape[1];
            var numBoxes = scores.shape[2];
            var maxNumOutput = O.shape[0];

            var selected = AllocTensorInt(scores.shape);
            var bitmask = AllocTensorInt(new TensorShape(numBatches, numBoxes, numBoxes));

            // create bitmask of boxes
            {
                var fn = centerPointBox == Layers.CenterPointBox.Center ? ComputeFunctions.k_NMSBitmaskCenter : ComputeFunctions.k_NMSBitmaskCorners;
                cb.SetComputeIntParam(fn.shader, k_ID_numBatches, numBatches);
                cb.SetComputeIntParam(fn.shader, k_ID_numBoxes, numBoxes);
                cb.SetComputeFloatParam(fn.shader, k_ID_iouThreshold, iouThreshold);
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(boxes));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(bitmask));
                cb.Dispatch(fn, numBatches, numBoxes, numBoxes);
            }

            // sort
            var order = AllocTensorInt(new TensorShape(numBoxes));
            Range(order, 0, 1);
            var indicesSorted = AllocTensorInt(scores.shape);
            Span<int> tiles = stackalloc int[1]; tiles[0] = numBatches * numClasses;
            Tile(order, indicesSorted, tiles);
            var scoresSorted = AllocTensorFloat(scores.shape);
            MemCopy(scores, scoresSorted);
            BitonicSort(scoresSorted, indicesSorted, true);

            var bitmaskOverlap = AllocTensorInt(scores.shape);

            // selection
            {
                var fn = ComputeFunctions.k_NMSSelect;
                cb.SetComputeIntParam(fn.shader, k_ID_numBatches, numBatches);
                cb.SetComputeIntParam(fn.shader, k_ID_numBoxes, numBoxes);
                cb.SetComputeIntParam(fn.shader, k_ID_numClasses, numClasses);
                cb.SetComputeFloatParam(fn.shader, k_ID_scoreThreshold, scoreThreshold);
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(bitmask));
                cb.SetTensorAsBuffer(fn, k_ID_Sptr, Pin(scoresSorted));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(indicesSorted));
                cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(bitmaskOverlap));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(selected));
                cb.Dispatch(fn, 1, numClasses, numBatches);
            }

            // compaction
            var numSelected = AllocTensorInt(new TensorShape());
            {
                var fn = ComputeFunctions.k_NMSCompact;
                cb.SetComputeIntParam(fn.shader, k_ID_numBatches, numBatches);
                cb.SetComputeIntParam(fn.shader, k_ID_numBoxes, numBoxes);
                cb.SetComputeIntParam(fn.shader, k_ID_numClasses, numClasses);
                cb.SetComputeIntParam(fn.shader, k_ID_maxNumOutput, maxNumOutput);
                cb.SetComputeIntParam(fn.shader, k_ID_maxOutputBoxesPerClass, maxOutputBoxesPerClass);
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(selected));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetTensorAsBuffer(fn, k_ID_Iptr, Pin(numSelected));
                cb.Dispatch(fn, 1, 1, 1);
            }

            ReleaseTensorFloat(scoresSorted);
            ReleaseTensorInt(order);
            ReleaseTensorInt(indicesSorted);
            ReleaseTensorInt(selected);
            ReleaseTensorInt(bitmask);
            ReleaseTensorInt(bitmaskOverlap);

            if (!InternalCommandBuffer())
                D.LogWarning("NMS: Need to download from ComputeBuffer, will flush CommandBuffer");
            ExecuteCommandBufferAndClear();
            var numSelectedCPU = numSelected.DownloadToNativeArray();
            ReleaseTensorInt(numSelected);

            O.Reshape(new TensorShape(numSelectedCPU[0], 3));
        }

        /// <inheritdoc/>
        public void SliceSet(Tensor X, Tensor O, int axis, int start, int step)
        {
            var strideX = X.shape.Length(axis);
            var strideO = O.shape.Length(axis) * step;
            var length = strideX;
            var count = X.shape.Length(0, axis);
            MemCopyStride(X, O, strideX, strideO, length, count, 0, O.shape.Strides(axis) * start);
        }

        /// <inheritdoc/>
        public void Slice(Tensor X, Tensor O, ReadOnlySpan<int> starts, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps)
        {
            var fn = ComputeFunctions.k_Slice;
            unsafe
            {
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                var pStarts = stackalloc int[8] { 0, 0, 0, 0, 0, 0, 0, 0 };
                var pSteps = stackalloc int[8] { 1, 1, 1, 1, 1, 1, 1, 1 };

                for (int i = 0; i < starts.Length; i++)
                {
                    int axis = axes[i];
                    pStarts[(TensorShape.maxRank - X.shape.rank) + axis] = starts[i];
                    pSteps[(TensorShape.maxRank - X.shape.rank) + axis] = steps[i];
                }
                cb.SetInt8(fn, k_ID_starts, pStarts);
                cb.SetInt8(fn, k_ID_steps, pSteps);
            }
            cb.SetComputeIntParam(fn.shader, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void SliceSet(Tensor X, Tensor values, Tensor O, ReadOnlySpan<int> starts, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps)
        {
            MemCopy(X, O);
            var fn = ComputeFunctions.k_SliceSet;
            unsafe
            {
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, values.shape);
                var pStarts = stackalloc int[8] { 0, 0, 0, 0, 0, 0, 0, 0 };
                var pSteps = stackalloc int[8] { 1, 1, 1, 1, 1, 1, 1, 1 };

                for (int i = 0; i < starts.Length; i++)
                {
                    int axis = axes[i];
                    pStarts[(TensorShape.maxRank - X.shape.rank) + axis] = starts[i];
                    pSteps[(TensorShape.maxRank - X.shape.rank) + axis] = steps[i];
                }
                cb.SetInt8(fn, k_ID_starts, pStarts);
                cb.SetInt8(fn, k_ID_steps, pSteps);
            }
            cb.SetComputeIntParam(fn.shader, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(values));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, values.shape.length);
        }

        /// <inheritdoc/>
        public void Split(Tensor X, Tensor O, int axis, int start)
        {
            var fn = ComputeFunctions.k_Split;
            cb.SetComputeIntParam(fn.shader, k_ID_start, start);
            cb.SetComputeIntParam(fn.shader, k_ID_lengthO, O.shape.length);
            cb.SetComputeIntParam(fn.shader, k_ID_strideLower, O.shape.Strides(axis));
            int strideUpperX = axis == 0 ? X.shape.length : X.shape.Strides(axis - 1);
            int strideUpperO = axis == 0 ? O.shape.length : O.shape.Strides(axis - 1);
            cb.SetComputeIntParam(fn.shader, k_ID_strideUpperX, strideUpperX);
            cb.SetComputeIntParam(fn.shader, k_ID_strideUpperO, strideUpperO);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            var numBlocksY = ComputeHelper.IDivC(O.shape.length, (int)ComputeHelper.SafeDispatchLimit);
            var numBlocksX = ComputeHelper.IDivC(O.shape.length, numBlocksY);
            cb.SetComputeIntParam(fn.shader, k_ID_MaxBlockIndexX, numBlocksX);
            cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
        }

        /// <inheritdoc/>
        public void Pad(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> pad, Layers.PadMode padMode, float constant)
        {
            ComputeFunction fn;
            switch (padMode)
            {
                case Layers.PadMode.Constant:
                    fn = ComputeFunctions.k_PadBorderND;
                    break;
                case Layers.PadMode.Reflect:
                    fn = ComputeFunctions.k_PadReflectND;
                    break;
                case Layers.PadMode.Edge:
                    fn = ComputeFunctions.k_PadEdgeND;
                    break;
                case Layers.PadMode.Symmetric:
                    fn = ComputeFunctions.k_PadSymmetricND;
                    break;
                case Layers.PadMode.Wrap:
                    fn = ComputeFunctions.k_PadWrapND;
                    break;
                default:
                    throw new NotImplementedException();
            }

            cb.SetComputeFloatParam(fn.shader, k_ID_Beta, constant);

            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
            cb.SetInt16(fn, k_ID_pad, pad);
            cb.SetComputeIntParam(fn.shader, k_ID_rank, X.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Pad(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> pad, Layers.PadMode padMode, int constant)
        {
            ComputeFunction fn;
            switch (padMode)
            {
                case Layers.PadMode.Constant:
                    fn = ComputeFunctions.k_PadBorderND;
                    break;
                case Layers.PadMode.Reflect:
                    fn = ComputeFunctions.k_PadReflectND;
                    break;
                case Layers.PadMode.Edge:
                    fn = ComputeFunctions.k_PadEdgeND;
                    break;
                case Layers.PadMode.Symmetric:
                    fn = ComputeFunctions.k_PadSymmetricND;
                    break;
                case Layers.PadMode.Wrap:
                    fn = ComputeFunctions.k_PadWrapND;
                    break;
                default:
                    throw new NotImplementedException();
            }

            cb.SetComputeFloatParam(fn.shader, k_ID_Beta, math.asfloat(constant));

            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
            cb.SetInt16(fn, k_ID_pad, pad);
            cb.SetComputeIntParam(fn.shader, k_ID_rank, X.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Transpose(Tensor X, Tensor O)
        {
            var fn = ComputeFunctions.k_Transpose;
            unsafe
            {
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);

                int* permutations = stackalloc int[TensorShape.maxRank];
                for (int i = 0; i < X.shape.rank; i++)
                    permutations[i] = (X.shape.rank - 1) - i;
                cb.SetInt8(fn, k_ID_permutations, permutations);
            }
            cb.SetComputeIntParam(fn.shader, k_ID_rank, X.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, X.shape.length);
        }

        /// <inheritdoc/>
        public void Transpose(Tensor X, Tensor O, ReadOnlySpan<int> permutations)
        {
            bool is2DTranspose = ShapeInference.IsTranspose2D(X.shape, permutations, out int equivalentXH, out int equivalentXW);

            if (is2DTranspose)
            {
                if (equivalentXW == 1 || equivalentXH == 1)
                {
                    MemCopy(X, O);
                    return;
                }

                var fn = ComputeFunctions.k_Transpose2D;
                cb.SetComputeIntParam(fn.shader, k_ID_X_width, equivalentXW);
                cb.SetComputeIntParam(fn.shader, k_ID_X_height, equivalentXH);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

                cb.Dispatch(fn, equivalentXW, equivalentXH, 1);
            }
            else
            {

                var fn = ComputeFunctions.k_Transpose;
                unsafe
                {
                    cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                    cb.SetInt8(fn, k_ID_permutations, permutations);
                }
                cb.SetComputeIntParam(fn.shader, k_ID_rank, X.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, X.shape.length);
            }
        }

        void ArgMaxTail(Tensor<float> X, Tensor<int> O, int axis)
        {
            int globalNonSpatialLength = X.shape.Length(0, axis);
            int globalSpatialDims = X.shape.length / globalNonSpatialLength;

            int localSpatialLength = globalSpatialDims;

            var Oshape = new TensorShape(globalNonSpatialLength, localSpatialLength);

            Tensor<int> Xindices = AllocTensorInt(X.shape); // save max(X)
            bool isFirstDispatch = true;

            // downsample with pyramid approach
            while (localSpatialLength > 64 * 4)
            {
                int spatialLengthO = ComputeHelper.IDivC(localSpatialLength, 64 * 4);
                Oshape[-1] = spatialLengthO;

                var Otemp = AllocTensorFloat(Oshape);
                var Oindicestemp = AllocTensorInt(Oshape);

                var fnPool = ComputeFunctions.k_ArgMaxReduce;
                cb.SetTensorAsBuffer(fnPool, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fnPool, k_ID_XIndices, Pin(Xindices));
                cb.SetTensorAsBuffer(fnPool, k_ID_Optr, Pin(Otemp));
                cb.SetTensorAsBuffer(fnPool, k_ID_OIndices, Pin(Oindicestemp));
                cb.SetComputeIntParam(fnPool.shader, k_ID_SpatialDims, localSpatialLength);
                cb.SetComputeIntParam(fnPool.shader, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetComputeIntParam(fnPool.shader, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(fnPool, globalNonSpatialLength, ComputeHelper.IDivC(localSpatialLength, 4), 1);

                if (!isFirstDispatch)
                {
                    ReleaseTensorFloat(X);
                    ReleaseTensorInt(Xindices);
                }
                else
                {
                    ReleaseTensorInt(Xindices);
                }
                X = Otemp;
                Xindices = Oindicestemp;
                localSpatialLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var fn = ComputeFunctions.k_GlobalArgMaxReduce;
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_XIndices, Pin(Xindices));
            cb.SetTensorAsBuffer(fn, k_ID_OIndices, Pin(O));
            cb.SetComputeIntParam(fn.shader, k_ID_SpatialDims, localSpatialLength);
            cb.SetComputeIntParam(fn.shader, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

            cb.Dispatch(fn, globalNonSpatialLength, 1, 1);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
            ReleaseTensorInt(Xindices);
        }

        /// <inheritdoc/>
        public void ArgMax(Tensor<float> X, Tensor<int> O, int axis, bool selectLastIndex)
        {
            int dimAxis = X.shape[axis];
            Assert.AreNotEqual(0, dimAxis, "ValueError: zero-size array to reduction operation maximum which has no identity.");

            if (!selectLastIndex && (dimAxis == X.shape.Length(axis)))
            {
                ArgMaxTail(X, O, axis);
                return;
            }

            var fn = (selectLastIndex ? ComputeFunctions.k_ArgMaxFloatLast : ComputeFunctions.k_ArgMaxFloatFirst);
            cb.SetComputeIntParam(fn.shader, k_ID_innerLength, X.shape.Strides(axis));
            cb.SetComputeIntParam(fn.shader, k_ID_reduceLength, dimAxis);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void ArgMax(Tensor<int> X, Tensor<int> O, int axis, bool selectLastIndex)
        {
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

            var fn = (selectLastIndex ? ComputeFunctions.k_ArgMaxIntLast : ComputeFunctions.k_ArgMaxIntFirst);
            cb.SetComputeIntParam(fn.shader, k_ID_innerLength, X.shape.Strides(axis));
            cb.SetComputeIntParam(fn.shader, k_ID_reduceLength, X.shape[axis]);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void ArgMin(Tensor<float> X, Tensor<int> O, int axis, bool selectLastIndex)
        {
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation minimum which has no identity.");

            var fn = (selectLastIndex ? ComputeFunctions.k_ArgMinFloatLast : ComputeFunctions.k_ArgMinFloatFirst);
            cb.SetComputeIntParam(fn.shader, k_ID_innerLength, X.shape.Strides(axis));
            cb.SetComputeIntParam(fn.shader, k_ID_reduceLength, X.shape[axis]);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void ArgMin(Tensor<int> X, Tensor<int> O, int axis, bool selectLastIndex)
        {
            var fn = (selectLastIndex ? ComputeFunctions.k_ArgMinIntLast : ComputeFunctions.k_ArgMinIntFirst);
            cb.SetComputeIntParam(fn.shader, k_ID_innerLength, X.shape.Strides(axis));
            cb.SetComputeIntParam(fn.shader, k_ID_reduceLength, X.shape[axis]);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Not(Tensor<int> X, Tensor<int> O)
        {
            var fn = ComputeFunctions.k_Not;
            cb.SetTensorAsBuffer(fn, k_ID_X_int_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_int_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void HardSwish(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_HardSwish;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Sign(Tensor<float> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_SignFloat;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Sign(Tensor<int> X, Tensor<int> O)
        {
            var fn = ComputeFunctions.k_SignInt;
            cb.SetTensorAsBuffer(fn, k_ID_X_int_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_int_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void IsInf(Tensor<float> X, Tensor<int> O, bool detectNegative, bool detectPositive)
        {
            var fn = ComputeFunctions.k_IsInf;
            cb.SetBool(fn, k_ID_detectNegative, detectNegative);
            cb.SetBool(fn, k_ID_detectPositive, detectPositive);
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void IsNaN(Tensor<float> X, Tensor<int> O)
        {
            var fn = ComputeFunctions.k_IsNaN;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_int_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Where(Tensor<int> C, Tensor A, Tensor B, Tensor O)
        {
            var fn = ComputeFunctions.k_Where;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeC, k_ID_stridesC, C.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
            cb.SetComputeIntParam(fn.shader, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(C));
            cb.SetTensorAsBuffer(fn, k_ID_Sptr, Pin(A));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.UnrolledDispatch(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Tile(Tensor X, Tensor O, ReadOnlySpan<int> repeats)
        {
            var fn = ComputeFunctions.k_Tile;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
            cb.SetComputeIntParam(fn.shader, k_ID_rank, O.shape.rank);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void MemClear(Tensor O)
        {
            var length = O.shape.length;
            var numWords = ComputeHelper.IDivC(length, 4);
            var wordsHeight = ComputeHelper.IDivC(numWords, (int)ComputeHelper.SafeDispatchLimit * 32 * 8);
            var wordsWidth = ComputeHelper.IDivC(numWords, wordsHeight);

            var fn = ComputeFunctions.k_MemSet;
            cb.SetComputeFloatParam(fn.shader, k_ID_memValueFloat, 0);
            cb.SetComputeIntParam(fn.shader, k_ID_offsetO, 0);
            cb.SetComputeIntParam(fn.shader, k_ID_count, length);
            cb.SetComputeIntParam(fn.shader, k_ID_O_width, wordsWidth * 4);
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, wordsWidth, wordsHeight, 1);
        }

        /// <inheritdoc/>
        public void MemSet(Tensor<float> O, float value)
        {
            var length = O.shape.length;
            var numWords = ComputeHelper.IDivC(length, 4);
            var wordsHeight = ComputeHelper.IDivC(numWords, (int)ComputeHelper.SafeDispatchLimit * 32 * 8);
            var wordsWidth = ComputeHelper.IDivC(numWords, wordsHeight);

            var fn = ComputeFunctions.k_MemSet;
            cb.SetComputeFloatParam(fn.shader, k_ID_memValueFloat, value);
            cb.SetComputeIntParam(fn.shader, k_ID_offsetO, 0);
            cb.SetComputeIntParam(fn.shader, k_ID_count, length);
            cb.SetComputeIntParam(fn.shader, k_ID_O_width, wordsWidth * 4);
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, wordsWidth, wordsHeight, 1);
        }

        /// <inheritdoc/>
        public void MemSet(Tensor<int> O, int value)
        {
            var length = O.shape.length;
            var numWords = ComputeHelper.IDivC(length, 4);
            var wordsHeight = ComputeHelper.IDivC(numWords, (int)ComputeHelper.SafeDispatchLimit * 32 * 8);
            var wordsWidth = ComputeHelper.IDivC(numWords, wordsHeight);

            var fn = ComputeFunctions.k_MemSet;
            cb.SetComputeFloatParam(fn.shader, k_ID_memValueFloat, math.asfloat(value));
            cb.SetComputeIntParam(fn.shader, k_ID_offsetO, 0);
            cb.SetComputeIntParam(fn.shader, k_ID_count, length);
            cb.SetComputeIntParam(fn.shader, k_ID_O_width, wordsWidth * 4);
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, wordsWidth, wordsHeight, 1);
        }

        /// <inheritdoc/>
        public void Expand(Tensor X, Tensor O)
        {
            var fn = ComputeFunctions.k_Expand;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
            cb.SetComputeIntParam(fn.shader, k_ID_rank, O.shape.rank);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void CompressWithIndices(Tensor X, Tensor<int> indices, Tensor O, int numIndices, int axis)
        {
            var fn = ComputeFunctions.k_Gather;
            cb.SetComputeIntParam(fn.shader, k_ID_endLength, X.shape.Strides(axis));
            cb.SetComputeIntParam(fn.shader, k_ID_indicesLength, numIndices);
            cb.SetComputeIntParam(fn.shader, k_ID_axisDim, X.shape[axis]);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(indices));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Gather(Tensor X, Tensor<int> indices, Tensor O, int axis)
        {
            var fn = ComputeFunctions.k_Gather;
            cb.SetComputeIntParam(fn.shader, k_ID_endLength, X.shape.Strides(axis));
            cb.SetComputeIntParam(fn.shader, k_ID_indicesLength, indices.shape.length);
            cb.SetComputeIntParam(fn.shader, k_ID_axisDim, X.shape[axis]);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(indices));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.UnrolledDispatch(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void GatherElements(Tensor X, Tensor<int> indices, Tensor O, int axis)
        {
            Logger.AssertIsTrue(indices.shape.rank == X.shape.rank, "GatherElements: input and indices rank should match");
            Logger.AssertIsTrue(O.shape == indices.shape, "GatherElements: output and indices shapes should match");
            axis = X.shape.Axis(axis); // note: this is safe since the ranks of X and indices match

            // See ScatterElements for more info
            bool fastPathPossible = ShapeInference.ScatterGatherElementsSupportsFastPath(indices.shape, X.shape, axis);
            var fn = fastPathPossible ? ComputeFunctions.k_GatherElementsFast : ComputeFunctions.k_GatherElements;

            cb.SetComputeIntParam(fn.shader, k_ID_inputAxisSize, X.shape[axis]);

            if (fastPathPossible)
            {
                cb.SetComputeIntParam(fn.shader, k_ID_indicesAxisElementStride, indices.shape.Strides(axis));
                cb.SetComputeIntParam(fn.shader, k_ID_inputAxisElementStride, X.shape.Strides(axis));
                cb.SetComputeIntParam(fn.shader, k_ID_indicesAxisMinusOneElementStride, indices.shape[axis] * indices.shape.Strides(axis));

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(indices));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

                cb.UnrolledDispatch(fn, indices.shape.length);
            }
            else
            {
                cb.SetTensorStridesCompactedAtHead(fn, k_ID_stridesO, indices.shape);
                cb.SetTensorStridesCompactedAtHead(fn, k_ID_stridesX, X.shape); // WARNING: Remember that X in the shader and here are inputs!
                cb.SetComputeIntParam(fn.shader, k_ID_posAxis, axis);
                cb.SetComputeIntParam(fn.shader, k_ID_rank, X.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(indices));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

                cb.UnrolledDispatch(fn, indices.shape.length);
            }
        }

        /// <inheritdoc/>
        public void GatherND(Tensor X, Tensor<int> indices, Tensor O, int batchDims)
        {
            var fn = ComputeFunctions.k_GatherND;
            cb.SetComputeIntParam(fn.shader, k_ID_rankX, X.shape.rank);
            cb.SetComputeIntParam(fn.shader, k_ID_rankO, O.shape.rank);
            cb.SetComputeIntParam(fn.shader, k_ID_rankIndices, indices.shape.rank);
            cb.SetComputeIntParam(fn.shader, k_ID_iStart, TensorShape.maxRank - O.shape.rank);
            cb.SetComputeIntParam(fn.shader, k_ID_iEndIndices, TensorShape.maxRank - O.shape.rank + indices.shape.rank - 1);
            cb.SetComputeIntParam(fn.shader, k_ID_iEndX, TensorShape.maxRank - O.shape.rank + batchDims);
            cb.SetComputeIntParam(fn.shader, k_ID_iStartB, TensorShape.maxRank - X.shape.rank + batchDims);
            cb.SetComputeIntParam(fn.shader, k_ID_iEndB, TensorShape.maxRank - X.shape.rank + batchDims + indices.shape[-1]);
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeIndices, k_ID_stridesIndices, indices.shape);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(indices));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.UnrolledDispatch(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void ScatterElements(Tensor X, Tensor<int> indices, Tensor updates, Tensor O, int axis, Layers.ScatterReductionMode reduction)
        {
            // TODO: The ONNX definition for ScatterElements allows duplicate indices when using the
            // reduction modes, but allowing this introduces race conditions for updating the output
            // tensor. As the current use cases for ScatterElements do not use reductions, fallback
            // to the single-threaded burst cpu implementation.
            if (reduction != Layers.ScatterReductionMode.None)
            {
                throw new NotImplementedException();
            }
            MemCopy(X, O);

            Logger.AssertIsTrue(indices.shape.rank == X.shape.rank, "ScatterElements: input and indices rank should match");
            axis = X.shape.Axis(axis); // note: this is safe since the ranks of X and indices match

            bool fastPathPossible = ShapeInference.ScatterGatherElementsSupportsFastPath(indices.shape, X.shape, axis);
            var fn = fastPathPossible ? ComputeFunctions.k_ScatterElementsFast : ComputeFunctions.k_ScatterElements;

            cb.SetComputeIntParam(fn.shader, k_ID_outAxisSize, X.shape[axis]);
            cb.SetComputeIntParam(fn.shader, k_ID_reductionType, (int)reduction);

            if (fastPathPossible)
            {
                cb.SetComputeIntParam(fn.shader, k_ID_indicesAxisElementStride, indices.shape.Strides(axis));
                cb.SetComputeIntParam(fn.shader, k_ID_outAxisElementStride, X.shape.Strides(axis));
                cb.SetComputeIntParam(fn.shader, k_ID_indicesAxisMinusOneElementStride, indices.shape[axis] * indices.shape.Strides(axis));

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(updates));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(indices));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

                cb.UnrolledDispatch(fn, indices.shape.length);
            }
            else
            {
                cb.SetTensorStridesCompactedAtHead(fn, k_ID_stridesO, O.shape);
                cb.SetTensorStridesCompactedAtHead(fn, k_ID_stridesX, indices.shape); // WARNING: Remember that X in the shader code is updates, but here X is the input tensor!
                cb.SetComputeIntParam(fn.shader, k_ID_posAxis, axis);
                cb.SetComputeIntParam(fn.shader, k_ID_rank, X.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(updates));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(indices));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

                cb.UnrolledDispatch(fn, indices.shape.length);
            }
        }

        /// <inheritdoc/>
        public void ScatterND(Tensor<float> X, Tensor<int> indices, Tensor<float> updates, Tensor<float> O, Layers.ScatterReductionMode reduction)
        {
            MemCopy(X, O);
            int indexRemapDim = indices.shape[-1];
            int indicesLength = indices.shape.Length(0, -1);
            int updatesLength = updates.shape.length / indicesLength;

            var fn = ComputeFunctions.k_ScatterNDFloat;
            cb.SetComputeIntParam(fn.shader, k_ID_updatesLength, updatesLength);
            cb.SetComputeIntParam(fn.shader, k_ID_indicesLength, indicesLength);
            cb.SetComputeIntParam(fn.shader, k_ID_indexRemapDim, indexRemapDim);
            cb.SetComputeIntParam(fn.shader, k_ID_reduction, (int)reduction);
            unsafe
            {
                var trailing = stackalloc int[8];
                int trailingDim = 1;
                for (int j = (indexRemapDim - 1); j >= 0; j--)
                {
                    trailing[j] = trailingDim;
                    trailingDim *= X.shape[j];
                }
                cb.SetInt8(fn, k_ID_trailing, trailing);
            }
            cb.SetTensorAsBuffer(fn, k_ID_Iptr, Pin(indices));
            cb.SetTensorAsBuffer(fn, k_ID_Uptr, Pin(updates));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.Dispatch(fn, updatesLength, indicesLength, 1);
        }

        /// <inheritdoc/>
        public void ScatterND(Tensor<int> X, Tensor<int> indices, Tensor<int> updates, Tensor<int> O, Layers.ScatterReductionMode reduction)
        {
            MemCopy(X, O);

            int indexRemapDim = indices.shape[-1];
            int indicesLength = indices.shape.Length(0, -1);
            int updatesLength = updates.shape.length / indicesLength;

            var fn = ComputeFunctions.k_ScatterNDInt;
            cb.SetComputeIntParam(fn.shader, k_ID_updatesLength, updatesLength);
            cb.SetComputeIntParam(fn.shader, k_ID_indicesLength, indicesLength);
            cb.SetComputeIntParam(fn.shader, k_ID_indexRemapDim, indexRemapDim);
            cb.SetComputeIntParam(fn.shader, k_ID_reduction, (int)reduction);
            unsafe
            {
                var trailing = stackalloc int[8];
                int trailingDim = 1;
                for (int j = indexRemapDim - 1; j >= 0; j--)
                {
                    trailing[j] = trailingDim;
                    trailingDim *= X.shape[j];
                }
                cb.SetInt8(fn, k_ID_trailing, trailing);
            }
            cb.SetTensorAsBuffer(fn, k_ID_Iptr, Pin(indices));
            cb.SetTensorAsBuffer(fn, k_ID_UIntptr, Pin(updates));
            cb.SetTensorAsBuffer(fn, k_ID_OIntptr, Pin(O));
            cb.Dispatch(fn, updatesLength, indicesLength, 1);
        }

        /// <inheritdoc/>
        public void OneHot(Tensor<int> X, Tensor<int> O, int axis, int depth, int offValue, int onValue)
        {
            axis = O.shape.Axis(axis);

            var fn = ComputeFunctions.k_OneHot;
            cb.SetComputeIntParam(fn.shader, k_ID_depth, depth);
            cb.SetComputeIntParam(fn.shader, k_ID_offValue, offValue);
            cb.SetComputeIntParam(fn.shader, k_ID_onValue, onValue);
            cb.SetComputeIntParam(fn.shader, k_ID_rankO, O.shape.rank);

            cb.SetComputeIntParam(fn.shader, k_ID_stridesToAxis, O.shape.Strides(axis));
            cb.SetComputeIntParam(fn.shader, k_ID_axisDim, O.shape[axis]);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, O.shape.length, 1, 1);
        }

        /// <inheritdoc/>
        public void OneHot(Tensor<int> X, Tensor<float> O, int axis, int depth, float offValue, float onValue)
        {
            axis = O.shape.Axis(axis);

            var fn = ComputeFunctions.k_OneHot;
            cb.SetComputeIntParam(fn.shader, k_ID_depth, depth);
            cb.SetComputeIntParam(fn.shader, k_ID_offValue, math.asint(offValue));
            cb.SetComputeIntParam(fn.shader, k_ID_onValue, math.asint(onValue));
            cb.SetComputeIntParam(fn.shader, k_ID_rankO, O.shape.rank);

            cb.SetComputeIntParam(fn.shader, k_ID_stridesToAxis, O.shape.Strides(axis));
            cb.SetComputeIntParam(fn.shader, k_ID_axisDim, O.shape[axis]);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, O.shape.length, 1, 1);
        }

        /// <inheritdoc/>
        public void TopK(Tensor<float> X, Tensor<float> values, Tensor<int> indices, int k, int axis, bool largest)
        {
            int reduceLength = X.shape[axis];
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.length / (reduceLength * innerLength);

            var fn = (largest ? ComputeFunctions.k_TopKLargest : ComputeFunctions.k_TopKSmallest);
            cb.SetComputeIntParam(fn.shader, k_ID_innerLength, innerLength);
            cb.SetComputeIntParam(fn.shader, k_ID_outerLength, outerLength);
            cb.SetComputeIntParam(fn.shader, k_ID_reduceLength, reduceLength);
            cb.SetComputeIntParam(fn.shader, k_ID_maxK, k);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Valuesptr, Pin(values));
            cb.SetTensorAsBuffer(fn, k_ID_Indicesptr, Pin(indices));
            cb.Dispatch(fn, innerLength, outerLength, 1);
        }

        /// <inheritdoc/>
        public void RoiAlign(Tensor<float> X, Tensor<float> rois, Tensor<int> indices, Tensor<float> O, Layers.RoiPoolingMode mode, int outputHeight, int outputWidth, int samplingRatio, float spatialScale)
        {
            var fn = (mode == Layers.RoiPoolingMode.Avg ? ComputeFunctions.k_RoiAlignAvg : ComputeFunctions.k_RoiAlignMax);
            cb.SetComputeIntParam(fn.shader, k_ID_numRois, rois.shape[0]);
            cb.SetComputeIntParam(fn.shader, k_ID_inputChannels, X.shape[1]);
            cb.SetComputeIntParam(fn.shader, k_ID_inputHeight, X.shape[2]);
            cb.SetComputeIntParam(fn.shader, k_ID_inputWidth, X.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_inputSpatialSize, X.shape[2] * X.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_inputBatchOffset, X.shape[1] * X.shape[2] * X.shape[3]);
            cb.SetComputeIntParam(fn.shader, k_ID_outputHeight, outputHeight);
            cb.SetComputeIntParam(fn.shader, k_ID_outputWidth, outputWidth);
            cb.SetComputeIntParam(fn.shader, k_ID_outputSpatialSize, outputHeight * outputWidth);
            cb.SetComputeFloatParam(fn.shader, k_ID_normalizeOHeight, 1.0f / outputHeight);
            cb.SetComputeFloatParam(fn.shader, k_ID_normalizeOWidth, 1.0f / outputWidth);
            cb.SetComputeIntParam(fn.shader, k_ID_samplingRatio, samplingRatio);
            cb.SetComputeFloatParam(fn.shader, k_ID_spatialScale, spatialScale);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Sptr, Pin(rois));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(indices));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, O.shape[0] * O.shape[1], O.shape[2] * O.shape[3], 1);
        }

        /// <inheritdoc/>
        public void RandomNormal(Tensor<float> O, float mean, float scale, int? seed)
        {
            var fn = ComputeFunctions.k_RandomNormal;
            cb.SetComputeIntParam(fn.shader, k_ID_lengthO, O.shape.length);
            cb.SetComputeIntParam(fn.shader, k_ID_seed, (int)Random.GetSeed(seed));
            cb.SetComputeFloatParam(fn.shader, k_ID_mean, mean);
            cb.SetComputeFloatParam(fn.shader, k_ID_scale, scale);

            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, O.shape.length, 1, 1);
        }

        /// <inheritdoc/>
        public void TopP(Tensor<float> X, Tensor<float> random, Tensor<int> O)
        {
            var batch = O.shape.length;

            var fn = ComputeFunctions.k_TopP;
            cb.SetComputeIntParam(fn.shader, k_ID_count, O.shape[-1]);
            cb.SetComputeIntParam(fn.shader, k_ID_innerLength, X.shape[-1]);
            cb.SetComputeIntParam(fn.shader, k_ID_outerLength, batch);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(random));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.Dispatch(fn, batch, 1, 1);
        }

        /// <inheritdoc/>
        public void RandomUniform(Tensor<float> O, float low, float high, int? seed)
        {
            var fn = ComputeFunctions.k_RandomUniform;
            cb.SetComputeIntParam(fn.shader, k_ID_lengthO, O.shape.length);
            cb.SetComputeIntParam(fn.shader, k_ID_seed, (int)Random.GetSeed(seed));
            cb.SetComputeFloatParam(fn.shader, k_ID_low, low);
            cb.SetComputeFloatParam(fn.shader, k_ID_high, high);

            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, O.shape.length, 1, 1);
        }

        /// <inheritdoc/>
        public void Bernoulli(Tensor<float> X, Tensor O, int? seed)
        {
            var fn = (O.dataType == DataType.Float ? ComputeFunctions.k_BernoulliFloat : ComputeFunctions.k_BernoulliInt);
            cb.SetComputeIntParam(fn.shader, k_ID_lengthO, O.shape.length);
            cb.SetComputeIntParam(fn.shader, k_ID_seed, (int)Random.GetSeed(seed));
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Cast(Tensor<int> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_CastIntToFloat;
            cb.SetTensorAsBuffer(fn, k_ID_X_int_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_float_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Cast(Tensor<float> X, Tensor<int> O)
        {
            var fn = ComputeFunctions.k_CastFloatToInt;
            cb.SetTensorAsBuffer(fn, k_ID_X_float_ptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_O_int_ptr, Pin(O));
            cb.UnrolledDispatchFast(fn, O.shape.length);
        }

        /// <inheritdoc/>
        public void Cast(Tensor<short> X, Tensor<float> O)
        {
            var fn = ComputeFunctions.k_CastHalfToFloat;
            cb.SetComputeIntParam(fn.shader, k_ID_lengthO, O.shape.length);
            cb.SetTensorAsBuffer(fn, k_ID_XIntptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, X.count);
        }

        /// <inheritdoc/>
        public virtual void MemCopy(Tensor X, Tensor O)
        {
            MemCopy(X, O, O.shape.length, 0, 0);
        }

        void MemCopy(Tensor X, Tensor O, int count, int offsetX, int offsetO)
        {
            var numWords = ComputeHelper.IDivC(count, 4);
            var wordsHeight = ComputeHelper.IDivC(numWords, (int)ComputeHelper.SafeDispatchLimit * 32 * 8);
            var wordsWidth = ComputeHelper.IDivC(numWords, wordsHeight);

            var fn = ComputeFunctions.k_MemCopy;
            cb.SetComputeIntParam(fn.shader, k_ID_offsetO, offsetO);
            cb.SetComputeIntParam(fn.shader, k_ID_offsetX, offsetX);
            cb.SetComputeIntParam(fn.shader, k_ID_count, count);
            cb.SetComputeIntParam(fn.shader, k_ID_O_width, wordsWidth * 4);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));

            cb.Dispatch(fn, wordsWidth, wordsHeight, 1);
        }

        /// <inheritdoc/>
        public void MemCopyStride(Tensor X, Tensor O, int strideX, int strideO, int length, int count, int offsetX, int offsetO)
        {
            if (length == 0 || count == 0)
                return;
            if (count == 1 || (strideX == length && strideO == length))
            {
                // contiguous memory can be copied together
                MemCopy(X, O, length * count, offsetX, offsetO);
                return;
            }
            Logger.AssertIsTrue(length > 0, "MemCopy.InputError: copy stride length must be greater than 0");
            Logger.AssertIsTrue(count > 0, "MemCopy.InputError: copy stride count must be greater than 0");
            Logger.AssertIsTrue(offsetX >= 0, "MemCopy.BoundsError: copy stride out of bounds for tensor X");
            Logger.AssertIsTrue(offsetX + (count - 1) * strideX + length <= X.shape.length, "MemCopy.BoundsError: copy stride out of bounds for tensor X");
            Logger.AssertIsTrue(offsetO >= 0, "MemCopy.BoundsError: copy stride out of bounds for tensor O");
            Logger.AssertIsTrue(offsetO + (count - 1) * strideO + length <= O.shape.length, "MemCopy.BoundsError: copy stride out of bounds for tensor O");
            var fn = ComputeFunctions.k_MemCopyStride;
            var copyLength = count * length;
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetComputeIntParam(fn.shader, k_ID_strideX, strideX);
            cb.SetComputeIntParam(fn.shader, k_ID_strideO, strideO);
            cb.SetComputeIntParam(fn.shader, k_ID_offsetX, offsetX);
            cb.SetComputeIntParam(fn.shader, k_ID_offsetO, offsetO);
            cb.SetComputeIntParam(fn.shader, k_ID_elementSize, length);
            cb.SetComputeIntParam(fn.shader, k_ID_count, copyLength);
            cb.Dispatch(fn, ComputeHelper.IDivC(copyLength, 4), 1, 1);
        }

        void Gemm(Tensor<float> X, Tensor<float> Y, Tensor<float> O, int M, int K, int N, bool transposeA = false, bool transposeB = false, Layers.FusableActivation fusedActivation = Layers.FusableActivation.None)
        {
            if (transposeA || transposeB)
            {
                ComputeFunction fn;

                if (transposeA)
                    fn = transposeB ? ComputeFunctions.k_GemmT_XT_WT_T8x8_R4x4 : ComputeFunctions.k_GemmT_XT_T8x8_R4x4;
                else
                    fn = ComputeFunctions.k_GemmT_WT_T8x8_R4x4;

                cb.SetComputeIntParam(fn.shader, k_ID_M, M);
                cb.SetComputeIntParam(fn.shader, k_ID_N, N);
                cb.SetComputeIntParam(fn.shader, k_ID_K, K);
                cb.SetComputeIntParam(fn.shader, k_ID_maxXIndex, M * K - 1);
                cb.SetComputeIntParam(fn.shader, k_ID_maxWIndex, K * N - 1);
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(Y));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetComputeFloatParam(fn.shader, k_ID__MinValue, fusedActivation == Layers.FusableActivation.Relu ? 0.0f : float.MinValue);

                cb.Dispatch(fn, ComputeHelper.IDivC(N, 4), ComputeHelper.IDivC(M, 4), 1);
            }
            else
            {
                int workItemsX, workItemsY, workItemsZ;
                ComputeFunction fn;

                if (M == 1)
                {
                    fn = ComputeFunctions.k_Gemm_V_L1Cached64;
                    workItemsX = ComputeHelper.IDivC(N, 4);
                    workItemsY = 1;
                    workItemsZ = 1;
                }
                else if (N % 64 == 0 && K % 16 == 0)
                {
                    fn = ComputeFunctions.k_Gemm_T16x16_R4x4;
                    workItemsX = ComputeHelper.IDivC(N, 4);
                    workItemsY = ComputeHelper.IDivC(M, 4);
                    workItemsZ = 1;
                }
                else
                {
                    fn = ComputeFunctions.k_Gemm_T8x8_R4x4;
                    workItemsX = ComputeHelper.IDivC(N, 4);
                    workItemsY = ComputeHelper.IDivC(M, 4);
                    workItemsZ = 1;
                }

                cb.SetComputeIntParam(fn.shader, k_ID_X_width, K);
                cb.SetComputeIntParam(fn.shader, k_ID_W_width, N);
                cb.SetComputeIntParam(fn.shader, k_ID_O_width, N);
                cb.SetComputeIntParam(fn.shader, k_ID_O_height, M);
                cb.SetComputeIntParam(fn.shader, k_ID_maxXIndex, M * K - 1);
                cb.SetComputeIntParam(fn.shader, k_ID_maxWIndex, K * N - 1);
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(Y));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetComputeFloatParam(fn.shader, k_ID__MinValue, fusedActivation == Layers.FusableActivation.Relu ? 0.0f : float.MinValue);

                cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);
            }
        }

        /// <inheritdoc/>
        protected void SinglePassLSTM(Tensor<float> X, Tensor<float> W, Tensor<float> R, Tensor<float> B, Tensor<int> sequenceLens, Tensor<float> P, Tensor<float> Y, Tensor<float> Y_h, Tensor<float> Y_c, Layers.RnnActivation[] activations, float[] activationAlpha, float[] activationBeta, bool inputForget, float clip, bool isReverse, int dirIndex, Layers.RnnLayout layout)
        {
            var numDirections = B.shape[0];
            var inputSize = X.shape[2];
            var hiddenSize = R.shape[2];

            var seqLength = X.shape[0];
            var batchSize = X.shape[1];

            var xStrideSeq = batchSize * 4 * hiddenSize;
            var xStrideBatch = 4 * hiddenSize;

            var yStrideDir = batchSize * hiddenSize;
            var yStrideSeq = numDirections * batchSize * hiddenSize;
            var yStrideBatch = hiddenSize;

            if (layout == Layers.RnnLayout.BatchFirst)
            {
                seqLength = X.shape[1];
                batchSize = X.shape[0];

                xStrideSeq = 4 * hiddenSize;
                xStrideBatch = seqLength * 4 * hiddenSize;

                yStrideDir = hiddenSize;
                yStrideSeq = numDirections * hiddenSize;
                yStrideBatch = seqLength * numDirections * hiddenSize;
            }

            var HtxRT = AllocTensorFloat(new TensorShape(batchSize * 4 * hiddenSize));
            var XsixWT = AllocTensorFloat(new TensorShape(seqLength * batchSize * 4 * hiddenSize));

            Gemm(X, W, XsixWT, seqLength * batchSize, inputSize, 4 * hiddenSize, transposeB: true);

            var endFn = ComputeFunctions.k_LSTMEnd;
            cb.SetComputeIntParam(endFn.shader, k_ID_hiddenSize, hiddenSize);
            cb.SetComputeIntParam(endFn.shader, k_ID_batchSize, batchSize);
            cb.SetComputeIntParam(endFn.shader, k_ID_xStride, xStrideBatch);
            cb.SetComputeIntParam(endFn.shader, k_ID_yStride, yStrideBatch);
            cb.SetBool(endFn, k_ID_inputForget, inputForget);
            cb.SetComputeFloatParam(endFn.shader, k_ID_clipValue, clip);
            cb.SetComputeIntParam(endFn.shader, k_ID_fActivation, (int)activations[3 * dirIndex + 0]);
            cb.SetComputeFloatParam(endFn.shader, k_ID_fAlpha, activationAlpha[3 * dirIndex + 0]);
            cb.SetComputeFloatParam(endFn.shader, k_ID_fBeta, activationAlpha[3 * dirIndex + 0]);
            cb.SetComputeIntParam(endFn.shader, k_ID_gActivation, (int)activations[3 * dirIndex + 1]);
            cb.SetComputeFloatParam(endFn.shader, k_ID_gAlpha, activationAlpha[3 * dirIndex + 1]);
            cb.SetComputeFloatParam(endFn.shader, k_ID_gBeta, activationAlpha[3 * dirIndex + 1]);
            cb.SetComputeIntParam(endFn.shader, k_ID_hActivation, (int)activations[3 * dirIndex + 2]);
            cb.SetComputeFloatParam(endFn.shader, k_ID_hAlpha, activationAlpha[3 * dirIndex + 2]);
            cb.SetComputeFloatParam(endFn.shader, k_ID_hBeta, activationAlpha[3 * dirIndex + 2]);
            cb.SetTensorAsBuffer(endFn, k_ID_Yptr, Pin(Y));
            cb.SetTensorAsBuffer(endFn, k_ID_YHptr, Pin(Y_h));
            cb.SetTensorAsBuffer(endFn, k_ID_YCptr, Pin(Y_c));
            cb.SetTensorAsBuffer(endFn, k_ID_Bptr, Pin(B));
            cb.SetComputeIntParam(endFn.shader, k_ID_bOffset, dirIndex * 8 * hiddenSize);
            cb.SetTensorAsBuffer(endFn, k_ID_Pptr, Pin(P));
            cb.SetComputeIntParam(endFn.shader, k_ID_pOffset, dirIndex * 3 * hiddenSize);
            cb.SetTensorAsBuffer(endFn, k_ID_XsixWTptr, Pin(XsixWT));
            cb.SetTensorAsBuffer(endFn, k_ID_HtxRTptr, Pin(HtxRT));
            cb.SetTensorAsBuffer(endFn, k_ID_SequenceLensptr, Pin(sequenceLens));

            for (var i = 0; i < seqLength; i++)
            {
                var seqIndex = isReverse ? seqLength - 1 - i : i;

                Gemm(Y_h, R, HtxRT, batchSize, hiddenSize, 4 * hiddenSize, transposeB: true);

                cb.SetComputeIntParam(endFn.shader, k_ID_seqIndex, seqIndex);
                cb.SetComputeIntParam(endFn.shader, k_ID_yOffset, dirIndex * yStrideDir + seqIndex * yStrideSeq);
                cb.SetComputeIntParam(endFn.shader, k_ID_xOffset, seqIndex * xStrideSeq);
                cb.Dispatch(endFn, batchSize, hiddenSize, 1);
            }

            ReleaseTensorFloat(HtxRT);
            ReleaseTensorFloat(XsixWT);
        }

        /// <summary>
        /// Sets final output tensor for W, R, initialH and initialC from provided input tensors
        /// if no input is provided the tensor is cleared to 0 as a default
        /// otherwise if the input tensor can be used directly in the calculation this will early out
        /// </summary>
        void SetRnnInput(Tensor<float> X, Tensor<float> O, int index, int count, int length, int strideX)
        {
            if (X == O)
                return;
            if (X == null)
                MemClear(O);
            else
                MemCopyStride(X, O, strideX, length, length, count, index * length, 0);
        }

        /// <summary>
        /// Sets intermediate input tensors for Y_h and Y_c from intermediate output tensor
        /// if the calculation is single direction and sequenceFirst layout then the output
        /// tensor will be used directly and this command early outs
        /// </summary>
        void SetRnnOutput(Tensor<float> X, Tensor<float> O, int index, int count, int length, int strideO)
        {
            if (X == O)
                return;
            MemCopyStride(X, O, length, strideO, length, count, 0, index * length);
        }

        /// <inheritdoc/>
        public void LSTM(Tensor<float> X, Tensor<float> W, Tensor<float> R, Tensor<float> B, Tensor<int> sequenceLens, Tensor<float> initialH, Tensor<float> initialC, Tensor<float> P, Tensor<float> Y, Tensor<float> Yh, Tensor<float> Yc, Layers.RnnDirection direction, Layers.RnnActivation[] activations, float[] activationAlpha, float[] activationBeta, bool inputForget, float clip, Layers.RnnLayout layout)
        {
            var seqLength = X.shape[layout == Layers.RnnLayout.SequenceFirst ? 0 : 1];
            var batchSize = X.shape[layout == Layers.RnnLayout.SequenceFirst ? 1 : 0];
            var inputSize = X.shape[2];
            var hiddenSize = R.shape[2];
            var numDirections = W.shape[0];

            var W1 = numDirections == 2 ? AllocTensorFloat(new TensorShape(1, 4 * hiddenSize, inputSize)) : W;
            var R1 = numDirections == 2 ? AllocTensorFloat(new TensorShape(1, 4 * hiddenSize, hiddenSize)) : R;

            var Bi = B;
            if (B == null)
            {
                Bi = AllocTensorFloat(new TensorShape(numDirections, 8 * hiddenSize));
                MemClear(Bi);
            }
            var sequenceLensi = sequenceLens;
            if (sequenceLens == null)
            {
                sequenceLensi = AllocTensorInt(new TensorShape(batchSize));
                MemSet(sequenceLensi, math.asint(seqLength));
            }
            var Pi = P;
            if (P == null)
            {
                Pi = AllocTensorFloat(new TensorShape(numDirections, 3 * hiddenSize));
                MemClear(Pi);
            }

            var Y_h1 = layout == Layers.RnnLayout.SequenceFirst ? (numDirections == 2 ? AllocTensorFloat(new TensorShape(1, batchSize, hiddenSize)) : Yh) : AllocTensorFloat(new TensorShape(batchSize, 1, hiddenSize));
            var Y_c1 = layout == Layers.RnnLayout.SequenceFirst ? (numDirections == 2 ? AllocTensorFloat(new TensorShape(1, batchSize, hiddenSize)) : Yc) : AllocTensorFloat(new TensorShape(batchSize, 1, hiddenSize));

            var Y_hcLower = layout == Layers.RnnLayout.SequenceFirst ? batchSize * hiddenSize : hiddenSize;
            var Y_hcUpper = layout == Layers.RnnLayout.SequenceFirst ? 1 : batchSize;

            for (var i = 0; i < numDirections; i++)
            {
                SetRnnInput(W, W1, i, 1, 4 * hiddenSize * inputSize, 0);
                SetRnnInput(R, R1, i, 1, 4 * hiddenSize * hiddenSize, 0);
                SetRnnInput(initialH, Y_h1, i, Y_hcUpper, Y_hcLower, numDirections * Y_hcLower);
                SetRnnInput(initialC, Y_c1, i, Y_hcUpper, Y_hcLower, numDirections * Y_hcLower);
                var isReverse = direction == Layers.RnnDirection.Reverse || (direction == Layers.RnnDirection.Bidirectional && i == 1);
                SinglePassLSTM(X, W1, R1, Bi, sequenceLensi, Pi, Y, Y_h1, Y_c1, activations, activationAlpha, activationBeta, inputForget, clip, isReverse, i, layout);
                SetRnnOutput(Y_h1, Yh, i, Y_hcUpper, Y_hcLower, numDirections * Y_hcLower);
                SetRnnOutput(Y_c1, Yc, i, Y_hcUpper, Y_hcLower, numDirections * Y_hcLower);
            }

            if (numDirections == 2)
            {
                ReleaseTensorFloat(W1);
                ReleaseTensorFloat(R1);
            }
            if (B == null)
            {
                ReleaseTensorFloat(Bi);
            }
            if (sequenceLens == null)
            {
                ReleaseTensorInt(sequenceLensi);
            }
            if (P == null)
            {
                ReleaseTensorFloat(Pi);
            }
            if (layout != Layers.RnnLayout.SequenceFirst || numDirections == 2)
            {
                ReleaseTensorFloat(Y_h1);
                ReleaseTensorFloat(Y_c1);
            }
        }

        /// <inheritdoc/>
        public void DequantizeLinear(Tensor<byte> X, Tensor<float> O, float scale, byte zeroPoint)
        {
            var fn = ComputeFunctions.k_DequantizeUint8;
            cb.SetComputeFloatParam(fn.shader, k_ID_scale, scale);
            cb.SetComputeIntParam(fn.shader, k_ID_zeroPoint, (int)zeroPoint);
            cb.SetTensorAsBuffer(fn, k_ID_XIntptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetComputeIntParam(fn.shader, k_ID_lengthO, O.shape.length);
            cb.UnrolledDispatch(fn, X.count);
        }

        /// <inheritdoc/>
        public void Reshape(Tensor X, Tensor O)
        {
            MemCopy(X, O);
        }

        /// <summary>
        /// In place sort value tensor given key tensor either in ascending or descending order.
        /// Both input value and key tensors will be mutated.
        /// </summary>
        /// <param name="Value">The input value tensor. Must be of shape (N0, ... , Nn, L). Sort is happening on the L dimension </param>
        /// <param name="Key">The input key tensor. Must be of shape (N0, ... , Nn, L). Optional, if not given will sort Value.  </param>
        /// <param name="descending">Whether to sort in a ascending or descending order.</param>
        internal void BitonicSort(Tensor<float> Value, Tensor<int> Key = null, bool descending = false)
        {
            // https://en.wikipedia.org/wiki/Bitonic_sorter
            // https://stackoverflow.com/questions/73147204/can-bitonic-sort-handle-non-power-of-2-data-in-a-non-recursive-implementation
            int N = Value.shape[-1];
            int L = Value.shape.Length(0, -1);

            ComputeFunction fn;
            if (Key == null)
                fn = ComputeFunctions.k_BitonicSortStep;
            else
            {
                fn = ComputeFunctions.k_BitonicSortKeyStep;
                cb.SetTensorAsBuffer(fn, k_ID_O_int_ptr, Pin(Key));
            }
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(Value));
            cb.SetComputeIntParam(fn.shader, k_ID_lengthO, N);
            if (descending)
                cb.EnableKeyword(fn.shader, new LocalKeyword(fn.shader, "DESCENDING"));
            else
                cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, "DESCENDING"));

            for (int k = 1; k < N; k <<= 1)
            {
                cb.SetComputeIntParam(fn.shader, k_ID_indexJ, (k << 1) - 1);
                cb.Dispatch(fn, N, L, 1);

                for (int j = k >> 1; j > 0; j >>= 1)
                {
                    cb.SetComputeIntParam(fn.shader, k_ID_indexJ, j);
                    cb.Dispatch(fn, N, L, 1);
                }
            }
        }
    }
}
