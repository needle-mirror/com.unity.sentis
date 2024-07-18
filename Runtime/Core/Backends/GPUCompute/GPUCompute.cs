using UnityEngine;
using UnityEngine.Assertions;
using System;
using System.Runtime.CompilerServices;
using Unity.Mathematics;
using static Unity.Sentis.ComputeTensorData;
using static Unity.Sentis.ShaderPropertyID;

[assembly: InternalsVisibleTo("Unity.Sentis.RuntimeTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis
{
    /// <summary>
    /// Represents a GPUCompute backend ops.
    /// </summary>
    public partial class GPUComputeBackend : IBackend
    {
        /// <summary>
        /// Initializes and returns an instance of `GPUComputeOps`.
        /// </summary>
        public GPUComputeBackend() { }

        // Do we need this class or operate on ComputeTensorData instead?
        TensorClassPool<TensorFloat> m_TensorFloatPool = new TensorClassPool<TensorFloat>();
        TensorClassPool<TensorInt> m_TensorIntPool = new TensorClassPool<TensorInt>();
        TensorDataPool<ComputeTensorData> m_MemoryPool = new TensorDataPool<ComputeTensorData>();

        TensorFloat AllocTensorFloat(TensorShape shape)
        {
            ComputeTensorData data = m_MemoryPool.AdoptFromPool(shape.length);
            if (data == null)
                data = new ComputeTensorData(shape.length);
            var tensor = m_TensorFloatPool.AdoptFromPool();
            if (tensor == null)
                tensor = TensorFloat.AllocNoData(shape);

            tensor.shape = shape;
            tensor.dataOnBackend = data;
            return tensor;
        }

        TensorInt AllocTensorInt(TensorShape shape)
        {
            ComputeTensorData data = m_MemoryPool.AdoptFromPool(shape.length);
            if (data == null)
                data = new ComputeTensorData(shape.length);
            var tensor = m_TensorIntPool.AdoptFromPool();
            if (tensor == null)
                tensor = TensorInt.AllocNoData(shape);

            tensor.shape = shape;
            tensor.dataOnBackend = data;
            return tensor;
        }

        void ReleaseTensorFloat(TensorFloat tensor)
        {
            if (tensor == null)
                return;
            m_MemoryPool.ReleaseToPool(tensor.dataOnBackend as ComputeTensorData);
            tensor.dataOnBackend = null;
            m_TensorFloatPool.ReleaseToPool(tensor as TensorFloat);
        }

        void ReleaseTensorInt(TensorInt tensor)
        {
            if (tensor == null)
                return;
            m_MemoryPool.ReleaseToPool(tensor.dataOnBackend as ComputeTensorData);
            tensor.dataOnBackend = null;
            m_TensorIntPool.ReleaseToPool(tensor as TensorInt);
        }

        /// <summary>
        /// Disposes of the ops and any associated memory.
        /// </summary>
        public void Dispose()
        {
            m_MemoryPool?.Dispose();
            m_MemoryPool = null;
        }

        /// <inheritdoc/>
        public BackendType backendType => BackendType.GPUCompute;

        /// <inheritdoc/>
        public void MatMul2D(TensorFloat X, TensorFloat Y, TensorFloat O, bool xTranspose, bool yTranspose)
        {
            Gemm(X, Y, O, O.shape[0], xTranspose ? X.shape[0] : X.shape[1], O.shape[1], xTranspose, yTranspose);
        }

        /// <inheritdoc/>
        public void MatMul(TensorFloat X, TensorFloat Y, TensorFloat O)
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

                fn.SetInt6(k_ID_shapeA, shapeA);
                fn.SetInt6(k_ID_stridesA, stridesA);
                fn.SetInt6(k_ID_shapeB, shapeB);
                fn.SetInt6(k_ID_stridesB, stridesB);
                fn.SetInt6(k_ID_shapeO, shapeO);
                fn.SetInt6(k_ID_stridesO, stridesO);
            }

            fn.SetInt(k_ID_AM, M);
            fn.SetInt(k_ID_AN, K);
            fn.SetInt(k_ID_BM, K);
            fn.SetInt(k_ID_BN, N);
            fn.SetInt(k_ID_CB, batch);
            fn.SetInt(k_ID_CM, M);
            fn.SetInt(k_ID_CN, N);
            fn.SetInt(k_ID_rank, oShape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(Y));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.Dispatch(batch, M, N);
        }

        void BatchedGemm(TensorFloat X, TensorFloat Y, TensorFloat O, int batch, int M, int K, int N)
        {
            ComputeFunction fn;
            if (N % 64 == 0 && K % 16 == 0)
                fn = ComputeFunctions.k_GemmBatched_T16x16_R4x4;
            else
                fn = ComputeFunctions.k_GemmBatched_T8x8_R4x4;


            fn.SetInt(k_ID_maxXIndex, X.shape.length - 1);
            fn.SetInt(k_ID_maxWIndex, Y.shape.length - 1);
            fn.SetInt(k_ID_X_width, K);
            fn.SetInt(k_ID_W_width, N);
            fn.SetInt(k_ID_O_width, N);
            fn.SetInt(k_ID_O_height, M);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Wptr, Pin(Y));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(ComputeHelper.IDivC(N, 4), ComputeHelper.IDivC(M, 4), batch);
        }

        /// <inheritdoc/>
        public void Dense(TensorFloat X, TensorFloat W, TensorFloat B, TensorFloat O, Layers.FusableActivation fusedActivation)
        {
            var Otmp = (fusedActivation != Layers.FusableActivation.None) ? AllocTensorFloat(O.shape) : O;
            var M = Otmp.shape.Length(0, -1);
            var K = X.shape[-1];
            var N = Otmp.shape[-1];
            if (B != null)
                Gemm(X, W, B, Otmp, M, K, N);
            else
                Gemm(X, W, Otmp, M, K, N);

            if (fusedActivation != Layers.FusableActivation.None)
            {
                ApplyFusedActivation(Otmp, O, fusedActivation);
                ReleaseTensorFloat(Otmp);
            }
        }

        /// <inheritdoc/>
        public void Tril(Tensor X, Tensor O, int k)
        {
            // Warning, for some reason shared mem implementation on intel gpu is x2 faster than regular one
            var fn = ComputeFunctions.k_Tril;
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.SetInt(k_ID_X_width, X.shape[-1]);
            fn.SetInt(k_ID_X_height, X.shape[-2]);
            fn.SetInt(k_ID_X_length, X.shape.length);
            fn.SetInt(k_ID_diagonalK, k);

            fn.Dispatch(ComputeHelper.IDivC(X.shape.length, 4), 1, 1);
        }

        /// <inheritdoc/>
        public void Triu(Tensor X, Tensor O, int k)
        {
            // Warning, for some reason shared mem implementation on intel gpu is x2 faster than regular one
            var fn = ComputeFunctions.k_Triu;
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.SetInt(k_ID_X_width, X.shape[-1]);
            fn.SetInt(k_ID_X_height, X.shape[-2]);
            fn.SetInt(k_ID_X_length, X.shape.length);
            fn.SetInt(k_ID_diagonalK, k);

            fn.Dispatch(ComputeHelper.IDivC(X.shape.length, 4), 1, 1);
        }

        void ApplyFusedActivation(TensorFloat X, TensorFloat O, Layers.FusableActivation fusedActivation)
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
        public void Conv(TensorFloat X, TensorFloat K, TensorFloat B, TensorFloat O, int groups, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
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
                fn.SetInt(k_ID_O_depth, O.shape[2]);
                fn.SetInt(k_ID_O_height, O.shape[3]);
                fn.SetInt(k_ID_O_width, O.shape[4]);
                fn.SetInt(k_ID_X_depth, X.shape[2]);
                fn.SetInt(k_ID_X_height, X.shape[3]);
                fn.SetInt(k_ID_X_width, X.shape[4]);
                fn.SetInt(k_ID_K_depth, K.shape[2]);
                fn.SetInt(k_ID_K_height, K.shape[3]);
                fn.SetInt(k_ID_K_width, K.shape[4]);
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Kptr, Pin(K));
                if (B != null)
                {
                    fn.EnableKeyword("USEBIAS");
                    fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                }
                else
                {
                    fn.DisableKeyword("USEBIAS");
                }
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_O_batch, O.shape[0]); fn.SetInt(k_ID_O_channels, O.shape[1]);
                fn.SetInt(k_ID_X_channels, X.shape[1]);
                fn.SetInt4(k_ID__Stride, strides);
                fn.SetInt4(k_ID__Pad, pads);
                fn.SetInt4(k_ID__Dilation, dilations);
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
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Wptr, Pin(K));
                if (B != null)
                {
                    fn.EnableKeyword("USEBIAS");
                    fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                }
                else
                {
                    fn.DisableKeyword("USEBIAS");
                }
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_inputChannels, X.shape[1]);
                fn.SetInt(k_ID_inputHeight, X.shape[2]);
                fn.SetInt(k_ID_inputWidth, X.shape[3]);
                fn.SetInt(k_ID_kernelHeight, K.shape[2]);
                fn.SetInt(k_ID_kernelWidth, K.shape[3]);
                fn.SetInt(k_ID_outputChannels, O.shape[1]);
                fn.SetInt(k_ID_outputHeight, O.shape[2]);
                fn.SetInt(k_ID_outputWidth, O.shape[3]);
                fn.SetInt(k_ID_strideHeight, strides[0]);
                fn.SetInt(k_ID_strideWidth, strides[1]);
                fn.SetInt(k_ID_padHeight, pads[0]);
                fn.SetInt(k_ID_padWidth, pads[1]);
                fn.SetInt(k_ID_dilationHeight, dilations != null ? dilations[0] : 1);
                fn.SetInt(k_ID_dilationWidth, dilations != null ? dilations[1] : 1);
                fn.SetInt(k_ID_inputChannelsSize, X.shape[1] * X.shape[2] * X.shape[3]);
                fn.SetInt(k_ID_outputChannelsSize, O.shape[1] * O.shape[2] * O.shape[3]);
                fn.SetInt(k_ID_kernelChannelSize, K.shape[1] * K.shape[2] * K.shape[3]);
                fn.SetInt(k_ID_inputSize, X.shape[2] * X.shape[3]);
                fn.SetInt(k_ID_outputSize, O.shape[2] * O.shape[3]);
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
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Wptr, Pin(K));
                if (B != null)
                {
                    fn.EnableKeyword("USEBIAS");
                    fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                }
                else
                {
                    fn.DisableKeyword("USEBIAS");
                }
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_inputChannels, X.shape[1]);
                fn.SetInt(k_ID_inputHeight, X.shape[2]);
                fn.SetInt(k_ID_kernelHeight, K.shape[2]);
                fn.SetInt(k_ID_outputChannels, O.shape[1]);
                fn.SetInt(k_ID_outputHeight, O.shape[2]);
                fn.SetInt(k_ID_strideHeight, strides[0]);
                fn.SetInt(k_ID_padHeight, pads[0]);
                fn.SetInt(k_ID_dilationHeight, dilations[0]);
                fn.SetInt(k_ID_inputChannelsSize, X.shape[1] * X.shape[2]);
                fn.SetInt(k_ID_outputChannelsSize, O.shape[1] * O.shape[2]);
                fn.SetInt(k_ID_kernelChannelSize, K.shape[1] * K.shape[2]);
                fn.SetInt(k_ID_inputSize, X.shape[2]);
                fn.SetInt(k_ID_outputSize, O.shape[2]);
            }

            fn.SetInt(k_ID_kernelLength, K.shape.length);
            fn.SetFloat(k_ID__MinValue, fusedActivation == Layers.FusableActivation.Relu ? 0.0f : float.MinValue);

            fn.Dispatch(workItemsX, workItemsY, workItemsZ);
        }

        void ConvMobile(TensorFloat X, TensorFloat K, TensorFloat B, TensorFloat O, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
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
                fn.SetInt(k_ID_O_depth, O.shape[2]); fn.SetInt(k_ID_O_height, O.shape[3]); fn.SetInt(k_ID_O_width, O.shape[4]);
                fn.SetInt(k_ID_X_depth, X.shape[2]); fn.SetInt(k_ID_X_height, X.shape[3]); fn.SetInt(k_ID_X_width, X.shape[4]);
                fn.SetInt(k_ID_K_depth, K.shape[2]); fn.SetInt(k_ID_K_height, K.shape[3]); fn.SetInt(k_ID_K_width, K.shape[4]);
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
                fn.SetInt(k_ID_O_height, O.shape[2]); fn.SetInt(k_ID_O_width, O.shape[3]);
                fn.SetInt(k_ID_X_height, X.shape[2]); fn.SetInt(k_ID_X_width, X.shape[3]);
                fn.SetInt(k_ID_K_height, K.shape[2]); fn.SetInt(k_ID_K_width, K.shape[3]);
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
                fn.SetInt(k_ID_O_width, O.shape[2]);
                fn.SetInt(k_ID_X_width, X.shape[2]);
                fn.SetInt(k_ID_K_width, K.shape[2]);
                workItemsX = ComputeHelper.IDivC(k, 4);
                workItemsY = ComputeHelper.IDivC(w, 4);
                workItemsZ = n;
            }

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Kptr, Pin(K));
            if (B != null)
            {
                fn.EnableKeyword("USEBIAS");
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            }
            else
            {
                fn.DisableKeyword("USEBIAS");
            }
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.SetInt(k_ID_O_batch, O.shape[0]);
            fn.SetInt(k_ID_O_channels, O.shape[1]);
            fn.SetInt(k_ID_X_channels, X.shape[1]);
            fn.SetInt4(k_ID__Stride, strides);
            fn.SetInt4(k_ID__Pad, pads);
            fn.SetInt4(k_ID__Dilation, dilations);

            fn.SetFloat(k_ID__MinValue, fusedActivation == Layers.FusableActivation.Relu ? 0.0f : float.MinValue);

            fn.Dispatch(workItemsX, workItemsY, workItemsZ);
        }

        /// <inheritdoc/>
        public void ConvTranspose(TensorFloat X, TensorFloat W, TensorFloat B, TensorFloat O, Span<int> strides, Span<int> pads, Span<int> outputPadding, Layers.FusableActivation fusedActivation)
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

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Wptr, Pin(W));
            if (B != null)
            {
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.EnableKeyword("USEBIAS");
            }
            else
            {
                fn.DisableKeyword("USEBIAS");
            }
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.SetInt(k_ID_inputChannels, X.shape[1]);
            fn.SetInt(k_ID_outputChannels, O.shape[1]);
            fn.SetInt(k_ID_dilationHeight, 1);
            fn.SetInt(k_ID_dilationWidth, 1);

            var kernelSize = W.shape.Length(2);
            var inputSize = X.shape.Length(2);
            var outputSize = O.shape.Length(2);
            fn.SetInt(k_ID_kernelLength, W.shape.length);
            fn.SetInt(k_ID_kernelSize, kernelSize);
            fn.SetInt(k_ID_inputSize, inputSize);
            fn.SetInt(k_ID_outputSize, outputSize);
            fn.SetInt(k_ID_inputChannelsSize, X.shape[1] * inputSize);
            fn.SetInt(k_ID_outputChannelsSize, O.shape[1] * outputSize);
            fn.SetInt(k_ID_kernelChannelSize, W.shape[0] * kernelSize);
            fn.SetInt(k_ID_inputWidth, X.shape[-1]);
            fn.SetInt(k_ID_kernelWidth, W.shape[-1]);
            fn.SetInt(k_ID_outputWidth, O.shape[-1]);
            fn.SetInt(k_ID_padWidth, W.shape[-1] - pads[numSpatialDims - 1] - 1);
            fn.SetInt(k_ID_strideWidth, strides[numSpatialDims - 1]);
            if (numSpatialDims > 1)
            {
                fn.SetInt(k_ID_inputHeight, X.shape[-2]);
                fn.SetInt(k_ID_kernelHeight, W.shape[-2]);
                fn.SetInt(k_ID_outputHeight, O.shape[-2]);
                fn.SetInt(k_ID_padHeight, W.shape[-2] - pads[numSpatialDims - 2] - 1);
                fn.SetInt(k_ID_strideHeight, strides[numSpatialDims - 2]);
            }

            fn.SetFloat(k_ID__MinValue, fusedActivation == Layers.FusableActivation.Relu ? 0.0f : float.MinValue);

            fn.Dispatch(workItemsX, workItemsY, workItemsZ);
        }

        void ConvTransposeMobile(TensorFloat X, TensorFloat W, TensorFloat B, TensorFloat O, Span<int> stride, Span<int> pad, Span<int> outputAdjustment, Layers.FusableActivation fusedActivation)
        {
            ComputeFunction fn;

            var numSpatialDims = X.shape.rank - 2;

            if (numSpatialDims == 1)
                fn = ComputeFunctions.k_ConvTranspose1D_T16x16_R4x4;
            else if (numSpatialDims == 2)
                fn = ComputeFunctions.k_ConvTranspose2D_T16x16_R4x4;
            else
                fn = ComputeFunctions.k_ConvTranspose3D_T16x16_R4x4;

            fn.SetInt(k_ID_O_channels, O.shape[1]);
            fn.SetInt(k_ID_X_channels, X.shape[1]);
            fn.SetInt(k_ID_maxXIndex, X.shape.length - 1);
            fn.SetInt(k_ID_maxKIndex, W.shape.length - 1);
            fn.SetInt4(k_ID__Pad, pad);
            fn.SetInt4(k_ID__Stride, stride);

            fn.SetInt(k_ID_O_width, O.shape[-1]);
            fn.SetInt(k_ID_X_width, X.shape[-1]);
            fn.SetInt(k_ID_K_width, W.shape[-1]);

            if (numSpatialDims > 1)
            {
                fn.SetInt(k_ID_O_height, O.shape[-2]);
                fn.SetInt(k_ID_X_height, X.shape[-2]);
                fn.SetInt(k_ID_K_height, W.shape[-2]);
            }

            if (numSpatialDims > 2)
            {
                fn.SetInt(k_ID_O_depth, O.shape[-3]);
                fn.SetInt(k_ID_X_depth, X.shape[-3]);
                fn.SetInt(k_ID_K_depth, W.shape[-3]);
            }

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Kptr, Pin(W));
            if (B != null)
            {
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetInt(k_ID_maxBIndex, B.shape.length - 1);
                fn.EnableKeyword("USEBIAS");
            }
            else
            {
                fn.DisableKeyword("USEBIAS");
            }
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            var workItemsX = ComputeHelper.IDivC(O.shape[1], 4);
            var workItemsY = ComputeHelper.IDivC(O.shape.Length(2), 4);
            var workItemsZ = O.shape[0];
            if (fusedActivation == Layers.FusableActivation.Relu)
                fn.SetFloat(k_ID__MinValue, 0.0f);
            else
                fn.SetFloat(k_ID__MinValue, float.MinValue);

            fn.Dispatch(workItemsX, workItemsY, workItemsZ);
        }

        /// <inheritdoc/>
        public void Resize(TensorFloat X, TensorFloat O, ReadOnlySpan<float> scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode, Layers.CoordTransformMode coordTransformMode)
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
        public void GridSample(TensorFloat X, TensorFloat grid, TensorFloat O, Layers.InterpolationMode mode, Layers.PaddingMode paddingMode, bool alignCorners)
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

                    fn.SetInt(k_ID_inDepth, xD);
                    fn.SetInt(k_ID_outDepth, oD);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            fn.SetInt(k_ID_outBatch, n);
            fn.SetInt(k_ID_outChannels, c);
            fn.SetInt(k_ID_inHeight, xH);
            fn.SetInt(k_ID_inWidth, xW);
            fn.SetInt(k_ID_outHeight, oH);
            fn.SetInt(k_ID_outWidth, oW);
            fn.SetInt(k_ID_inSpatialSize, xSpatialDim);
            fn.SetInt(k_ID_outSpatialSize, oSpatialDim);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Sptr, Pin(grid));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            switch (mode)
            {
                case Layers.InterpolationMode.Nearest:
                    fn.DisableKeyword("LINEAR");
                    break;
                case Layers.InterpolationMode.Linear:
                    fn.EnableKeyword("LINEAR");
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(mode), mode, null);
            }

            switch (paddingMode)
            {
                case Layers.PaddingMode.Zeros:
                    fn.DisableKeyword("BORDER");
                    fn.DisableKeyword("REFLECTION");
                    break;
                case Layers.PaddingMode.Border:
                    fn.DisableKeyword("REFLECTION");
                    fn.EnableKeyword("BORDER");
                    break;
                case Layers.PaddingMode.Reflection:
                    fn.DisableKeyword("BORDER");
                    fn.EnableKeyword("REFLECTION");
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(paddingMode), paddingMode, null);
            }

            if (alignCorners)
                fn.EnableKeyword("ALIGN_CORNERS");
            else
                fn.DisableKeyword("ALIGN_CORNERS");

            fn.Dispatch(oSpatialDim, c, n);
        }

        void ResizeND(TensorFloat X, TensorFloat O, ReadOnlySpan<float> scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode, Layers.CoordTransformMode coordTransformMode)
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

        void Resize1D(TensorFloat X, TensorFloat O, int axis, float scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode, Layers.CoordTransformMode coordTransformMode)
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

            fn.SetFloat(k_ID_scale1D, outputScale);
            fn.SetFloat(k_ID_bias1D, outputBias);
            fn.SetInt(k_ID_innerLength, innerLength);
            fn.SetInt(k_ID_outerLength, outerLength);
            fn.SetInt(k_ID_inWidth, X.shape[axis]);
            fn.SetInt(k_ID_outWidth, O.shape[axis]);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(outerLength, O.shape[axis], innerLength);
        }

        void Upsample1D(TensorFloat X, TensorFloat O, ReadOnlySpan<float> scale, Layers.NearestMode nearestMode, Layers.InterpolationMode interpolationMode, Layers.CoordTransformMode coordTransformMode)
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

            fn.SetFloat(k_ID_scale1D, outputScale);
            fn.SetFloat(k_ID_bias1D, outputBias);
            fn.SetInt(k_ID_inWidth, X.shape[2]);
            fn.SetInt(k_ID_outWidth, O.shape[2]);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(O.shape[0] * O.shape[1], O.shape[2], 1);
        }

        void Upsample2D(TensorFloat X, TensorFloat O, ReadOnlySpan<float> scale, Layers.NearestMode nearestMode, Layers.InterpolationMode interpolationMode, Layers.CoordTransformMode coordTransformMode)
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

            fn.SetVector(k_ID_scale, scaleXY);
            fn.SetVector(k_ID_bias, biasXY);
            fn.SetInt(k_ID_inHeight, X.shape[2]);
            fn.SetInt(k_ID_inWidth, X.shape[3]);
            fn.SetInt(k_ID_outHeight, O.shape[2]);
            fn.SetInt(k_ID_outWidth, O.shape[3]);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(O.shape[0] * O.shape[1], O.shape[2], O.shape[3]);
        }

        void Upsample3D(TensorFloat X, TensorFloat O, ReadOnlySpan<float> scale, Layers.NearestMode nearestMode, Layers.InterpolationMode interpolationMode, Layers.CoordTransformMode coordTransformMode)
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

            fn.SetVector(k_ID_scale, scaleXYD);
            fn.SetVector(k_ID_bias, biasXYD);
            fn.SetInt(k_ID_inDepth, X.shape[2]);
            fn.SetInt(k_ID_inHeight, X.shape[3]);
            fn.SetInt(k_ID_inWidth, X.shape[4]);
            fn.SetInt(k_ID_outBatch, O.shape[0]);
            fn.SetInt(k_ID_outChannels, O.shape[1]);
            fn.SetInt(k_ID_outDepth, O.shape[2]);
            fn.SetInt(k_ID_outHeight, O.shape[3]);
            fn.SetInt(k_ID_outWidth, O.shape[4]);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(O.shape[2], O.shape[3], O.shape[4]);
        }

        /// <inheritdoc/>
        public void DepthToSpace(TensorFloat X, TensorFloat O, int blocksize, Layers.DepthToSpaceMode mode)
        {
            var fn = (mode == Layers.DepthToSpaceMode.DepthColumnRow) ? ComputeFunctions.k_DepthToSpaceDepthColumnRow : ComputeFunctions.k_DepthToSpaceColumnRowDepth;
            fn.SetInt(k_ID_blocksize, blocksize);
            fn.SetInt(k_ID_inputChannels, X.shape[1]);
            fn.SetInt(k_ID_inputHeight, X.shape[2]);
            fn.SetInt(k_ID_inputWidth, X.shape[3]);
            fn.SetInt(k_ID_inputSpatialSize, X.shape[2] * X.shape[3]);
            fn.SetInt(k_ID_inputBatchOffset, X.shape[1] * X.shape[2] * X.shape[3]);
            fn.SetInt(k_ID_outputChannels, O.shape[1]);
            fn.SetInt(k_ID_outputHeight, O.shape[2]);
            fn.SetInt(k_ID_outputWidth, O.shape[3]);
            fn.SetInt(k_ID_outputSpatialSize, O.shape[2] * O.shape[3]);
            fn.SetInt(k_ID_outputBatch, O.shape[0]);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(O.shape[0] * O.shape[1], O.shape[2] * O.shape[3], 1);
        }

        /// <inheritdoc/>
        public void SpaceToDepth(TensorFloat X, TensorFloat O, int blocksize)
        {
            var fn = ComputeFunctions.k_SpaceToDepth;
            fn.SetInt(k_ID_blocksize, blocksize);
            fn.SetInt(k_ID_inputChannels, X.shape[1]);
            fn.SetInt(k_ID_inputHeight, X.shape[2]);
            fn.SetInt(k_ID_inputWidth, X.shape[3]);
            fn.SetInt(k_ID_inputSpatialSize, X.shape[2] * X.shape[3]);
            fn.SetInt(k_ID_inputBatchOffset, X.shape[1] * X.shape[2] * X.shape[3]);
            fn.SetInt(k_ID_outputChannels, O.shape[1]);
            fn.SetInt(k_ID_outputHeight, O.shape[2]);
            fn.SetInt(k_ID_outputWidth, O.shape[3]);
            fn.SetInt(k_ID_outputSpatialSize, O.shape[2] * O.shape[3]);
            fn.SetInt(k_ID_outputBatch, O.shape[0]);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(O.shape[0] * O.shape[1], O.shape[2] * O.shape[3], 1);
        }

        /// <inheritdoc/>
        public void GlobalAverageVariancePool(TensorFloat X, TensorFloat O, int axis)
        {
            int globalNonSpatialLength = X.shape.Length(0, axis);
            int globalSpatialDims = X.shape.length / globalNonSpatialLength;

            int localSpatialLength = globalSpatialDims;

            var Oshape = new TensorShape(globalNonSpatialLength, localSpatialLength);

            TensorFloat X2 = X; // save a X^2 and do it in the first dispatch
            bool isFirstDispatch = true;

            // downsample with pyramid approach
            while (localSpatialLength > 64 * 4)
            {
                int spatialLengthO = ComputeHelper.IDivC(localSpatialLength, 64 * 4);
                Oshape[-1] = spatialLengthO;
                var Otemp = AllocTensorFloat(Oshape);
                var O2temp = AllocTensorFloat(Oshape);

                var fnPool = ComputeFunctions.k_AverageVariancePoolReduce;
                fnPool.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fnPool.SetTensorAsBuffer(k_ID_X2ptr, Pin(X2));
                fnPool.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                fnPool.SetTensorAsBuffer(k_ID_O2ptr, Pin(O2temp));
                fnPool.SetInt(k_ID_SpatialDims, localSpatialLength);
                fnPool.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                fnPool.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                fnPool.Dispatch(globalNonSpatialLength, ComputeHelper.IDivC(localSpatialLength, 4), 1);

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
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_X2ptr, Pin(X2));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.SetInt(k_ID_SpatialDims, localSpatialLength);
            fn.SetInt(k_ID_GlobalSpatialDims, globalSpatialDims);
            fn.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

            fn.Dispatch(globalNonSpatialLength, 1, 1);

            if (!isFirstDispatch)
            {
                ReleaseTensorFloat(X);
                ReleaseTensorFloat(X2);
            }
        }

        void GroupedConv(TensorFloat X, TensorFloat K, TensorFloat B, TensorFloat O, int groups, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
        {
            var Otmp = (fusedActivation != Layers.FusableActivation.None) ? AllocTensorFloat(O.shape) : O;

            int outputGroupedChannels = Otmp.shape[1] / groups;

            ComputeFunction fn;

            if (X.shape.rank == 5)
            {
                fn = (outputGroupedChannels < 64) ? ComputeFunctions.k_GroupedConv3D : ComputeFunctions.k_GroupedConv3D_GroupLower64;
                fn.SetInt(k_ID_O_depth, Otmp.shape[2]);
                fn.SetInt(k_ID_O_height, Otmp.shape[3]);
                fn.SetInt(k_ID_O_width, Otmp.shape[4]);
                fn.SetInt(k_ID_X_depth, X.shape[2]);
                fn.SetInt(k_ID_X_height, X.shape[3]);
                fn.SetInt(k_ID_X_width, X.shape[4]);
                fn.SetInt(k_ID_K_depth, K.shape[2]);
                fn.SetInt(k_ID_K_height, K.shape[3]);
                fn.SetInt(k_ID_K_width, K.shape[4]);
            }
            else if (X.shape.rank == 4)
            {
                fn = (outputGroupedChannels < 64) ? ComputeFunctions.k_GroupedConv2D : ComputeFunctions.k_GroupedConv2D_GroupLower64;
                fn.SetInt(k_ID_O_height, Otmp.shape[2]);
                fn.SetInt(k_ID_O_width, Otmp.shape[3]);
                fn.SetInt(k_ID_X_height, X.shape[2]);
                fn.SetInt(k_ID_X_width, X.shape[3]);
                fn.SetInt(k_ID_K_height, K.shape[2]);
                fn.SetInt(k_ID_K_width, K.shape[3]);
            }
            else
            {
                fn = (outputGroupedChannels < 64) ? ComputeFunctions.k_GroupedConv1D : ComputeFunctions.k_GroupedConv1D_GroupLower64;
                fn.SetInt(k_ID_O_width, Otmp.shape[2]);
                fn.SetInt(k_ID_X_width, X.shape[2]);
                fn.SetInt(k_ID_K_width, K.shape[2]);
            }

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Kptr, Pin(K));
            if (B != null)
            {
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.EnableKeyword("USEBIAS");
            }
            else
            {
                fn.DisableKeyword("USEBIAS");
            }
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(Otmp));
            fn.SetInt(k_ID_O_channels, O.shape[1]);
            fn.SetInt(k_ID_X_channels, X.shape[1]);
            fn.SetInt4(k_ID__Stride, strides);
            fn.SetInt4(k_ID__Pad, pads);
            fn.SetInt4(k_ID__Dilation, dilations);
            fn.SetInt(k_ID__Groups, groups);
            fn.SetInt(k_ID_strideX, X.shape.Length(2));
            fn.SetInt(k_ID_strideO, Otmp.shape.Length(2));
            fn.SetInt(k_ID_strideK, K.shape.Length(2));
            fn.SetInt(k_ID_inputGroupedChannels, X.shape[1] / groups);
            fn.SetInt(k_ID_outputGroupedChannels, Otmp.shape[1] / groups);

            fn.Dispatch(ComputeHelper.IDivC(Otmp.shape[1], 4), ComputeHelper.IDivC(Otmp.shape.Length(2), 4), Otmp.shape[0]);

            if (fusedActivation != Layers.FusableActivation.None)
            {
                ApplyFusedActivation(Otmp, O, fusedActivation);
                ReleaseTensorFloat(Otmp);
            }
        }

        void DepthwiseConv2D(TensorFloat X, TensorFloat K, TensorFloat B, TensorFloat O, int group, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
        {
            var Otmp = (fusedActivation != Layers.FusableActivation.None) ? AllocTensorFloat(O.shape) : O;

            ComputeFunction fn;
            int workItemsX, workItemsY, workItemsZ;

            TensorFloat KWE = null;
            if (K.shape[2] == 3 && K.shape[3] == 3 && strides[0] == 1 && strides[1] == 1 && dilations[0] == 1 && dilations[1] == 1)
            {
                KWE = AllocTensorFloat(new TensorShape(Otmp.shape[1], 4, 4));

                ComputeFunction fnKE = ComputeFunctions.k_KernelWinoExpand;
                fnKE.SetTensorAsBuffer(k_ID_Kptr, Pin(K));
                fnKE.SetTensorAsBuffer(k_ID_Optr, Pin(KWE));
                fnKE.SetInt(k_ID_O_channels, O.shape[1]);
                fnKE.Dispatch(O.shape[1], 1, 1);

                fn = ComputeFunctions.k_DepthwiseConv2DWinograd;

                fn.SetTensorAsBuffer(k_ID_KWEptr, Pin(KWE));

                workItemsX = ComputeHelper.IDivC(Otmp.shape[3], 2);
                workItemsY = ComputeHelper.IDivC(Otmp.shape[2], 2);
                workItemsZ = Otmp.shape[0] * Otmp.shape[1];
            }
            else
            {
                fn = ComputeFunctions.k_DepthwiseConv2DDirect;

                fn.SetTensorAsBuffer(k_ID_Kptr, Pin(K));

                fn.SetInt(k_ID_K_heightDiv4, ComputeHelper.IDivC(K.shape[2], 4));
                fn.SetInt(k_ID_K_widthDiv4, ComputeHelper.IDivC(K.shape[3], 4));
                fn.SetInt(k_ID_K_height, K.shape[2]);
                fn.SetInt(k_ID_K_width, K.shape[3]);
                fn.SetInt(k_ID_StrideK, K.shape[2] * K.shape[3]);

                workItemsX = Otmp.shape[3];
                workItemsY = Otmp.shape[2];
                workItemsZ = Otmp.shape[0] * Otmp.shape[1];
            }

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            if (B != null)
            {
                fn.EnableKeyword("USEBIAS");
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            }
            else
            {
                fn.DisableKeyword("USEBIAS");
            }
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(Otmp));
            fn.SetInt(k_ID_X_channels, X.shape[1]);
            fn.SetInt(k_ID_X_height, X.shape[2]);
            fn.SetInt(k_ID_X_width, X.shape[3]);
            fn.SetInt(k_ID_O_batch, O.shape[0]);
            fn.SetInt(k_ID_O_channels, O.shape[1]);
            fn.SetInt(k_ID_O_height, O.shape[2]);
            fn.SetInt(k_ID_O_width, O.shape[3]);
            fn.SetInt4(k_ID_Stride, strides);
            fn.SetInt4(k_ID_Pad, pads);
            fn.SetInt4(k_ID_Dilation, dilations);
            fn.SetInt(k_ID_StrideX, X.shape[2] * X.shape[3]);
            fn.SetInt(k_ID_MaxLengthX, X.shape.length - 1);
            fn.SetInt(k_ID_MaxLengthK, K.shape.length - 1);
            fn.SetInt(k_ID_StrideO, Otmp.shape[2] * Otmp.shape[3]);
            fn.SetInt(k_ID_StrideFeaturesO, Otmp.shape[0] * Otmp.shape[1]);

            fn.Dispatch(workItemsX, workItemsY, workItemsZ);
            ReleaseTensorFloat(KWE);

            if (fusedActivation != Layers.FusableActivation.None)
            {
                ApplyFusedActivation(Otmp, O, fusedActivation);
                ReleaseTensorFloat(Otmp);
            }
        }

        /// <inheritdoc/>
        public void ScaleBias(TensorFloat X, TensorFloat S, TensorFloat B, TensorFloat O)
        {
            int batch = X.shape[0];
            int channels = X.shape[1];
            int spatialDims = X.shape.Length(2);

            var fn = ComputeFunctions.k_ScaleBias;

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Sptr, Pin(S));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.SetInt(k_ID_LengthO, O.shape.length);
            fn.SetInt(k_ID_batch, batch);
            fn.SetInt(k_ID_channels, channels);
            fn.SetInt(k_ID_spatialDims, spatialDims);
            fn.Dispatch(spatialDims, ComputeHelper.IDivC(channels, 4), batch);
        }

        /// <inheritdoc/>
        public void InstanceNormalization(TensorFloat X, TensorFloat S, TensorFloat B, TensorFloat O, float epsilon)
        {
            var reduceOpShape = ShapeInference.GlobalAverageVariancePool(X.shape);
            var meanVariance = AllocTensorFloat(reduceOpShape);
            GlobalAverageVariancePool(X, meanVariance, 2);

            var fn = ComputeFunctions.k_InstanceNormalizationTail;

            fn.SetInt(k_ID_channels, X.shape[1]);
            fn.SetInt(k_ID_spatialDims, X.shape.length / (X.shape[0] * X.shape[1]));
            fn.SetFloat(k_ID_epsilon, epsilon);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Sptr, Pin(S));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Wptr, Pin(meanVariance));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.UnrolledDispatch(O.shape.length);
            ReleaseTensorFloat(meanVariance);
        }

        /// <inheritdoc/>
        public void LayerNormalization(TensorFloat X, TensorFloat S, TensorFloat B, TensorFloat O, float epsilon)
        {
            int axis = X.shape.Axis(-1);

            var reducedShape = X.shape.Reduce(axis);
            reducedShape[axis] = 2;

            int axisDim = X.shape[axis];
            int outerLength = X.shape.Length(0, -1);

            var meanVariance = AllocTensorFloat(reducedShape);
            GlobalAverageVariancePool(X, meanVariance, -1);

            var fn = ComputeFunctions.k_LayerNormalizationTail;
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Wptr, Pin(meanVariance));
            fn.SetTensorAsBuffer(k_ID_Sptr, Pin(S));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.SetInt(k_ID_axisDim, axisDim);
            fn.SetInt(k_ID_outerLength, outerLength);
            fn.SetFloat(k_ID_epsilon, epsilon);
            fn.Dispatch(axisDim, outerLength, 1);

            ReleaseTensorFloat(meanVariance);
        }

        /// <inheritdoc/>
        public void BatchNormalization(TensorFloat X, TensorFloat S, TensorFloat B, TensorFloat mean, TensorFloat variance, TensorFloat O, float epsilon)
        {
            var batch = X.shape[0];
            var channels = X.shape.rank == 1 ? 1 : X.shape[1];
            var spatialDims = X.shape.Length(2);

            var fn = ComputeFunctions.k_BatchNormalization;

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Wptr, Pin(mean));
            fn.SetTensorAsBuffer(k_ID_Zptr, Pin(variance));
            fn.SetTensorAsBuffer(k_ID_Sptr, Pin(S));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.SetInt(k_ID_LengthO, O.shape.length);
            fn.SetInt(k_ID_batch, batch);
            fn.SetInt(k_ID_channels, channels);
            fn.SetInt(k_ID_spatialDims, spatialDims);
            fn.SetFloat(k_ID_epsilon, epsilon);
            fn.Dispatch(spatialDims, ComputeHelper.IDivC(channels, 4), batch);
        }

        /// <inheritdoc/>
        public void Range(TensorFloat O, float start, float delta)
        {
            var fn = ComputeFunctions.k_RangeFloat;
            fn.SetFloat(k_ID_alpha, start);
            fn.SetFloat(k_ID_beta, delta);
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Range(TensorInt O, int start, int delta)
        {
            var fn = ComputeFunctions.k_RangeInt;
            fn.SetInt(k_ID_alphai, start);
            fn.SetInt(k_ID_betai, delta);
            fn.SetTensorAsBuffer(k_ID_O_int_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Relu(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Relu;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Relu6(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Relu6;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void LeakyRelu(TensorFloat X, TensorFloat O, float alpha)
        {
            var fn = ComputeFunctions.k_LeakyRelu;
            fn.SetFloat(k_ID_alpha, 0.5f * (1f + alpha));
            fn.SetFloat(k_ID_beta, 0.5f * (1f - alpha));
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Tanh(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Tanh;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Softplus(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Softplus;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Sigmoid(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Sigmoid;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void HardSigmoid(TensorFloat X, TensorFloat O, float alpha, float beta)
        {
            var fn = ComputeFunctions.k_HardSigmoid;
            fn.SetFloat(k_ID_alpha, alpha);
            fn.SetFloat(k_ID_beta, beta);
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Elu(TensorFloat X, TensorFloat O, float alpha)
        {
            var fn = ComputeFunctions.k_Elu;
            fn.SetFloat(k_ID_alpha, alpha);
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Gelu(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Gelu;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void GeluFast(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_GeluFast;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Selu(TensorFloat X, TensorFloat O, float alpha, float gamma)
        {
            var fn = ComputeFunctions.k_Selu;
            fn.SetFloat(k_ID_alpha, alpha);
            fn.SetFloat(k_ID_gamma, gamma);
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Swish(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Swish;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Abs(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_AbsFloat;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Abs(TensorInt X, TensorInt O)
        {
            var fn = ComputeFunctions.k_AbsInt;
            fn.SetTensorAsBuffer(k_ID_X_int_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_int_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Neg(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_NegFloat;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Neg(TensorInt X, TensorInt O)
        {
            var fn = ComputeFunctions.k_NegInt;
            fn.SetTensorAsBuffer(k_ID_X_int_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_int_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Ceil(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Ceil;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Clip(TensorFloat X, TensorFloat O, float min, float max)
        {
            var fn = ComputeFunctions.k_ClipFloat;
            fn.SetFloat(k_ID_alpha, min);
            fn.SetFloat(k_ID_beta, max);
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Clip(TensorInt X, TensorInt O, int min, int max)
        {
            var fn = ComputeFunctions.k_ClipInt;
            fn.SetInt(k_ID_alphai, min);
            fn.SetInt(k_ID_betai, max);
            fn.SetTensorAsBuffer(k_ID_X_int_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_int_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Floor(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Floor;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Round(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Round;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Reciprocal(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Reciprocal;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Square(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_SquareFloat;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Square(TensorInt X, TensorInt O)
        {
            var fn = ComputeFunctions.k_SquareInt;
            fn.SetTensorAsBuffer(k_ID_X_int_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_int_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Exp(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Exp;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Log(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Log;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Sqrt(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Sqrt;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Acos(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Acos;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Acosh(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Acosh;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Asin(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Asin;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Asinh(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Asinh;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Atan(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Atan;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Atanh(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Atanh;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Cos(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Cos;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Cosh(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Cosh;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Sin(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Sin;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Sinh(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Sinh;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Tan(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Tan;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Erf(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Erf;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Celu(TensorFloat X, TensorFloat O, float alpha)
        {
            var fn = ComputeFunctions.k_Celu;
            fn.SetFloat(k_ID_alpha, alpha);
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Shrink(TensorFloat X, TensorFloat O, float bias, float lambd)
        {
            var fn = ComputeFunctions.k_Shrink;
            fn.SetFloat(k_ID_alpha, bias);
            fn.SetFloat(k_ID_beta, lambd);
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Softsign(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_Softsign;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void ThresholdedRelu(TensorFloat X, TensorFloat O, float alpha)
        {
            var fn = ComputeFunctions.k_ThresholdedRelu;
            fn.SetFloat(k_ID_alpha, alpha);
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Softmax(TensorFloat X, TensorFloat O, int axis)
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
            fn.SetInt(k_ID_innerLength, innerLength);
            fn.SetInt(k_ID_reduceLength, reduceLength);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Sptr, Pin(XexpSums));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(Xmax));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.UnrolledDispatch(O.shape.length);

            ReleaseTensorFloat(Xmax);
            ReleaseTensorFloat(XexpSums);
        }

        /// <inheritdoc/>
        public void LogSoftmax(TensorFloat X, TensorFloat O, int axis)
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
            fn.SetInt(k_ID_innerLength, innerLength);
            fn.SetInt(k_ID_reduceLength, reduceLength);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(XexpSums));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.UnrolledDispatch(O.shape.length);

            ReleaseTensorFloat(Xmax);
            ReleaseTensorFloat(XexpSums);
        }

        /// <inheritdoc/>
        public void Hardmax(TensorFloat X, TensorFloat O, int axis)
        {
            //Allocate temp tensors
            var reduceOpShape = X.shape.Reduce(axis);
            var argMax = AllocTensorFloat(reduceOpShape);

            int offsetReduce = X.shape.Strides(axis);

            // argmax
            {
                var fn = ComputeFunctions.k_ArgMaxFloatFirst;
                fn.SetInt(k_ID_innerLength, offsetReduce);
                fn.SetInt(k_ID_reduceLength, X.shape[axis]);
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(argMax));
                fn.UnrolledDispatch(reduceOpShape.length);
            }
            // one hot from argmax
            {
                var fn = ComputeFunctions.k_HardmaxEnd;
                fn.SetInt(k_ID_innerLength, offsetReduce);
                fn.SetInt(k_ID_reduceLength, X.shape[axis]);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(argMax));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }

            ReleaseTensorFloat(argMax);
        }

        /// <inheritdoc/>
        public void ScalarMad(TensorFloat X, TensorFloat O, float s, float b)
        {
            var fn = ComputeFunctions.k_ScalarMadFloat;
            fn.SetFloat(k_ID_alpha, s);
            fn.SetFloat(k_ID_beta, b);
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void ScalarMad(TensorInt X, TensorInt O, int s, int b)
        {
            var fn = ComputeFunctions.k_ScalarMadInt;
            fn.SetInt(k_ID_alphai, s);
            fn.SetInt(k_ID_betai, b);
            fn.SetTensorAsBuffer(k_ID_X_int_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_int_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void CumSum(TensorFloat X, TensorFloat O, int axis, bool reverse, bool exclusive)
        {
            var reduceOpShape = X.shape.Reduce(axis);
            var offsetReduce = X.shape.Strides(axis);

            var fn = (reverse ? (exclusive ? ComputeFunctions.k_CumSumFloatReverseExclusive : ComputeFunctions.k_CumSumFloatReverseInclusive) : (exclusive ? ComputeFunctions.k_CumSumFloatForwardExclusive : ComputeFunctions.k_CumSumFloatForwardInclusive));
            fn.SetInt(k_ID_innerLength, offsetReduce);
            fn.SetInt(k_ID_reduceLength, X.shape[axis]);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(reduceOpShape.length);
        }

        /// <inheritdoc/>
        public void CumSum(TensorInt X, TensorInt O, int axis, bool reverse, bool exclusive)
        {
            var reduceOpShape = X.shape.Reduce(axis);
            var offsetReduce = X.shape.Strides(axis);

            var fn = (reverse ? (exclusive ? ComputeFunctions.k_CumSumIntReverseExclusive : ComputeFunctions.k_CumSumIntReverseInclusive) : (exclusive ? ComputeFunctions.k_CumSumIntForwardExclusive : ComputeFunctions.k_CumSumIntForwardInclusive));
            fn.SetInt(k_ID_innerLength, offsetReduce);
            fn.SetInt(k_ID_reduceLength, X.shape[axis]);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(reduceOpShape.length);
        }

        /// <inheritdoc/>
        public void Einsum(TensorFloat[] inputTensors, TensorFloat O, TensorIndex[] operandIndices, TensorIndex outputIndices, TensorIndex sumIndices, TensorShape sumShape)
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
                        fn.SetInt8(k_ID_outStridesA, outStridesA);
                        fn.SetInt8(k_ID_sumStridesA, sumStridesA);

                        fn.SetTensorShapeStrides(k_ID_outLengths, k_ID_outStrides, O.shape);
                        fn.SetTensorShapeStrides(k_ID_sumLengths, k_ID_sumStrides, sumShape);
                    }

                    fn.SetInt(k_ID_sumSize, sumShape.length);
                    fn.SetInt(k_ID_sumRank, sumShape.rank);
                    fn.SetInt(k_ID_outRank, O.shape.rank);

                    fn.SetTensorAsBuffer(k_ID_Xptr, Pin(inputTensors[0]));
                    fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                    fn.UnrolledDispatch(O.shape.length);
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
                        fn.SetInt8(k_ID_outStridesA, outStridesA);
                        fn.SetInt8(k_ID_sumStridesA, sumStridesA);

                        var outStridesB = stackalloc int[TensorShape.maxRank];
                        var sumStridesB = stackalloc int[TensorShape.maxRank];
                        EinsumHelper.PinOperandStrides(inputTensors[1].shape, operandIndices[1], outputIndices, sumIndices, outStridesB, sumStridesB);
                        fn.SetInt8(k_ID_outStridesB, outStridesB);
                        fn.SetInt8(k_ID_sumStridesB, sumStridesB);

                        fn.SetTensorShapeStrides(k_ID_outLengths, k_ID_outStrides, O.shape);
                        fn.SetTensorShapeStrides(k_ID_sumLengths, k_ID_sumStrides, sumShape);
                    }

                    fn.SetInt(k_ID_sumSize, sumShape.length);
                    fn.SetInt(k_ID_sumRank, sumShape.rank);
                    fn.SetInt(k_ID_outRank, O.shape.rank);

                    fn.SetTensorAsBuffer(k_ID_Xptr, Pin(inputTensors[0]));
                    fn.SetTensorAsBuffer(k_ID_Bptr, Pin(inputTensors[1]));
                    fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                    fn.UnrolledDispatch(O.shape.length);
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
        public void NonMaxSuppression(TensorFloat boxes, TensorFloat scores, TensorInt O, int maxOutputBoxesPerClass, float iouThreshold, float scoreThreshold, Layers.CenterPointBox centerPointBox)
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
                fn.SetInt(k_ID_numBatches, numBatches);
                fn.SetInt(k_ID_numBoxes, numBoxes);
                fn.SetFloat(k_ID_iouThreshold, iouThreshold);
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(boxes));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(bitmask));
                fn.Dispatch(numBatches, numBoxes, numBoxes);
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
                fn.SetInt(k_ID_numBatches, numBatches);
                fn.SetInt(k_ID_numBoxes, numBoxes);
                fn.SetInt(k_ID_numClasses, numClasses);
                fn.SetFloat(k_ID_scoreThreshold, scoreThreshold);
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(bitmask));
                fn.SetTensorAsBuffer(k_ID_Sptr, Pin(scoresSorted));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(indicesSorted));
                fn.SetTensorAsBuffer(k_ID_Wptr, Pin(bitmaskOverlap));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(selected));
                fn.Dispatch(1, numClasses, numBatches);
            }

            // compaction
            var numSelected = AllocTensorInt(new TensorShape());
            {
                var fn = ComputeFunctions.k_NMSCompact;
                fn.SetInt(k_ID_numBatches, numBatches);
                fn.SetInt(k_ID_numBoxes, numBoxes);
                fn.SetInt(k_ID_numClasses, numClasses);
                fn.SetInt(k_ID_maxNumOutput, maxNumOutput);
                fn.SetInt(k_ID_maxOutputBoxesPerClass, maxOutputBoxesPerClass);
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(selected));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetTensorAsBuffer(k_ID_Iptr, Pin(numSelected));
                fn.Dispatch(1, 1, 1);
            }

            ReleaseTensorFloat(scoresSorted);
            ReleaseTensorInt(order);
            ReleaseTensorInt(indicesSorted);
            ReleaseTensorInt(selected);
            ReleaseTensorInt(bitmask);
            ReleaseTensorInt(bitmaskOverlap);
            var numSelectedCPU = numSelected.DownloadToNativeArray<int>();
            ReleaseTensorInt(numSelected);

            O.shape = new TensorShape(numSelectedCPU[0], 3);
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
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
                var pStarts = stackalloc int[8] { 0, 0, 0, 0, 0, 0, 0, 0 };
                var pSteps = stackalloc int[8] { 1, 1, 1, 1, 1, 1, 1, 1 };

                for (int i = 0; i < starts.Length; i++)
                {
                    int axis = axes != null ? X.shape.Axis(axes[i]) : i;
                    int start = Math.Min(starts[i], X.shape[axis] - 1);
                    start = start < 0 ? X.shape[axis] + start : start;
                    int step = steps != null ? steps[i] : 1;
                    pStarts[(TensorShape.maxRank - X.shape.rank) + axis] = start;
                    pSteps[(TensorShape.maxRank - X.shape.rank) + axis] = step;
                }
                fn.SetInt8(k_ID_starts, pStarts);
                fn.SetInt8(k_ID_steps, pSteps);
            }
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }

        /// <inheritdoc/>
        public void SliceSet(Tensor X, Tensor values, Tensor O, ReadOnlySpan<int> starts, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps)
        {
            MemCopy(X, O);
            var fn = ComputeFunctions.k_SliceSet;
            unsafe
            {
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, values.shape);
                var pStarts = stackalloc int[8] { 0, 0, 0, 0, 0, 0, 0, 0 };
                var pSteps = stackalloc int[8] { 1, 1, 1, 1, 1, 1, 1, 1 };

                for (int i = 0; i < starts.Length; i++)
                {
                    int axis = axes != null ? X.shape.Axis(axes[i]) : i;
                    int start = Math.Min(starts[i], X.shape[axis] - 1);
                    start = start < 0 ? X.shape[axis] + start : start;
                    int step = steps != null ? steps[i] : 1;
                    pStarts[(TensorShape.maxRank - X.shape.rank) + axis] = start;
                    pSteps[(TensorShape.maxRank - X.shape.rank) + axis] = step;
                }
                fn.SetInt8(k_ID_starts, pStarts);
                fn.SetInt8(k_ID_steps, pSteps);
            }
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(values));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(values.shape.length);
        }

        /// <inheritdoc/>
        public void Split(Tensor X, Tensor O, int axis, int start)
        {
            axis = X.shape.Axis(axis);

            var fn = ComputeFunctions.k_Split;
            fn.SetInt(k_ID_start, start);
            fn.SetInt(k_ID_lengthO, O.shape.length);
            fn.SetInt(k_ID_strideLower, O.shape.Strides(axis));
            int strideUpperX = axis == 0 ? X.shape.length : X.shape.Strides(axis - 1);
            int strideUpperO = axis == 0 ? O.shape.length : O.shape.Strides(axis - 1);
            fn.SetInt(k_ID_strideUpperX, strideUpperX);
            fn.SetInt(k_ID_strideUpperO, strideUpperO);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            var numBlocksY = ComputeHelper.IDivC(O.shape.length, (int)ComputeHelper.SafeDispatchLimit);
            var numBlocksX = ComputeHelper.IDivC(O.shape.length, numBlocksY);
            fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX);
            fn.Dispatch(numBlocksX, numBlocksY, 1);
        }

        /// <inheritdoc/>
        public void Pad(TensorFloat X, TensorFloat O, ReadOnlySpan<int> pad, Layers.PadMode padMode, float constant)
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

            fn.SetFloat(k_ID_Beta, constant);

            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
            fn.SetInt16(k_ID_pad, pad);
            fn.SetInt(k_ID_rank, X.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }

        /// <inheritdoc/>
        public void Pad(TensorInt X, TensorInt O, ReadOnlySpan<int> pad, Layers.PadMode padMode, int constant)
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

            fn.SetFloat(k_ID_Beta, math.asfloat(constant));

            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
            fn.SetInt16(k_ID_pad, pad);
            fn.SetInt(k_ID_rank, X.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }

        /// <inheritdoc/>
        public void Transpose(Tensor X, Tensor O)
        {
            var fn = ComputeFunctions.k_Transpose;
            unsafe
            {
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);

                int* permutations = stackalloc int[TensorShape.maxRank];
                for (int i = 0; i < X.shape.rank; i++)
                    permutations[i] = (X.shape.rank - 1) - i;
                fn.SetInt8(k_ID_permutations, permutations);
            }
            fn.SetInt(k_ID_rank, X.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(X.shape.length);
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
                fn.SetInt(k_ID_X_width, equivalentXW);
                fn.SetInt(k_ID_X_height, equivalentXH);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

                fn.Dispatch(equivalentXW, equivalentXH, 1);
            }
            else
            {

                var fn = ComputeFunctions.k_Transpose;
                unsafe
                {
                    fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
                    fn.SetInt8(k_ID_permutations, permutations);
                }
                fn.SetInt(k_ID_rank, X.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(X.shape.length);
            }
        }

        void ArgMaxTail(TensorFloat X, TensorInt O, int axis)
        {
            int globalNonSpatialLength = X.shape.Length(0, axis);
            int globalSpatialDims = X.shape.length / globalNonSpatialLength;

            int localSpatialLength = globalSpatialDims;

            var Oshape = new TensorShape(globalNonSpatialLength, localSpatialLength);

            TensorInt Xindices = AllocTensorInt(X.shape); // save max(X)
            bool isFirstDispatch = true;

            // downsample with pyramid approach
            while (localSpatialLength > 64 * 4)
            {
                int spatialLengthO = ComputeHelper.IDivC(localSpatialLength, 64 * 4);
                Oshape[-1] = spatialLengthO;

                var Otemp = AllocTensorFloat(Oshape);
                var Oindicestemp = AllocTensorInt(Oshape);

                var fnPool = ComputeFunctions.k_ArgMaxReduce;
                fnPool.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fnPool.SetTensorAsBuffer(k_ID_XIndices, Pin(Xindices));
                fnPool.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                fnPool.SetTensorAsBuffer(k_ID_OIndices, Pin(Oindicestemp));
                fnPool.SetInt(k_ID_SpatialDims, localSpatialLength);
                fnPool.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                fnPool.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                fnPool.Dispatch(globalNonSpatialLength, ComputeHelper.IDivC(localSpatialLength, 4), 1);

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
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_XIndices, Pin(Xindices));
            fn.SetTensorAsBuffer(k_ID_OIndices, Pin(O));
            fn.SetInt(k_ID_SpatialDims, localSpatialLength);
            fn.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

            fn.Dispatch(globalNonSpatialLength, 1, 1);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
            ReleaseTensorInt(Xindices);
        }

        /// <inheritdoc/>
        public void ArgMax(TensorFloat X, TensorInt O, int axis, bool selectLastIndex)
        {
            int dimAxis = X.shape[axis];
            Assert.AreNotEqual(0, dimAxis, "ValueError: zero-size array to reduction operation maximum which has no identity.");

            if (!selectLastIndex && (dimAxis == X.shape.Length(axis)))
            {
                ArgMaxTail(X, O, axis);
                return;
            }

            var fn = (selectLastIndex ? ComputeFunctions.k_ArgMaxFloatLast : ComputeFunctions.k_ArgMaxFloatFirst);
            fn.SetInt(k_ID_innerLength, X.shape.Strides(axis));
            fn.SetInt(k_ID_reduceLength, dimAxis);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }

        /// <inheritdoc/>
        public void ArgMax(TensorInt X, TensorInt O, int axis, bool selectLastIndex)
        {
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

            var fn = (selectLastIndex ? ComputeFunctions.k_ArgMaxIntLast : ComputeFunctions.k_ArgMaxIntFirst);
            fn.SetInt(k_ID_innerLength, X.shape.Strides(axis));
            fn.SetInt(k_ID_reduceLength, X.shape[axis]);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }

        /// <inheritdoc/>
        public void ArgMin(TensorFloat X, TensorInt O, int axis, bool selectLastIndex)
        {
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation minimum which has no identity.");

            var fn = (selectLastIndex ? ComputeFunctions.k_ArgMinFloatLast : ComputeFunctions.k_ArgMinFloatFirst);
            fn.SetInt(k_ID_innerLength, X.shape.Strides(axis));
            fn.SetInt(k_ID_reduceLength, X.shape[axis]);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }

        /// <inheritdoc/>
        public void ArgMin(TensorInt X, TensorInt O, int axis, bool selectLastIndex)
        {
            var fn = (selectLastIndex ? ComputeFunctions.k_ArgMinIntLast : ComputeFunctions.k_ArgMinIntFirst);
            fn.SetInt(k_ID_innerLength, X.shape.Strides(axis));
            fn.SetInt(k_ID_reduceLength, X.shape[axis]);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }

        /// <inheritdoc/>
        public void Not(TensorInt X, TensorInt O)
        {
            var fn = ComputeFunctions.k_Not;
            fn.SetTensorAsBuffer(k_ID_X_int_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_int_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void HardSwish(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_HardSwish;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Sign(TensorFloat X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_SignFloat;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Sign(TensorInt X, TensorInt O)
        {
            var fn = ComputeFunctions.k_SignInt;
            fn.SetTensorAsBuffer(k_ID_X_int_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_int_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void IsInf(TensorFloat X, TensorInt O, bool detectNegative, bool detectPositive)
        {
            var fn = ComputeFunctions.k_IsInf;
            fn.SetBool(k_ID_detectNegative, detectNegative);
            fn.SetBool(k_ID_detectPositive, detectPositive);
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void IsNaN(TensorFloat X, TensorInt O)
        {
            var fn = ComputeFunctions.k_IsNaN;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_int_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Where(TensorInt C, Tensor A, Tensor B, Tensor O)
        {
            var fn = ComputeFunctions.k_Where;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeC, k_ID_stridesC, C.shape);
            fn.SetTensorShapeStrides(k_ID_shapeA, k_ID_stridesA, A.shape);
            fn.SetTensorShapeStrides(k_ID_shapeB, k_ID_stridesB, B.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(C));
            fn.SetTensorAsBuffer(k_ID_Sptr, Pin(A));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.UnrolledDispatch(O.shape.length);
        }

        /// <inheritdoc/>
        public void Tile(Tensor X, Tensor O, ReadOnlySpan<int> repeats)
        {
            var fn = ComputeFunctions.k_Tile;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }

        /// <inheritdoc/>
        public void MemClear(Tensor O)
        {
            var length = O.shape.length;
            var numWords = ComputeHelper.IDivC(length, 4);
            var wordsHeight = ComputeHelper.IDivC(numWords, (int)ComputeHelper.SafeDispatchLimit * 32 * 8);
            var wordsWidth = ComputeHelper.IDivC(numWords, wordsHeight);

            var fn = ComputeFunctions.k_MemSet;
            fn.SetFloat(k_ID_memValueFloat, 0);
            fn.SetInt(k_ID_offsetO, 0);
            fn.SetInt(k_ID_count, length);
            fn.SetInt(k_ID_O_width, wordsWidth * 4);
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(wordsWidth, wordsHeight, 1);
        }

        /// <inheritdoc/>
        public void MemSet(TensorFloat O, float value)
        {
            var length = O.shape.length;
            var numWords = ComputeHelper.IDivC(length, 4);
            var wordsHeight = ComputeHelper.IDivC(numWords, (int)ComputeHelper.SafeDispatchLimit * 32 * 8);
            var wordsWidth = ComputeHelper.IDivC(numWords, wordsHeight);

            var fn = ComputeFunctions.k_MemSet;
            fn.SetFloat(k_ID_memValueFloat, value);
            fn.SetInt(k_ID_offsetO, 0);
            fn.SetInt(k_ID_count, length);
            fn.SetInt(k_ID_O_width, wordsWidth * 4);
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(wordsWidth, wordsHeight, 1);
        }

        /// <inheritdoc/>
        public void MemSet(TensorInt O, int value)
        {
            var length = O.shape.length;
            var numWords = ComputeHelper.IDivC(length, 4);
            var wordsHeight = ComputeHelper.IDivC(numWords, (int)ComputeHelper.SafeDispatchLimit * 32 * 8);
            var wordsWidth = ComputeHelper.IDivC(numWords, wordsHeight);

            var fn = ComputeFunctions.k_MemSet;
            fn.SetFloat(k_ID_memValueFloat, math.asfloat(value));
            fn.SetInt(k_ID_offsetO, 0);
            fn.SetInt(k_ID_count, length);
            fn.SetInt(k_ID_O_width, wordsWidth * 4);
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(wordsWidth, wordsHeight, 1);
        }

        /// <inheritdoc/>
        public void Expand(Tensor X, Tensor O)
        {
            var fn = ComputeFunctions.k_Expand;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }

        /// <inheritdoc/>
        public void CompressWithIndices(Tensor X, TensorInt indices, Tensor O, int numIndices, int axis)
        {
            var fn = ComputeFunctions.k_Gather;
            fn.SetInt(k_ID_endLength, X.shape.Strides(axis));
            fn.SetInt(k_ID_indicesLength, numIndices);
            fn.SetInt(k_ID_axisDim, X.shape[axis]);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(indices));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }

        /// <inheritdoc/>
        public void Gather(Tensor X, TensorInt indices, Tensor O, int axis)
        {
            var fn = ComputeFunctions.k_Gather;
            fn.SetInt(k_ID_endLength, X.shape.Strides(axis));
            fn.SetInt(k_ID_indicesLength, indices.shape.length);
            fn.SetInt(k_ID_axisDim, X.shape[axis]);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(indices));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.UnrolledDispatch(O.shape.length);
        }

        /// <inheritdoc/>
        public void GatherElements(Tensor X, TensorInt indices, Tensor O, int axis)
        {
            Logger.AssertIsTrue(indices.shape.rank == X.shape.rank, "GatherElements: input and indices rank should match");
            Logger.AssertIsTrue(O.shape == indices.shape, "GatherElements: output and indices shapes should match");
            axis = X.shape.Axis(axis); // note: this is safe since the ranks of X and indices match

            // See ScatterElements for more info
            bool fastPathPossible = ShapeInference.ScatterGatherElementsSupportsFastPath(indices.shape, X.shape, axis);
            var fn = fastPathPossible ? ComputeFunctions.k_GatherElementsFast : ComputeFunctions.k_GatherElements;

            fn.SetInt(k_ID_inputAxisSize, X.shape[axis]);

            if (fastPathPossible)
            {
                fn.SetInt(k_ID_indicesAxisElementStride, indices.shape.Strides(axis));
                fn.SetInt(k_ID_inputAxisElementStride, X.shape.Strides(axis));
                fn.SetInt(k_ID_indicesAxisMinusOneElementStride, indices.shape[axis] * indices.shape.Strides(axis));

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(indices));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

                fn.UnrolledDispatch(indices.shape.length);
            }
            else
            {
                fn.SetTensorStridesCompactedAtHead(k_ID_stridesO, indices.shape);
                fn.SetTensorStridesCompactedAtHead(k_ID_stridesX, X.shape); // WARNING: Remember that X in the shader and here are inputs!
                fn.SetInt(k_ID_posAxis, axis);
                fn.SetInt(k_ID_rank, X.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(indices));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

                fn.UnrolledDispatch(indices.shape.length);
            }
        }

        /// <inheritdoc/>
        public void GatherND(Tensor X, TensorInt indices, Tensor O, int batchDims)
        {
            var fn = ComputeFunctions.k_GatherND;
            fn.SetInt(k_ID_rankX, X.shape.rank);
            fn.SetInt(k_ID_rankO, O.shape.rank);
            fn.SetInt(k_ID_rankIndices, indices.shape.rank);
            fn.SetInt(k_ID_iStart, TensorShape.maxRank - O.shape.rank);
            fn.SetInt(k_ID_iEndIndices, TensorShape.maxRank - O.shape.rank + indices.shape.rank - 1);
            fn.SetInt(k_ID_iEndX, TensorShape.maxRank - O.shape.rank + batchDims);
            fn.SetInt(k_ID_iEndMin, TensorShape.maxRank - O.shape.rank + Math.Min(batchDims, indices.shape.rank - 1));
            fn.SetInt(k_ID_iStartB, TensorShape.maxRank - X.shape.rank + batchDims);
            fn.SetInt(k_ID_iEndB, TensorShape.maxRank - X.shape.rank + batchDims + indices.shape[-1]);
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
            fn.SetTensorShapeStrides(k_ID_shapeIndices, k_ID_stridesIndices, indices.shape);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(indices));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.UnrolledDispatch(O.shape.length);
        }

        /// <inheritdoc/>
        public void ScatterElements(Tensor X, TensorInt indices, Tensor updates, Tensor O, int axis, Layers.ScatterReductionMode reduction)
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

            fn.SetInt(k_ID_outAxisSize, X.shape[axis]);
            fn.SetInt(k_ID_reductionType, (int)reduction);

            if (fastPathPossible)
            {
                fn.SetInt(k_ID_indicesAxisElementStride, indices.shape.Strides(axis));
                fn.SetInt(k_ID_outAxisElementStride, X.shape.Strides(axis));
                fn.SetInt(k_ID_indicesAxisMinusOneElementStride, indices.shape[axis] * indices.shape.Strides(axis));

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(updates));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(indices));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

                fn.UnrolledDispatch(indices.shape.length);
            }
            else
            {
                fn.SetTensorStridesCompactedAtHead(k_ID_stridesO, O.shape);
                fn.SetTensorStridesCompactedAtHead(k_ID_stridesX, indices.shape); // WARNING: Remember that X in the shader code is updates, but here X is the input tensor!
                fn.SetInt(k_ID_posAxis, axis);
                fn.SetInt(k_ID_rank, X.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(updates));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(indices));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

                fn.UnrolledDispatch(indices.shape.length);
            }
        }

        /// <inheritdoc/>
        public void ScatterND(TensorFloat X, TensorInt indices, TensorFloat updates, TensorFloat O, Layers.ScatterReductionMode reduction)
        {
            MemCopy(X, O);
            int indexRemapDim = indices.shape[-1];
            int indicesLength = indices.shape.Length(0, -1);
            int updatesLength = updates.shape.length / indicesLength;

            var fn = ComputeFunctions.k_ScatterNDFloat;
            fn.SetInt(k_ID_updatesLength, updatesLength);
            fn.SetInt(k_ID_indicesLength, indicesLength);
            fn.SetInt(k_ID_indexRemapDim, indexRemapDim);
            fn.SetInt(k_ID_reduction, (int)reduction);
            unsafe
            {
                var trailing = stackalloc int[8];
                int trailingDim = 1;
                for (int j = (indexRemapDim - 1); j >= 0; j--)
                {
                    trailing[j] = trailingDim;
                    trailingDim *= X.shape[j];
                }
                fn.SetInt8(k_ID_trailing, trailing);
            }
            fn.SetTensorAsBuffer(k_ID_Iptr, Pin(indices));
            fn.SetTensorAsBuffer(k_ID_Uptr, Pin(updates));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.Dispatch(updatesLength, indicesLength, 1);
        }

        /// <inheritdoc/>
        public void ScatterND(TensorInt X, TensorInt indices, TensorInt updates, TensorInt O, Layers.ScatterReductionMode reduction)
        {
            MemCopy(X, O);

            int indexRemapDim = indices.shape[-1];
            int indicesLength = indices.shape.Length(0, -1);
            int updatesLength = updates.shape.length / indicesLength;

            var fn = ComputeFunctions.k_ScatterNDInt;
            fn.SetInt(k_ID_updatesLength, updatesLength);
            fn.SetInt(k_ID_indicesLength, indicesLength);
            fn.SetInt(k_ID_indexRemapDim, indexRemapDim);
            fn.SetInt(k_ID_reduction, (int)reduction);
            unsafe
            {
                var trailing = stackalloc int[8];
                int trailingDim = 1;
                for (int j = indexRemapDim - 1; j >= 0; j--)
                {
                    trailing[j] = trailingDim;
                    trailingDim *= X.shape[j];
                }
                fn.SetInt8(k_ID_trailing, trailing);
            }
            fn.SetTensorAsBuffer(k_ID_Iptr, Pin(indices));
            fn.SetTensorAsBuffer(k_ID_UIntptr, Pin(updates));
            fn.SetTensorAsBuffer(k_ID_OIntptr, Pin(O));
            fn.Dispatch(updatesLength, indicesLength, 1);
        }

        /// <inheritdoc/>
        public void OneHot(TensorInt X, TensorInt O, int axis, int depth, int offValue, int onValue)
        {
            axis = O.shape.Axis(axis);

            var fn = ComputeFunctions.k_OneHot;
            fn.SetInt(k_ID_depth, depth);
            fn.SetInt(k_ID_offValue, offValue);
            fn.SetInt(k_ID_onValue, onValue);
            fn.SetInt(k_ID_rankO, O.shape.rank);

            fn.SetInt(k_ID_stridesToAxis, O.shape.Strides(axis));
            fn.SetInt(k_ID_axisDim, O.shape[axis]);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(O.shape.length, 1, 1);
        }

        /// <inheritdoc/>
        public void OneHot(TensorInt X, TensorFloat O, int axis, int depth, float offValue, float onValue)
        {
            axis = O.shape.Axis(axis);

            var fn = ComputeFunctions.k_OneHot;
            fn.SetInt(k_ID_depth, depth);
            fn.SetInt(k_ID_offValue, math.asint(offValue));
            fn.SetInt(k_ID_onValue, math.asint(onValue));
            fn.SetInt(k_ID_rankO, O.shape.rank);

            fn.SetInt(k_ID_stridesToAxis, O.shape.Strides(axis));
            fn.SetInt(k_ID_axisDim, O.shape[axis]);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(O.shape.length, 1, 1);
        }

        /// <inheritdoc/>
        public void TopK(TensorFloat X, TensorFloat values, TensorInt indices, int k, int axis, bool largest)
        {
            int reduceLength = X.shape[axis];
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.length / (reduceLength * innerLength);

            var fn = (largest ? ComputeFunctions.k_TopKLargest : ComputeFunctions.k_TopKSmallest);
            fn.SetInt(k_ID_innerLength, innerLength);
            fn.SetInt(k_ID_outerLength, outerLength);
            fn.SetInt(k_ID_reduceLength, reduceLength);
            fn.SetInt(k_ID_maxK, k);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Valuesptr, Pin(values));
            fn.SetTensorAsBuffer(k_ID_Indicesptr, Pin(indices));
            fn.Dispatch(innerLength, outerLength, 1);
        }

        /// <inheritdoc/>
        public void RoiAlign(TensorFloat X, TensorFloat rois, TensorInt indices, TensorFloat O, Layers.RoiPoolingMode mode, int outputHeight, int outputWidth, int samplingRatio, float spatialScale)
        {
            var fn = (mode == Layers.RoiPoolingMode.Avg ? ComputeFunctions.k_RoiAlignAvg : ComputeFunctions.k_RoiAlignMax);
            fn.SetInt(k_ID_numRois, rois.shape[0]);
            fn.SetInt(k_ID_inputChannels, X.shape[1]);
            fn.SetInt(k_ID_inputHeight, X.shape[2]);
            fn.SetInt(k_ID_inputWidth, X.shape[3]);
            fn.SetInt(k_ID_inputSpatialSize, X.shape[2] * X.shape[3]);
            fn.SetInt(k_ID_inputBatchOffset, X.shape[1] * X.shape[2] * X.shape[3]);
            fn.SetInt(k_ID_outputHeight, outputHeight);
            fn.SetInt(k_ID_outputWidth, outputWidth);
            fn.SetInt(k_ID_outputSpatialSize, outputHeight * outputWidth);
            fn.SetFloat(k_ID_normalizeOHeight, 1.0f / outputHeight);
            fn.SetFloat(k_ID_normalizeOWidth, 1.0f / outputWidth);
            fn.SetInt(k_ID_samplingRatio, samplingRatio);
            fn.SetFloat(k_ID_spatialScale, spatialScale);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Sptr, Pin(rois));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(indices));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(O.shape[0] * O.shape[1], O.shape[2] * O.shape[3], 1);
        }

        /// <inheritdoc/>
        public void RandomNormal(TensorFloat O, float mean, float scale, int? seed)
        {
            var fn = ComputeFunctions.k_RandomNormal;
            fn.SetInt(k_ID_lengthO, O.shape.length);
            fn.SetInt(k_ID_seed, (int)Random.GetSeed(seed));
            fn.SetFloat(k_ID_mean, mean);
            fn.SetFloat(k_ID_scale, scale);

            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(O.shape.length, 1, 1);
        }

        /// <inheritdoc/>
        public void TopP(TensorFloat X, TensorFloat random, TensorInt O)
        {
            var batch = O.shape.length;

            var fn = ComputeFunctions.k_TopP;
            fn.SetInt(k_ID_count, O.shape[-1]);
            fn.SetInt(k_ID_innerLength, X.shape[-1]);
            fn.SetInt(k_ID_outerLength, batch);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(random));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.Dispatch(batch, 1, 1);
        }

        /// <inheritdoc/>
        public void RandomUniform(TensorFloat O, float low, float high, int? seed)
        {
            var fn = ComputeFunctions.k_RandomUniform;
            fn.SetInt(k_ID_lengthO, O.shape.length);
            fn.SetInt(k_ID_seed, (int)Random.GetSeed(seed));
            fn.SetFloat(k_ID_low, low);
            fn.SetFloat(k_ID_high, high);

            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(O.shape.length, 1, 1);
        }

        /// <inheritdoc/>
        public void Bernoulli(TensorFloat X, Tensor O, int? seed)
        {
            var fn = (O.dataType == DataType.Float ? ComputeFunctions.k_BernoulliFloat : ComputeFunctions.k_BernoulliInt);
            fn.SetInt(k_ID_lengthO, O.shape.length);
            fn.SetInt(k_ID_seed, (int)Random.GetSeed(seed));
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }

        /// <inheritdoc/>
        public void Cast(TensorInt X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_CastIntToFloat;
            fn.SetTensorAsBuffer(k_ID_X_int_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_float_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Cast(TensorFloat X, TensorInt O)
        {
            var fn = ComputeFunctions.k_CastFloatToInt;
            fn.SetTensorAsBuffer(k_ID_X_float_ptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_O_int_ptr, Pin(O));
            fn.UnrolledDispatchFast(O.shape.length);
        }

        /// <inheritdoc/>
        public void Cast(TensorShort X, TensorFloat O)
        {
            var fn = ComputeFunctions.k_CastHalfToFloat;
            fn.SetInt(k_ID_lengthO, O.shape.length);
            fn.SetTensorAsBuffer(k_ID_XIntptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(X.count);
        }

        /// <inheritdoc/>
        public void MemCopy(Tensor X, Tensor O)
        {
            MemCopy(X, O, O.shape.length, 0, 0);
        }

        void MemCopy(Tensor X, Tensor O, int count, int offsetX, int offsetO)
        {
            var numWords = ComputeHelper.IDivC(count, 4);
            var wordsHeight = ComputeHelper.IDivC(numWords, (int)ComputeHelper.SafeDispatchLimit * 32 * 8);
            var wordsWidth = ComputeHelper.IDivC(numWords, wordsHeight);

            var fn = ComputeFunctions.k_MemCopy;
            fn.SetInt(k_ID_offsetO, offsetO);
            fn.SetInt(k_ID_offsetX, offsetX);
            fn.SetInt(k_ID_count, count);
            fn.SetInt(k_ID_O_width, wordsWidth * 4);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

            fn.Dispatch(wordsWidth, wordsHeight, 1);
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
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.SetInt(k_ID_strideX, strideX);
            fn.SetInt(k_ID_strideO, strideO);
            fn.SetInt(k_ID_offsetX, offsetX);
            fn.SetInt(k_ID_offsetO, offsetO);
            fn.SetInt(k_ID_elementSize, length);
            fn.SetInt(k_ID_count, copyLength);
            fn.Dispatch(ComputeHelper.IDivC(copyLength, 4), 1, 1);
        }

        void Gemm(TensorFloat X, TensorFloat Y, TensorFloat B, TensorFloat O, int M, int K, int N)
        {
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

            fn.SetInt(k_ID_X_width, K);
            fn.SetInt(k_ID_W_width, N);
            fn.SetInt(k_ID_O_height, M);
            fn.SetInt(k_ID_O_width, N);
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Wptr, Pin(Y));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.SetInt(k_ID_maxXIndex, M * K - 1);
            fn.SetInt(k_ID_maxWIndex, K * N - 1);

            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetInt(k_ID_maxBIndex, N - 1);

            fn.Dispatch(workItemsX, workItemsY, 1);
        }

        void Gemm(TensorFloat X, TensorFloat Y, TensorFloat O, int M, int K, int N, bool transposeA = false, bool transposeB = false)
        {
            if (transposeA || transposeB)
            {
                ComputeFunction fn;

                if (transposeA)
                    fn = transposeB ? ComputeFunctions.k_GemmT_XT_WT_T8x8_R4x4 : ComputeFunctions.k_GemmT_XT_T8x8_R4x4;
                else
                    fn = ComputeFunctions.k_GemmT_WT_T8x8_R4x4;


                fn.SetInt(k_ID_M, M);
                fn.SetInt(k_ID_N, N);
                fn.SetInt(k_ID_K, K);
                fn.SetInt(k_ID_maxXIndex, M * K - 1);
                fn.SetInt(k_ID_maxWIndex, K * N - 1);
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Wptr, Pin(Y));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

                fn.Dispatch(ComputeHelper.IDivC(N, 4), ComputeHelper.IDivC(M, 4), 1);
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

                fn.SetInt(k_ID_X_width, K);
                fn.SetInt(k_ID_W_width, N);
                fn.SetInt(k_ID_O_width, N);
                fn.SetInt(k_ID_O_height, M);
                fn.SetInt(k_ID_maxXIndex, M * K - 1);
                fn.SetInt(k_ID_maxWIndex, K * N - 1);
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Wptr, Pin(Y));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));

                fn.Dispatch(workItemsX, workItemsY, workItemsZ);
            }
        }

        /// <inheritdoc/>
        protected void SinglePassLSTM(TensorFloat X, TensorFloat W, TensorFloat R, TensorFloat B, TensorInt sequenceLens, TensorFloat P, TensorFloat Y, TensorFloat Y_h, TensorFloat Y_c, Layers.RnnActivation[] activations, float[] activationAlpha, float[] activationBeta, bool inputForget, float clip, bool isReverse, int dirIndex, Layers.RnnLayout layout)
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
            endFn.SetInt(k_ID_hiddenSize, hiddenSize);
            endFn.SetInt(k_ID_batchSize, batchSize);
            endFn.SetInt(k_ID_xStride, xStrideBatch);
            endFn.SetInt(k_ID_yStride, yStrideBatch);
            endFn.SetBool(k_ID_inputForget, inputForget);
            endFn.SetFloat(k_ID_clipValue, clip);
            endFn.SetInt(k_ID_fActivation, (int)activations[3 * dirIndex + 0]);
            endFn.SetFloat(k_ID_fAlpha, activationAlpha[3 * dirIndex + 0]);
            endFn.SetFloat(k_ID_fBeta, activationAlpha[3 * dirIndex + 0]);
            endFn.SetInt(k_ID_gActivation, (int)activations[3 * dirIndex + 1]);
            endFn.SetFloat(k_ID_gAlpha, activationAlpha[3 * dirIndex + 1]);
            endFn.SetFloat(k_ID_gBeta, activationAlpha[3 * dirIndex + 1]);
            endFn.SetInt(k_ID_hActivation, (int)activations[3 * dirIndex + 2]);
            endFn.SetFloat(k_ID_hAlpha, activationAlpha[3 * dirIndex + 2]);
            endFn.SetFloat(k_ID_hBeta, activationAlpha[3 * dirIndex + 2]);
            endFn.SetTensorAsBuffer(k_ID_Yptr, Pin(Y));
            endFn.SetTensorAsBuffer(k_ID_YHptr, Pin(Y_h));
            endFn.SetTensorAsBuffer(k_ID_YCptr, Pin(Y_c));
            endFn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            endFn.SetInt(k_ID_bOffset, dirIndex * 8 * hiddenSize);
            endFn.SetTensorAsBuffer(k_ID_Pptr, Pin(P));
            endFn.SetInt(k_ID_pOffset, dirIndex * 3 * hiddenSize);
            endFn.SetTensorAsBuffer(k_ID_XsixWTptr, Pin(XsixWT));
            endFn.SetTensorAsBuffer(k_ID_HtxRTptr, Pin(HtxRT));
            endFn.SetTensorAsBuffer(k_ID_SequenceLensptr, Pin(sequenceLens));

            for (var i = 0; i < seqLength; i++)
            {
                var seqIndex = isReverse ? seqLength - 1 - i : i;

                Gemm(Y_h, R, HtxRT, batchSize, hiddenSize, 4 * hiddenSize, transposeB: true);

                endFn.SetInt(k_ID_seqIndex, seqIndex);
                endFn.SetInt(k_ID_yOffset, dirIndex * yStrideDir + seqIndex * yStrideSeq);
                endFn.SetInt(k_ID_xOffset, seqIndex * xStrideSeq);
                endFn.Dispatch(batchSize, hiddenSize, 1);
            }

            ReleaseTensorFloat(HtxRT);
            ReleaseTensorFloat(XsixWT);
        }

        /// <summary>
        /// Sets final output tensor for W, R, initialH and initialC from provided input tensors
        /// if no input is provided the tensor is cleared to 0 as a default
        /// otherwise if the input tensor can be used directly in the calculation this will early out
        /// </summary>
        void SetRnnInput(TensorFloat X, TensorFloat O, int index, int count, int length, int strideX)
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
        void SetRnnOutput(TensorFloat X, TensorFloat O, int index, int count, int length, int strideO)
        {
            if (X == O)
                return;
            MemCopyStride(X, O, length, strideO, length, count, 0, index * length);
        }

        /// <inheritdoc/>
        public void LSTM(TensorFloat X, TensorFloat W, TensorFloat R, TensorFloat B, TensorInt sequenceLens, TensorFloat initialH, TensorFloat initialC, TensorFloat P, TensorFloat Y, TensorFloat Yh, TensorFloat Yc, Layers.RnnDirection direction, Layers.RnnActivation[] activations, float[] activationAlpha, float[] activationBeta, bool inputForget, float clip, Layers.RnnLayout layout)
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
        public void DequantizeLinear(TensorByte X, TensorFloat O, float scale, byte zeroPoint)
        {
            var fn = ComputeFunctions.k_DequantizeUint8;
            fn.SetFloat(k_ID_scale, scale);
            fn.SetInt(k_ID_zeroPoint, (int)zeroPoint);
            fn.SetTensorAsBuffer(k_ID_XIntptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.SetInt(k_ID_lengthO, O.shape.length);
            fn.UnrolledDispatch(X.count);
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
        internal void BitonicSort(TensorFloat Value, TensorInt Key = null, bool descending = false)
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
                fn.SetTensorAsBuffer(k_ID_O_int_ptr, Pin(Key));
            }
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(Value));
            fn.SetInt(k_ID_lengthO, N);
            if (descending)
                fn.EnableKeyword("DESCENDING");
            else
                fn.DisableKeyword("DESCENDING");

            for (int k = 1; k < N; k <<= 1)
            {
                fn.SetInt(k_ID_indexJ, (k << 1) - 1);
                fn.Dispatch(N, L, 1);

                for (int j = k >> 1; j > 0; j >>= 1)
                {
                    fn.SetInt(k_ID_indexJ, j);
                    fn.Dispatch(N, L, 1);
                }
            }
        }

        /// <inheritdoc/>
        public Tensor PinToDevice(Tensor X, bool clearOnInit = false)
        {
            Pin(X, clearOnInit);
            return X;
        }
    }
}
