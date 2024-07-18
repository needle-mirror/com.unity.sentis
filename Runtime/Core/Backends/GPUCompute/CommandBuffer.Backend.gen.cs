// This is auto-generated -- do not modify directly

using System;
using Unity.Sentis;
using UnityEngine.Assertions;
using static Unity.Sentis.ComputeTensorData;
using static Unity.Sentis.ShaderPropertyID;

namespace Unity.Sentis
{
    public partial class GPUCommandBufferBackend
    {
        // Binary Broadcast

        /// <inheritdoc/>
        public void PRelu(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastPRelu;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastPRelu;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwisePRelu;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Pow(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastPowFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastPowFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwisePowFloat;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Add(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastAddFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastAddFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseAddFloat;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Sub(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastSubFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastSubFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseSubFloat;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Mul(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastMulFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastMulFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseMulFloat;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Div(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastDivFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastDivFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseDivFloat;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Mod(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastModFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastModFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseModFloat;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void FMod(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastFModFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastFModFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseFModFloat;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Pow(TensorFloat A, TensorInt B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastPowInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastPowInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwisePowInt;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Add(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastAddInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastAddInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseAddInt;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Sub(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastSubInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastSubInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseSubInt;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Mul(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastMulInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastMulInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseMulInt;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Div(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastDivInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastDivInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseDivInt;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Mod(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastModInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastModInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseModInt;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void FMod(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastFModInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastFModInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseFModInt;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Min(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastMinFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastMinFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseMinFloat;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Max(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastMaxFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastMaxFloat;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseMaxFloat;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Min(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastMinInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastMinInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseMinInt;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Max(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastMaxInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastMaxInt;
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseMaxInt;
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fn, O.shape.length);
            }
        }


        // Pool ops
        /// <inheritdoc/>
        public void MaxPool(TensorFloat X, TensorFloat O, int[] kernelShape, int[] strides, int[] pads)
        {
            ComputeFunction fn;
            switch (X.shape.rank)
            {
                case 3:
                    fn = ComputeFunctions.k_MaxPool1D;
                    cb.SetInt(fn, k_ID_stride, strides[0]);
                    cb.SetInt(fn, k_ID_pad, pads[0]);
                    cb.SetInt(fn, k_ID_inHeight, X.shape[2]);
                    cb.SetInt(fn, k_ID_pool, kernelShape[0]);
                    cb.SetInt(fn, k_ID_outHeight, O.shape[2]);
                    break;
                case 4:
                    fn = ComputeFunctions.k_MaxPool2D;
                    cb.SetInt(fn, k_ID_strideX, strides[1]);
                    cb.SetInt(fn, k_ID_strideY, strides[0]);
                    cb.SetInt(fn, k_ID_padX, pads[1]);
                    cb.SetInt(fn, k_ID_padY, pads[0]);

                    cb.SetInt(fn, k_ID_inHeight, X.shape[2]);
                    cb.SetInt(fn, k_ID_inWidth, X.shape[3]);

                    cb.SetInt(fn, k_ID_poolX, kernelShape[1]);
                    cb.SetInt(fn, k_ID_poolY, kernelShape[0]);

                    cb.SetInt(fn, k_ID_outHeight, O.shape[2]);
                    cb.SetInt(fn, k_ID_outWidth, O.shape[3]);
                    break;
                default:
                    throw new NotImplementedException();
            }
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }
        /// <inheritdoc/>
        public void AveragePool(TensorFloat X, TensorFloat O, int[] kernelShape, int[] strides, int[] pads)
        {
            ComputeFunction fn;
            switch (X.shape.rank)
            {
                case 3:
                    fn = ComputeFunctions.k_AveragePool1D;
                    cb.SetInt(fn, k_ID_stride, strides[0]);
                    cb.SetInt(fn, k_ID_pad, pads[0]);
                    cb.SetInt(fn, k_ID_inHeight, X.shape[2]);
                    cb.SetInt(fn, k_ID_pool, kernelShape[0]);
                    cb.SetInt(fn, k_ID_outHeight, O.shape[2]);
                    break;
                case 4:
                    fn = ComputeFunctions.k_AveragePool2D;
                    cb.SetInt(fn, k_ID_strideX, strides[1]);
                    cb.SetInt(fn, k_ID_strideY, strides[0]);
                    cb.SetInt(fn, k_ID_padX, pads[1]);
                    cb.SetInt(fn, k_ID_padY, pads[0]);

                    cb.SetInt(fn, k_ID_inHeight, X.shape[2]);
                    cb.SetInt(fn, k_ID_inWidth, X.shape[3]);

                    cb.SetInt(fn, k_ID_poolX, kernelShape[1]);
                    cb.SetInt(fn, k_ID_poolY, kernelShape[0]);

                    cb.SetInt(fn, k_ID_outHeight, O.shape[2]);
                    cb.SetInt(fn, k_ID_outWidth, O.shape[3]);
                    break;
                default:
                    throw new NotImplementedException();
            }
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }

        // Global pool ops
        /// <inheritdoc/>
        public void GlobalMaxPool(TensorFloat X, TensorFloat O)
        {
            int globalSpatialDims = X.shape.Length(2);
            int globalNonSpatialLength = X.shape[0] * X.shape[1];

            int localSpatialLength = globalSpatialDims;

            var Oshape = new TensorShape(X.shape[0], X.shape[1], localSpatialLength);
            bool isTempAlloc = false;

            // downsample with pyramid approach
            while (localSpatialLength > 64 * 4)
            {
                int spatialLengthO = ComputeHelper.IDivC(localSpatialLength, 64 * 4);
                Oshape[2] = spatialLengthO;
                var Otemp = AllocTensorFloat(Oshape);

                var fnPool = ComputeFunctions.k_MaxPoolReduce;
                cb.SetTensorAsBuffer(fnPool, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fnPool, k_ID_Optr, Pin(Otemp));
                cb.SetInt(fnPool, k_ID_SpatialDims, localSpatialLength);
                cb.SetInt(fnPool, k_ID_SpatialDimsO, spatialLengthO);

                cb.Dispatch(fnPool, globalNonSpatialLength, ComputeHelper.IDivC(localSpatialLength, 4), 1);

                if (isTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otemp;
                localSpatialLength = spatialLengthO;
                isTempAlloc = true;
            }

            var fn = ComputeFunctions.k_GlobalMaxPool;
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetInt(fn, k_ID_SpatialDims, localSpatialLength);
            cb.SetInt(fn, k_ID_GlobalSpatialDims, globalSpatialDims);

            cb.Dispatch(fn, globalNonSpatialLength, 1, 1);

            if (isTempAlloc)
                ReleaseTensorFloat(X);
        }
        /// <inheritdoc/>
        public void GlobalAveragePool(TensorFloat X, TensorFloat O)
        {
            int globalSpatialDims = X.shape.Length(2);
            int globalNonSpatialLength = X.shape[0] * X.shape[1];

            int localSpatialLength = globalSpatialDims;

            var Oshape = new TensorShape(X.shape[0], X.shape[1], localSpatialLength);
            bool isTempAlloc = false;

            // downsample with pyramid approach
            while (localSpatialLength > 64 * 4)
            {
                int spatialLengthO = ComputeHelper.IDivC(localSpatialLength, 64 * 4);
                Oshape[2] = spatialLengthO;
                var Otemp = AllocTensorFloat(Oshape);

                var fnPool = ComputeFunctions.k_AveragePoolReduce;
                cb.SetTensorAsBuffer(fnPool, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fnPool, k_ID_Optr, Pin(Otemp));
                cb.SetInt(fnPool, k_ID_SpatialDims, localSpatialLength);
                cb.SetInt(fnPool, k_ID_SpatialDimsO, spatialLengthO);

                cb.Dispatch(fnPool, globalNonSpatialLength, ComputeHelper.IDivC(localSpatialLength, 4), 1);

                if (isTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otemp;
                localSpatialLength = spatialLengthO;
                isTempAlloc = true;
            }

            var fn = ComputeFunctions.k_GlobalAveragePool;
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.SetInt(fn, k_ID_SpatialDims, localSpatialLength);
            cb.SetInt(fn, k_ID_GlobalSpatialDims, globalSpatialDims);

            cb.Dispatch(fn, globalNonSpatialLength, 1, 1);

            if (isTempAlloc)
                ReleaseTensorFloat(X);
        }

        // Compare ops
        /// <inheritdoc/>
        public void Greater(TensorFloat A, TensorFloat B, TensorInt O)
        {
            var fn = ComputeFunctions.k_GreaterFloat;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
            cb.SetInt(fn, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }
        /// <inheritdoc/>
        public void Greater(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_GreaterInt;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
            cb.SetInt(fn, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }
        /// <inheritdoc/>
        public void GreaterOrEqual(TensorFloat A, TensorFloat B, TensorInt O)
        {
            var fn = ComputeFunctions.k_GreaterOrEqualFloat;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
            cb.SetInt(fn, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }
        /// <inheritdoc/>
        public void GreaterOrEqual(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_GreaterOrEqualInt;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
            cb.SetInt(fn, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }
        /// <inheritdoc/>
        public void Less(TensorFloat A, TensorFloat B, TensorInt O)
        {
            var fn = ComputeFunctions.k_LessFloat;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
            cb.SetInt(fn, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }
        /// <inheritdoc/>
        public void Less(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_LessInt;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
            cb.SetInt(fn, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }
        /// <inheritdoc/>
        public void LessOrEqual(TensorFloat A, TensorFloat B, TensorInt O)
        {
            var fn = ComputeFunctions.k_LessOrEqualFloat;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
            cb.SetInt(fn, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }
        /// <inheritdoc/>
        public void LessOrEqual(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_LessOrEqualInt;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
            cb.SetInt(fn, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }
        /// <inheritdoc/>
        public void Equal(TensorFloat A, TensorFloat B, TensorInt O)
        {
            var fn = ComputeFunctions.k_EqualFloat;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
            cb.SetInt(fn, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }
        /// <inheritdoc/>
        public void Equal(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_EqualInt;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
            cb.SetInt(fn, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }
        /// <inheritdoc/>
        public void Or(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_OrInt;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
            cb.SetInt(fn, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }
        /// <inheritdoc/>
        public void And(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_AndInt;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
            cb.SetInt(fn, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }
        /// <inheritdoc/>
        public void Xor(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_XorInt;
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
            cb.SetInt(fn, k_ID_rank, O.shape.rank);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
            cb.UnrolledDispatch(fn, O.shape.length);
        }

        // Reduction
        internal void ReduceMin(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceMinFloat;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceMinFloat;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceMinFloat;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        internal void ReduceMax(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceMaxFloat;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceMaxFloat;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceMaxFloat;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        internal void ReduceSum(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceSumFloat;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceSumFloat;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceSumFloat;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        internal void ReduceSumSquare(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceSumSquareFloat;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceSumSquareFloat;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceSumSquareFloat;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        internal void ReduceMean(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceMeanFloat;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceMeanFloat;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceMeanFloat;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        internal void ReduceProd(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceProdFloat;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceProdFloat;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceProdFloat;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        internal void ReduceL1(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceL1Float;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceL1Float;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceL1Float;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        internal void ReduceL2(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceL2Float;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceL2Float;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceL2Float;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        internal void ReduceSqrt(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceSqrtFloat;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceSqrtFloat;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceSqrtFloat;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        internal void ReduceLogSum(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceLogSumFloat;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceLogSumFloat;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceLogSumFloat;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        internal void ReduceLogSum(TensorFloat X, TensorFloat Xmax, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceLogSumFloat;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Bptr, Pin(Xmax));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceLogSumFloat;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Bptr, Pin(Xmax));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceLogSumFloat;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Bptr, Pin(Xmax));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        internal void ReduceLogSumExp(TensorFloat X, TensorFloat Xmax, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceLogSumExpFloat;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Bptr, Pin(Xmax));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceLogSumExpFloat;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Bptr, Pin(Xmax));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceLogSumExpFloat;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Bptr, Pin(Xmax));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        internal void ReduceSumExp(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceSumExpFloat;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceSumExpFloat;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceSumExpFloat;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        internal void ReduceSumExp(TensorFloat X, TensorFloat Xmax, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceSumExpFloat;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Bptr, Pin(Xmax));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceSumExpFloat;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Bptr, Pin(Xmax));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceSumExpFloat;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Bptr, Pin(Xmax));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        internal void ReduceMin(TensorInt X, TensorInt O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceMinInt;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorInt(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceMinInt;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorInt(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceMinInt;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        internal void ReduceMax(TensorInt X, TensorInt O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceMaxInt;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorInt(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceMaxInt;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorInt(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceMaxInt;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        internal void ReduceSum(TensorInt X, TensorInt O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceSumInt;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorInt(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceSumInt;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorInt(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceSumInt;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        internal void ReduceSumSquare(TensorInt X, TensorInt O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceSumSquareInt;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorInt(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceSumSquareInt;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorInt(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceSumSquareInt;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        internal void ReduceProd(TensorInt X, TensorInt O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceProdInt;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorInt(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceProdInt;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorInt(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceProdInt;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        internal void ReduceL1(TensorInt X, TensorInt O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceL1Int;
                cb.SetInt(fallbackKernel, k_ID_ReducedDim, reduceLength);
                cb.SetInt(fallbackKernel, k_ID_InnerDim, innerLength);
                cb.SetFloat(fallbackKernel, k_ID_Normalization, 1.0f / reduceLength);

                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fallbackKernel, k_ID_Optr, Pin(O));
                cb.UnrolledDispatch(fallbackKernel, outerLength * innerLength);
                return;
            }

            int localReduceLength = reduceLength;
            bool isFirstDispatch = true;

            const int kernelReductionThreadCount = 64 * 4;

            // downsample with pyramid approach
            while (localReduceLength > kernelReductionThreadCount)
            {
                int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);

                var Otemp = AllocTensorInt(new TensorShape(outerLength * spatialLengthO * innerLength));

                var localKernel = ComputeFunctions.k_ReduceL1Int;
                cb.SetTensorAsBuffer(localKernel, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(localKernel, k_ID_Optr, Pin(Otemp));
                cb.SetInt(localKernel, k_ID_ReducedDim, localReduceLength);
                cb.SetInt(localKernel, k_ID_InnerDim, innerLength);
                cb.SetInt(localKernel, k_ID_SpatialDimsO, spatialLengthO);
                cb.SetInt(localKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                cb.Dispatch(localKernel, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorInt(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceL1Int;
            cb.SetTensorAsBuffer(globalKernel, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(globalKernel, k_ID_Optr, Pin(O));
            cb.SetInt(globalKernel, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(globalKernel, k_ID_InnerDim, innerLength);
            cb.SetInt(globalKernel, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            cb.SetFloat(globalKernel, k_ID_Normalization, 1.0f / reduceLength);

            cb.Dispatch(globalKernel, outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        public void ReduceMin(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceMin(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else
                {
                    var Otmp = AllocTensorFloat(shapeXReduced);

                    ReduceMin(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorFloat(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                ReduceMin(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceMax(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceMax(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else
                {
                    var Otmp = AllocTensorFloat(shapeXReduced);

                    ReduceMax(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorFloat(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                ReduceMax(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceSum(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceSum(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else
                {
                    var Otmp = AllocTensorFloat(shapeXReduced);

                    ReduceSum(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorFloat(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                ReduceSum(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceSumSquare(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceSumSquare(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isInitial = true;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else if (isInitial)
                {
                    var Otmp = AllocTensorFloat(shapeXReduced);

                    ReduceSumSquare(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorFloat(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isInitial = false;
                    isXTempAlloc = true;
                }
                else
                {
                    var Otmp = AllocTensorFloat(shapeXReduced);

                    ReduceSum(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorFloat(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            if (isInitial)
            {
                ReduceSumSquare(X, O, outerLength, reduceLength, innerLength);
            }
            else
            {
                ReduceSum(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceMean(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceMean(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else
                {
                    var Otmp = AllocTensorFloat(shapeXReduced);

                    ReduceMean(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorFloat(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                ReduceMean(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceProd(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceProd(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else
                {
                    var Otmp = AllocTensorFloat(shapeXReduced);

                    ReduceProd(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorFloat(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                ReduceProd(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceL1(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceL1(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isInitial = true;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else if (isInitial)
                {
                    var Otmp = AllocTensorFloat(shapeXReduced);

                    ReduceL1(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorFloat(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isInitial = false;
                    isXTempAlloc = true;
                }
                else
                {
                    var Otmp = AllocTensorFloat(shapeXReduced);

                    ReduceSum(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorFloat(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            if (isInitial)
            {
                ReduceL1(X, O, outerLength, reduceLength, innerLength);
            }
            else
            {
                ReduceSum(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceL2(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceL2(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isInitial = true;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else if (isInitial)
                {
                    var Otmp = AllocTensorFloat(shapeXReduced);

                    ReduceSumSquare(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorFloat(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isInitial = false;
                    isXTempAlloc = true;
                }
                else
                {
                    var Otmp = AllocTensorFloat(shapeXReduced);

                    ReduceSum(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorFloat(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            if (isInitial)
            {
                ReduceL2(X, O, outerLength, reduceLength, innerLength);
            }
            else
            {
                ReduceSqrt(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceLogSum(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceLogSum(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else
                {
                    var Otmp = AllocTensorFloat(shapeXReduced);

                    ReduceSum(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorFloat(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                ReduceLogSum(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceLogSumExp(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                var Xmax = AllocTensorFloat(O.shape);
                ReduceMax(X, Xmax, 1, X.shape.length, 1);
                ReduceLogSumExp(X, Xmax, O, 1, X.shape.length, 1);
                ReleaseTensorFloat(Xmax);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else
                {
                    var Otmp = AllocTensorFloat(shapeXReduced);

                    var Xmax = AllocTensorFloat(shapeXReduced);
                    ReduceMax(X, Xmax, outerLength, reduceLength, innerLength);
                    ReduceLogSumExp(X, Xmax, Otmp, outerLength, reduceLength, innerLength);
                    ReleaseTensorFloat(Xmax);

                    if (isXTempAlloc)
                        ReleaseTensorFloat(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var Xmax = AllocTensorFloat(shapeXReduced);
                ReduceMax(X, Xmax, outerLength, reduceLength, innerLength);
                ReduceLogSumExp(X, Xmax, O, outerLength, reduceLength, innerLength);
                ReleaseTensorFloat(Xmax);
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceSumExp(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceSumExp(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else
                {
                    var Otmp = AllocTensorFloat(shapeXReduced);

                    ReduceSumExp(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorFloat(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                ReduceSumExp(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceMin(TensorInt X, TensorInt O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceMin(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else
                {
                    var Otmp = AllocTensorInt(shapeXReduced);

                    ReduceMin(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorInt(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                ReduceMin(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        public void ReduceMax(TensorInt X, TensorInt O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceMax(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else
                {
                    var Otmp = AllocTensorInt(shapeXReduced);

                    ReduceMax(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorInt(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                ReduceMax(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        public void ReduceSum(TensorInt X, TensorInt O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceSum(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else
                {
                    var Otmp = AllocTensorInt(shapeXReduced);

                    ReduceSum(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorInt(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                ReduceSum(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        public void ReduceSumSquare(TensorInt X, TensorInt O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceSumSquare(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isInitial = true;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else if (isInitial)
                {
                    var Otmp = AllocTensorInt(shapeXReduced);

                    ReduceSumSquare(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorInt(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isInitial = false;
                    isXTempAlloc = true;
                }
                else
                {
                    var Otmp = AllocTensorInt(shapeXReduced);

                    ReduceSum(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorInt(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            if (isInitial)
            {
                ReduceSumSquare(X, O, outerLength, reduceLength, innerLength);
            }
            else
            {
                ReduceSum(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        public void ReduceProd(TensorInt X, TensorInt O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceProd(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else
                {
                    var Otmp = AllocTensorInt(shapeXReduced);

                    ReduceProd(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorInt(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                ReduceProd(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        public void ReduceL1(TensorInt X, TensorInt O, ReadOnlySpan<int> axes)
        {
            if (axes == null || axes.Length == 0)
            {
                ReduceL1(X, O, 1, X.shape.length, 1);
                return;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
            int outerLength = X.shape.Length(0, axis);
            int dimX = X.shape[axis];
            int reduceLength = dimX;
            TensorShape shapeXReduced = X.shape;
            shapeXReduced[axis] = 1;
            int prevAxis = axis;
            bool isInitial = true;
            bool isXTempAlloc = false;

            for (int i = 1; i < axes.Length; i++)
            {
                axis = X.shape.Axis(axes[i]);
                dimX = X.shape[axis];
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                if ((axis == (prevAxis + 1)))
                {
                    innerLength /= dimX;
                    reduceLength *= dimX;
                }
                else if (isInitial)
                {
                    var Otmp = AllocTensorInt(shapeXReduced);

                    ReduceL1(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorInt(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isInitial = false;
                    isXTempAlloc = true;
                }
                else
                {
                    var Otmp = AllocTensorInt(shapeXReduced);

                    ReduceSum(X, Otmp, outerLength, reduceLength, innerLength);

                    if (isXTempAlloc)
                        ReleaseTensorInt(X);
                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isXTempAlloc = true;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            if (isInitial)
            {
                ReduceL1(X, O, outerLength, reduceLength, innerLength);
            }
            else
            {
                ReduceSum(X, O, outerLength, reduceLength, innerLength);
            }
            if (isXTempAlloc)
                ReleaseTensorInt(X);
        }
    }
}
