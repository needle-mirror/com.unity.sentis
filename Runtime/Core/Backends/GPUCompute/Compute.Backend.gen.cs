// This is auto-generated -- do not modify directly

using System;
using Unity.Sentis;
using UnityEngine.Assertions;
using static Unity.Sentis.ComputeTensorData;
using static Unity.Sentis.ShaderPropertyID;

namespace Unity.Sentis
{
    public partial class GPUComputeBackend
    {
        // Binary Broadcast

        /// <inheritdoc/>
        public void PRelu(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastPRelu;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastPRelu;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwisePRelu;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Pow(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastPowFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastPowFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwisePowFloat;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Add(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastAddFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastAddFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseAddFloat;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Sub(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastSubFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastSubFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseSubFloat;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Mul(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastMulFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastMulFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseMulFloat;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Div(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastDivFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastDivFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseDivFloat;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Mod(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastModFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastModFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseModFloat;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void FMod(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastFModFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastFModFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseFModFloat;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Pow(TensorFloat A, TensorInt B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastPowInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastPowInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwisePowInt;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Add(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastAddInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastAddInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseAddInt;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Sub(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastSubInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastSubInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseSubInt;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Mul(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastMulInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastMulInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseMulInt;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Div(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastDivInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastDivInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseDivInt;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Mod(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastModInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastModInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseModInt;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void FMod(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastFModInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastFModInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseFModInt;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Min(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastMinFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastMinFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseMinFloat;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Max(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastMaxFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastMaxFloat;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseMaxFloat;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Min(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastMinInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastMinInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseMinInt;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Max(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFunctions.k_ScalarBroadcastMaxInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFunctions.k_BroadcastMaxInt;
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeHelper.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFunctions.k_ElementwiseMaxInt;
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fn.UnrolledDispatch(O.shape.length);
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
                    fn.SetInt(k_ID_stride, strides[0]);
                    fn.SetInt(k_ID_pad, pads[0]);
                    fn.SetInt(k_ID_inHeight, X.shape[2]);
                    fn.SetInt(k_ID_pool, kernelShape[0]);
                    fn.SetInt(k_ID_outHeight, O.shape[2]);
                    break;
                case 4:
                    fn = ComputeFunctions.k_MaxPool2D;
                    fn.SetInt(k_ID_strideX, strides[1]);
                    fn.SetInt(k_ID_strideY, strides[0]);
                    fn.SetInt(k_ID_padX, pads[1]);
                    fn.SetInt(k_ID_padY, pads[0]);

                    fn.SetInt(k_ID_inHeight, X.shape[2]);
                    fn.SetInt(k_ID_inWidth, X.shape[3]);

                    fn.SetInt(k_ID_poolX, kernelShape[1]);
                    fn.SetInt(k_ID_poolY, kernelShape[0]);

                    fn.SetInt(k_ID_outHeight, O.shape[2]);
                    fn.SetInt(k_ID_outWidth, O.shape[3]);
                    break;
                default:
                    throw new NotImplementedException();
            }
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }
        /// <inheritdoc/>
        public void AveragePool(TensorFloat X, TensorFloat O, int[] kernelShape, int[] strides, int[] pads)
        {
            ComputeFunction fn;
            switch (X.shape.rank)
            {
                case 3:
                    fn = ComputeFunctions.k_AveragePool1D;
                    fn.SetInt(k_ID_stride, strides[0]);
                    fn.SetInt(k_ID_pad, pads[0]);
                    fn.SetInt(k_ID_inHeight, X.shape[2]);
                    fn.SetInt(k_ID_pool, kernelShape[0]);
                    fn.SetInt(k_ID_outHeight, O.shape[2]);
                    break;
                case 4:
                    fn = ComputeFunctions.k_AveragePool2D;
                    fn.SetInt(k_ID_strideX, strides[1]);
                    fn.SetInt(k_ID_strideY, strides[0]);
                    fn.SetInt(k_ID_padX, pads[1]);
                    fn.SetInt(k_ID_padY, pads[0]);

                    fn.SetInt(k_ID_inHeight, X.shape[2]);
                    fn.SetInt(k_ID_inWidth, X.shape[3]);

                    fn.SetInt(k_ID_poolX, kernelShape[1]);
                    fn.SetInt(k_ID_poolY, kernelShape[0]);

                    fn.SetInt(k_ID_outHeight, O.shape[2]);
                    fn.SetInt(k_ID_outWidth, O.shape[3]);
                    break;
                default:
                    throw new NotImplementedException();
            }
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
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
                fnPool.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fnPool.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                fnPool.SetInt(k_ID_SpatialDims, localSpatialLength);
                fnPool.SetInt(k_ID_SpatialDimsO, spatialLengthO);

                fnPool.Dispatch(globalNonSpatialLength, ComputeHelper.IDivC(localSpatialLength, 4), 1);

                if (isTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otemp;
                localSpatialLength = spatialLengthO;
                isTempAlloc = true;
            }

            var fn = ComputeFunctions.k_GlobalMaxPool;
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.SetInt(k_ID_SpatialDims, localSpatialLength);
            fn.SetInt(k_ID_GlobalSpatialDims, globalSpatialDims);

            fn.Dispatch(globalNonSpatialLength, 1, 1);

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
                fnPool.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fnPool.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                fnPool.SetInt(k_ID_SpatialDims, localSpatialLength);
                fnPool.SetInt(k_ID_SpatialDimsO, spatialLengthO);

                fnPool.Dispatch(globalNonSpatialLength, ComputeHelper.IDivC(localSpatialLength, 4), 1);

                if (isTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otemp;
                localSpatialLength = spatialLengthO;
                isTempAlloc = true;
            }

            var fn = ComputeFunctions.k_GlobalAveragePool;
            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.SetInt(k_ID_SpatialDims, localSpatialLength);
            fn.SetInt(k_ID_GlobalSpatialDims, globalSpatialDims);

            fn.Dispatch(globalNonSpatialLength, 1, 1);

            if (isTempAlloc)
                ReleaseTensorFloat(X);
        }

        // Compare ops
        /// <inheritdoc/>
        public void Greater(TensorFloat A, TensorFloat B, TensorInt O)
        {
            var fn = ComputeFunctions.k_GreaterFloat;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeA, k_ID_stridesA, A.shape);
            fn.SetTensorShapeStrides(k_ID_shapeB, k_ID_stridesB, B.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }
        /// <inheritdoc/>
        public void Greater(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_GreaterInt;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeA, k_ID_stridesA, A.shape);
            fn.SetTensorShapeStrides(k_ID_shapeB, k_ID_stridesB, B.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }
        /// <inheritdoc/>
        public void GreaterOrEqual(TensorFloat A, TensorFloat B, TensorInt O)
        {
            var fn = ComputeFunctions.k_GreaterOrEqualFloat;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeA, k_ID_stridesA, A.shape);
            fn.SetTensorShapeStrides(k_ID_shapeB, k_ID_stridesB, B.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }
        /// <inheritdoc/>
        public void GreaterOrEqual(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_GreaterOrEqualInt;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeA, k_ID_stridesA, A.shape);
            fn.SetTensorShapeStrides(k_ID_shapeB, k_ID_stridesB, B.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }
        /// <inheritdoc/>
        public void Less(TensorFloat A, TensorFloat B, TensorInt O)
        {
            var fn = ComputeFunctions.k_LessFloat;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeA, k_ID_stridesA, A.shape);
            fn.SetTensorShapeStrides(k_ID_shapeB, k_ID_stridesB, B.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }
        /// <inheritdoc/>
        public void Less(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_LessInt;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeA, k_ID_stridesA, A.shape);
            fn.SetTensorShapeStrides(k_ID_shapeB, k_ID_stridesB, B.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }
        /// <inheritdoc/>
        public void LessOrEqual(TensorFloat A, TensorFloat B, TensorInt O)
        {
            var fn = ComputeFunctions.k_LessOrEqualFloat;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeA, k_ID_stridesA, A.shape);
            fn.SetTensorShapeStrides(k_ID_shapeB, k_ID_stridesB, B.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }
        /// <inheritdoc/>
        public void LessOrEqual(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_LessOrEqualInt;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeA, k_ID_stridesA, A.shape);
            fn.SetTensorShapeStrides(k_ID_shapeB, k_ID_stridesB, B.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }
        /// <inheritdoc/>
        public void Equal(TensorFloat A, TensorFloat B, TensorInt O)
        {
            var fn = ComputeFunctions.k_EqualFloat;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeA, k_ID_stridesA, A.shape);
            fn.SetTensorShapeStrides(k_ID_shapeB, k_ID_stridesB, B.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }
        /// <inheritdoc/>
        public void Equal(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_EqualInt;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeA, k_ID_stridesA, A.shape);
            fn.SetTensorShapeStrides(k_ID_shapeB, k_ID_stridesB, B.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }
        /// <inheritdoc/>
        public void Or(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_OrInt;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeA, k_ID_stridesA, A.shape);
            fn.SetTensorShapeStrides(k_ID_shapeB, k_ID_stridesB, B.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }
        /// <inheritdoc/>
        public void And(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_AndInt;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeA, k_ID_stridesA, A.shape);
            fn.SetTensorShapeStrides(k_ID_shapeB, k_ID_stridesB, B.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }
        /// <inheritdoc/>
        public void Xor(TensorInt A, TensorInt B, TensorInt O)
        {
            var fn = ComputeFunctions.k_XorInt;
            fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
            fn.SetTensorShapeStrides(k_ID_shapeA, k_ID_stridesA, A.shape);
            fn.SetTensorShapeStrides(k_ID_shapeB, k_ID_stridesB, B.shape);
            fn.SetInt(k_ID_rank, O.shape.rank);

            fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
            fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
            fn.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            fn.UnrolledDispatch(O.shape.length);
        }

        // Reduction
        internal void ReduceMin(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceMinFloat;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceMinFloat;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }
        internal void ReduceMax(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceMaxFloat;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceMaxFloat;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }
        internal void ReduceSum(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceSumFloat;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceSumFloat;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }
        internal void ReduceSumSquare(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceSumSquareFloat;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceSumSquareFloat;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }
        internal void ReduceMean(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceMeanFloat;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceMeanFloat;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }
        internal void ReduceProd(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceProdFloat;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceProdFloat;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }
        internal void ReduceL1(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceL1Float;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceL1Float;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }
        internal void ReduceL2(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceL2Float;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceL2Float;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }
        internal void ReduceSqrt(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceSqrtFloat;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceSqrtFloat;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }
        internal void ReduceLogSum(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceLogSumFloat;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceLogSumFloat;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }
        internal void ReduceLogSum(TensorFloat X, TensorFloat Xmax, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceLogSumFloat;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.SetTensorAsBuffer(k_ID_Bptr, Pin(Xmax));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Bptr, Pin(Xmax));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceLogSumFloat;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Bptr, Pin(Xmax));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }
        internal void ReduceLogSumExp(TensorFloat X, TensorFloat Xmax, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceLogSumExpFloat;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.SetTensorAsBuffer(k_ID_Bptr, Pin(Xmax));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Bptr, Pin(Xmax));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceLogSumExpFloat;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Bptr, Pin(Xmax));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }
        internal void ReduceSumExp(TensorFloat X, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceSumExpFloat;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceSumExpFloat;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }
        internal void ReduceSumExp(TensorFloat X, TensorFloat Xmax, TensorFloat O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceSumExpFloat;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.SetTensorAsBuffer(k_ID_Bptr, Pin(Xmax));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Bptr, Pin(Xmax));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorFloat(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceSumExpFloat;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Bptr, Pin(Xmax));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorFloat(X);
        }
        internal void ReduceMin(TensorInt X, TensorInt O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceMinInt;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorInt(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceMinInt;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorInt(X);
        }
        internal void ReduceMax(TensorInt X, TensorInt O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceMaxInt;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorInt(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceMaxInt;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorInt(X);
        }
        internal void ReduceSum(TensorInt X, TensorInt O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceSumInt;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorInt(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceSumInt;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorInt(X);
        }
        internal void ReduceSumSquare(TensorInt X, TensorInt O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceSumSquareInt;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorInt(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceSumSquareInt;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorInt(X);
        }
        internal void ReduceProd(TensorInt X, TensorInt O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceProdInt;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorInt(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceProdInt;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

            if (!isFirstDispatch)
                ReleaseTensorInt(X);
        }
        internal void ReduceL1(TensorInt X, TensorInt O, int outerLength, int reduceLength, int innerLength)
        {
            if (innerLength > (int)ComputeHelper.SafeDispatchLimit || outerLength > (int)ComputeHelper.SafeDispatchLimit)
            {
                var fallbackKernel = ComputeFunctions.k_UnrolledReduceL1Int;
                fallbackKernel.SetInt(k_ID_ReducedDim, reduceLength);
                fallbackKernel.SetInt(k_ID_InnerDim, innerLength);
                fallbackKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

                fallbackKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fallbackKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
                fallbackKernel.UnrolledDispatch(outerLength * innerLength);
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
                localKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                localKernel.SetTensorAsBuffer(k_ID_Optr, Pin(Otemp));
                localKernel.SetInt(k_ID_ReducedDim, localReduceLength);
                localKernel.SetInt(k_ID_InnerDim, innerLength);
                localKernel.SetInt(k_ID_SpatialDimsO, spatialLengthO);
                localKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

                localKernel.Dispatch(outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

                if (!isFirstDispatch)
                    ReleaseTensorInt(X);

                X = Otemp;
                localReduceLength = spatialLengthO;
                isFirstDispatch = false;
            }

            var globalKernel = ComputeFunctions.k_GlobalReduceL1Int;
            globalKernel.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
            globalKernel.SetTensorAsBuffer(k_ID_Optr, Pin(O));
            globalKernel.SetInt(k_ID_ReducedDim, localReduceLength);
            globalKernel.SetInt(k_ID_InnerDim, innerLength);
            globalKernel.SetInt(k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
            globalKernel.SetFloat(k_ID_Normalization, 1.0f / reduceLength);

            globalKernel.Dispatch(outerLength, 1, innerLength);

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
