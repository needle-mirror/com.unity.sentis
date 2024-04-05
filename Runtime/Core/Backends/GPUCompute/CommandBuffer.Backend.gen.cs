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
        public void Pow(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastPowFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastPowFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwisePowFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Add(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastAddFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastAddFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseAddFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Sub(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastSubFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastSubFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseSubFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Mul(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastMulFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastMulFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseMulFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Div(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastDivFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastDivFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseDivFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Mod(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastModFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastModFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseModFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void FMod(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastFModFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastFModFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseFModFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Pow(TensorFloat A, TensorInt B, TensorFloat O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastPowInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastPowInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwisePowInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Add(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastAddInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastAddInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseAddInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Sub(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastSubInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastSubInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseSubInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Mul(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastMulInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastMulInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseMulInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Div(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastDivInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastDivInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseDivInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Mod(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastModInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastModInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseModInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void FMod(TensorInt A, TensorInt B, TensorInt O)
        {
            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastFModInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastFModInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(A));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseFModInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O), O.shape.length);
            }
        }

        // Variadic Broadcast

        void BroadcastMin(TensorFloat X, TensorFloat Y, TensorFloat O)
        {
            if (X.shape == O.shape && Y.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastMinFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Y));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastMinFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Y));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseMinFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, Y.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Min(TensorFloat[] tensors, TensorFloat O)
        {
            var Otmp = (tensors.Length > 2) ? AllocTensorFloat(O.shape) : null;

            var curX = tensors[0];
            var curO = tensors.Length % 2 == 0 ? O : Otmp;
            for (var t = 1; t < tensors.Length; t++)
            {
                BroadcastMin(curX, tensors[t], curO);
                curX = curO;
                curO = curO == O ? Otmp : O;
            }

            ReleaseTensorFloat(Otmp);
            Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");
        }

        void BroadcastMax(TensorFloat X, TensorFloat Y, TensorFloat O)
        {
            if (X.shape == O.shape && Y.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastMaxFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Y));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastMaxFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Y));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseMaxFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, Y.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Max(TensorFloat[] tensors, TensorFloat O)
        {
            var Otmp = (tensors.Length > 2) ? AllocTensorFloat(O.shape) : null;

            var curX = tensors[0];
            var curO = tensors.Length % 2 == 0 ? O : Otmp;
            for (var t = 1; t < tensors.Length; t++)
            {
                BroadcastMax(curX, tensors[t], curO);
                curX = curO;
                curO = curO == O ? Otmp : O;
            }

            ReleaseTensorFloat(Otmp);
            Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");
        }

        void BroadcastMean(TensorFloat X, TensorFloat Y, TensorFloat O, float normalizationX, float normalizationY)
        {
            if (X.shape == O.shape && Y.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastMeanFloat");
                cb.SetFloat(fn, k_ID_alpha, normalizationX);
                cb.SetFloat(fn, k_ID_beta, normalizationY);
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Y));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastMeanFloat");
                cb.SetFloat(fn, k_ID_alpha, normalizationX);
                cb.SetFloat(fn, k_ID_beta, normalizationY);
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Y));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseMeanFloat");
                cb.SetFloat(fn, k_ID_alpha, normalizationX);
                cb.SetFloat(fn, k_ID_beta, normalizationY);
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, Y.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Mean(TensorFloat[] tensors, TensorFloat O)
        {
            var Otmp = (tensors.Length > 2) ? AllocTensorFloat(O.shape) : null;

            var curX = tensors[0];
            var curO = tensors.Length % 2 == 0 ? O : Otmp;
            for (var t = 1; t < tensors.Length; t++)
            {
                BroadcastMean(curX, tensors[t], curO, t == 1 ? 1.0f / tensors.Length : 1.0f, 1.0f / tensors.Length);
                curX = curO;
                curO = curO == O ? Otmp : O;
            }

            ReleaseTensorFloat(Otmp);
            Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");
        }

        void BroadcastSum(TensorFloat X, TensorFloat Y, TensorFloat O)
        {
            if (X.shape == O.shape && Y.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastAddFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Y));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastAddFloat");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Y));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseAddFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, Y.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Sum(TensorFloat[] tensors, TensorFloat O)
        {
            var Otmp = (tensors.Length > 2) ? AllocTensorFloat(O.shape) : null;

            var curX = tensors[0];
            var curO = tensors.Length % 2 == 0 ? O : Otmp;
            for (var t = 1; t < tensors.Length; t++)
            {
                BroadcastSum(curX, tensors[t], curO);
                curX = curO;
                curO = curO == O ? Otmp : O;
            }

            ReleaseTensorFloat(Otmp);
            Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");
        }

        void BroadcastMin(TensorInt X, TensorInt Y, TensorInt O)
        {
            if (X.shape == O.shape && Y.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastMinInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Y));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastMinInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Y));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseMinInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, Y.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Min(TensorInt[] tensors, TensorInt O)
        {
            var Otmp = (tensors.Length > 2) ? AllocTensorInt(O.shape) : null;

            var curX = tensors[0];
            var curO = tensors.Length % 2 == 0 ? O : Otmp;
            for (var t = 1; t < tensors.Length; t++)
            {
                BroadcastMin(curX, tensors[t], curO);
                curX = curO;
                curO = curO == O ? Otmp : O;
            }

            ReleaseTensorInt(Otmp);
            Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");
        }

        void BroadcastMax(TensorInt X, TensorInt Y, TensorInt O)
        {
            if (X.shape == O.shape && Y.shape.length == 1)
            {
                var fn = ComputeFuncSingleton.Instance.Get("ScalarBroadcastMaxInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Y));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = ComputeFuncSingleton.Instance.Get("BroadcastMaxInt");
                cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Y));
                cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O));
                cb.SetInt(fn, k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                cb.SetInt(fn, k_ID_MaxBlockIndexX, numBlocksX * 4);
                cb.Dispatch(fn, numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = ComputeFuncSingleton.Instance.Get("ElementwiseMaxInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, Y.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public void Max(TensorInt[] tensors, TensorInt O)
        {
            var Otmp = (tensors.Length > 2) ? AllocTensorInt(O.shape) : null;

            var curX = tensors[0];
            var curO = tensors.Length % 2 == 0 ? O : Otmp;
            for (var t = 1; t < tensors.Length; t++)
            {
                BroadcastMax(curX, tensors[t], curO);
                curX = curO;
                curO = curO == O ? Otmp : O;
            }

            ReleaseTensorInt(Otmp);
            Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");
        }

        // Reduction

        /// <inheritdoc/>
        public void ReduceMin(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceMinFloat", "GlobalReduceMinFloat", "UnrolledReduceMinFloat");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceMinFloat", "GlobalReduceMinFloat", "UnrolledReduceMinFloat");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceMinFloat", "GlobalReduceMinFloat", "UnrolledReduceMinFloat");
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceMax(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceMaxFloat", "GlobalReduceMaxFloat", "UnrolledReduceMaxFloat");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceMaxFloat", "GlobalReduceMaxFloat", "UnrolledReduceMaxFloat");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceMaxFloat", "GlobalReduceMaxFloat", "UnrolledReduceMaxFloat");
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceSum(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceSumSquare(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceSumSquareFloat", "GlobalReduceSumSquareFloat", "UnrolledReduceSumSquareFloat");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumSquareFloat", "GlobalReduceSumSquareFloat", "UnrolledReduceSumSquareFloat");

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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceSumSquareFloat", "GlobalReduceSumSquareFloat", "UnrolledReduceSumSquareFloat");
            }
            else
            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceMean(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceMeanFloat", "GlobalReduceMeanFloat", "UnrolledReduceMeanFloat");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceMeanFloat", "GlobalReduceMeanFloat", "UnrolledReduceMeanFloat");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceMeanFloat", "GlobalReduceMeanFloat", "UnrolledReduceMeanFloat");
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceProd(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceProdFloat", "GlobalReduceProdFloat", "UnrolledReduceProdFloat");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceProdFloat", "GlobalReduceProdFloat", "UnrolledReduceProdFloat");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceProdFloat", "GlobalReduceProdFloat", "UnrolledReduceProdFloat");
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceL1(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceL1Float", "GlobalReduceL1Float", "UnrolledReduceL1Float");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceL1Float", "GlobalReduceL1Float", "UnrolledReduceL1Float");

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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceL1Float", "GlobalReduceL1Float", "UnrolledReduceL1Float");
            }
            else
            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceL2(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceL2Float", "GlobalReduceL2Float", "UnrolledReduceL2Float");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumSquareFloat", "GlobalReduceSumSquareFloat", "UnrolledReduceSumSquareFloat");

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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceL2Float", "GlobalReduceL2Float", "UnrolledReduceL2Float");
            }
            else
            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceSqrtFloat", "GlobalReduceSqrtFloat", "UnrolledReduceSqrtFloat");
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceLogSum(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceLogSumFloat", "GlobalReduceLogSumFloat", "UnrolledReduceLogSumFloat");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceLogSumFloat", "GlobalReduceLogSumFloat", "UnrolledReduceLogSumFloat");
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceLogSumExp(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                var Xmax = AllocTensorFloat(O.shape);
                Reduce(X, Xmax, 1, X.shape.length, 1, "ReduceMaxFloat", "GlobalReduceMaxFloat", "UnrolledReduceMaxFloat");
                Reduce(X, Xmax, O, 1, X.shape.length, 1, "ReduceLogSumExpFloat", "GlobalReduceLogSumExpFloat", "UnrolledReduceLogSumExpFloat");
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
                    Reduce(X, Xmax, outerLength, reduceLength, innerLength, "ReduceMaxFloat", "GlobalReduceMaxFloat", "UnrolledReduceMaxFloat");
                    Reduce(X, Xmax, Otmp, outerLength, reduceLength, innerLength, "ReduceLogSumExpFloat", "GlobalReduceLogSumExpFloat", "UnrolledReduceLogSumExpFloat");
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
                Reduce(X, Xmax, outerLength, reduceLength, innerLength, "ReduceMaxFloat", "GlobalReduceMaxFloat", "UnrolledReduceMaxFloat");
                Reduce(X, Xmax, O, outerLength, reduceLength, innerLength, "ReduceLogSumExpFloat", "GlobalReduceLogSumExpFloat", "UnrolledReduceLogSumExpFloat");
                ReleaseTensorFloat(Xmax);
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceSumExp(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceSumExpFloat", "GlobalReduceSumExpFloat", "UnrolledReduceSumExpFloat");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumExpFloat", "GlobalReduceSumExpFloat", "UnrolledReduceSumExpFloat");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceSumExpFloat", "GlobalReduceSumExpFloat", "UnrolledReduceSumExpFloat");
            }
            if (isXTempAlloc)
                ReleaseTensorFloat(X);
        }

        /// <inheritdoc/>
        public void ReduceMin(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceMinInt", "GlobalReduceMinInt", "UnrolledReduceMinInt");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceMinInt", "GlobalReduceMinInt", "UnrolledReduceMinInt");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceMinInt", "GlobalReduceMinInt", "UnrolledReduceMinInt");
            }
            if (isXTempAlloc)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        public void ReduceMax(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceMaxInt", "GlobalReduceMaxInt", "UnrolledReduceMaxInt");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceMaxInt", "GlobalReduceMaxInt", "UnrolledReduceMaxInt");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceMaxInt", "GlobalReduceMaxInt", "UnrolledReduceMaxInt");
            }
            if (isXTempAlloc)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        public void ReduceSum(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceSumInt", "GlobalReduceSumInt", "UnrolledReduceSumInt");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumInt", "GlobalReduceSumInt", "UnrolledReduceSumInt");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceSumInt", "GlobalReduceSumInt", "UnrolledReduceSumInt");
            }
            if (isXTempAlloc)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        public void ReduceSumSquare(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceSumSquareInt", "GlobalReduceSumSquareInt", "UnrolledReduceSumSquareInt");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumSquareInt", "GlobalReduceSumSquareInt", "UnrolledReduceSumSquareInt");

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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumInt", "GlobalReduceSumInt", "UnrolledReduceSumInt");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceSumSquareInt", "GlobalReduceSumSquareInt", "UnrolledReduceSumSquareInt");
            }
            else
            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceSumInt", "GlobalReduceSumInt", "UnrolledReduceSumInt");
            }
            if (isXTempAlloc)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        public void ReduceProd(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceProdInt", "GlobalReduceProdInt", "UnrolledReduceProdInt");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceProdInt", "GlobalReduceProdInt", "UnrolledReduceProdInt");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceProdInt", "GlobalReduceProdInt", "UnrolledReduceProdInt");
            }
            if (isXTempAlloc)
                ReleaseTensorInt(X);
        }

        /// <inheritdoc/>
        public void ReduceL1(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
        {
            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceL1Int", "GlobalReduceL1Int", "UnrolledReduceL1Int");
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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceL1Int", "GlobalReduceL1Int", "UnrolledReduceL1Int");

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

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumInt", "GlobalReduceSumInt", "UnrolledReduceSumInt");

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
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceL1Int", "GlobalReduceL1Int", "UnrolledReduceL1Int");
            }
            else
            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceSumInt", "GlobalReduceSumInt", "UnrolledReduceSumInt");
            }
            if (isXTempAlloc)
                ReleaseTensorInt(X);
        }
    }
}
