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
        public override TensorFloat Pow(TensorFloat A, TensorFloat B)
        {
            var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;

            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastPowFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastPowFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwisePowFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
            }
            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat Add(TensorFloat A, TensorFloat B)
        {
            var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;

            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastAddFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastAddFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseAddFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
            }
            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat Sub(TensorFloat A, TensorFloat B)
        {
            var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;

            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastSubFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastSubFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseSubFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
            }
            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat Mul(TensorFloat A, TensorFloat B)
        {
            var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;

            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastMulFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMulFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMulFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
            }
            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat Div(TensorFloat A, TensorFloat B)
        {
            var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;

            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastDivFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastDivFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseDivFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
            }
            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat FMod(TensorFloat A, TensorFloat B)
        {
            var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;

            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastFModFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastFModFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseFModFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
            }
            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat Pow(TensorFloat A, TensorInt B)
        {
            var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;

            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastPowInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastPowInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwisePowInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
            }
            return O;
        }

        /// <inheritdoc/>
        public override TensorInt Add(TensorInt A, TensorInt B)
        {
            var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;

            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastAddInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastAddInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseAddInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
            }
            return O;
        }

        /// <inheritdoc/>
        public override TensorInt Sub(TensorInt A, TensorInt B)
        {
            var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;

            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastSubInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastSubInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseSubInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
            }
            return O;
        }

        /// <inheritdoc/>
        public override TensorInt Mul(TensorInt A, TensorInt B)
        {
            var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;

            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastMulInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMulInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMulInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
            }
            return O;
        }

        /// <inheritdoc/>
        public override TensorInt Div(TensorInt A, TensorInt B)
        {
            var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;

            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastDivInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastDivInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseDivInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
            }
            return O;
        }

        /// <inheritdoc/>
        public override TensorInt Mod(TensorInt A, TensorInt B)
        {
            var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;

            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastModInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastModInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseModInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
            }
            return O;
        }

        /// <inheritdoc/>
        public override TensorInt FMod(TensorInt A, TensorInt B)
        {
            var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
            if (O.shape.HasZeroDims())
                return O;

            if (A.shape == O.shape && B.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastFModInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastFModInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(A));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(B));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseFModInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
            }
            return O;
        }

        // Variadic Broadcast

        void BroadcastMin(TensorFloat O, TensorFloat X, TensorFloat Y)
        {
            if (X.shape == O.shape && Y.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastMinFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(Y));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMinFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(Y));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMinFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, Y.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, clearOnInit: false), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public override TensorFloat Min(TensorFloat[] tensors)
        {
            var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(tensors));
            if (O.shape.HasZeroDims())
                return O;

            var Otmp = (tensors.Length > 2) ? NewTempTensorFloat(O.shape) : null;

            var curX = tensors[0];
            var curO = tensors.Length % 2 == 0 ? O : Otmp;
            for (var t = 1; t < tensors.Length; t++)
            {
                BroadcastMin(curO, curX, tensors[t]);
                curX = curO;
                curO = curO == O ? Otmp : O;
            }

            Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");

            return O;
        }

        void BroadcastMax(TensorFloat O, TensorFloat X, TensorFloat Y)
        {
            if (X.shape == O.shape && Y.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastMaxFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(Y));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMaxFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(Y));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMaxFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, Y.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, clearOnInit: false), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public override TensorFloat Max(TensorFloat[] tensors)
        {
            var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(tensors));
            if (O.shape.HasZeroDims())
                return O;

            var Otmp = (tensors.Length > 2) ? NewTempTensorFloat(O.shape) : null;

            var curX = tensors[0];
            var curO = tensors.Length % 2 == 0 ? O : Otmp;
            for (var t = 1; t < tensors.Length; t++)
            {
                BroadcastMax(curO, curX, tensors[t]);
                curX = curO;
                curO = curO == O ? Otmp : O;
            }

            Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");

            return O;
        }

        void BroadcastMean(TensorFloat O, TensorFloat X, TensorFloat Y, float normalizationX, float normalizationY)
        {
            if (X.shape == O.shape && Y.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastMeanFloat");
                fn.SetFloat(k_ID_alpha, normalizationX);
                fn.SetFloat(k_ID_beta, normalizationY);
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(Y));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMeanFloat");
                fn.SetFloat(k_ID_alpha, normalizationX);
                fn.SetFloat(k_ID_beta, normalizationY);
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(Y));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMeanFloat");
                fn.SetFloat(k_ID_alpha, normalizationX);
                fn.SetFloat(k_ID_beta, normalizationY);
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, Y.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, clearOnInit: false), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public override TensorFloat Mean(TensorFloat[] tensors)
        {
            var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(tensors));
            if (O.shape.HasZeroDims())
                return O;

            var Otmp = (tensors.Length > 2) ? NewTempTensorFloat(O.shape) : null;

            var curX = tensors[0];
            var curO = tensors.Length % 2 == 0 ? O : Otmp;
            for (var t = 1; t < tensors.Length; t++)
            {
                BroadcastMean(curO, curX, tensors[t], t == 1 ? 1.0f / tensors.Length : 1.0f, 1.0f / tensors.Length);
                curX = curO;
                curO = curO == O ? Otmp : O;
            }

            Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");

            return O;
        }

        void BroadcastSum(TensorFloat O, TensorFloat X, TensorFloat Y)
        {
            if (X.shape == O.shape && Y.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastAddFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(Y));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastAddFloat");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(Y));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseAddFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, Y.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, clearOnInit: false), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public override TensorFloat Sum(TensorFloat[] tensors)
        {
            var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(tensors));
            if (O.shape.HasZeroDims())
                return O;

            var Otmp = (tensors.Length > 2) ? NewTempTensorFloat(O.shape) : null;

            var curX = tensors[0];
            var curO = tensors.Length % 2 == 0 ? O : Otmp;
            for (var t = 1; t < tensors.Length; t++)
            {
                BroadcastSum(curO, curX, tensors[t]);
                curX = curO;
                curO = curO == O ? Otmp : O;
            }

            Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");

            return O;
        }

        void BroadcastMin(TensorInt O, TensorInt X, TensorInt Y)
        {
            if (X.shape == O.shape && Y.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastMinInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(Y));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMinInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(Y));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMinInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, Y.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, clearOnInit: false), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public override TensorInt Min(TensorInt[] tensors)
        {
            var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(tensors));
            if (O.shape.HasZeroDims())
                return O;

            var Otmp = (tensors.Length > 2) ? NewTempTensorInt(O.shape) : null;

            var curX = tensors[0];
            var curO = tensors.Length % 2 == 0 ? O : Otmp;
            for (var t = 1; t < tensors.Length; t++)
            {
                BroadcastMin(curO, curX, tensors[t]);
                curX = curO;
                curO = curO == O ? Otmp : O;
            }

            Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");

            return O;
        }

        void BroadcastMax(TensorInt O, TensorInt X, TensorInt Y)
        {
            if (X.shape == O.shape && Y.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastMaxInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(Y));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMaxInt");
                fn.SetTensorAsBuffer(k_ID_Xptr, Pin(X));
                fn.SetTensorAsBuffer(k_ID_Bptr, Pin(Y));
                fn.SetTensorAsBuffer(k_ID_Optr, Pin(O, clearOnInit: false));
                fn.SetInt(k_ID_LengthO, O.shape.length - 1);
                var numThreads = ComputeHelper.IDivC(O.shape.length, 4);
                var numBlocksY = ComputeHelper.IDivC(numThreads, (int)ComputeFunc.SafeDispatchLimit);
                var numBlocksX = ComputeHelper.IDivC(numThreads, numBlocksY);
                fn.SetInt(k_ID_MaxBlockIndexX, numBlocksX * 4);
                fn.Dispatch(numBlocksX, numBlocksY, 1);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMaxInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, Y.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, clearOnInit: false), O.shape.length);
            }
        }

        /// <inheritdoc/>
        public override TensorInt Max(TensorInt[] tensors)
        {
            var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(tensors));
            if (O.shape.HasZeroDims())
                return O;

            var Otmp = (tensors.Length > 2) ? NewTempTensorInt(O.shape) : null;

            var curX = tensors[0];
            var curO = tensors.Length % 2 == 0 ? O : Otmp;
            for (var t = 1; t < tensors.Length; t++)
            {
                BroadcastMax(curO, curX, tensors[t]);
                curX = curO;
                curO = curO == O ? Otmp : O;
            }

            Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");

            return O;
        }

        // Reduction

        /// <inheritdoc/>
        public override TensorFloat ReduceMin(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceMinFloat", "GlobalReduceMinFloat", "UnrolledReduceMinFloat");
                return O;
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
                    var Otmp = NewTempTensorFloat(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceMinFloat", "GlobalReduceMinFloat", "UnrolledReduceMinFloat");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceMinFloat", "GlobalReduceMinFloat", "UnrolledReduceMinFloat");
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceMax(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceMaxFloat", "GlobalReduceMaxFloat", "UnrolledReduceMaxFloat");
                return O;
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
                    var Otmp = NewTempTensorFloat(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceMaxFloat", "GlobalReduceMaxFloat", "UnrolledReduceMaxFloat");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceMaxFloat", "GlobalReduceMaxFloat", "UnrolledReduceMaxFloat");
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceSum(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");
                return O;
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
                    var Otmp = NewTempTensorFloat(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceSumSquare(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceSumSquareFloat", "GlobalReduceSumSquareFloat", "UnrolledReduceSumSquareFloat");
                return O;
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
                    var Otmp = NewTempTensorFloat(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumSquareFloat", "GlobalReduceSumSquareFloat", "UnrolledReduceSumSquareFloat");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorFloat(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
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

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceMean(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceMeanFloat", "GlobalReduceMeanFloat", "UnrolledReduceMeanFloat");
                return O;
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
                    var Otmp = NewTempTensorFloat(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceMeanFloat", "GlobalReduceMeanFloat", "UnrolledReduceMeanFloat");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceMeanFloat", "GlobalReduceMeanFloat", "UnrolledReduceMeanFloat");
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceProd(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceProdFloat", "GlobalReduceProdFloat", "UnrolledReduceProdFloat");
                return O;
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
                    var Otmp = NewTempTensorFloat(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceProdFloat", "GlobalReduceProdFloat", "UnrolledReduceProdFloat");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceProdFloat", "GlobalReduceProdFloat", "UnrolledReduceProdFloat");
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceL1(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceL1Float", "GlobalReduceL1Float", "UnrolledReduceL1Float");
                return O;
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
                    var Otmp = NewTempTensorFloat(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceL1Float", "GlobalReduceL1Float", "UnrolledReduceL1Float");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorFloat(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
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

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceL2(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceL2Float", "GlobalReduceL2Float", "UnrolledReduceL2Float");
                return O;
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
                    var Otmp = NewTempTensorFloat(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumSquareFloat", "GlobalReduceSumSquareFloat", "UnrolledReduceSumSquareFloat");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorFloat(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
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

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceLogSum(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceLogSumFloat", "GlobalReduceLogSumFloat", "UnrolledReduceLogSumFloat");
                return O;
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
                    var Otmp = NewTempTensorFloat(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumFloat", "GlobalReduceSumFloat", "UnrolledReduceSumFloat");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceLogSumFloat", "GlobalReduceLogSumFloat", "UnrolledReduceLogSumFloat");
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceLogSumExp(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var Xmax = NewTempTensorFloat(Oshape);
                Reduce(X, Xmax, 1, X.shape.length, 1, "ReduceMaxFloat", "GlobalReduceMaxFloat", "UnrolledReduceMaxFloat");
                Reduce(X, Xmax, O, 1, X.shape.length, 1, "ReduceLogSumExpFloat", "GlobalReduceLogSumExpFloat", "UnrolledReduceLogSumExpFloat");
                return O;
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
                    var Otmp = NewTempTensorFloat(shapeXReduced);

                    var Xmax = NewTempTensorFloat(shapeXReduced);
                    Reduce(X, Xmax, outerLength, reduceLength, innerLength, "ReduceMaxFloat", "GlobalReduceMaxFloat", "UnrolledReduceMaxFloat");
                    Reduce(X, Xmax, Otmp, outerLength, reduceLength, innerLength, "ReduceLogSumExpFloat", "GlobalReduceLogSumExpFloat", "UnrolledReduceLogSumExpFloat");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var Xmax = NewTempTensorFloat(shapeXReduced);
                Reduce(X, Xmax, outerLength, reduceLength, innerLength, "ReduceMaxFloat", "GlobalReduceMaxFloat", "UnrolledReduceMaxFloat");
                Reduce(X, Xmax, O, outerLength, reduceLength, innerLength, "ReduceLogSumExpFloat", "GlobalReduceLogSumExpFloat", "UnrolledReduceLogSumExpFloat");
            }

            return O;
        }

        /// <inheritdoc/>
        public TensorFloat ReduceSumExp(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceSumExpFloat", "GlobalReduceSumExpFloat", "UnrolledReduceSumExpFloat");
                return O;
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
                    var Otmp = NewTempTensorFloat(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumExpFloat", "GlobalReduceSumExpFloat", "UnrolledReduceSumExpFloat");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceSumExpFloat", "GlobalReduceSumExpFloat", "UnrolledReduceSumExpFloat");
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorInt ReduceMin(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorInt(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceMinInt", "GlobalReduceMinInt", "UnrolledReduceMinInt");
                return O;
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
                    var Otmp = NewTempTensorInt(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceMinInt", "GlobalReduceMinInt", "UnrolledReduceMinInt");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceMinInt", "GlobalReduceMinInt", "UnrolledReduceMinInt");
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorInt ReduceMax(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorInt(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceMaxInt", "GlobalReduceMaxInt", "UnrolledReduceMaxInt");
                return O;
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
                    var Otmp = NewTempTensorInt(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceMaxInt", "GlobalReduceMaxInt", "UnrolledReduceMaxInt");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceMaxInt", "GlobalReduceMaxInt", "UnrolledReduceMaxInt");
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorInt ReduceSum(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorInt(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceSumInt", "GlobalReduceSumInt", "UnrolledReduceSumInt");
                return O;
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
                    var Otmp = NewTempTensorInt(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumInt", "GlobalReduceSumInt", "UnrolledReduceSumInt");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceSumInt", "GlobalReduceSumInt", "UnrolledReduceSumInt");
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorInt ReduceSumSquare(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorInt(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceSumSquareInt", "GlobalReduceSumSquareInt", "UnrolledReduceSumSquareInt");
                return O;
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
                    var Otmp = NewTempTensorInt(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumSquareInt", "GlobalReduceSumSquareInt", "UnrolledReduceSumSquareInt");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorInt(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumInt", "GlobalReduceSumInt", "UnrolledReduceSumInt");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
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

            return O;
        }

        /// <inheritdoc/>
        public override TensorInt ReduceProd(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorInt(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceProdInt", "GlobalReduceProdInt", "UnrolledReduceProdInt");
                return O;
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
                    var Otmp = NewTempTensorInt(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceProdInt", "GlobalReduceProdInt", "UnrolledReduceProdInt");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                Reduce(X, O, outerLength, reduceLength, innerLength, "ReduceProdInt", "GlobalReduceProdInt", "UnrolledReduceProdInt");
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorInt ReduceL1(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorInt(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                Reduce(X, O, 1, X.shape.length, 1, "ReduceL1Int", "GlobalReduceL1Int", "UnrolledReduceL1Int");
                return O;
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
                    var Otmp = NewTempTensorInt(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceL1Int", "GlobalReduceL1Int", "UnrolledReduceL1Int");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorInt(shapeXReduced);

                    Reduce(X, Otmp, outerLength, reduceLength, innerLength, "ReduceSumInt", "GlobalReduceSumInt", "UnrolledReduceSumInt");

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    outerLength = X.shape.Length(0, axis);
                    reduceLength = dimX;
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

            return O;
        }
    }
}
