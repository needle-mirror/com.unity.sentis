// This is auto-generated -- do not modify directly

using System;
using Unity.Sentis;
using UnityEngine.Assertions;
using static Unity.Sentis.ComputeTensorData;
using static Unity.Sentis.ShaderPropertyID;

namespace Unity.Sentis
{
    public partial class GPUCommandBufferOps
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
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastPowFloat");
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwisePowFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastAddFloat");
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseAddFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastSubFloat");
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseSubFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMulFloat");
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMulFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastDivFloat");
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseDivFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastFModFloat");
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseFModFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastPowInt");
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwisePowInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastAddInt");
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseAddInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastSubInt");
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseSubInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMulInt");
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMulInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastDivInt");
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseDivInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastModInt");
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseModInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastFModInt");
                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseFModInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, A.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, B.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            return O;
        }

        // Variadic Broadcast

        void BroadcastMin(TensorFloat O, TensorFloat X, TensorFloat Y)
        {
            if (X.shape == O.shape && Y.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastMinFloat");
                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMinFloat");
                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMinFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, Y.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMaxFloat");
                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMaxFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, Y.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
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
                cb.SetFloat(fn, k_ID_alpha, normalizationX);
                cb.SetFloat(fn, k_ID_beta, normalizationY);
                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMeanFloat");
                cb.SetFloat(fn, k_ID_alpha, normalizationX);
                cb.SetFloat(fn, k_ID_beta, normalizationY);
                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMeanFloat");
                cb.SetFloat(fn, k_ID_alpha, normalizationX);
                cb.SetFloat(fn, k_ID_beta, normalizationY);
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, Y.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastAddFloat");
                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseAddFloat");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, Y.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMinInt");
                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMinInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, Y.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
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
                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMaxInt");
                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMaxInt");
                cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                cb.SetTensorShapeStrides(fn, k_ID_shapeY, k_ID_stridesY, Y.shape);
                cb.SetInt(fn, k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                cb.ScheduleXBO(fn, Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
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
        public override TensorFloat ReduceMin(TensorFloat X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceMinFloat");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceMinFloat");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceMinFloat");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceMax(TensorFloat X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceMaxFloat");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceMaxFloat");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceMaxFloat");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceSum(TensorFloat X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceSumFloat");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceSumFloat");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceSumFloat");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceSumSquare(TensorFloat X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceSumSquareFloat");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceSumSquareFloat");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorFloat(shapeXReduced);
                    var fn = new ComputeFunc("ReduceSumFloat");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            if (isInitial)
            {
                var fn = new ComputeFunc("ReduceSumSquareFloat");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ReduceSumFloat");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceMean(TensorFloat X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceMeanFloat");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceMeanFloat");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceMeanFloat");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceProd(TensorFloat X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceProdFloat");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceProdFloat");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceProdFloat");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceL1(TensorFloat X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceL1Float");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceL1Float");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorFloat(shapeXReduced);
                    var fn = new ComputeFunc("ReduceSumFloat");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            if (isInitial)
            {
                var fn = new ComputeFunc("ReduceL1Float");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ReduceSumFloat");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceL2(TensorFloat X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceL2Float");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceSumSquareFloat");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorFloat(shapeXReduced);
                    var fn = new ComputeFunc("ReduceSumFloat");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            if (isInitial)
            {
                var fn = new ComputeFunc("ReduceL2Float");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ReduceSqrtFloat");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceLogSum(TensorFloat X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceLogSumFloat");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceSumFloat");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceLogSumFloat");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorFloat ReduceLogSumExp(TensorFloat X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorFloat(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceLogSumExpFloat");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceLogSumExpFloat");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceLogSumExpFloat");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorInt ReduceMin(TensorInt X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorInt(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceMinInt");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceMinInt");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceMinInt");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorInt ReduceMax(TensorInt X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorInt(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceMaxInt");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceMaxInt");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceMaxInt");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorInt ReduceSum(TensorInt X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorInt(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceSumInt");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceSumInt");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceSumInt");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorInt ReduceSumSquare(TensorInt X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorInt(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceSumSquareInt");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceSumSquareInt");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorInt(shapeXReduced);
                    var fn = new ComputeFunc("ReduceSumInt");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            if (isInitial)
            {
                var fn = new ComputeFunc("ReduceSumSquareInt");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ReduceSumInt");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorInt ReduceProd(TensorInt X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorInt(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceProdInt");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceProdInt");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceProdInt");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }

        /// <inheritdoc/>
        public override TensorInt ReduceL1(TensorInt X, int[] axes, bool keepdim)
        {
            TensorShape Oshape = X.shape.Reduce(axes, keepdim);
            var O = NewOutputTensorInt(Oshape);
            if (Oshape.HasZeroDims())
                return O;

            if (axes == null || axes.Length == 0)
            {
                var fn = new ComputeFunc("ReduceL1Int");
                cb.SetInt(fn, k_ID_innerLength, 1);
                cb.SetInt(fn, k_ID_reduceLength, X.shape.length);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
                return O;
            }

            // Accumulate reduce axis until non contiguity
            // X: (2 3 4 5 6), reduce 0,1,4
            // reduce 0 + 1 will result in a fused reduce on axis 2*3
            // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

            int axis = X.shape.Axis(axes[0]);
            int innerLength = X.shape.Strides(axis);
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
                    var fn = new ComputeFunc("ReduceL1Int");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorInt(shapeXReduced);
                    var fn = new ComputeFunc("ReduceSumInt");
                    cb.SetInt(fn, k_ID_innerLength, innerLength);
                    cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                    cb.ScheduleXO(fn, Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            if (isInitial)
            {
                var fn = new ComputeFunc("ReduceL1Int");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ReduceSumInt");
                cb.SetInt(fn, k_ID_innerLength, innerLength);
                cb.SetInt(fn, k_ID_reduceLength, reduceLength);
                cb.ScheduleXO(fn, Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }
    }
}
