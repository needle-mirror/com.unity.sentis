// This is auto-generated -- do not modify directly

using System;
using Unity.Sentis;
using UnityEngine.Assertions;
using static Unity.Sentis.ComputeTensorData;
using static Unity.Sentis.ShaderPropertyID;

namespace Unity.Sentis
{
    public partial class GPUComputeOps
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
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastPowFloat");
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwisePowFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastAddFloat");
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseAddFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastSubFloat");
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseSubFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMulFloat");
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMulFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastDivFloat");
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseDivFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastFModFloat");
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseFModFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastPowInt");
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwisePowInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastAddInt");
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseAddInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastSubInt");
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseSubInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMulInt");
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMulInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastDivInt");
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseDivInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastModInt");
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseModInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (A.shape == O.shape && B.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastFModInt");
                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseFModInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, A.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, B.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(A), Pin(B), Pin(O, uploadCache: false), O.shape.length);
            }
            return O;
        }

        // Variadic Broadcast

        void BroadcastMin(TensorFloat O, TensorFloat X, TensorFloat Y)
        {
            if (X.shape == O.shape && Y.shape.length == 1)
            {
                var fn = new ComputeFunc("ScalarBroadcastMinFloat");
                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMinFloat");
                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMinFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, Y.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMaxFloat");
                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMaxFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, Y.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMeanFloat");
                fn.SetFloat(k_ID_alpha, normalizationX);
                fn.SetFloat(k_ID_beta, normalizationY);
                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
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

                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastAddFloat");
                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseAddFloat");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, Y.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMinInt");
                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMinInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, Y.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
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
                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else if (X.shape == O.shape && Y.shape == O.shape)
            {
                var fn = new ComputeFunc("BroadcastMaxInt");
                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ElementwiseMaxInt");
                fn.SetTensorShapeStrides(k_ID_shapeO, k_ID_stridesO, O.shape);
                fn.SetTensorShapeStrides(k_ID_shapeX, k_ID_stridesX, X.shape);
                fn.SetTensorShapeStrides(k_ID_shapeY, k_ID_stridesY, Y.shape);
                fn.SetInt(k_ID_rank, (TensorShape.maxRank - 1) - O.shape.rank);

                fn.ScheduleXBO(Pin(X), Pin(Y), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceMinFloat");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceMaxFloat");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceSumFloat");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorFloat(shapeXReduced);
                    var fn = new ComputeFunc("ReduceSumFloat");
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

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
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ReduceSumFloat");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceMeanFloat");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceProdFloat");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorFloat(shapeXReduced);
                    var fn = new ComputeFunc("ReduceSumFloat");
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

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
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ReduceSumFloat");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorFloat(shapeXReduced);
                    var fn = new ComputeFunc("ReduceSumFloat");
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

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
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ReduceSqrtFloat");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceLogSumFloat");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceLogSumExpFloat");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceMinInt");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceMaxInt");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceSumInt");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorInt(shapeXReduced);
                    var fn = new ComputeFunc("ReduceSumInt");
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

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
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ReduceSumInt");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                }

                shapeXReduced[axis] = 1;
                prevAxis = axis;
            }

            {
                var fn = new ComputeFunc("ReduceProdInt");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                fn.SetInt(k_ID_innerLength, 1);
                fn.SetInt(k_ID_reduceLength, X.shape.length);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
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
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

                    X = Otmp;
                    innerLength = X.shape.Strides(axis);
                    reduceLength = dimX;
                    isInitial = false;
                }
                else
                {
                    var Otmp = NewTempTensorInt(shapeXReduced);
                    var fn = new ComputeFunc("ReduceSumInt");
                    fn.SetInt(k_ID_innerLength, innerLength);
                    fn.SetInt(k_ID_reduceLength, reduceLength);
                    fn.ScheduleXO(Pin(X), Pin(Otmp, uploadCache: false), Otmp.shape.length);

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
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }
            else
            {
                var fn = new ComputeFunc("ReduceSumInt");
                fn.SetInt(k_ID_innerLength, innerLength);
                fn.SetInt(k_ID_reduceLength, reduceLength);
                fn.ScheduleXO(Pin(X), Pin(O, uploadCache: false), O.shape.length);
            }

            return O;
        }
    }
}
