using System;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a local pooling layer.
    /// </summary>
    abstract class LocalPool : Layer
    {
        public int[] kernelShape;
        public int[] strides;
        public int[] pads;
        public AutoPad autopad;

        protected LocalPool(int output, int input, int[] kernelShape, int[] strides, int[] pads, AutoPad autopad = AutoPad.NotSet)
            : base(new[] { output }, new[] { input })
        {
            this.kernelShape = kernelShape;
            this.strides = strides;
            if (this.strides == null)
            {
                this.strides = new int[this.kernelShape.Length];
                for (var i = 0; i < this.strides.Length; i++)
                {
                    this.strides[i] = 1;
                }
            }
            this.pads = pads;
            this.pads ??= new int[2 * this.kernelShape.Length];
            this.autopad = autopad;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            shapeX.DeclareRank(2 + kernelShape.Length);

            Logger.AssertIsTrue(strides == null || shapeX.rank - 2 == strides.Length, "Pool.InputError: strides must have same number of values as spatial dimensions or be null");
            Logger.AssertIsTrue(pads == null || (shapeX.rank - 2) * 2 == pads.Length, "Pool.InputError: padding must have twice the number of values as spatial dimensions or be null");

            var shapeOut = new DynamicTensorShape(shapeX);

            for (var i = 2; i < shapeOut.rank; i++)
            {
                var s = strides == null ? 1 : strides[i - 2];
                var p = (pads == null || autopad != AutoPad.NotSet) ? 0 : (pads[i - 2] + pads[i - 2 + (shapeX.rank - 2)]);
                shapeOut[i] = shapeX[i].Pool(kernelShape[i - 2], s, p, 1, false, autopad);
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
        }

        public override string ToString()
        {
            return $"{base.ToString()}, kernelShape: [{string.Join(", ", kernelShape)}], strides: [{string.Join(", ", strides)}], pads: [{string.Join(", ", pads)}], autopad: {autopad}";
        }
    }

    /// <summary>
    /// Represents a global pooling layer.
    /// </summary>
    abstract class GlobalPool : Layer
    {
        protected GlobalPool(int output, int input)
            : base(new[] { output }, new[] { input }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType));
                return;
            }

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 3 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 3, shapeX.rank);

            var shapeOut = new DynamicTensorShape(shapeX);

            for (var i = 2; i < shapeOut.rank; i++)
            {
                shapeOut[i] = DynamicTensorDim.One;
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
        }
    }

    /// <summary>
    /// Represents an `AveragePool` pooling layer. This calculates an output tensor by pooling the mean values of the input tensor across its spatial dimensions according to the given pool and stride values.
    /// </summary>
    class AveragePool : LocalPool
    {
        public AveragePool(int output, int input, int[] kernelShape, int[] strides, int[] pads, AutoPad autopad = AutoPad.NotSet)
            : base(output, input, kernelShape, strides, pads, autopad) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            ShapeInference.UpdatePadForPoolAutoPadding(X.shape, kernelShape, strides, pads, false, autopad);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.ApplyPool(X.shape, kernelShape, strides, pads), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.AveragePool(X, O, kernelShape, strides, pads);
        }

        internal override string profilerTag => "AveragePool";
    }

    /// <summary>
    /// Represents a `GlobalAveragePool` pooling layer. This calculates an output tensor by pooling the mean values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
    /// </summary>
    class GlobalAveragePool : GlobalPool
    {
        public GlobalAveragePool(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.GlobalPool(X.shape), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.GlobalAveragePool(X, O);
        }

        internal override string profilerTag => "GlobalAveragePool";
    }

    /// <summary>
    /// Represents a `GlobalMaxPool` pooling layer. This calculates an output tensor by pooling the maximum values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
    /// </summary>
    class GlobalMaxPool : GlobalPool
    {
        public GlobalMaxPool(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.GlobalPool(X.shape), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.GlobalMaxPool(X, O);
        }

        internal override string profilerTag => "GlobalMaxPool";
    }

    /// <summary>
    /// Represents a `MaxPool` pooling layer. This calculates an output tensor by pooling the maximum values of the input tensor across its spatial dimensions according to the given pool and stride values.
    /// </summary>
    class MaxPool : LocalPool
    {
        public MaxPool(int output, int input, int[] kernelShape, int[] strides, int[] pads, AutoPad autopad = AutoPad.NotSet)
            : base(output, input, kernelShape, strides, pads, autopad) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            ShapeInference.UpdatePadForPoolAutoPadding(X.shape, kernelShape, strides, pads, false, autopad);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.ApplyPool(X.shape, kernelShape, strides, pads), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.MaxPool(X, O, kernelShape, strides, pads);
        }

        internal override string profilerTag => "MaxPool";
    }
}
