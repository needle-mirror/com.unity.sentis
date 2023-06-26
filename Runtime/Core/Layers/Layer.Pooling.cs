using System;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an `AveragePool` pooling layer. This calculates an output tensor by pooling the mean values of the input tensor across its spatial dimensions according to the given pool and stride values.
    /// </summary>
    [Serializable]
    public class AveragePool : Layer
    {
        /// <summary>
        /// The size of the kernel along each spatial axis.
        /// </summary>
        public int[] pool;
        /// <summary>
        /// The stride along each spatial axis.
        ///
        /// If this is `null` the layer uses a default of [1, 1, ..., 1].
        /// </summary>
        public int[] stride;
        /// <summary>
        /// The lower and upper padding values for each spatial dimension. For example [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.
        ///
        /// If this is `null` the layer uses a default of [0, 0, ..., 0].
        /// </summary>
        public int[] pad;
        /// <summary>
        /// The auto padding mode of the pool as an `AutoPad`.
        /// </summary>
        public AutoPad autopad;

        /// <summary>
        /// Initializes and returns an instance of `AveragePool` pooling layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="pool">The size of the kernel along each spatial axis.</param>
        /// <param name="stride">The stride along each spatial axis.</param>
        /// <param name="pad">The lower and upper padding values for each spatial dimension, [pad_left, pad_right] for 1D, [pad_top, pad_bottom, pad_left, pad_right] for 2D, etc.</param>
        /// <param name="autopad">The auto padding mode of the pool as an `AutoPad`. The default value is `AutoPad.NotSet`.</param>
        public AveragePool(string name, string input, int[] pool, int[] stride, int[] pad, AutoPad autopad = AutoPad.NotSet)
        {
            this.name = name;
            inputs = new[] { input };
            this.pool = pool;
            this.stride = stride;
            this.pad = pad;
            this.autopad = autopad;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.Pool(inputShapes[0], pool, stride, pad, autopad, false);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            ShapeInference.UpdatePadForPoolAutoPadding(inputTensors[0].shape, pool, stride, false, autopad, pad);
            return ctx.ops.AveragePool(inputTensors[0] as TensorFloat, pool, stride, pad);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, pool: [{string.Join(", ", pool)}], stride: [{string.Join(", ", stride)}], pad: [{string.Join(", ", pad)}], autopad: {autopad}";
        }

        internal override string profilerTag => "AveragePool";
    }

    /// <summary>
    /// Represents a `GlobalAveragePool` pooling layer. This calculates an output tensor by pooling the mean values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
    /// </summary>
    [Serializable]
    public class GlobalAveragePool : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `GlobalAveragePool` pooling layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public GlobalAveragePool(string name, string input)
        {
            this.name = name;
            inputs = new[] { input };
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.GlobalPool(inputShapes[0]);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.GlobalAveragePool(inputTensors[0] as TensorFloat);
        }

        internal override string profilerTag => "GlobalAveragePool";
    }

    /// <summary>
    /// Represents a `GlobalMaxPool` pooling layer. This calculates an output tensor by pooling the maximum values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
    /// </summary>
    [Serializable]
    public class GlobalMaxPool : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `GlobalMaxPool` pooling layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public GlobalMaxPool(string name, string input)
        {
            this.name = name;
            inputs = new[] { input };
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.GlobalPool(inputShapes[0]);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.GlobalMaxPool(inputTensors[0] as TensorFloat);
        }

        internal override string profilerTag => "GlobalMaxPool";
    }

    /// <summary>
    /// Represents a `MaxPool` pooling layer. This calculates an output tensor by pooling the maximum values of the input tensor across its spatial dimensions according to the given pool and stride values.
    /// </summary>
    [Serializable]
    public class MaxPool : Layer
    {
        /// <summary>
        /// The size of the kernel along each spatial axis.
        /// </summary>
        public int[] pool;
        /// <summary>
        /// The stride along each spatial axis.
        ///
        /// If this is `null` the layer uses a default of [1, 1, ..., 1].
        /// </summary>
        public int[] stride;
        /// <summary>
        /// The lower and upper padding values for each spatial dimension, [pad_left, pad_right] for 1D, [pad_top, pad_bottom, pad_left, pad_right] for 2D, etc.
        ///
        /// If this is `null` the layer uses a default of [0, 0, ..., 0].
        /// </summary>
        public int[] pad;
        /// <summary>
        /// The auto padding mode of the pool as an `AutoPad`.
        /// </summary>
        public AutoPad autopad;

        /// <summary>
        /// Initializes and returns an instance of `MaxPool` pooling layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="pool">The size of the kernel along each spatial axis.</param>
        /// <param name="stride">The stride along each spatial axis.</param>
        /// <param name="pad">The lower and upper padding values for each spatial dimension. For example [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.</param>
        /// <param name="autopad">The auto padding mode of the pool as an `AutoPad`. The default value is `AutoPad.NotSet`.</param>
        public MaxPool(string name, string input, int[] pool, int[] stride, int[] pad, AutoPad autopad = AutoPad.NotSet)
        {
            this.name = name;
            inputs = new[] { input };
            this.pool = pool;
            this.stride = stride;
            this.pad = pad;
            this.autopad = autopad;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.Pool(inputShapes[0], pool, stride, pad, autopad, false);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            ShapeInference.UpdatePadForPoolAutoPadding(inputTensors[0].shape, pool, stride, false, autopad, pad);
            return ctx.ops.MaxPool(inputTensors[0] as TensorFloat, pool, stride, pad);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, pool: [{string.Join(", ", pool)}], stride: [{string.Join(", ", stride)}], pad: [{string.Join(", ", pad)}], autopad: {autopad}";
        }

        internal override string profilerTag => "MaxPool";
    }
}
