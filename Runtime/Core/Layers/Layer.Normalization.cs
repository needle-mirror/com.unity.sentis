using System;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an element-wise `ScaleBias` normalization layer: f(x, s, b) = x * s + b.
    /// </summary>
    [Serializable]
    public class ScaleBias : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `ScaleBias` normalization layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="scale">The name to use for the scale tensor of the layer.</param>
        /// <param name="bias">The name to use for the bias tensor of the layer.</param>
        public ScaleBias(string name, string input, string scale, string bias)
        {
            this.name = name;
            inputs = new[] { input, scale, bias };
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.ScaleBias(inputShapes[0], inputShapes[1], inputShapes[2]);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.ScaleBias(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, inputTensors[2] as TensorFloat);
        }

        internal override string profilerTag => "ScaleBias";
    }

    /// <summary>
    /// Represents an `InstanceNormalization` normalization layer. This computes the mean variance on the spatial dims of the input tensor and normalizes them according to `scale` and `bias` tensors.
    /// </summary>
    [Serializable]
    public class InstanceNormalization : Layer
    {
        /// <summary>
        /// The epsilon value the layer uses to avoid division by zero.
        /// </summary>
        public float epsilon;

        /// <summary>
        /// Initializes and returns an instance of `InstanceNormalization` normalization layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="scale">The name to use for the scale tensor of the layer.</param>
        /// <param name="bias">The name to use for the bias tensor of the layer.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero. The default value is 1e-5f.</param>
        public InstanceNormalization(string name, string input, string scale, string bias, float epsilon = 1e-5f)
        {
            this.name = name;
            inputs = new[] { input, scale, bias };
            this.epsilon = epsilon;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.Normalization(inputShapes[0], inputShapes[1], inputShapes[2]);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            // @TODO: support other types of Normalization at test time.
            // Currently supported only pool=1 (InstanceNormalization)

            // NOTE: beta is used to retrieve epsilon value
            // because beta is 0 by default (while alpha is 1 by default)
            // 0 value is more inline with very small epsilon
            if (epsilon == 0)
                epsilon = Mathf.Epsilon; // safety check to prevent division by zero
            return ctx.ops.InstanceNormalization(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, inputTensors[2] as TensorFloat, epsilon);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, epsilon: {epsilon}";
        }

        internal override string profilerTag => "Normalization";
    }

    /// <summary>
    /// Represents an `AxisNormalization` normalization layer. This computes the mean variance on the last dims of the input tensor and normalizes them according to `scale` and `bias` tensors.
    /// </summary>
    [Serializable]
    public class AxisNormalization : Layer
    {
        /// <summary>
        /// The epsilon value the layer uses to avoid division by zero.
        /// </summary>
        public float epsilon;

        /// <summary>
        /// Initializes and returns an instance of `AxisNormalization` normalization layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="scale">The name to use for the scale tensor of the layer.</param>
        /// <param name="bias">The name to use for the bias tensor of the layer.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero. The default value is 1e-5f.</param>
        public AxisNormalization(string name, string input, string scale, string bias, float epsilon = 1e-5f)
        {
            this.name = name;
            inputs = new[] { input, scale, bias };

            if (epsilon == 0)
                epsilon = Mathf.Epsilon; // safety check to prevent division by zero
            this.epsilon = epsilon;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            SymbolicTensorShape shapeX = inputShapes[0];

            if (!shapeX.hasRank)
                return SymbolicTensorShape.UnknownShape;

            Logger.AssertIsTrue(shapeX.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            SymbolicTensorShape shapeScale = inputShapes[1], shapeBias = inputShapes[2];
            shapeScale.DeclareRank(1);
            shapeBias.DeclareRank(1);

            var shapeOut = new SymbolicTensorShape(shapeX);
            shapeOut[-1] = SymbolicTensorDim.MaxDefinedDim(shapeOut[-1], SymbolicTensorDim.MaxDefinedDim(shapeScale[0], shapeBias[0]));
            return shapeOut;
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.AxisNormalization(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, inputTensors[2] as TensorFloat, epsilon);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, epsilon: {epsilon}";
        }

        internal override string profilerTag => "AxisNormalization";
    }

    /// <summary>
    /// Represents an `LRN` local response normalization layer. This normalizes the input tensor over local input regions.
    /// </summary>
    [Serializable]
    public class LRN : Layer
    {
        /// <summary>
        /// The scaling parameter to use for the normalization.
        /// </summary>
        public float alpha;
        /// <summary>
        /// The exponent to use for the normalization.
        /// </summary>
        public float beta;
        /// <summary>
        /// The bias value to use for the normalization.
        /// </summary>
        public float bias;
        /// <summary>
        /// The number of channels to sum over.
        /// </summary>
        public int count;

        /// <summary>
        /// Initializes and returns an instance of `LRN` local response normalization layer layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="alpha">The scaling parameter to use for the normalization.</param>
        /// <param name="beta">The exponent to use for the normalization.</param>
        /// <param name="bias">The bias value to use for the normalization.</param>
        /// <param name="count">The number of channels to sum over.</param>
        public LRN(string name, string input, float alpha, float beta, float bias, int count)
        {
            this.name = name;
            inputs = new[] { input };
            this.alpha = alpha;
            this.beta = beta;
            this.bias = bias;
            this.count = count;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.LRN(inputShapes[0]);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.LRN(inputTensors[0] as TensorFloat, alpha, beta, bias, count);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, alpha: {alpha}, beta: {beta}, bias: {bias}, count: {count}";
        }

        internal override string profilerTag => "LRN";
    }
}
