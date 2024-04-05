using System;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an element-wise `ScaleBias` normalization layer: f(x, s, b) = x * s + b.
    /// </summary>
    [Serializable]
    class ScaleBias : Layer
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
            this.index = name;
            inputs = new[] { input, scale, bias };
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var scale = ctx.GetPartialTensor(inputs[1]);
            var bias = ctx.GetPartialTensor(inputs[2]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeScale = scale.shape;
            var shapeBias = bias.shape;
            var c = SymbolicTensorDim.Unknown;
            shapeScale.DeclareRank(1);
            c = SymbolicTensorDim.MaxDefinedDim(c, shapeScale[0]);
            shapeBias.DeclareRank(1);
            c = SymbolicTensorDim.MaxDefinedDim(c, shapeBias[0]);
            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(index, new PartialTensor(dataType));
                return;
            }

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 2 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 2, shapeX.rank);

            var shapeOut = new SymbolicTensorShape(shapeX);
            shapeOut[1] = SymbolicTensorDim.MaxDefinedDim(shapeOut[1], c);
            ctx.AddPartialTensor(index, new PartialTensor(dataType, shapeOut));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.ScaleBias(X as TensorFloat, ctx.vars.GetTensor(inputs[1]) as TensorFloat, ctx.vars.GetTensor(inputs[2]) as TensorFloat, O);
        }

        internal override string profilerTag => "ScaleBias";
    }

    /// <summary>
    /// Represents an `InstanceNormalization` normalization layer. This computes the mean variance on the spatial dims of the input tensor and normalizes them according to `scale` and `bias` tensors.
    /// </summary>
    [Serializable]
    class InstanceNormalization : Layer
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
            this.index = name;
            inputs = new[] { input, scale, bias };
            if (epsilon == 0)
                epsilon = Mathf.Epsilon; // safety check to prevent division by zero
            this.epsilon = epsilon;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var scale = ctx.GetPartialTensor(inputs[1]);
            var bias = ctx.GetPartialTensor(inputs[2]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeScale = scale.shape;
            var shapeBias = bias.shape;
            var c = SymbolicTensorDim.Unknown;
            shapeScale.DeclareRank(1);
            c = SymbolicTensorDim.MaxDefinedDim(c, shapeScale[0]);
            shapeBias.DeclareRank(1);
            c = SymbolicTensorDim.MaxDefinedDim(c, shapeBias[0]);
            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(index, new PartialTensor(dataType));
                return;
            }

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 2 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 2, shapeX.rank);
            shapeScale.DeclareRank(1);

            var shapeOut = new SymbolicTensorShape(shapeX);
            shapeOut[1] = SymbolicTensorDim.MaxDefinedDim(shapeOut[1], c);
            ctx.AddPartialTensor(index, new PartialTensor(dataType, shapeOut));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.InstanceNormalization(X as TensorFloat, ctx.vars.GetTensor(inputs[1]) as TensorFloat, ctx.vars.GetTensor(inputs[2]) as TensorFloat, O, epsilon);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, epsilon: {epsilon}";
        }

        internal override string profilerTag => "InstanceNormalization";
    }

    /// <summary>
    /// Represents an `LayerNormalization` normalization layer. This computes the mean variance on the last dimension of the input tensor and normalizes it according to `scale` and `bias` tensors.
    /// </summary>
    [Serializable]
    class LayerNormalization : Layer
    {
        /// <summary>
        /// The epsilon value the layer uses to avoid division by zero.
        /// </summary>
        public float epsilon;

        /// <summary>
        /// Initializes and returns an instance of `LayerNormalization` normalization layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="scale">The name to use for the scale tensor of the layer.</param>
        /// <param name="bias">The name to use for the bias tensor of the layer.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero. The default value is 1e-5f.</param>
        public LayerNormalization(string name, string input, string scale, string bias, float epsilon = 1e-5f)
        {
            this.index = name;
            inputs = new[] { input, scale, bias };

            if (epsilon == 0)
                epsilon = Mathf.Epsilon; // safety check to prevent division by zero
            this.epsilon = epsilon;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var scale = ctx.GetPartialTensor(inputs[1]);
            var bias = ctx.GetPartialTensor(inputs[2]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeScale = scale.shape;
            var shapeBias = bias.shape;

            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(index, new PartialTensor(dataType, SymbolicTensorShape.UnknownShape));
                return;
            }

            Logger.AssertIsTrue(shapeX.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            shapeScale.DeclareRank(1);
            shapeBias.DeclareRank(1);

            var shape = new SymbolicTensorShape(shapeX);
            shape[-1] = SymbolicTensorDim.MaxDefinedDim(shape[-1], SymbolicTensorDim.MaxDefinedDim(shapeScale[0], shapeBias[0]));
            ctx.AddPartialTensor(index, new PartialTensor(dataType, shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.LayerNormalization(X as TensorFloat, ctx.vars.GetTensor(inputs[1]) as TensorFloat, ctx.vars.GetTensor(inputs[2]) as TensorFloat, O, epsilon);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, epsilon: {epsilon}";
        }

        internal override string profilerTag => "LayerNormalization";
    }

    /// <summary>
    /// Represents an `BatchNormalization` normalization layer. This computes the mean variance on the second dimension of the input tensor and normalizes it according to `scale` and `bias` tensors.
    /// </summary>
    [Serializable]
    class BatchNormalization : Layer
    {
        /// <summary>
        /// The epsilon value the layer uses to avoid division by zero.
        /// </summary>
        public float epsilon;

        /// <summary>
        /// Initializes and returns an instance of `BatchNormalization` normalization layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="scale">The name to use for the scale tensor of the layer.</param>
        /// <param name="bias">The name to use for the bias tensor of the layer.</param>
        /// <param name="mean">The name to use for the mean tensor of the layer.</param>
        /// <param name="variance">The name to use for the variance tensor of the layer.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero. The default value is 1e-5f.</param>
        public BatchNormalization(string name, string input, string scale, string bias, string mean, string variance, float epsilon = 1e-5f)
        {
            this.index = name;
            inputs = new[] { input, scale, bias, mean, variance };
            this.epsilon = epsilon;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var scale = ctx.GetPartialTensor(inputs[1]);
            var bias = ctx.GetPartialTensor(inputs[2]);
            var mean = ctx.GetPartialTensor(inputs[3]);
            var var = ctx.GetPartialTensor(inputs[4]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeScale = scale.shape;
            var shapeBias = bias.shape;
            var shapeMean = mean.shape;
            var shapeVar = var.shape;

            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(index, new PartialTensor(dataType, SymbolicTensorShape.UnknownShape));
                return;
            }

            Logger.AssertIsTrue(shapeX.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            shapeScale.DeclareRank(1);
            shapeBias.DeclareRank(1);
            shapeMean.DeclareRank(1);
            shapeVar.DeclareRank(1);

            var shape = new SymbolicTensorShape(shapeX);
            if (shapeX.rank > 1)
                shape[1] = SymbolicTensorDim.MaxDefinedDim(shape[1], SymbolicTensorDim.MaxDefinedDim(shapeScale[0], SymbolicTensorDim.MaxDefinedDim(shapeBias[0], SymbolicTensorDim.MaxDefinedDim(shapeMean[0], shapeVar[0]))));
            ctx.AddPartialTensor(index, new PartialTensor(dataType, shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.BatchNormalization(X as TensorFloat, ctx.vars.GetTensor(inputs[1]) as TensorFloat, ctx.vars.GetTensor(inputs[2]) as TensorFloat, ctx.vars.GetTensor(inputs[3]) as TensorFloat, ctx.vars.GetTensor(inputs[4]) as TensorFloat, O, epsilon);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, epsilon: {epsilon}";
        }

        internal override string profilerTag => "BatchNormalization";
    }

    /// <summary>
    /// Represents an `LRN` local response normalization layer. This normalizes the input tensor over local input regions.
    /// </summary>
    [Serializable]
    class LRN : Layer
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
            this.index = name;
            inputs = new[] { input };
            this.alpha = alpha;
            this.beta = beta;
            this.bias = bias;
            this.count = count;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            Logger.AssertIsTrue(X.shape.hasRank ? X.shape.rank >= 2 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 2, X.shape.rank);

            ctx.AddPartialTensor(index, new PartialTensor(X.dataType, X.shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            Logger.AssertIsFalse(ctx.backend is GPUCommandBufferBackend, "BackendTypeError: GPUCommandBuffer is not supported on the LRN layer");
            var X = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;

            // https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
            // However divide the sum by size to follow onnx and pytorch implementation
            // ONNX https://github.com/onnx/onnx/blob/master/docs/Operators.md#LRN
            // PYTORCH https://github.com/pytorch/pytorch/blob/1465970a343e61f2f2b104859ca7f5d7e03f5d02/torch/nn/functional.py#L2069
            // Tensorflow don't and follow the paper to the letter https://github.com/tensorflow/tensorflow/blob/e6faa845c51bb69465146d93646947fd2ba53efa/tensorflow/python/kernel_tests/lrn_op_test.py#L53
            // However they bake the division to alpha when exporting to ONNX https://github.com/onnx/tensorflow-onnx/blob/7c37ccb97e0fd478ce093910c4a1411b18e44fd7/tf2onnx/onnx_opset/math.py

            BurstTensorData.Pin(X);
            BurstTensorData.Pin(O);

            float sizef = count;

            var itRemap = new TensorNDIterator(O.shape);
            for (var it = new TensorNDIterator(O.shape); it.HasNext(); it.MoveNext())
            {
                int c = it[1];
                float regionCenter = (sizef - 1.0f) / 2.0f;
                int regionStart = Math.Max(0, c - (int)Mathf.Floor(regionCenter));
                int regionEnd = Math.Min(X.shape[1], c + (int)Mathf.Ceil(regionCenter) + 1);
                float sumOfSquared = 0.0f;
                for (int ci = regionStart; ci < regionEnd; ++ci)
                {
                    itRemap.CopyNDIndex(it);
                    itRemap[1] = ci;
                    float regionValue = X[itRemap.index];
                    sumOfSquared += regionValue * regionValue;
                }

                O[it.index] = X[it.index] / Mathf.Pow(bias + alpha * sumOfSquared / sizef, beta);
            }
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, alpha: {alpha}, beta: {beta}, bias: {bias}, count: {count}";
        }

        internal override string profilerTag => "LRN";
    }
}
