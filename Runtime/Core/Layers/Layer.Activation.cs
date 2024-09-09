using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an element-wise activation layer.
    /// </summary>
    abstract class Activation : Layer
    {
        protected Activation(int output, int input)
            : base(new[] { output }, new[] { input }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, X.shape));
        }
    }

    /// <summary>
    /// Represents an element-wise `Celu` activation layer: f(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1)).
    /// </summary>
    class Celu : Activation
    {
        public float alpha;

        public Celu(int output, int input, float alpha)
            : base(output, input)
        {
            this.alpha = alpha;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Celu(X, O, alpha);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, alpha: {alpha}";
        }

        public override string opName => "Celu";
    }

    /// <summary>
    /// Represents an element-wise `Elu` activation layer: f(x) = x if x >= 0, otherwise f(x) = alpha * (e^x - 1).
    /// </summary>
    class Elu : Activation
    {
        public float alpha;

        public Elu(int output, int input, float alpha = 1.0f)
            : base(output, input)
        {
            this.alpha = alpha;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Elu(X as Tensor<float>, O, alpha);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, alpha: {alpha}";
        }

        public override string opName => "Elu";
    }

    /// <summary>
    /// Represents an element-wise `Gelu` activation layer: f(x) = x / 2 * (1 + erf(x / sqrt(2))).
    /// </summary>
    class Gelu : Activation
    {
        public Gelu(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Gelu(X, O);
        }

        public override string opName => "Gelu";
    }

    class GeluFast : Activation
    {
        public GeluFast(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.GeluFast(X, O);
        }

        public override string opName => "GeluFast";
    }

    /// <summary>
    /// Represents an element-wise `Erf` activation layer: f(x) = erf(x).
    /// </summary>
    class Erf : Activation
    {
        public Erf(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Erf(X as Tensor<float>, O);
        }

        public override string opName => "Erf";
    }

    /// <summary>
    /// Represents a `Hardmax` activation layer along an axis: f(x, axis) = 1 if x is the first maximum value along the specified axis, otherwise f(x) = 0.
    /// </summary>
    class Hardmax : Activation
    {
        public int axis;

        public Hardmax(int output, int input, int axis = -1)
            : base(output, input)
        {
            this.axis = axis;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Hardmax(X as Tensor<float>, O, axis);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        public override string opName => "Hardmax";
    }

    /// <summary>
    /// Represents an element-wise `HardSigmoid` activation layer: f(x) = clamp(alpha * x + beta, 0, 1).
    /// </summary>
    class HardSigmoid : Activation
    {
        public float alpha;
        public float beta;

        public HardSigmoid(int output, int input, float alpha = 0.2f, float beta = 0.5f)
            : base(output, input)
        {
            this.alpha = alpha;
            this.beta = beta;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.HardSigmoid(X, O, alpha, beta);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, alpha: {alpha}, beta: {beta}";
        }

        public override string opName => "HardSigmoid";
    }

    /// <summary>
    /// Represents an element-wise `HardSwish` activation layer: f(x) = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid(x, alpha, beta), where alpha = 1/6 and beta = 0.5.
    /// </summary>
    class HardSwish : Activation
    {
        public HardSwish(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.HardSwish(X, O);
        }

        public override string opName => "HardSwish";
    }

    /// <summary>
    /// Represents an element-wise `LeakyRelu` activation layer: f(x) = x if x >= 0, otherwise f(x) = alpha * x.
    /// </summary>
    class LeakyRelu : Activation
    {
        public float alpha;

        public LeakyRelu(int output, int input, float alpha = 0.01f)
            : base(output, input)
        {
            this.alpha = alpha;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.LeakyRelu(X, O, alpha);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, alpha: {alpha}";
        }

        public override string opName => "LeakyRelu";
    }

    /// <summary>
    /// Represents an element-wise `PRelu` activation layer: f(x) = x if x >= 0, otherwise f(x) = slope * x.
    ///
    /// The slope tensor must be unidirectional broadcastable to x.
    /// </summary>
    class PRelu : Layer
    {
        public PRelu(int output, int input, int slope)
            : base(new[] { output }, new[] { input, slope }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var slope = ctx.GetPartialTensor(inputs[1]);
            var shapeX = X.shape;
            var shapeSlope = slope.shape;
            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float));
                return;
            }

            if (!shapeSlope.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, shapeX));
                return;
            }

            Logger.AssertIsTrue(shapeSlope.rank <= shapeX.rank, "PRelu.InputError: slope shape must be unidirectional broadcastable to input");
            var numInitialDims = shapeX.rank - shapeSlope.rank;
            var shapeOut = new DynamicTensorShape(shapeX);

            for (var i = 0; i < shapeSlope.rank; i++)
            {
                if (shapeSlope[i] == 1)
                    continue;
                shapeOut[numInitialDims + i] = DynamicTensorDim.MaxDefinedDim(shapeOut[numInitialDims + i], shapeSlope[i]);
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var slope = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.PRelu(X as Tensor<float>, slope as Tensor<float>, O);
        }

        public override string opName => "PRelu";
    }

    /// <summary>
    /// Represents an element-wise `Relu` activation layer: f(x) = max(0, x).
    /// </summary>
    class Relu : Activation
    {
        public Relu(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Relu(X as Tensor<float>, O);
        }

        public override string opName => "Relu";
    }

    /// <summary>
    /// Represents an element-wise `Relu6` activation layer: f(x) = clamp(x, 0, 6).
    /// </summary>
    class Relu6 : Activation
    {
        public Relu6(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Relu6(X, O);
        }

        public override string opName => "Relu6";
    }

    /// <summary>
    /// Represents an element-wise `Selu` activation layer: f(x) = gamma * x if x >= 0, otherwise f(x) = (alpha * e^x - alpha).
    /// </summary>
    class Selu : Activation
    {
        public float alpha;
        public float gamma;

        public Selu(int output, int input, float alpha = 1.67326f, float gamma = 1.0507f)
            : base(output, input)
        {
            this.alpha = alpha;
            this.gamma = gamma;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Selu(X, O, alpha, gamma);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, alpha: {alpha}, gamma: {gamma}";
        }

        public override string opName => "Selu";
    }

    /// <summary>
    /// Represents an element-wise `Sigmoid` activation layer: f(x) = 1/(1 + e^(-x)).
    /// </summary>
    class Sigmoid : Activation
    {
        public Sigmoid(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Sigmoid(X as Tensor<float>, O);
        }

        public override string opName => "Sigmoid";
    }

    /// <summary>
    /// Represents an element-wise `Softplus` activation layer: f(x) = ln(e^x + 1).
    /// </summary>
    class Softplus : Activation
    {
        public Softplus(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Softplus(X as Tensor<float>, O);
        }

        public override string opName => "Softplus";
    }

    /// <summary>
    /// Represents an element-wise `Softsign` activation layer: f(x) = x/(|x| + 1).
    /// </summary>
    class Softsign : Activation
    {
        public Softsign(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Softsign(X as Tensor<float>, O);
        }

        public override string opName => "Softsign";
    }

    /// <summary>
    /// Represents an element-wise `Swish` activation layer. f(x) = sigmoid(x) * x = x / (1 + e^{-x}).
    /// </summary>
    class Swish : Activation
    {
        public Swish(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Swish(X as Tensor<float>, O);
        }

        public override string opName => "Swish";
    }

    /// <summary>
    /// Represents an element-wise `Tanh` activation layer: f(x) = tanh(x).
    /// </summary>
    class Tanh : Activation
    {
        public Tanh(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Tanh(X as Tensor<float>, O);
        }

        public override string opName => "Tanh";
    }

    /// <summary>
    /// Represents an element-wise `ThresholdedRelu` activation layer: f(x) = x if x > alpha, otherwise f(x) = 0.
    /// </summary>
    class ThresholdedRelu : Activation
    {
        public float alpha;

        public ThresholdedRelu(int output, int input, float alpha)
            : base(output, input)
        {
            this.alpha = alpha;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.ThresholdedRelu(X as Tensor<float>, O, alpha);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, alpha: {alpha}";
        }

        public override string opName => "ThresholdedRelu";
    }
}
