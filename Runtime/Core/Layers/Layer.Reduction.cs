using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents the abstract base class for reduction layers.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    public abstract class Reduce : Layer
    {
        /// <summary>
        /// Whether to keep the axis dimension in the output tensor.
        /// </summary>
        public bool keepdims;
        /// <summary>
        /// Whether to perform an identity operation if the input axes tensor is empty.
        ///
        /// If this is `false` and the input axes tensor is empty then the reduction is applied on all axes of the input tensor.
        /// </summary>
        public bool noopWithEmptyAxes;

        /// <summary>
        /// Initializes and returns an instance of `Reduce` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        protected Reduce(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
        {
            this.name = name;
            this.inputs = inputs;
            this.keepdims = keepdims;
            this.noopWithEmptyAxes = noopWithEmptyAxes;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            if (inputShapes.Length < 2)
                return noopWithEmptyAxes ? inputShapes[0] : SymbolicInference.Reduce(inputShapes[0], keepdims, noopWithEmptyAxes);
            return SymbolicInference.Reduce(inputShapes[0], inputShapes[1], ctx.GetPartialTensor(inputs[1]), keepdims, noopWithEmptyAxes);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, keepdims: {keepdims}, noopWithEmptyAxes: {noopWithEmptyAxes}";
        }
    }

    /// <summary>
    /// Represents a `ReduceL1` reduction layer along the given axes: f(x1, x2 ... xn) = |x1| + |x2| + ... + |xn|.
    /// </summary>
    [Serializable]
    public class ReduceL1 : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceL1` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceL1(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputs.Length > 1 ? (inputTensors[1] as TensorInt).ToReadOnlyArray() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            if (inputTensors[0] is TensorInt)
                return ctx.ops.ReduceL1(inputTensors[0] as TensorInt, axes, keepdims);
            else
                return ctx.ops.ReduceL1(inputTensors[0] as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceL1";
    }

    /// <summary>
    /// Represents a `ReduceL2` reduction layer along the given axes: f(x1, x2 ... xn) = sqrt(x1² + x2² + ... + xn²).
    /// </summary>
    [Serializable]
    public class ReduceL2 : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceL2` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceL2(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputs.Length > 1 ? (inputTensors[1] as TensorInt).ToReadOnlyArray() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            return ctx.ops.ReduceL2(inputTensors[0] as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceL2";
    }

    /// <summary>
    /// Represents a `ReduceLogSum` reduction layer along the given axes: f(x1, x2 ... xn) = log(x1 + x2 + ... + xn).
    /// </summary>
    [Serializable]
    public class ReduceLogSum : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceLogSum` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceLogSum(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputs.Length > 1 ? (inputTensors[1] as TensorInt).ToReadOnlyArray() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            return ctx.ops.ReduceLogSum(inputTensors[0] as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceLogSum";
    }

    /// <summary>
    /// Represents a `ReduceLogSumExp` reduction layer along the given axes: f(x1, x2 ... xn) = log(e^x1 + e^x2 + ... + e^xn).
    /// </summary>
    [Serializable]
    public class ReduceLogSumExp : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceLogSumExp` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceLogSumExp(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputs.Length > 1 ? (inputTensors[1] as TensorInt).ToReadOnlyArray() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            return ctx.ops.ReduceLogSumExp(inputTensors[0] as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceLogSumExp";
    }

    /// <summary>
    /// Represents a `ReduceMax` reduction layer along the given axes: f(x1, x2 ... xn) = max(x1, x2, ... , xn).
    /// </summary>
    [Serializable]
    public class ReduceMax : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceMax` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceMax(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputs.Length > 1 ? (inputTensors[1] as TensorInt).ToReadOnlyArray() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            if (inputTensors[0] is TensorInt)
                return ctx.ops.ReduceMax(inputTensors[0] as TensorInt, axes, keepdims);
            else
                return ctx.ops.ReduceMax(inputTensors[0] as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceMax";
    }

    /// <summary>
    /// Represents a `ReduceMean` reduction layer along the given axes: f(x1, x2 ... xn) = (x1 + x2 + ... + xn) / n.
    /// </summary>
    [Serializable]
    public class ReduceMean : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceMean` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceMean(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputs.Length > 1 ? (inputTensors[1] as TensorInt).ToReadOnlyArray() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            return ctx.ops.ReduceMean(inputTensors[0] as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceMean";
    }

    /// <summary>
    /// Represents a `ReduceMin` reduction layer along the given axes: f(x1, x2 ... xn) = min(x1, x2, ... , xn).
    /// </summary>
    [Serializable]
    public class ReduceMin : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceMin` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceMin(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputs.Length > 1 ? (inputTensors[1] as TensorInt).ToReadOnlyArray() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            if (inputTensors[0] is TensorInt)
                return ctx.ops.ReduceMin(inputTensors[0] as TensorInt, axes, keepdims);
            else
                return ctx.ops.ReduceMin(inputTensors[0] as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceMin";
    }

    /// <summary>
    /// Represents a `ReduceProd` reduction layer along the given axes: f(x1, x2 ... xn) = x1 * x2 * ... * xn.
    /// </summary>
    [Serializable]
    public class ReduceProd : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceProd` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceProd(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputs.Length > 1 ? (inputTensors[1] as TensorInt).ToReadOnlyArray() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            if (inputTensors[0] is TensorInt)
                return ctx.ops.ReduceProd(inputTensors[0] as TensorInt, axes, keepdims);
            else
                return ctx.ops.ReduceProd(inputTensors[0] as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceProd";
    }

    /// <summary>
    /// Represents a `ReduceSum` reduction layer along the given axes: f(x1, x2 ... xn) = x1 + x2 + ... + xn.
    /// </summary>
    [Serializable]
    public class ReduceSum : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceSum` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceSum(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputs.Length > 1 ? (inputTensors[1] as TensorInt).ToReadOnlyArray() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            if (inputTensors[0] is TensorInt)
                return ctx.ops.ReduceSum(inputTensors[0] as TensorInt, axes, keepdims);
            else
                return ctx.ops.ReduceSum(inputTensors[0] as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceSum";
    }

    /// <summary>
    /// Represents a `ReduceSumSquare` reduction layer along the given axes: f(x1, x2 ... xn) = x1² + x2² + ... + xn².
    /// </summary>
    [Serializable]
    public class ReduceSumSquare : Reduce
    {
        /// <summary>
        /// Initializes and returns an instance of `ReduceSumSquare` reduction layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The name of the input tensor and optionally the axes tensor. If no axes tensor is provided, the layer performs the reduction according to the value of `noopWithEmptyAxes`.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="noopWithEmptyAxes">Whether to perform an identity operation if the input axes tensor is empty. The default value is `false`.</param>
        public ReduceSumSquare(string name, string[] inputs, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(name, inputs, keepdims, noopWithEmptyAxes) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputs.Length > 1 ? (inputTensors[1] as TensorInt).ToReadOnlyArray() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
                return inputTensors[0].ShallowCopy();
            if (inputTensors[0] is TensorInt)
                return ctx.ops.ReduceSumSquare(inputTensors[0] as TensorInt, axes, keepdims);
            else
                return ctx.ops.ReduceSumSquare(inputTensors[0] as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceSumSquare";
    }
}
