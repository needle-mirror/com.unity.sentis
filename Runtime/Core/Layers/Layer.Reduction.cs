using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents the abstract base class for reduction layers.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    abstract class Reduce : Layer
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
            this.index = name;
            this.inputs = inputs;
            this.keepdims = keepdims;
            this.noopWithEmptyAxes = noopWithEmptyAxes;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var axes = inputs.Length == 2 ? ctx.GetPartialTensor(inputs[1]) : null;
            var shapeAxes = axes?.shape ?? new SymbolicTensorShape(SymbolicTensorDim.Zero);
            if (axes != null && axes.isPartiallyKnown && axes.length != 0)
            {
                var reducedShape = new SymbolicTensorShape(shapeX);
                if (!axes.IsFullyKnown() && reducedShape.hasRank)
                {
                    // replace any non 0 or 1 dims with unknown (0 and 1 stay the same whether reduced or not)
                    for (var i = 0; i < reducedShape.rank; i++)
                    {
                        if (reducedShape[i] == 0 || reducedShape[i] == 1)
                            continue;
                        reducedShape[i] = SymbolicTensorDim.Unknown;
                    }
                }

                for (var i = 0; i < axes.length; i++)
                {
                    if (!axes[i].isIntValue)
                        continue;
                    var axis = axes[i].intValue;
                    // reducing on a zero axis will result in a zero rather than a one
                    if (shapeX[axis].isValue)
                        reducedShape[axis] = shapeX[axis].value == 0 ? SymbolicTensorDim.Zero : SymbolicTensorDim.One;
                    else
                        reducedShape[axis] = SymbolicTensorDim.Unknown;
                }

                var tensorOut = new PartialTensor(dataType, reducedShape);
                if (!keepdims)
                {
                    tensorOut = tensorOut.Reshape(!axes.IsFullyKnown() ? SymbolicTensorShape.UnknownOfRank(tensorOut.shape.rank - axes.length) : tensorOut.shape.Squeeze(axes));
                }

                ctx.AddPartialTensor(index, tensorOut);
                return;
            }

            if (shapeAxes.IsFullyKnown())
            {
                if (shapeAxes[0].value != 0)
                    ctx.AddPartialTensor(index, new PartialTensor(dataType, keepdims ? SymbolicTensorShape.UnknownOfRankLike(shapeX) : SymbolicTensorShape.UnknownShape));
                else if (noopWithEmptyAxes)
                    ctx.AddPartialTensor(index, new PartialTensor(dataType, shapeX));
                else
                    ctx.AddPartialTensor(index, new PartialTensor(dataType, keepdims ? SymbolicTensorShape.OnesLike(shapeX) : new SymbolicTensorShape()));
                return;
            }

            ctx.AddPartialTensor(index, new PartialTensor(dataType, keepdims && !noopWithEmptyAxes ? SymbolicTensorShape.UnknownOfRankLike(shapeX) : SymbolicTensorShape.UnknownShape));
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
    class ReduceL1 : Reduce
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
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var axes = inputs.Length > 1 && inputs[1] != null ? ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is TensorInt)
                ctx.backend.ReduceL1(X as TensorInt, O as TensorInt, axes, keepdims);
            else
                ctx.backend.ReduceL1(X as TensorFloat, O as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceL1";
    }

    /// <summary>
    /// Represents a `ReduceL2` reduction layer along the given axes: f(x1, x2 ... xn) = sqrt(x1² + x2² + ... + xn²).
    /// </summary>
    [Serializable]
    class ReduceL2 : Reduce
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
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var axes = inputs.Length > 1 && inputs[1] != null ? ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Reduce(axes, keepdims), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.ReduceL2(X, O as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceL2";
    }

    /// <summary>
    /// Represents a `ReduceLogSum` reduction layer along the given axes: f(x1, x2 ... xn) = log(x1 + x2 + ... + xn).
    /// </summary>
    [Serializable]
    class ReduceLogSum : Reduce
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
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var axes = inputs.Length > 1 && inputs[1] != null ? ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Reduce(axes, keepdims), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.ReduceLogSum(X, O as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceLogSum";
    }

    /// <summary>
    /// Represents a `ReduceLogSumExp` reduction layer along the given axes: f(x1, x2 ... xn) = log(e^x1 + e^x2 + ... + e^xn).
    /// </summary>
    [Serializable]
    class ReduceLogSumExp : Reduce
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
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var axes = inputs.Length > 1 && inputs[1] != null ? ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Reduce(axes, keepdims), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.ReduceLogSumExp(X, O as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceLogSumExp";
    }

    /// <summary>
    /// Represents a `ReduceMax` reduction layer along the given axes: f(x1, x2 ... xn) = max(x1, x2, ... , xn).
    /// </summary>
    [Serializable]
    class ReduceMax : Reduce
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
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var axes = inputs.Length > 1 && inputs[1] != null ? ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is TensorInt)
                ctx.backend.ReduceMax(X as TensorInt, O as TensorInt, axes, keepdims);
            else
                ctx.backend.ReduceMax(X as TensorFloat, O as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceMax";
    }

    /// <summary>
    /// Represents a `ReduceMean` reduction layer along the given axes: f(x1, x2 ... xn) = (x1 + x2 + ... + xn) / n.
    /// </summary>
    [Serializable]
    class ReduceMean : Reduce
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
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var axes = inputs.Length > 1 && inputs[1] != null ? ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Reduce(axes, keepdims), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.ReduceMean(X, O as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceMean";
    }

    /// <summary>
    /// Represents a `ReduceMin` reduction layer along the given axes: f(x1, x2 ... xn) = min(x1, x2, ... , xn).
    /// </summary>
    [Serializable]
    class ReduceMin : Reduce
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
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var axes = inputs.Length > 1 && inputs[1] != null ? ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is TensorInt)
                ctx.backend.ReduceMin(X as TensorInt, O as TensorInt, axes, keepdims);
            else
                ctx.backend.ReduceMin(X as TensorFloat, O as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceMin";
    }

    /// <summary>
    /// Represents a `ReduceProd` reduction layer along the given axes: f(x1, x2 ... xn) = x1 * x2 * ... * xn.
    /// </summary>
    [Serializable]
    class ReduceProd : Reduce
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
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var axes = inputs.Length > 1 && inputs[1] != null ? ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is TensorInt)
                ctx.backend.ReduceProd(X as TensorInt, O as TensorInt, axes, keepdims);
            else
                ctx.backend.ReduceProd(X as TensorFloat, O as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceProd";
    }

    /// <summary>
    /// Represents a `ReduceSum` reduction layer along the given axes: f(x1, x2 ... xn) = x1 + x2 + ... + xn.
    /// </summary>
    [Serializable]
    class ReduceSum : Reduce
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
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var axes = inputs.Length > 1 && inputs[1] != null ? ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is TensorInt)
                ctx.backend.ReduceSum(X as TensorInt, O as TensorInt, axes, keepdims);
            else
                ctx.backend.ReduceSum(X as TensorFloat, O as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceSum";
    }

    /// <summary>
    /// Represents a `ReduceSumSquare` reduction layer along the given axes: f(x1, x2 ... xn) = x1² + x2² + ... + xn².
    /// </summary>
    [Serializable]
    class ReduceSumSquare : Reduce
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
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var axes = inputs.Length > 1 && inputs[1] != null ? ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is TensorInt)
                ctx.backend.ReduceSumSquare(X as TensorInt, O as TensorInt, axes, keepdims);
            else
                ctx.backend.ReduceSumSquare(X as TensorFloat, O as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceSumSquare";
    }
}
