using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents the abstract base class for reduction layers.
    /// </summary>
    [Optimization.CPUFallback.CPUReadInputs(1)]
    abstract class Reduce : Layer
    {
        public bool keepdims;
        public bool noopWithEmptyAxes;

        protected Reduce(string output, string data, string axes = "", bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(new[] { output }, new[] { data, axes })
        {
            this.keepdims = keepdims;
            this.noopWithEmptyAxes = noopWithEmptyAxes;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var axes = ctx.GetPartialTensor(inputs[1]);
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

                ctx.AddPartialTensor(outputs[0], tensorOut);
                return;
            }

            if (shapeAxes.IsFullyKnown())
            {
                if (shapeAxes[0].value != 0)
                    ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, keepdims ? SymbolicTensorShape.UnknownOfRankLike(shapeX) : SymbolicTensorShape.UnknownShape));
                else if (noopWithEmptyAxes)
                    ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeX));
                else
                    ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, keepdims ? SymbolicTensorShape.OnesLike(shapeX) : new SymbolicTensorShape()));
                return;
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, keepdims && !noopWithEmptyAxes ? SymbolicTensorShape.UnknownOfRankLike(shapeX) : SymbolicTensorShape.UnknownShape));
        }

        public override string ToString()
        {
            return $"{base.ToString()}, keepdims: {keepdims}, noopWithEmptyAxes: {noopWithEmptyAxes}";
        }
    }

    /// <summary>
    /// Represents a `ReduceL1` reduction layer along the given axes: f(x1, x2 ... xn) = |x1| + |x2| + ... + |xn|.
    /// </summary>
    class ReduceL1 : Reduce
    {
        public ReduceL1(string output, string data, string axes = "", bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = !string.IsNullOrEmpty(inputs[1]) ? ctx.storage.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
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
    class ReduceL2 : Reduce
    {
        public ReduceL2(string output, string data, string axes = "", bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var axes = !string.IsNullOrEmpty(inputs[1]) ? ctx.storage.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.ReduceL2(X, O as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceL2";
    }

    /// <summary>
    /// Represents a `ReduceLogSum` reduction layer along the given axes: f(x1, x2 ... xn) = log(x1 + x2 + ... + xn).
    /// </summary>
    class ReduceLogSum : Reduce
    {
        public ReduceLogSum(string output, string data, string axes = "", bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var axes = !string.IsNullOrEmpty(inputs[1]) ? ctx.storage.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.ReduceLogSum(X, O as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceLogSum";
    }

    /// <summary>
    /// Represents a `ReduceLogSumExp` reduction layer along the given axes: f(x1, x2 ... xn) = log(e^x1 + e^x2 + ... + e^xn).
    /// </summary>
    class ReduceLogSumExp : Reduce
    {
        public ReduceLogSumExp(string output, string data, string axes = "", bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var axes = !string.IsNullOrEmpty(inputs[1]) ? ctx.storage.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.ReduceLogSumExp(X, O as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceLogSumExp";
    }

    /// <summary>
    /// Represents a `ReduceMax` reduction layer along the given axes: f(x1, x2 ... xn) = max(x1, x2, ... , xn).
    /// </summary>
    class ReduceMax : Reduce
    {
        public ReduceMax(string output, string data, string axes = "", bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = !string.IsNullOrEmpty(inputs[1]) ? ctx.storage.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
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
    class ReduceMean : Reduce
    {
        public ReduceMean(string output, string data, string axes = "", bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var axes = !string.IsNullOrEmpty(inputs[1]) ? ctx.storage.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.ReduceMean(X, O as TensorFloat, axes, keepdims);
        }

        internal override string profilerTag => "ReduceMean";
    }

    /// <summary>
    /// Represents a `ReduceMin` reduction layer along the given axes: f(x1, x2 ... xn) = min(x1, x2, ... , xn).
    /// </summary>
    class ReduceMin : Reduce
    {
        public ReduceMin(string output, string data, string axes = "", bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = !string.IsNullOrEmpty(inputs[1]) ? ctx.storage.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
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
    class ReduceProd : Reduce
    {
        public ReduceProd(string output, string data, string axes = "", bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = !string.IsNullOrEmpty(inputs[1]) ? ctx.storage.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
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
    class ReduceSum : Reduce
    {
        public ReduceSum(string output, string data, string axes = "", bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = !string.IsNullOrEmpty(inputs[1]) ? ctx.storage.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
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
    class ReduceSumSquare : Reduce
    {
        public ReduceSumSquare(string output, string data, string axes = "", bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = !string.IsNullOrEmpty(inputs[1]) ? ctx.storage.GetTensor(inputs[1]).ToReadOnlySpan<int>() : null;
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType) as TensorFloat;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
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
