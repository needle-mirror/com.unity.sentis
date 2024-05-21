using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an element-wise comparison layer.
    /// </summary>
    abstract class Comparison : Broadcast
    {
        protected Comparison(string output, string a, string b)
            : base(output, a, b) { }

        internal override DataType InferPartialDataType(PartialTensor[] inputTensors)
        {
            return DataType.Int;
        }
    }

    /// <summary>
    /// Represents an element-wise `And` logical operation layer: f(a, b) = a &amp; b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class And : Broadcast
    {
        public And(string output, string a, string b)
            : base(output, a, b) { }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as TensorInt;
            var B = ctx.storage.GetTensor(inputs[1]) as TensorInt;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as TensorInt;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.And(A, B, O);
        }

        internal override string profilerTag => "And";
    }

    /// <summary>
    /// Represents a `Compress` logical layer that selects slices of an input tensor along a given axis according to a condition tensor.
    /// If you don't provide an axis, the layer flattens the input tensor.
    /// </summary>
    class Compress : Layer
    {
        public bool hasAxis;
        public int axis;

        public Compress(string output, string input, string condition, int? axis)
            : base(new[] { output }, new[] { input, condition })
        {
            hasAxis = axis.HasValue;
            this.axis = axis ?? 0;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var condition = ctx.GetPartialTensor(inputs[1]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var isZero = shapeX.Length() * condition.shape.Length() == 0;
            if (!hasAxis)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, new SymbolicTensorShape(isZero ? SymbolicTensorDim.Zero : SymbolicTensorDim.Unknown)));
                return;
            }

            var shapeOut = shapeX;
            shapeOut[axis] = isZero ? SymbolicTensorDim.Zero : SymbolicTensorDim.Unknown;
            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            if (!hasAxis)
            {
                var flattenedShape = new TensorShape(X.shape.length);
                X.shape = flattenedShape;
            }

            var condition = ctx.storage.GetTensor(inputs[1]) as TensorInt;
            var numCondition = condition.shape.length;

            var indices = ctx.storage.AllocateTensor(condition.shape, DataType.Int, BackendType.CPU) as TensorInt;
            BurstTensorData.Pin(indices);

            var numIndices = 0;
            for (var i = 0; i < numCondition; i++)
            {
                if (condition[i] == 0)
                    continue;
                indices[numIndices] = i;
                numIndices++;
            }

            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Compress(X.shape, numIndices, axis), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.CompressWithIndices(X, indices, O, numIndices, axis);
            ctx.storage.Dispose(indices);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, hasAxis: {hasAxis}, axis: {axis}";
        }

        internal override string profilerTag => "Compress";
    }

    /// <summary>
    /// Represents an element-wise `Equal` logical operation layer: f(a, b) = 1 if a == b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Equal : Comparison
    {
        public Equal(string output, string a, string b)
            : base(output, a, b) { }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as TensorInt;
            if (O.shape.HasZeroDims())
                return;
            if (A is TensorInt)
                ctx.backend.Equal(A as TensorInt, B as TensorInt, O);
            else
                ctx.backend.Equal(A as TensorFloat, B as TensorFloat, O);
        }

        internal override string profilerTag => "Equal";
    }

    /// <summary>
    /// Represents an element-wise `Greater` logical operation layer: f(a, b) = 1 if a > b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Greater : Comparison
    {
        public Greater(string output, string a, string b)
            : base(output, a, b) { }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as TensorInt;
            if (O.shape.HasZeroDims())
                return;
            if (A is TensorInt)
                ctx.backend.Greater(A as TensorInt, B as TensorInt, O);
            else
                ctx.backend.Greater(A as TensorFloat, B as TensorFloat, O);
        }

        internal override string profilerTag => "Greater";
    }

    /// <summary>
    /// Represents an element-wise `GreaterOrEqual` logical operation layer: f(a, b) = 1 if a >= b, otherwise f(a,b) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class GreaterOrEqual : Comparison
    {
        public GreaterOrEqual(string output, string a, string b)
            : base(output, a, b) { }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as TensorInt;
            if (O.shape.HasZeroDims())
                return;
            if (A is TensorInt)
                ctx.backend.GreaterOrEqual(A as TensorInt, B as TensorInt, O);
            else
                ctx.backend.GreaterOrEqual(A as TensorFloat, B as TensorFloat, O);
        }

        internal override string profilerTag => "GreaterOrEqual";
    }

    /// <summary>
    /// Represents an element-wise `IsInf` logical layer: f(x) = 1 elementwise if x is +Inf and detectPositive, or x is -Inf and `detectNegative` is true. Otherwise f(x) = 0.
    /// </summary>
    class IsInf : Layer
    {
        public bool detectNegative;
        public bool detectPositive;

        public IsInf(string output, string input, bool detectNegative, bool detectPositive)
            : base(new[] { output }, new[] { input })
        {
            this.detectNegative = detectNegative;
            this.detectPositive = detectPositive;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, ctx.GetPartialTensor(inputs[0]).shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape, DataType.Int, ctx.backend.backendType) as TensorInt;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.IsInf(A, O, detectNegative, detectPositive);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, detectNegative: {detectNegative}, detectPositive: {detectPositive}";
        }

        internal override string profilerTag => "IsInf";
    }

    /// <summary>
    /// Represents an element-wise `IsNaN` logical layer: f(x) = 1 if x is NaN, otherwise f(x) = 0.
    /// </summary>
    class IsNaN : Layer
    {
        public IsNaN(string output, string input)
            : base(new[] { output }, new[] { input }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, ctx.GetPartialTensor(inputs[0]).shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape, DataType.Int, ctx.backend.backendType) as TensorInt;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.IsNaN(A, O);
        }

        internal override string profilerTag => "IsNaN";
    }

    /// <summary>
    /// Represents an element-wise `Less` logical operation layer: f(a, b) = 1 if a &lt; b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Less : Comparison
    {
        public Less(string output, string a, string b)
            : base(output, a, b) { }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as TensorInt;
            if (O.shape.HasZeroDims())
                return;
            if (A is TensorInt)
                ctx.backend.Less(A as TensorInt, B as TensorInt, O);
            else
                ctx.backend.Less(A as TensorFloat, B as TensorFloat, O);
        }

        internal override string profilerTag => "Less";
    }

    /// <summary>
    /// Represents an element-wise `LessOrEqual` logical operation layer: f(a, b) = 1 if a &lt;= b, otherwise f(a,b) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class LessOrEqual : Comparison
    {
        public LessOrEqual(string output, string a, string b)
            : base(output, a, b) { }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as TensorInt;
            if (O.shape.HasZeroDims())
                return;
            if (A is TensorInt)
                ctx.backend.LessOrEqual(A as TensorInt, B as TensorInt, O);
            else
                ctx.backend.LessOrEqual(A as TensorFloat, B as TensorFloat, O);
        }

        internal override string profilerTag => "LessOrEqual";
    }

    /// <summary>
    /// Represents an element-wise `Not` logical layer: f(x) = ~x.
    /// </summary>
    class Not : Layer
    {
        public Not(string output, string input)
            : base(new[] { output }, new[] { input }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, ctx.GetPartialTensor(inputs[0]).shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as TensorInt;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape, DataType.Int, ctx.backend.backendType) as TensorInt;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Not(A, O);
        }
        internal override string profilerTag => "Not";
    }

    /// <summary>
    /// Represents an element-wise `Or` logical operation layer: f(a, b) = a | b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Or : Broadcast
    {
        public Or(string output, string a, string b)
            : base(output, a, b) { }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as TensorInt;
            var B = ctx.storage.GetTensor(inputs[1]) as TensorInt;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as TensorInt;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Or(A, B, O);
        }

        internal override string profilerTag => "Or";
    }

    /// <summary>
    /// Represents an element-wise `Xor` logical operation layer: f(a, b) = a ^ b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Xor : Broadcast
    {
        public Xor(string output, string a, string b)
            : base(output, a, b) { }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as TensorInt;
            var B = ctx.storage.GetTensor(inputs[1]) as TensorInt;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as TensorInt;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Xor(A, B, O);
        }

        internal override string profilerTag => "Xor";
    }

    /// <summary>
    /// Represents an element-wise `Where` logical operation layer: f(condition, a, b) = a if `condition`, otherwise f(condition, a, b) = b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Where : Broadcast
    {
        public Where(string output, string condition, string input1, string input2)
            : base(output, condition, input1, input2) { }

        internal override DataType InferPartialDataType(PartialTensor[] inputTensors)
        {
            return inputTensors[1].dataType;
        }

        public override void Execute(ExecutionContext ctx)
        {
            var C = ctx.storage.GetTensor(inputs[0]) as TensorInt;
            var A = ctx.storage.GetTensor(inputs[1]);
            var B = ctx.storage.GetTensor(inputs[2]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape.Broadcast(C.shape)), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Where(C, A, B, O);
        }

        internal override string profilerTag => "Where";
    }
}
