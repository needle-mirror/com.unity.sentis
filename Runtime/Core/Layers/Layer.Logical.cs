using System;
using Unity.Profiling;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an element-wise comparison layer.
    /// </summary>
    abstract class Comparison : Broadcast
    {
        protected Comparison(int output, int a, int b)
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
        static readonly string k_OpName = "And";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public And(int output, int a, int b)
            : base(output, a, b) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as Tensor<int>;
            var B = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.And(A, B, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `Compress` logical layer that selects slices of an input tensor along a given axis according to a condition tensor.
    /// If you don't provide an axis, the layer flattens the input tensor.
    /// </summary>
    class Compress : Layer
    {
        static readonly string k_OpName = "Compress";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public bool hasAxis;
        public int axis;

        public Compress(int output, int input, int condition, int? axis)
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
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, new DynamicTensorShape(isZero ? DynamicTensorDim.Zero : DynamicTensorDim.Unknown)));
                return;
            }

            var shapeOut = shapeX;
            shapeOut[axis] = isZero ? DynamicTensorDim.Zero : DynamicTensorDim.Unknown;
            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var isTempX = !hasAxis;
            if (isTempX)
            {
                var flattenedShape = new TensorShape(X.shape.length);
                var tempX = ctx.storage.AllocateTensor(flattenedShape, X.dataType, ctx.backend.backendType);
                ctx.backend.Reshape(X, tempX);
                X = tempX;
            }

            var condition = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var numCondition = condition.shape.length;

            var indices = ctx.storage.AllocateTensor(condition.shape, DataType.Int, BackendType.CPU) as Tensor<int>;
            CPUTensorData.Pin(indices);

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
            {
                if (isTempX)
                    ctx.storage.Dispose(X);
                ctx.storage.Dispose(indices);
                return;
            }
            ctx.backend.CompressWithIndices(X, indices, O, numIndices, axis);
            if (isTempX)
                ctx.storage.Dispose(X);
            ctx.storage.Dispose(indices);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, hasAxis: {hasAxis}, axis: {axis}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Equal` logical operation layer: f(a, b) = 1 if a == b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Equal : Comparison
    {
        static readonly string k_OpName = "Equal";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Equal(int output, int a, int b)
            : base(output, a, b) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Equal(A as Tensor<int>, B as Tensor<int>, O);
            else
                ctx.backend.Equal(A as Tensor<float>, B as Tensor<float>, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Greater` logical operation layer: f(a, b) = 1 if a > b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Greater : Comparison
    {
        static readonly string k_OpName = "Greater";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Greater(int output, int a, int b)
            : base(output, a, b) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Greater(A as Tensor<int>, B as Tensor<int>, O);
            else
                ctx.backend.Greater(A as Tensor<float>, B as Tensor<float>, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `GreaterOrEqual` logical operation layer: f(a, b) = 1 if a >= b, otherwise f(a,b) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class GreaterOrEqual : Comparison
    {
        static readonly string k_OpName = "GreaterOrEqual";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public GreaterOrEqual(int output, int a, int b)
            : base(output, a, b) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.GreaterOrEqual(A as Tensor<int>, B as Tensor<int>, O);
            else
                ctx.backend.GreaterOrEqual(A as Tensor<float>, B as Tensor<float>, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `IsInf` logical layer: f(x) = 1 elementwise if x is +Inf and detectPositive, or x is -Inf and `detectNegative` is true. Otherwise f(x) = 0.
    /// </summary>
    class IsInf : Layer
    {
        static readonly string k_OpName = "IsInf";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public bool detectNegative;
        public bool detectPositive;

        public IsInf(int output, int input, bool detectNegative, bool detectPositive)
            : base(new[] { output }, new[] { input })
        {
            this.detectNegative = detectNegative;
            this.detectPositive = detectPositive;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, ctx.GetPartialTensor(inputs[0]).shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape, DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.IsInf(A, O, detectNegative, detectPositive);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, detectNegative: {detectNegative}, detectPositive: {detectPositive}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `IsNaN` logical layer: f(x) = 1 if x is NaN, otherwise f(x) = 0.
    /// </summary>
    class IsNaN : Layer
    {
        static readonly string k_OpName = "IsNaN";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public IsNaN(int output, int input)
            : base(new[] { output }, new[] { input }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, ctx.GetPartialTensor(inputs[0]).shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape, DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.IsNaN(A, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Less` logical operation layer: f(a, b) = 1 if a &lt; b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Less : Comparison
    {
        static readonly string k_OpName = "Less";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Less(int output, int a, int b)
            : base(output, a, b) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Less(A as Tensor<int>, B as Tensor<int>, O);
            else
                ctx.backend.Less(A as Tensor<float>, B as Tensor<float>, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `LessOrEqual` logical operation layer: f(a, b) = 1 if a &lt;= b, otherwise f(a,b) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class LessOrEqual : Comparison
    {
        static readonly string k_OpName = "LessOrEqual";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public LessOrEqual(int output, int a, int b)
            : base(output, a, b) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.LessOrEqual(A as Tensor<int>, B as Tensor<int>, O);
            else
                ctx.backend.LessOrEqual(A as Tensor<float>, B as Tensor<float>, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Not` logical layer: f(x) = ~x.
    /// </summary>
    class Not : Layer
    {
        static readonly string k_OpName = "Not";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Not(int output, int input)
            : base(new[] { output }, new[] { input }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, ctx.GetPartialTensor(inputs[0]).shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as Tensor<int>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape, DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Not(A, O);
        }
        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Or` logical operation layer: f(a, b) = a | b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Or : Broadcast
    {
        static readonly string k_OpName = "Or";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Or(int output, int a, int b)
            : base(output, a, b) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as Tensor<int>;
            var B = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Or(A, B, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Xor` logical operation layer: f(a, b) = a ^ b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Xor : Broadcast
    {
        static readonly string k_OpName = "Xor";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Xor(int output, int a, int b)
            : base(output, a, b) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as Tensor<int>;
            var B = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Xor(A, B, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Where` logical operation layer: f(condition, a, b) = a if `condition`, otherwise f(condition, a, b) = b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Where : Broadcast
    {
        static readonly string k_OpName = "Where";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Where(int output, int condition, int input1, int input2)
            : base(output, condition, input1, input2) { }

        internal override DataType InferPartialDataType(PartialTensor[] inputTensors)
        {
            return inputTensors[1].dataType;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var C = ctx.storage.GetTensor(inputs[0]) as Tensor<int>;
            var A = ctx.storage.GetTensor(inputs[1]);
            var B = ctx.storage.GetTensor(inputs[2]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape.Broadcast(C.shape)), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Where(C, A, B, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }
}
