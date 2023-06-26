using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an element-wise `And` logical operation layer: f(a, b) = a & b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class And : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `And` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public And(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.And(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt);
        }

        internal override string profilerTag => "And";
    }

    /// <summary>
    /// Represents a `Compress` logical layer that selects slices of an input tensor along a given axis according to a condition tensor.
    /// If you don't provide an axis, the layer flattens the input tensor.
    /// </summary>
    [Serializable]
    public class Compress : Layer
    {
        /// <summary>
        /// Whether to perform the `Compress` along an axis. If `false`, the layer flattens the input tensor.
        /// </summary>
        public bool hasAxis;
        /// <summary>
        /// The axis along which to apply the Compress when `hasAxis` is `true`.
        /// </summary>
        public int axis;

        /// <summary>
        /// Initializes and returns an instance of `Compress` logical layer without an axis. The layer flattens the input tensor.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="condition">The name to use for the condition tensor of the layer.</param>
        public Compress(string name, string input, string condition)
        {
            this.name = name;
            inputs = new[] { input, condition };
        }

        /// <summary>
        /// Initializes and returns an instance of `Compress` logical layer along an axis.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="condition">The name to use for the condition tensor of the layer.</param>
        /// <param name="axis">The axis along which to apply the `Compress`.</param>
        public Compress(string name, string input, string condition, int axis)
        {
            this.name = name;
            inputs = new[] { input, condition };
            hasAxis = true;
            this.axis = axis;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            if (hasAxis)
                return SymbolicInference.Compress(inputShapes[0], inputShapes[1], axis);
            return SymbolicInference.Compress(inputShapes[0], inputShapes[1]);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (hasAxis)
                return ctx.ops.Compress(inputTensors[0], inputTensors[1] as TensorInt, axis);
            return ctx.ops.Compress(ctx.ops.Reshape(inputTensors[0], new TensorShape(inputTensors[0].shape.length)), inputTensors[1] as TensorInt, 0);
        }

        /// <inheritdoc/>
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
    [Serializable]
    public class Equal : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Equal` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Equal(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (inputTensors[0] is TensorInt)
                return ctx.ops.Equal(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt);
            else
                return ctx.ops.Equal(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat);
        }

        internal override string profilerTag => "Equal";
    }

    /// <summary>
    /// Represents an element-wise `Greater` logical operation layer: f(a, b) = 1 if a > b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Greater : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Greater` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Greater(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (inputTensors[0] is TensorInt)
                return ctx.ops.Greater(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt);
            else
                return ctx.ops.Greater(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat);
        }

        internal override string profilerTag => "Greater";
    }

    /// <summary>
    /// Represents an element-wise `GreaterOrEqual` logical operation layer: f(a, b) = 1 if a >= b, otherwise f(a,b) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class GreaterOrEqual : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `GreaterOrEqual` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public GreaterOrEqual(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (inputTensors[0] is TensorInt)
                return ctx.ops.GreaterOrEqual(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt);
            else
                return ctx.ops.GreaterOrEqual(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat);
        }

        internal override string profilerTag => "GreaterOrEqual";
    }

    /// <summary>
    /// Represents an element-wise `IsInf` logical layer: f(x) = 1 elementwise if x is +Inf and detectPositive, or x is -Inf and `detectNegative` is true. Otherwise f(x) = 0.
    /// </summary>
    [Serializable]
    public class IsInf : Layer
    {
        /// <summary>
        /// Whether to detect negative infinities in the `IsInf` function.
        /// </summary>
        public bool detectNegative;
        /// <summary>
        /// Whether to detect positive infinities in the `IsInf` function.
        /// </summary>
        public bool detectPositive;

        /// <summary>
        /// Initializes and returns an instance of `IsInf` logical layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="detectNegative">Whether to detect negative infinities in the `IsInf` function.</param>
        /// <param name="detectPositive">Whether to detect positive infinities in the `IsInf` function.</param>
        public IsInf(string name, string input, bool detectNegative, bool detectPositive)
        {
            this.name = name;
            inputs = new[] { input };
            this.detectNegative = detectNegative;
            this.detectPositive = detectPositive;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return inputShapes[0];
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.IsInf(inputTensors[0] as TensorFloat, detectNegative, detectPositive);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, detectNegative: {detectNegative}, detectPositive: {detectPositive}";
        }

        internal override string profilerTag => "IsInf";
    }

    /// <summary>
    /// Represents an element-wise `IsNaN` logical layer: f(x) = 1 if x is NaN, otherwise f(x) = 0.
    /// </summary>
    [Serializable]
    public class IsNaN : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `IsNaN` logical layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public IsNaN(string name, string input)
        {
            this.name = name;
            inputs = new[] { input };
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return inputShapes[0];
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.IsNaN(inputTensors[0] as TensorFloat);
        }

        internal override string profilerTag => "IsNaN";
    }

    /// <summary>
    /// Represents an element-wise `Less` logical operation layer: f(a, b) = 1 if a &lt; b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Less : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Less` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Less(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (inputTensors[0] is TensorInt)
                return ctx.ops.Less(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt);
            else
                return ctx.ops.Less(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat);
        }

        internal override string profilerTag => "Less";
    }

    /// <summary>
    /// Represents an element-wise `LessOrEqual` logical operation layer: f(a, b) = 1 if a &lt;= b, otherwise f(a,b) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class LessOrEqual : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `LessOrEqual` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public LessOrEqual(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (inputTensors[0] is TensorInt)
                return ctx.ops.LessOrEqual(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt);
            else
                return ctx.ops.LessOrEqual(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat);
        }

        internal override string profilerTag => "LessOrEqual";
    }

    /// <summary>
    /// Represents an element-wise `Not` logical layer: f(x) = ~x.
    /// </summary>
    [Serializable]
    public class Not : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Not` logical layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Not(string name, string input)
        {
            this.name = name;
            this.inputs = new[] { input };
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return inputShapes[0];
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.Not(inputTensors[0] as TensorInt);
        }

        internal override string profilerTag => "Not";
    }

    /// <summary>
    /// Represents an element-wise `Or` logical operation layer: f(a, b) = a | b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Or : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Or` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Or(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.Or(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt);
        }

        internal override string profilerTag => "Or";
    }

    /// <summary>
    /// Represents an element-wise `Xor` logical operation layer: f(a, b) = a ^ b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Xor : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Xor` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Xor(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.Xor(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt);
        }

        internal override string profilerTag => "Xor";
    }

    /// <summary>
    /// Represents an element-wise `Where` logical operation layer: f(condition, a, b) = a if `condition`, otherwise f(condition, a, b) = b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    public class Where : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Where` logical operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="condition">The name to use for the condition input tensor of the layer.</param>
        /// <param name="input1">The name to use for the first input tensor of the layer.</param>
        /// <param name="input2">The name to use for the second input tensor of the layer.</param>
        public Where(string name, string condition, string input1, string input2)
            : base(name, condition, input1, input2) { }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.Where(inputTensors[0] as TensorInt, inputTensors[1], inputTensors[2]);
        }

        internal override string profilerTag => "Where";
    }
}
