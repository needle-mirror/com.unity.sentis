using System;
using System.Linq;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an element-wise `Abs` math layer: f(x) = |x|.
    /// </summary>
    [Serializable]
    class Abs : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Abs` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Abs(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is TensorInt)
                ctx.backend.Abs(X as TensorInt, O as TensorInt);
            else
                ctx.backend.Abs(X as TensorFloat, O as TensorFloat);
        }

        internal override string profilerTag => "Abs";
    }

    /// <summary>
    /// Represents an element-wise `Add` math operation layer: f(a, b) = a + b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    class Add : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Add` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Add(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        internal override Func<PartialTensorElement, PartialTensorElement, PartialTensorElement> InferPartialOp => (a, b) => a + b;

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.vars.GetTensor(inputs[0]);
            var B = ctx.vars.GetTensor(inputs[1]);
            var O = ctx.vars.AllocateTensorAndStore(index, TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is TensorInt)
                ctx.backend.Add(A as TensorInt, B as TensorInt, O as TensorInt);
            else
                ctx.backend.Add(A as TensorFloat, B as TensorFloat, O as TensorFloat);
        }

        internal override string profilerTag => "Add";
    }

    /// <summary>
    /// Represents an element-wise `Ceil` math layer: f(x) = ceil(x).
    /// </summary>
    [Serializable]
    class Ceil : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Ceil` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Ceil(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Ceil(X, O);
        }

        internal override string profilerTag => "Ceil";
    }

    /// <summary>
    /// Represents an element-wise `Clip` math layer: f(x, xmin, xmax) = min(max(x, xmin), xmax)
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1, 2)]
    class Clip : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Clip` math layer with no min or max tensors.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Clip(string name, string input)
        {
            this.index = name;
            this.inputs = new[] { input };
        }

        /// <summary>
        /// Initializes and returns an instance of `Clip` math layer with no max tensor.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="min">The name to use for the min scalar tensor of the layer.</param>
        public Clip(string name, string input, string min)
        {
            this.index = name;
            this.inputs = new[] { input, min };
        }

        /// <summary>
        /// Initializes and returns an instance of `Clip` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="min">The name to use for the min scalar tensor of the layer.</param>
        /// <param name="max">The name to use for the max scalar tensor of the layer.</param>
        public Clip(string name, string input, string min, string max)
        {
            this.index = name;
            this.inputs = new[] { input, min, max };
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(index, new PartialTensor(X.dataType, X.shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is TensorInt)
            {
                var min = inputs.Length > 1 && ctx.vars.GetTensor(inputs[1]) != null ? ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>()[0] : int.MinValue;
                var max = inputs.Length > 2 && ctx.vars.GetTensor(inputs[2]) != null ? ctx.vars.GetTensor(inputs[2]).ToReadOnlySpan<int>()[0] : int.MaxValue;
                ctx.backend.Clip(X as TensorInt, O as TensorInt, min, max);
            }
            else
            {
                var min = inputs.Length > 1 && ctx.vars.GetTensor(inputs[1]) != null ? ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<float>()[0] : float.MinValue;
                var max = inputs.Length > 2 && ctx.vars.GetTensor(inputs[2]) != null ? ctx.vars.GetTensor(inputs[2]).ToReadOnlySpan<float>()[0] : float.MaxValue;
                ctx.backend.Clip(X as TensorFloat, O as TensorFloat, min, max);
            }
        }

        internal override string profilerTag => "Clip";
    }

    /// <summary>
    /// Represents a `CumSum` math layer that performs the cumulative sum along a given axis.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    class CumSum : Layer
    {
        /// <summary>
        /// Whether to perform the cumulative sum from the end of the axis.
        /// </summary>
        public bool reverse;
        /// <summary>
        /// Whether to include the respective input element in the cumulative sum.
        /// </summary>
        public bool exclusive;

        /// <summary>
        /// Initializes and returns an instance of `CumSum` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axis">The name to use for the axis scalar tensor along which to perform the cumulative sum.</param>
        /// <param name="reverse">Whether to perform the cumulative sum from the end of the axis.</param>
        /// <param name="exclusive">Whether to include the respective input element in the cumulative sum.</param>
        public CumSum(string name, string input, string axis, bool reverse, bool exclusive)
        {
            this.index = name;
            this.inputs = new[] { input, axis };
            this.reverse = reverse;
            this.exclusive = exclusive;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(index, new PartialTensor(X.dataType, X.shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var axis = ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>()[0];
            if (X is TensorInt)
                ctx.backend.CumSum(X as TensorInt, O as TensorInt, axis, reverse, exclusive);
            else
                ctx.backend.CumSum(X as TensorFloat, O as TensorFloat, axis, reverse, exclusive);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, reverse: {reverse}, exclusive: {exclusive}";
        }

        internal override string profilerTag => "CumSum";
    }

    /// <summary>
    /// Represents a `Dense` math operation layer which performs a matrix multiplication operation: f(x, w, b) = X x W + B.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    class Dense : FusedActivation
    {
        /// <summary>
        /// Initializes and returns an instance of `Dense` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the first input tensor of the layer.</param>
        /// <param name="weights">The name to use for the weights input tensor of the layer.</param>
        /// <param name="bias">The name to use for the bias input tensor of the layer.</param>
        /// <param name="fusedActivation">The fusable activation to apply to the output tensor of the layer.</param>
        public Dense(string name, string input, string weights, string bias, FusableActivation fusedActivation = FusableActivation.None)
        {
            this.index = name;
            inputs = new[] { input, weights, bias };
            this.fusedActivation = fusedActivation;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var W = ctx.GetPartialTensor(inputs[1]);
            var B = ctx.GetPartialTensor(inputs[2]);
            var shapeOut = X.shape.MatMul(W.shape);
            if (shapeOut.hasRank)
                shapeOut[-1] = SymbolicTensorDim.MaxDefinedDim(B.shape[0], shapeOut[-1]);
            ctx.AddPartialTensor(index, new PartialTensor(DataType.Float, shapeOut));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var W = ctx.vars.GetTensor(inputs[1]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.MatMul(W.shape), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Dense(X as TensorFloat, W as TensorFloat, ctx.vars.GetTensor(inputs[2]) as TensorFloat, O, fusedActivation);
        }

        internal override string profilerTag => "Dense";
    }

    /// <summary>
    /// Represents an element-wise `Div` math operation layer: f(a, b) = a / b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    class Div : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Div` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the numerator input tensor of the layer.</param>
        /// <param name="b">The name to use for the denominator input tensor of the layer.</param>
        public Div(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.vars.GetTensor(inputs[0]);
            var B = ctx.vars.GetTensor(inputs[1]);
            var O = ctx.vars.AllocateTensorAndStore(index, TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is TensorInt)
                ctx.backend.Div(A as TensorInt, B as TensorInt, O as TensorInt);
            else
                ctx.backend.Div(A as TensorFloat, B as TensorFloat, O as TensorFloat);
        }

        internal override string profilerTag => "Div";
    }

    /// <summary>
    /// Represents an `Einsum` math operation layer.
    /// </summary>
    /// <description>
    /// The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to an operand tensor, and the characters within the terms correspond to operands dimensions.
    /// This sequence may be followed by "->" to separate the left and right hand side of the equation. If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases, output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the equation.
    /// When a dimension character is repeated in the left-hand side, it represents summation along the dimension.
    /// The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions. Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions. The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the beginning of the output. The equation string may contain space (U+0020) character.
    /// </description>
    [Serializable]
    class Einsum : Layer
    {
        /// <summary>
        /// The equation of the Einstein summation as a comma-separated list of subscript labels.
        /// </summary>
        public string equation;

        TensorShape[] operandShapes;
        TensorIndex[] operandIndices;

        /// <summary>
        /// Initializes and returns an instance of `Einsum` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The names to use for the input tensors of the layer.</param>
        /// <param name="equation">The equation of the Einstein summation as a comma-separated list of subscript labels.</param>
        public Einsum(string name, string[] inputs, string equation)
        {
            this.index = name;
            this.inputs = inputs;
            this.equation = equation;
            operandShapes = new TensorShape[inputs.Length];
            operandIndices = new TensorIndex[inputs.Length];
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var inputTensors = ctx.GetPartialTensors(inputs);
            var operandIndices = new TensorIndex[inputTensors.Length];
            var shape = EinsumHelper.ParseEquationStringShape(equation, inputTensors.Select(i => i.shape).ToArray(), ref operandIndices, out _, out _);
            ctx.AddPartialTensor(index, new PartialTensor(DataType.Float, shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            TensorFloat[] tensors = new TensorFloat[inputs.Length];
            for (var i = 0; i < inputs.Length; i++)
            {
                var tensor = ctx.vars.GetTensor(inputs[i]) as TensorFloat;
                tensors[i] = tensor;
                operandShapes[i] = tensor.shape;
            }
            EinsumHelper.ParseEquationString(equation, operandShapes, ref operandIndices, out var outputIndices, out var outputShape, out var sumIndices, out var sumShape, out var numIndices);
            var O = ctx.vars.AllocateTensorAndStore(index, outputShape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            if (tensors.Length > 2)
                CPUBackend.EinsumND(tensors, O, operandShapes, operandIndices, outputIndices, outputShape, sumIndices, sumShape, numIndices);
            else
                ctx.backend.Einsum(tensors, O, operandIndices, outputIndices, sumIndices, sumShape);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, equation: {equation}";
        }

        internal override string profilerTag => "Einsum";
    }

    /// <summary>
    /// Represents an element-wise `Exp` math layer: f(x) = e^{x}.
    /// </summary>
    [Serializable]
    class Exp : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Exp` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Exp(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Exp(X, O);
        }

        internal override string profilerTag => "Exp";
    }

    /// <summary>
    /// Represents an element-wise `Floor` math layer: f(x) = floor(x).
    /// </summary>
    [Serializable]
    class Floor : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Floor` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Floor(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Floor(X, O);
        }

        internal override string profilerTag => "Floor";
    }

    /// <summary>
    /// Represents an element-wise `Log` math layer: f(x) = log(x).
    /// </summary>
    [Serializable]
    class Log : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Log` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Log(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Log(X, O);
        }

        internal override string profilerTag => "Log";
    }

    /// <summary>
    /// Represents a `MatMul` math operation layer which performs a matrix multiplication operation: f(a, b) = a x b.
    /// </summary>
    [Serializable]
    class MatMul : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `MatMul` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input0">The name to use for the first input tensor of the layer.</param>
        /// <param name="input1">The name to use for the second input tensor of the layer.</param>
        public MatMul(string name, string input0, string input1)
        {
            this.index = name;
            this.inputs = new[] { input0, input1 };
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var A = ctx.GetPartialTensor(inputs[0]);
            var B = ctx.GetPartialTensor(inputs[1]);
            ctx.AddPartialTensor(index, new PartialTensor(A.dataType, A.shape.MatMul(B.shape)));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.vars.GetTensor(inputs[0]);
            var B = ctx.vars.GetTensor(inputs[1]);
            var O = ctx.vars.AllocateTensorAndStore(index, A.shape.MatMul(B.shape), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            if (A.shape.HasZeroDims() || B.shape.HasZeroDims())
                ctx.backend.MemSet(O, 0.0f);
            else
                ctx.backend.MatMul(A as TensorFloat, B as TensorFloat, O);
        }

        internal override string profilerTag => "MatMul";
    }

    /// <summary>
    /// Represents a `MatMul2D` math operation layer which performs a matrix multiplication operation with optional transposes: f(a, b) = a' x b'.
    /// </summary>
    [Serializable]
    class MatMul2D : Layer
    {
        /// <summary>
        /// Whether to transpose the first input before performing the matrix multiplication.
        /// </summary>
        public bool transposeA;
        /// <summary>
        /// Whether to transpose the second input before performing the matrix multiplication.
        /// </summary>
        public bool transposeB;

        /// <summary>
        /// Initializes and returns an instance of `MatMul2D` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input0">The name to use for the first input tensor of the layer.</param>
        /// <param name="transpose0">Whether to transpose the first input before performing the matrix multiplication.</param>
        /// <param name="input1">The name to use for the second input tensor of the layer.</param>
        /// <param name="transpose1">Whether to transpose the second input before performing the matrix multiplication.</param>
        public MatMul2D(string name, string input0, bool transpose0, string input1, bool transpose1)
        {
            this.index = name;
            this.inputs = new[] { input0, input1 };
            this.transposeA = transpose0;
            this.transposeB = transpose1;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var A = ctx.GetPartialTensor(inputs[0]);
            var B = ctx.GetPartialTensor(inputs[1]);

            var shapeA = A.shape;
            var shapeB = B.shape;

            shapeA.DeclareRank(2);
            shapeB.DeclareRank(2);

            var mulXDim = transposeA ? shapeA[0] : shapeA[1];
            var mulYDim = transposeB ? shapeB[1] : shapeB[0];
            Logger.AssertIsFalse(mulXDim != mulYDim, "MatMul2D.ValueError: failed, dims not equal");

            var shapeOut = new SymbolicTensorShape(transposeA ? shapeA[1] : shapeA[0], transposeB ? shapeB[0] : shapeB[1]);
            ctx.AddPartialTensor(index, new PartialTensor(A.dataType, shapeOut));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.vars.GetTensor(inputs[0]);
            var B = ctx.vars.GetTensor(inputs[1]);
            var O = ctx.vars.AllocateTensorAndStore(index, ShapeInference.Gemm(A.shape, B.shape, transposeA, transposeB), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            if (A.shape.HasZeroDims() || B.shape.HasZeroDims())
                ctx.backend.MemSet(O, 0.0f);
            else
                ctx.backend.MatMul2D(A as TensorFloat, B as TensorFloat, O, transposeA, transposeB);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, transposeA: {transposeA}, transposeB: {transposeB}";
        }

        internal override string profilerTag => "MatMul2D";
    }

    /// <summary>
    /// Represents an element-wise `Max` math operation layer: f(x1, x2 ... xn) = max(x1, x2 ... xn).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    class Max : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Max` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The array of names to use for the input tensors of the layer.</param>
        public Max(string name, string[] inputs)
            : base(name, inputs) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            Tensor[] tensors = new Tensor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                tensors[i] = ctx.vars.GetTensor(inputs[i]);
            }
            var O = ctx.vars.AllocateTensorAndStore(index, TensorShapeHelper.BroadcastShape(tensors), ctx.vars.GetTensor(inputs[0]).dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (ctx.vars.GetTensor(inputs[0]) is TensorInt)
                ctx.backend.Max(Array.ConvertAll(tensors, i => i as TensorInt), O as TensorInt);
            else
                ctx.backend.Max(Array.ConvertAll(tensors, i => i as TensorFloat), O as TensorFloat);
        }

        internal override string profilerTag => "Max";
    }

    /// <summary>
    /// Represents an element-wise `Mean` math operation layer: f(x1, x2 ... xn) = (x1 + x2 ... xn) / n.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    class Mean : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Mean` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The array of names to use for the input tensors of the layer.</param>
        public Mean(string name, string[] inputs)
            : base(name, inputs) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            TensorFloat[] tensors = new TensorFloat[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                tensors[i] = ctx.vars.GetTensor(inputs[i]) as TensorFloat;
            }
            var O = ctx.vars.AllocateTensorAndStore(index, TensorShapeHelper.BroadcastShape(tensors), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Mean(tensors, O);
        }

        internal override string profilerTag => "Mean";
    }

    /// <summary>
    /// Represents an element-wise `Min` math operation layer: f(x1, x2 ... xn) = min(x1, x2 ... xn).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    class Min : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Min` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The array of names to use for the input tensors of the layer.</param>
        public Min(string name, string[] inputs)
            : base(name, inputs) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            Tensor[] tensors = new Tensor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                tensors[i] = ctx.vars.GetTensor(inputs[i]);
            }
            var O = ctx.vars.AllocateTensorAndStore(index, TensorShapeHelper.BroadcastShape(tensors), ctx.vars.GetTensor(inputs[0]).dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (ctx.vars.GetTensor(inputs[0]) is TensorInt)
                ctx.backend.Min(Array.ConvertAll(tensors, i => i as TensorInt), O as TensorInt);
            else
                ctx.backend.Min(Array.ConvertAll(tensors, i => i as TensorFloat), O as TensorFloat);
        }

        internal override string profilerTag => "Min";
    }

    /// <summary>
    /// Represents an element-wise `Max` math operation layer: f(a, b) = a % b.
    ///
    /// If fmod is false the sign of the remainder is the same as that of the divisor as in Python.
    ///
    /// If fmod is true the sign of the remainder is the same as that of the dividend as in C#.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    class Mod : Broadcast
    {
        /// <summary>
        /// Whether to have the sign of the remainder the same as that of the dividend rather than that of the divisor.
        /// </summary>
        public bool fmod;

        /// <summary>
        /// Initializes and returns an instance of `Mod` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the divisor input tensor of the layer.</param>
        /// <param name="b">The name to use for the dividend input tensor of the layer.</param>
        /// <param name="fmod">Whether to have the sign of the remainder the same as that of the dividend rather than that of the divisor.</param>
        public Mod(string name, string a, string b, bool fmod = false)
            : base(name, a, b)
        {
            this.fmod = fmod;
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.vars.GetTensor(inputs[0]);
            var B = ctx.vars.GetTensor(inputs[1]);
            var O = ctx.vars.AllocateTensorAndStore(index, TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (!fmod)
            {
                if (A is TensorInt)
                    ctx.backend.Mod(A as TensorInt, B as TensorInt, O as TensorInt);
                else
                    ctx.backend.Mod(A as TensorFloat, B as TensorFloat, O as TensorFloat);
            }
            else
            {
                if (A is TensorInt)
                    ctx.backend.FMod(A as TensorInt, B as TensorInt, O as TensorInt);
                else
                    ctx.backend.FMod(A as TensorFloat, B as TensorFloat, O as TensorFloat);
            }
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, fmod: {fmod}";
        }

        internal override string profilerTag => "Mod";
    }

    /// <summary>
    /// Represents an element-wise `Mul` math operation layer: f(a, b) = a * b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    class Mul : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Mul` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Mul(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        internal override Func<PartialTensorElement, PartialTensorElement, PartialTensorElement> InferPartialOp => (a, b) => a * b;

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.vars.GetTensor(inputs[0]);
            var B = ctx.vars.GetTensor(inputs[1]);
            var O = ctx.vars.AllocateTensorAndStore(index, TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is TensorInt)
                ctx.backend.Mul(A as TensorInt, B as TensorInt, O as TensorInt);
            else
                ctx.backend.Mul(A as TensorFloat, B as TensorFloat, O as TensorFloat);
        }

        internal override string profilerTag => "Mul";
    }

    /// <summary>
    /// Represents an element-wise `Neg` math layer: f(x) = -x.
    /// </summary>
    [Serializable]
    class Neg : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Neg` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Neg(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is TensorInt)
                ctx.backend.Neg(X as TensorInt, O as TensorInt);
            else
                ctx.backend.Neg(X as TensorFloat, O as TensorFloat);
        }

        internal override string profilerTag => "Neg";
    }

    /// <summary>
    /// Represents an element-wise `Pow` math operation layer: f(a, b) = pow(a, b).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    class Pow : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Pow` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Pow(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var B = ctx.vars.GetTensor(inputs[1]);
            var O = ctx.vars.AllocateTensorAndStore(index, TensorShapeHelper.BroadcastShape(A, B), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            if (B is TensorInt)
                ctx.backend.Pow(A, B as TensorInt, O);
            else
                ctx.backend.Pow(A, B as TensorFloat, O);
        }

        internal override string profilerTag => "Pow";
    }

    /// <summary>
    /// Represents an element-wise `Reciprocal` math layer: f(x) = 1 / x.
    /// </summary>
    [Serializable]
    class Reciprocal : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Reciprocal` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Reciprocal(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reciprocal(X as TensorFloat, O);
        }

        internal override string profilerTag => "Reciprocal";
    }

    /// <summary>
    /// Represents an element-wise `Round` math layer: f(x) = round(x).
    /// </summary>
    [Serializable]
    class Round : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Round` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Round(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Round(X as TensorFloat, O);
        }

        internal override string profilerTag => "Round";
    }

    /// <summary>
    /// Represents an element-wise `Mad` math operation: multiplies and adds bias to a tensor: f(T, s, b) = s * T + b.
    /// </summary>
    [Serializable]
    class ScalarMad : Activation
    {
        /// <summary>
        /// Whether to do int or float scalar operation.
        /// </summary>
        public DataType dataType;
        /// <summary>
        /// Input float scalar for multiplication.
        /// </summary>
        public float sFloat;
        /// <summary>
        /// Input float bias for addition.
        /// </summary>
        public float bFloat;
        /// <summary>
        /// Input int scalar for multiplication.
        /// </summary>
        public int sInt;
        /// <summary>
        /// Input int bias for addition.
        /// </summary>
        public int bInt;

        /// <summary>
        /// Initializes and returns an instance of `ScalarMad` math layer with float scale and bias.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="s">The value of the scale for the scalarmad function.</param>
        /// <param name="b">The value of the bias for the scalarmad function.</param>
        public ScalarMad(string name, string input, float s, float b)
            : base(name, input)
        {
            dataType = DataType.Float;
            sFloat = s;
            bFloat = b;
        }

        /// <summary>
        /// Initializes and returns an instance of `ScalarMad` math layer with int scale and bias.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="s">The value of the scale for the scalarmad function.</param>
        /// <param name="b">The value of the bias for the scalarmad function.</param>
        public ScalarMad(string name, string input, int s, int b)
            : base(name, input)
        {
            dataType = DataType.Int;
            sInt = s;
            bInt = b;
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (dataType == DataType.Float)
                ctx.backend.ScalarMad(X as TensorFloat, O as TensorFloat, sFloat, bFloat);
            else
                ctx.backend.ScalarMad(X as TensorInt, O as TensorInt, sInt, bInt);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return dataType == DataType.Float ? $"{base.ToString()}, sFloat: {sFloat}, bFloat: {bFloat}" : $"{base.ToString()}, sInt: {sInt}, bInt: {bInt}";
        }

        internal override string profilerTag => "ScalarMad";
    }

    /// <summary>
    /// Represents an element-wise `Shrink` math layer: f(x) = x + bias if x &lt; lambd. f(x) = x - bias if x &gt; lambd. Otherwise f(x) = 0.
    /// </summary>
    [Serializable]
    class Shrink : Layer
    {
        /// <summary>
        /// The value of the bias for the shrink function.
        /// </summary>
        public float bias;
        /// <summary>
        /// The value of lambda for the shrink function.
        /// </summary>
        public float lambd;

        /// <summary>
        /// Initializes and returns an instance of `Shrink` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="bias">The value of the bias for the shrink function.</param>
        /// <param name="lambd">The value of lambda for the shrink function.</param>
        public Shrink(string name, string input, float bias, float lambd)
        {
            this.index = name;
            inputs = new[] { input };
            this.bias = bias;
            this.lambd = lambd;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(index, new PartialTensor(X.dataType, X.shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Shrink(X as TensorFloat, O, bias, lambd);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, bias: {bias}, lambd: {lambd}";
        }

        internal override string profilerTag => "Shrink";
    }

    /// <summary>
    /// Represents an element-wise `Sign` math layer: f(x) = 1 if x > 0. f(x) = -1 if x &lt; 0. Otherwise f(x) = 0.
    /// </summary>
    [Serializable]
    class Sign : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Sign` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Sign(string name, string input)
        {
            this.index = name;
            this.inputs = new[] { input };
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(index, new PartialTensor(X.dataType, X.shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is TensorInt)
                ctx.backend.Sign(X as TensorInt, O as TensorInt);
            else
                ctx.backend.Sign(X as TensorFloat, O as TensorFloat);
        }

        internal override string profilerTag => "Sign";
    }

    /// <summary>
    /// Represents an element-wise `Sqrt` math layer: f(x) = sqrt(x).
    /// </summary>
    [Serializable]
    class Sqrt : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Sqrt` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Sqrt(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Sqrt(X as TensorFloat, O);
        }

        internal override string profilerTag => "Sqrt";
    }

    /// <summary>
    /// Represents an element-wise `Square` math layer: f(x) = x * x.
    /// </summary>
    [Serializable]
    class Square : Activation
    {
        /// <summary>
        /// Initializes and returns an instance of `Square` math layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Square(string name, string input)
            : base(name, input) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is TensorInt)
                ctx.backend.Square(X as TensorInt, O as TensorInt);
            else
                ctx.backend.Square(X as TensorFloat, O as TensorFloat);
        }

        internal override string profilerTag => "Square";
    }

    /// <summary>
    /// Represents an element-wise `Sub` math operation layer: f(a, b) = a - b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    class Sub : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Sub` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="a">The name to use for the first input tensor of the layer.</param>
        /// <param name="b">The name to use for the second input tensor of the layer.</param>
        public Sub(string name, string a, string b)
            : base(name, a, b) { }

        /// <inheritdoc/>
        internal override Func<PartialTensorElement, PartialTensorElement, PartialTensorElement> InferPartialOp => (a, b) => a - b;

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.vars.GetTensor(inputs[0]);
            var B = ctx.vars.GetTensor(inputs[1]);
            var O = ctx.vars.AllocateTensorAndStore(index, TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is TensorInt)
                ctx.backend.Sub(A as TensorInt, B as TensorInt, O as TensorInt);
            else
                ctx.backend.Sub(A as TensorFloat, B as TensorFloat, O as TensorFloat);
        }

        internal override string profilerTag => "Sub";
    }

    /// <summary>
    /// Represents an element-wise `Sum` math operation layer: f(x1, x2 ... xn) = x1 + x2 ... xn.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Serializable]
    class Sum : Broadcast
    {
        /// <summary>
        /// Initializes and returns an instance of `Sum` math operation layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The array of names to use for the input tensors of the layer.</param>
        public Sum(string name, string[] inputs)
            : base(name, inputs) { }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            TensorFloat[] tensors = new TensorFloat[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                tensors[i] = ctx.vars.GetTensor(inputs[i]) as TensorFloat;
            }
            var O = ctx.vars.AllocateTensorAndStore(index, TensorShapeHelper.BroadcastShape(tensors), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Sum(tensors, O);
        }

        internal override string profilerTag => "Sum";
    }
}
