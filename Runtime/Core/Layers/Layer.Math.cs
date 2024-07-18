using System;
using System.Linq;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an element-wise `Abs` math layer: f(x) = |x|.
    /// </summary>
    class Abs : Activation
    {
        public Abs(int output, int input)
            : base(output, input) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
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
    class Add : Broadcast
    {
        public Add(int output, int a, int b)
            : base(output, a, b) { }

        internal override Func<PartialTensorElement, PartialTensorElement, PartialTensorElement> InferPartialOp => (a, b) => a + b;

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
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
    class Ceil : Activation
    {
        public Ceil(int output, int input)
            : base(output, input) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Ceil(X, O);
        }

        internal override string profilerTag => "Ceil";
    }

    /// <summary>
    /// Represents an element-wise `Clip` math layer: f(x, xmin, xmax) = min(max(x, xmin), xmax)
    /// </summary>
    class Clip : Layer
    {
        public Clip(int output, int input, int min = -1, int max = -1)
            : base(new[] { output }, new[] { input, min, max }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, X.shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            // TODO don't switch data type at runtime
            if (X is TensorInt)
            {
                var min = ctx.storage.GetInt(inputs[1], int.MinValue);
                var max = ctx.storage.GetInt(inputs[2], int.MaxValue);
                ctx.backend.Clip(X as TensorInt, O as TensorInt, min, max);
            }
            else
            {
                var min = ctx.storage.GetFloat(inputs[1], float.MinValue);
                var max = ctx.storage.GetFloat(inputs[2], float.MaxValue);
                ctx.backend.Clip(X as TensorFloat, O as TensorFloat, min, max);
            }
        }

        internal override string profilerTag => "Clip";
    }

    /// <summary>
    /// Represents a `CumSum` math layer that performs the cumulative sum along a given axis.
    /// </summary>
    class CumSum : Layer
    {
        public bool reverse;
        public bool exclusive;

        public CumSum(int output, int input, int axis, bool reverse, bool exclusive)
            : base(new[] { output }, new[] { input, axis })
        {
            this.reverse = reverse;
            this.exclusive = exclusive;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, X.shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var axis = ctx.storage.GetInt(inputs[1]);
            if (X is TensorInt)
                ctx.backend.CumSum(X as TensorInt, O as TensorInt, axis, reverse, exclusive);
            else
                ctx.backend.CumSum(X as TensorFloat, O as TensorFloat, axis, reverse, exclusive);
        }

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
    class Dense : FusedActivation
    {
        public Dense(int output, int input, int weights, int bias, FusableActivation fusedActivation = FusableActivation.None)
            : base(new[] { output }, new[] { input, weights, bias })
        {
            this.fusedActivation = fusedActivation;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var W = ctx.GetPartialTensor(inputs[1]);
            var B = ctx.GetPartialTensor(inputs[2]);
            var shapeOut = X.shape.MatMul(W.shape);
            if (shapeOut.hasRank)
                shapeOut[-1] = SymbolicTensorDim.MaxDefinedDim(B.shape[0], shapeOut[-1]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, shapeOut));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var W = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.MatMul(W.shape), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Dense(X as TensorFloat, W as TensorFloat, ctx.storage.GetTensor(inputs[2]) as TensorFloat, O, fusedActivation);
        }

        internal override string profilerTag => "Dense";
    }

    /// <summary>
    /// Represents an element-wise `Div` math operation layer: f(a, b) = a / b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Div : Broadcast
    {
        public Div(int output, int a, int b)
            : base(output, a, b) { }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
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
    class Einsum : Layer
    {
        public string equation;

        TensorShape[] operandShapes;
        TensorIndex[] operandIndices;
        TensorFloat[] operandTensors;

        public Einsum(int output, int[] inputs, string equation)
            : base(new[] { output }, inputs)
        {
            this.equation = equation;
            operandShapes = new TensorShape[inputs.Length];
            operandIndices = new TensorIndex[inputs.Length];
            operandTensors = new TensorFloat[inputs.Length];
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var inputTensors = ctx.GetPartialTensors(inputs);
            var operandIndices = new TensorIndex[inputTensors.Length];
            var shape = EinsumHelper.ParseEquationStringShape(equation, inputTensors.Select(i => i.shape).ToArray(), ref operandIndices, out _, out _);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                operandTensors[i] = ctx.storage.GetTensor(inputs[i]) as TensorFloat;
                operandShapes[i] = operandTensors[i].shape;
            }
            EinsumHelper.ParseEquationString(equation, operandShapes, ref operandIndices, out var outputIndices, out var outputShape, out var sumIndices, out var sumShape, out var numIndices);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], outputShape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            if (inputs.Length > 2)
                CPUBackend.EinsumND(operandTensors, O, operandShapes, operandIndices, outputIndices, outputShape, sumIndices, sumShape, numIndices);
            else
                ctx.backend.Einsum(operandTensors, O, operandIndices, outputIndices, sumIndices, sumShape);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, equation: {equation}";
        }

        internal override string profilerTag => "Einsum";
    }

    /// <summary>
    /// Represents an element-wise `Exp` math layer: f(x) = e^{x}.
    /// </summary>
    class Exp : Activation
    {
        public Exp(int output, int input)
            : base(output, input) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Exp(X, O);
        }

        internal override string profilerTag => "Exp";
    }

    /// <summary>
    /// Represents an element-wise `Floor` math layer: f(x) = floor(x).
    /// </summary>
    class Floor : Activation
    {
        public Floor(int output, int input)
            : base(output, input) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Floor(X, O);
        }

        internal override string profilerTag => "Floor";
    }

    /// <summary>
    /// Represents an element-wise `Log` math layer: f(x) = log(x).
    /// </summary>
    class Log : Activation
    {
        public Log(int output, int input)
            : base(output, input) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Log(X, O);
        }

        internal override string profilerTag => "Log";
    }

    /// <summary>
    /// Represents a `MatMul` math operation layer which performs a matrix multiplication operation: f(a, b) = a x b.
    /// </summary>
    class MatMul : Layer
    {
        public MatMul(int output, int input0, int input1)
            : base(new[] { output }, new[] { input0, input1 }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var A = ctx.GetPartialTensor(inputs[0]);
            var B = ctx.GetPartialTensor(inputs[1]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(A.dataType, A.shape.MatMul(B.shape)));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.MatMul(B.shape), DataType.Float, ctx.backend.backendType) as TensorFloat;
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
    class MatMul2D : Layer
    {
        public bool transposeA;
        public bool transposeB;

        public MatMul2D(int output, int input0, bool transpose0, int input1, bool transpose1)
            : base(new[] { output }, new[] { input0, input1 })
        {
            this.transposeA = transpose0;
            this.transposeB = transpose1;
        }

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
            ctx.AddPartialTensor(outputs[0], new PartialTensor(A.dataType, shapeOut));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Gemm(A.shape, B.shape, transposeA, transposeB), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            if (A.shape.HasZeroDims() || B.shape.HasZeroDims())
                ctx.backend.MemSet(O, 0.0f);
            else
                ctx.backend.MatMul2D(A as TensorFloat, B as TensorFloat, O, transposeA, transposeB);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, transposeA: {transposeA}, transposeB: {transposeB}";
        }

        internal override string profilerTag => "MatMul2D";
    }

    /// <summary>
    /// Represents an element-wise `Min` math operation layer: f(a, b) = min(a, b).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Min : Broadcast
    {
        public Min(int output, int a, int b)
            : base(output, a, b) { }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is TensorInt)
                ctx.backend.Min(A as TensorInt, B as TensorInt, O as TensorInt);
            else
                ctx.backend.Min(A as TensorFloat, B as TensorFloat, O as TensorFloat);
        }

        internal override string profilerTag => "Min";
    }

    /// <summary>
    /// Represents an element-wise `Max` math operation layer: f(a, b) = max(a, b).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Max : Broadcast
    {
        public Max(int output, int a, int b)
            : base(output, a, b) { }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is TensorInt)
                ctx.backend.Max(A as TensorInt, B as TensorInt, O as TensorInt);
            else
                ctx.backend.Max(A as TensorFloat, B as TensorFloat, O as TensorFloat);
        }

        internal override string profilerTag => "Max";
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
    class Mod : Broadcast
    {
        public bool fmod;

        public Mod(int output, int a, int b, bool fmod = false)
            : base(output, a, b)
        {
            this.fmod = fmod;
        }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
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
    class Mul : Broadcast
    {
        public Mul(int output, int a, int b)
            : base(output, a, b) { }

        internal override Func<PartialTensorElement, PartialTensorElement, PartialTensorElement> InferPartialOp => (a, b) => a * b;

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
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
    class Neg : Activation
    {
        public Neg(int output, int input)
            : base(output, input) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
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
    class Pow : Broadcast
    {
        public Pow(int output, int a, int b)
            : base(output, a, b) { }

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), DataType.Float, ctx.backend.backendType) as TensorFloat;
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
    class Reciprocal : Activation
    {
        public Reciprocal(int output, int input)
            : base(output, input) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reciprocal(X as TensorFloat, O);
        }

        internal override string profilerTag => "Reciprocal";
    }

    /// <summary>
    /// Represents an element-wise `Round` math layer: f(x) = round(x).
    /// </summary>
    class Round : Activation
    {
        public Round(int output, int input)
            : base(output, input) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Round(X as TensorFloat, O);
        }

        internal override string profilerTag => "Round";
    }

    /// <summary>
    /// Represents an element-wise `Mad` math operation: multiplies and adds bias to a tensor: f(T, s, b) = s * T + b.
    /// </summary>
    class ScalarMad : Activation
    {
        public DataType dataType;
        public float sFloat;
        public float bFloat;
        public int sInt;
        public int bInt;

        public ScalarMad(int output, int input, float s, float b)
            : base(output, input)
        {
            dataType = DataType.Float;
            sFloat = s;
            bFloat = b;
        }

        public ScalarMad(int output, int input, int s, int b)
            : base(output, input)
        {
            dataType = DataType.Int;
            sInt = s;
            bInt = b;
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (dataType == DataType.Float)
                ctx.backend.ScalarMad(X as TensorFloat, O as TensorFloat, sFloat, bFloat);
            else
                ctx.backend.ScalarMad(X as TensorInt, O as TensorInt, sInt, bInt);
        }

        public override string ToString()
        {
            return dataType == DataType.Float ? $"{base.ToString()}, sFloat: {sFloat}, bFloat: {bFloat}" : $"{base.ToString()}, sInt: {sInt}, bInt: {bInt}";
        }

        internal override string profilerTag => "ScalarMad";
    }

    /// <summary>
    /// Represents an element-wise `Shrink` math layer: f(x) = x + bias if x &lt; lambd. f(x) = x - bias if x &gt; lambd. Otherwise f(x) = 0.
    /// </summary>
    class Shrink : Layer
    {
        public float bias;
        public float lambd;

        public Shrink(int output, int input, float bias, float lambd)
            : base(new[] { output }, new[] { input })
        {
            this.bias = bias;
            this.lambd = lambd;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, X.shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Shrink(X as TensorFloat, O, bias, lambd);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, bias: {bias}, lambd: {lambd}";
        }

        internal override string profilerTag => "Shrink";
    }

    /// <summary>
    /// Represents an element-wise `Sign` math layer: f(x) = 1 if x > 0. f(x) = -1 if x &lt; 0. Otherwise f(x) = 0.
    /// </summary>
    class Sign : Layer
    {
        public Sign(int output, int input)
            : base(new[] { output }, new[] { input }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, X.shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
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
    class Sqrt : Activation
    {
        public Sqrt(int output, int input)
            : base(output, input) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Sqrt(X as TensorFloat, O);
        }

        internal override string profilerTag => "Sqrt";
    }

    /// <summary>
    /// Represents an element-wise `Square` math layer: f(x) = x * x.
    /// </summary>
    class Square : Activation
    {
        public Square(int output, int input)
            : base(output, input) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
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
    class Sub : Broadcast
    {
        public Sub(int output, int a, int b)
            : base(output, a, b) { }

        internal override Func<PartialTensorElement, PartialTensorElement, PartialTensorElement> InferPartialOp => (a, b) => a - b;

        public override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is TensorInt)
                ctx.backend.Sub(A as TensorInt, B as TensorInt, O as TensorInt);
            else
                ctx.backend.Sub(A as TensorFloat, B as TensorFloat, O as TensorFloat);
        }

        internal override string profilerTag => "Sub";
    }
}
