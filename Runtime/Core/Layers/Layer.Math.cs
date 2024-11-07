using System;
using System.Linq;
using Unity.Profiling;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an element-wise `Abs` math layer: f(x) = |x|.
    /// </summary>
    class Abs : Activation
    {
        static readonly string k_OpName = "Abs";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Abs(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
                ctx.backend.Abs(X as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Abs(X as Tensor<float>, O as Tensor<float>);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Add` math operation layer: f(a, b) = a + b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Add : Broadcast
    {
        static readonly string k_OpName = "Add";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Add(int output, int a, int b)
            : base(output, a, b) { }

        internal override Func<PartialTensorElement, PartialTensorElement, PartialTensorElement> InferPartialOp => (a, b) => a + b;

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Add(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Add(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Ceil` math layer: f(x) = ceil(x).
    /// </summary>
    class Ceil : Activation
    {
        static readonly string k_OpName = "Ceil";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Ceil(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Ceil(X, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Clip` math layer: f(x, xmin, xmax) = min(max(x, xmin), xmax)
    /// </summary>
    class Clip : Layer
    {
        static readonly string k_OpName = "Clip";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Clip(int output, int input, int min = -1, int max = -1)
            : base(new[] { output }, new[] { input, min, max }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, X.shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            // TODO don't switch data type at runtime
            if (X is Tensor<int>)
            {
                var min = ctx.storage.GetInt(inputs[1], int.MinValue);
                var max = ctx.storage.GetInt(inputs[2], int.MaxValue);
                ctx.backend.Clip(X as Tensor<int>, O as Tensor<int>, min, max);
            }
            else
            {
                var min = ctx.storage.GetFloat(inputs[1], float.MinValue);
                var max = ctx.storage.GetFloat(inputs[2], float.MaxValue);
                ctx.backend.Clip(X as Tensor<float>, O as Tensor<float>, min, max);
            }
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `CumSum` math layer that performs the cumulative sum along a given axis.
    /// </summary>
    class CumSum : Layer
    {
        static readonly string k_OpName = "CumSum";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

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

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var axis = ctx.storage.GetInt(inputs[1]);
            if (X is Tensor<int>)
                ctx.backend.CumSum(X as Tensor<int>, O as Tensor<int>, axis, reverse, exclusive);
            else
                ctx.backend.CumSum(X as Tensor<float>, O as Tensor<float>, axis, reverse, exclusive);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, reverse: {reverse}, exclusive: {exclusive}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `Dense` math operation layer which performs a matrix multiplication operation: f(x, w, b) = X x W + B.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Dense : FusedActivation
    {
        static readonly string k_OpName = "Dense";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

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
                shapeOut[-1] = DynamicTensorDim.MaxDefinedDim(B.shape[0], shapeOut[-1]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var W = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.MatMul(W.shape), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Dense(X as Tensor<float>, W as Tensor<float>, ctx.storage.GetTensor(inputs[2]) as Tensor<float>, O, fusedActivation);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    class DenseBatched : FusedActivation
    {
        static readonly string k_OpName = "DenseBatched";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public DenseBatched(int output, int input, int weights, int bias, FusableActivation fusedActivation = FusableActivation.None)
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
                shapeOut[-1] = DynamicTensorDim.MaxDefinedDim(B.shape[-1], shapeOut[-1]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var W = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.MatMul(W.shape), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.DenseBatched(X as Tensor<float>, W as Tensor<float>, ctx.storage.GetTensor(inputs[2]) as Tensor<float>, O, fusedActivation);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Div` math operation layer: f(a, b) = a / b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Div : Broadcast
    {
        static readonly string k_OpName = "Div";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Div(int output, int a, int b)
            : base(output, a, b) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Div(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Div(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
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
        static readonly string k_OpName = "Einsum";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public string equation;

        TensorShape[] operandShapes;
        TensorIndex[] operandIndices;
        Tensor<float>[] operandTensors;

        public Einsum(int output, int[] inputs, string equation)
            : base(new[] { output }, inputs)
        {
            this.equation = equation;
            operandShapes = new TensorShape[inputs.Length];
            operandIndices = new TensorIndex[inputs.Length];
            operandTensors = new Tensor<float>[inputs.Length];
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var inputTensors = ctx.GetPartialTensors(inputs);
            var operandIndices = new TensorIndex[inputTensors.Length];
            var shape = EinsumHelper.ParseEquationStringShape(equation, inputTensors.Select(i => i.shape).ToArray(), ref operandIndices, out _, out _);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                operandTensors[i] = ctx.storage.GetTensor(inputs[i]) as Tensor<float>;
                operandShapes[i] = operandTensors[i].shape;
            }
            EinsumHelper.ParseEquationString(equation, operandShapes, ref operandIndices, out var outputIndices, out var outputShape, out var sumIndices, out var sumShape, out var numIndices);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], outputShape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
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

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Exp` math layer: f(x) = e^{x}.
    /// </summary>
    class Exp : Activation
    {
        static readonly string k_OpName = "Exp";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Exp(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Exp(X, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Floor` math layer: f(x) = floor(x).
    /// </summary>
    class Floor : Activation
    {
        static readonly string k_OpName = "Floor";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Floor(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Floor(X, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Log` math layer: f(x) = log(x).
    /// </summary>
    class Log : Activation
    {
        static readonly string k_OpName = "Log";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Log(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Log(X, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `MatMul` math operation layer which performs a matrix multiplication operation: f(a, b) = a x b.
    /// </summary>
    class MatMul : Layer
    {
        static readonly string k_OpName = "MatMul";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public MatMul(int output, int input0, int input1)
            : base(new[] { output }, new[] { input0, input1 }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var A = ctx.GetPartialTensor(inputs[0]);
            var B = ctx.GetPartialTensor(inputs[1]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(A.dataType, A.shape.MatMul(B.shape)));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.MatMul(B.shape), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            if (A.shape.HasZeroDims() || B.shape.HasZeroDims())
                ctx.backend.MemSet(O, 0.0f);
            else
                ctx.backend.MatMul(A as Tensor<float>, B as Tensor<float>, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `MatMul2D` math operation layer which performs a matrix multiplication operation with optional transposes: f(a, b) = a' x b'.
    /// </summary>
    class MatMul2D : Layer
    {
        static readonly string k_OpName = "MatMul2D";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

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

            var shapeOut = new DynamicTensorShape(transposeA ? shapeA[1] : shapeA[0], transposeB ? shapeB[0] : shapeB[1]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(A.dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Gemm(A.shape, B.shape, transposeA, transposeB), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            if (A.shape.HasZeroDims() || B.shape.HasZeroDims())
                ctx.backend.MemSet(O, 0.0f);
            else
                ctx.backend.MatMul2D(A as Tensor<float>, B as Tensor<float>, O, transposeA, transposeB);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, transposeA: {transposeA}, transposeB: {transposeB}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Min` math operation layer: f(a, b) = min(a, b).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Min : Broadcast
    {
        static readonly string k_OpName = "Min";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Min(int output, int a, int b)
            : base(output, a, b) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Min(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Min(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Max` math operation layer: f(a, b) = max(a, b).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Max : Broadcast
    {
        static readonly string k_OpName = "Max";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Max(int output, int a, int b)
            : base(output, a, b) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Max(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Max(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
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
        static readonly string k_OpName = "Mod";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public bool fmod;

        public Mod(int output, int a, int b, bool fmod = false)
            : base(output, a, b)
        {
            this.fmod = fmod;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (!fmod)
            {
                if (A is Tensor<int>)
                    ctx.backend.Mod(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
                else
                    ctx.backend.Mod(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
            }
            else
            {
                if (A is Tensor<int>)
                    ctx.backend.FMod(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
                else
                    ctx.backend.FMod(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
            }
        }

        public override string ToString()
        {
            return $"{base.ToString()}, fmod: {fmod}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Mul` math operation layer: f(a, b) = a * b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Mul : Broadcast
    {
        static readonly string k_OpName = "Mul";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Mul(int output, int a, int b)
            : base(output, a, b) { }

        internal override Func<PartialTensorElement, PartialTensorElement, PartialTensorElement> InferPartialOp => (a, b) => a * b;

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Mul(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Mul(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Neg` math layer: f(x) = -x.
    /// </summary>
    class Neg : Activation
    {
        static readonly string k_OpName = "Neg";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Neg(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
                ctx.backend.Neg(X as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Neg(X as Tensor<float>, O as Tensor<float>);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Pow` math operation layer: f(a, b) = pow(a, b).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Pow : Broadcast
    {
        static readonly string k_OpName = "Pow";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Pow(int output, int a, int b)
            : base(output, a, b) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            if (B is Tensor<int>)
                ctx.backend.Pow(A, B as Tensor<int>, O);
            else
                ctx.backend.Pow(A, B as Tensor<float>, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Reciprocal` math layer: f(x) = 1 / x.
    /// </summary>
    class Reciprocal : Activation
    {
        static readonly string k_OpName = "Reciprocal";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Reciprocal(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reciprocal(X as Tensor<float>, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Round` math layer: f(x) = round(x).
    /// </summary>
    class Round : Activation
    {
        static readonly string k_OpName = "Round";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Round(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Round(X as Tensor<float>, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Mad` math operation: multiplies and adds bias to a tensor: f(T, s, b) = s * T + b.
    /// </summary>
    class ScalarMad : Activation
    {
        static readonly string k_OpName = "ScalarMad";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
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

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (dataType == DataType.Float)
                ctx.backend.ScalarMad(X as Tensor<float>, O as Tensor<float>, sFloat, bFloat);
            else
                ctx.backend.ScalarMad(X as Tensor<int>, O as Tensor<int>, sInt, bInt);
        }

        public override string ToString()
        {
            return dataType == DataType.Float ? $"{base.ToString()}, sFloat: {sFloat}, bFloat: {bFloat}" : $"{base.ToString()}, sInt: {sInt}, bInt: {bInt}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Shrink` math layer: f(x) = x + bias if x &lt; lambd. f(x) = x - bias if x &gt; lambd. Otherwise f(x) = 0.
    /// </summary>
    class Shrink : Layer
    {
        static readonly string k_OpName = "Shrink";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
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

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Shrink(X as Tensor<float>, O, bias, lambd);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, bias: {bias}, lambd: {lambd}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Sign` math layer: f(x) = 1 if x > 0. f(x) = -1 if x &lt; 0. Otherwise f(x) = 0.
    /// </summary>
    class Sign : Layer
    {
        static readonly string k_OpName = "Sign";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Sign(int output, int input)
            : base(new[] { output }, new[] { input }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, X.shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
                ctx.backend.Sign(X as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Sign(X as Tensor<float>, O as Tensor<float>);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Sqrt` math layer: f(x) = sqrt(x).
    /// </summary>
    class Sqrt : Activation
    {
        static readonly string k_OpName = "Sqrt";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Sqrt(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Sqrt(X as Tensor<float>, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Square` math layer: f(x) = x * x.
    /// </summary>
    class Square : Activation
    {
        static readonly string k_OpName = "Square";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Square(int output, int input)
            : base(output, input) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
                ctx.backend.Square(X as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Square(X as Tensor<float>, O as Tensor<float>);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Sub` math operation layer: f(a, b) = a - b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Sub : Broadcast
    {
        static readonly string k_OpName = "Sub";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Sub(int output, int a, int b)
            : base(output, a, b) { }

        internal override Func<PartialTensorElement, PartialTensorElement, PartialTensorElement> InferPartialOp => (a, b) => a - b;

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Sub(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Sub(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }
}
