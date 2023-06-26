using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a `ConstantOfShape` layer. This generates a tensor with the shape given by the `input` tensor and filled with a given value.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(0)]
    public class ConstantOfShape : Layer
    {
        /// <summary>
        /// The data type of the layer as a `DataType`.
        /// </summary>
        public DataType dataType;
        /// <summary>
        /// The float value to use to fill the output tensor. The layer only uses this when the `dataType` equals `DataType.Float`.
        /// </summary>
        public float floatValue;
        /// <summary>
        /// The int value to use to fill the output tensor. The layer only uses this when the `dataType` equals `DataType.Int`.
        /// </summary>
        public int intValue;

        /// <summary>
        /// Initializes and returns an instance of `ConstantOfShape` layer with a float value.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the shape tensor of the layer.</param>
        /// <param name="value">The float value to use to fill the output tensor.</param>
        public ConstantOfShape(string name, string input, float value)
        {
            this.name = name;
            this.inputs = new[] { input };
            this.floatValue = value;
            this.dataType = DataType.Float;
        }

        /// <summary>
        /// Initializes and returns an instance of `ConstantOfShape` layer with an int value.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the shape tensor of the layer.</param>
        /// <param name="value">The int value to use to fill the output tensor.</param>
        public ConstantOfShape(string name, string input, int value)
        {
            this.name = name;
            this.inputs = new[] { input };
            this.intValue = value;
            this.dataType = DataType.Int;
        }

        internal override PartialTensor InferPartialTensor(PartialTensor[] partialTensors, ShapeInferenceContext ctx)
        {
            var outputSymbolicShape = partialTensors[0].ToSymbolicTensorShape();
            ctx.AddShape(name, outputSymbolicShape);
            if (outputSymbolicShape.IsFullyKnown() && outputSymbolicShape.rank <= 1 && dataType == DataType.Int)
                return PartialTensor.ConstantOfShape(outputSymbolicShape.ToTensorShape(), intValue);
            return PartialTensor.Unknown;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.FromShape(inputShapes[0]);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            TensorShape shape = new TensorShape((inputTensors[0] as TensorInt).ToReadOnlyArray());
            if (dataType == DataType.Int)
                return ctx.ops.ConstantOfShape(shape, intValue);
            else
                return ctx.ops.ConstantOfShape(shape, floatValue);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"ConstantOfShape{dataType.ToString()} - name: {name}, value: {floatValue}";
        }

        internal override string profilerTag => "ConstantOfShape";
    }

    /// <summary>
    /// Represents a `OneHot` layer. This generates a one-hot tensor with a given `depth`, `indices` and `values`.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1, 2)]
    public class OneHot : Layer
    {
        /// <summary>
        /// The axis along which the layer adds the one-hot representation.
        /// </summary>
        public int axis;

        /// <summary>
        /// Initializes and returns an instance of `OneHot` layer with a float value.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="indices">The name to use for the indices tensor of the layer.</param>
        /// <param name="depth">The name to use for the scalar depth tensor of the layer.</param>
        /// <param name="values">The name to use for the two-element off/on values tensor of the layer.</param>
        /// <param name="axis">The axis along which the layer adds the one-hot representation.</param>
        public OneHot(string name, string indices, string depth, string values, int axis)
        {
            this.name = name;
            inputs = new[] { indices, depth, values };
            this.axis = axis;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            var depth = ctx.GetPartialTensor(inputs[1]);
            return SymbolicInference.OneHot(inputShapes[0], axis, depth);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var values = inputTensors[2] as TensorInt;
            return ctx.ops.OneHot(inputTensors[0] as TensorInt, axis, (inputTensors[1] as TensorInt)[0], values[0], values[1]);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "OneHot";
    }

    /// <summary>
    /// Represents a `Range` layer. This generates a 1D output tensor where the values form an arithmetic progression defined by the `start`, `limit` and `delta` scalar input tensors.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(0, 1, 2)]
    public class Range : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Range` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="start">The name to use for the scalar start value tensor of the layer.</param>
        /// <param name="limit">The name to use for the scalar limit value tensor of the layer.</param>
        /// <param name="delta">The name to use for the scalar delta value tensor of the layer.</param>
        public Range(string name, string start, string limit, string delta)
        {
            this.name = name;
            this.inputs = new[] { start, limit, delta };
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            var start = ctx.GetPartialTensor(inputs[0]);
            var limit = ctx.GetPartialTensor(inputs[1]);
            var delta = ctx.GetPartialTensor(inputs[2]);

            inputShapes[0].DeclareRank(0);
            inputShapes[1].DeclareRank(0);
            inputShapes[2].DeclareRank(0);

            var shape = SymbolicTensorShape.UnknownOfRank(1);

            if (start[0] == 0 && delta[0] == 1)
                shape[0] = limit[0].ToSymbolicTensorDim();

            return shape;
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (inputTensors[0] is TensorInt)
            {
                int start = (inputTensors[0] as TensorInt)[0];
                int limit = (inputTensors[1] as TensorInt)[0];
                int delta = (inputTensors[2] as TensorInt)[0];
                return ctx.ops.Range(start, limit, delta);
            }
            else
            {
                float start = (inputTensors[0] as TensorFloat)[0];
                float limit = (inputTensors[1] as TensorFloat)[0];
                float delta = (inputTensors[2] as TensorFloat)[0];
                return ctx.ops.Range(start, limit, delta);
            }
        }

        internal override string profilerTag => "Range";
    }
}
