using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a `ConstantOfShape` layer. This generates a tensor with the shape given by the `input` tensor and filled with a given value.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(0)]
    class ConstantOfShape : Layer
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
            this.index = name;
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
            this.index = name;
            this.inputs = new[] { input };
            this.intValue = value;
            this.dataType = DataType.Int;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var shape = ctx.GetPartialTensor(inputs[0]).ToSymbolicTensorShape();
            var tensorOut = new PartialTensor(dataType, shape);
            if (!tensorOut.isPartiallyKnown)
            {
                ctx.AddPartialTensor(index, tensorOut);
                return;
            }

            for (var i = 0; i < tensorOut.length; i++)
            {
                tensorOut[i] = dataType == DataType.Float ? PartialTensorElement.FloatValue(floatValue) : PartialTensorElement.IntValue(intValue);
            }

            ctx.AddPartialTensor(index, tensorOut);
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            TensorShape shape = new TensorShape(ctx.vars.GetTensor(inputs[0]).ToReadOnlySpan<int>());
            var O = ctx.vars.AllocateTensorAndStore(index, shape, dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (dataType == DataType.Int)
                ctx.backend.MemSet(O as TensorInt, intValue);
            else
                ctx.backend.MemSet(O as TensorFloat, floatValue);
            return;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"ConstantOfShape{dataType.ToString()} - name: {index}, value: {floatValue}";
        }

        internal override string profilerTag => "ConstantOfShape";
    }

    /// <summary>
    /// Represents a `OneHot` layer. This generates a one-hot tensor with a given `depth`, `indices` and `values`.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1, 2)]
    class OneHot : Layer
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
            this.index = name;
            inputs = new[] { indices, depth, values };
            this.axis = axis;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var values = ctx.GetPartialTensor(inputs[2]);
            var shapeX = X.shape;
            var dataType = values.dataType;
            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(index, new PartialTensor(dataType));
                return;
            }

            var shapeOut = shapeX.Unsqueeze(axis);
            shapeOut[axis] = (SymbolicTensorDim)ctx.GetPartialTensor(inputs[1])[0];

            ctx.AddPartialTensor(index, new PartialTensor(dataType, shapeOut));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var indices = ctx.vars.GetTensor(inputs[0]) as TensorInt;
            var depth = ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>()[0];
            var values = ctx.vars.GetTensor(inputs[2]);
            var O = ctx.vars.AllocateTensorAndStore(index, ShapeInference.OneHot(indices.shape, axis, depth), values.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (values.dataType == DataType.Int)
            {
                var valuesi = values.ToReadOnlySpan<int>();
                ctx.backend.OneHot(indices, O as TensorInt, axis, depth, valuesi[0], valuesi[1]);
            }
            else
            {
                var valuesf = values.ToReadOnlySpan<float>();
                ctx.backend.OneHot(indices, O as TensorFloat, axis, depth, valuesf[0], valuesf[1]);
            }
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
    class Range : Layer
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
            this.index = name;
            this.inputs = new[] { start, limit, delta };
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var start = ctx.GetPartialTensor(inputs[0]);
            var limit = ctx.GetPartialTensor(inputs[1]);
            var delta = ctx.GetPartialTensor(inputs[2]);

            start.shape.DeclareRank(0);
            limit.shape.DeclareRank(0);
            delta.shape.DeclareRank(0);

            var shape = SymbolicTensorShape.UnknownOfRank(1);

            if (start[0] == 0 && delta[0] == 1)
                shape[0] = (SymbolicTensorDim)limit[0];

            ctx.AddPartialTensor(index, new PartialTensor(start.dataType, shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var start = ctx.vars.GetTensor(inputs[0]);
            var limit = ctx.vars.GetTensor(inputs[1]);
            var delta = ctx.vars.GetTensor(inputs[2]);

            if (start is TensorInt)
            {
                int starti = start.ToReadOnlySpan<int>()[0];
                int limiti = limit.ToReadOnlySpan<int>()[0];
                int deltai = delta.ToReadOnlySpan<int>()[0];
                var O = ctx.vars.AllocateTensorAndStore(index, ShapeInference.Range(starti, limiti, deltai), DataType.Int, ctx.backend.backendType) as TensorInt;
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Range(O, starti, deltai);
            }
            else
            {
                float startf = start.ToReadOnlySpan<float>()[0];
                float limitf = limit.ToReadOnlySpan<float>()[0];
                float deltaf = delta.ToReadOnlySpan<float>()[0];
                var O = ctx.vars.AllocateTensorAndStore(index, ShapeInference.Range(startf, limitf, deltaf), DataType.Float, ctx.backend.backendType) as TensorFloat;
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Range(O, startf, deltaf);
            }
        }

        internal override string profilerTag => "Range";
    }
}
