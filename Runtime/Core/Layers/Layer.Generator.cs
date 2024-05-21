using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a `ConstantOfShape` layer. This generates a tensor with the shape given by the `input` tensor and filled with a given value.
    /// </summary>
    [Optimization.CPUFallback.CPUReadInputs(0)]
    class ConstantOfShape : Layer
    {
        public DataType dataType;
        public float floatValue;
        public int intValue;

        public ConstantOfShape(string output, string input, int value)
            : base(new[] { output }, new[] { input })
        {
            dataType = DataType.Int;
            intValue = value;
        }

        public ConstantOfShape(string output, string input, float value)
            : base(new[] { output }, new[] { input })
        {
            dataType = DataType.Float;
            floatValue = value;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var shape = ctx.GetPartialTensor(inputs[0]).ToSymbolicTensorShape();
            var tensorOut = new PartialTensor(dataType, shape);
            if (!tensorOut.isPartiallyKnown)
            {
                ctx.AddPartialTensor(outputs[0], tensorOut);
                return;
            }

            for (var i = 0; i < tensorOut.length; i++)
            {
                tensorOut[i] = dataType == DataType.Float ? PartialTensorElement.FloatValue(floatValue) : PartialTensorElement.IntValue(intValue);
            }

            ctx.AddPartialTensor(outputs[0], tensorOut);
        }

        public override void Execute(ExecutionContext ctx)
        {
            TensorShape shape = new TensorShape(ctx.storage.GetTensor(inputs[0]).ToReadOnlySpan<int>());
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shape, dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (dataType == DataType.Int)
                ctx.backend.MemSet(O as TensorInt, intValue);
            else
                ctx.backend.MemSet(O as TensorFloat, floatValue);
            return;
        }

        public override string ToString()
        {
            return $"{base.ToString()}, dataType: {dataType}, floatValue: {floatValue}, intValue: {intValue}";
        }

        internal override string profilerTag => "ConstantOfShape";
    }

    /// <summary>
    /// Represents a `OneHot` layer. This generates a one-hot tensor with a given `depth`, `indices` and `values`.
    /// </summary>
    [Optimization.CPUFallback.CPUReadInputs(1, 2)]
    class OneHot : Layer
    {
        public int axis;

        public OneHot(string output, string indices, string depth, string values, int axis)
            : base(new[] { output }, new[] { indices, depth, values })
        {
            this.axis = axis;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var values = ctx.GetPartialTensor(inputs[2]);
            var shapeX = X.shape;
            var dataType = values.dataType;
            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType));
                return;
            }

            var shapeOut = shapeX.Unsqueeze(axis);
            shapeOut[axis] = (SymbolicTensorDim)ctx.GetPartialTensor(inputs[1])[0];

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var indices = ctx.storage.GetTensor(inputs[0]) as TensorInt;
            var depth = ctx.storage.GetTensor(inputs[1]).ToReadOnlySpan<int>()[0];
            var values = ctx.storage.GetTensor(inputs[2]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.OneHot(indices.shape, axis, depth), values.dataType, ctx.backend.backendType);
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

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "OneHot";
    }

    /// <summary>
    /// Represents a `Range` layer. This generates a 1D output tensor where the values form an arithmetic progression defined by the `start`, `limit` and `delta` scalar input tensors.
    /// </summary>
    [Optimization.CPUFallback.CPUReadInputs(0, 1, 2)]
    class Range : Layer
    {
        public Range(string output, string start, string limit, string delta)
            : base(new[] { output }, new[] { start, limit, delta }) { }

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

            ctx.AddPartialTensor(outputs[0], new PartialTensor(start.dataType, shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var start = ctx.storage.GetTensor(inputs[0]);
            var limit = ctx.storage.GetTensor(inputs[1]);
            var delta = ctx.storage.GetTensor(inputs[2]);

            if (start is TensorInt)
            {
                int starti = start.ToReadOnlySpan<int>()[0];
                int limiti = limit.ToReadOnlySpan<int>()[0];
                int deltai = delta.ToReadOnlySpan<int>()[0];
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Range(starti, limiti, deltai), DataType.Int, ctx.backend.backendType) as TensorInt;
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Range(O, starti, deltai);
            }
            else
            {
                float startf = start.ToReadOnlySpan<float>()[0];
                float limitf = limit.ToReadOnlySpan<float>()[0];
                float deltaf = delta.ToReadOnlySpan<float>()[0];
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Range(startf, limitf, deltaf), DataType.Float, ctx.backend.backendType) as TensorFloat;
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Range(O, startf, deltaf);
            }
        }

        internal override string profilerTag => "Range";
    }
}
