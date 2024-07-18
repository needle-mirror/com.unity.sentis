using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a `ConstantOfShape` layer. This generates a tensor with the shape given by the `input` tensor and filled with a given value.
    /// </summary>
    class ConstantOfShape : Layer
    {
        public DataType dataType;
        public float floatValue;
        public int intValue;

        public ConstantOfShape(int output, int input, int value)
            : base(new[] { output }, new[] { input })
        {
            dataType = DataType.Int;
            intValue = value;
        }

        public ConstantOfShape(int output, int input, float value)
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
            TensorShape shape = new TensorShape(ctx.storage.GetInts(inputs[0]));
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
    class OneHot : Layer
    {
        public int axis;

        public OneHot(int output, int indices, int depth, int values, int axis)
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
            var depth = ctx.storage.GetInt(inputs[1]);
            var dataType = ctx.storage.GetDataType(inputs[2]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.OneHot(indices.shape, axis, depth), dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (dataType == DataType.Int)
            {
                var valuesi = ctx.storage.GetInts(inputs[2]);
                ctx.backend.OneHot(indices, O as TensorInt, axis, depth, valuesi[0], valuesi[1]);
            }
            else
            {
                var valuesf = ctx.storage.GetFloats(inputs[2]);
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
    class Range : Layer
    {
        public Range(int output, int start, int limit, int delta)
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
            if (ctx.storage.GetDataType(inputs[0]) == DataType.Int)
            {
                int starti = ctx.storage.GetInt(inputs[0]);
                int limiti = ctx.storage.GetInt(inputs[1]);
                int deltai = ctx.storage.GetInt(inputs[2]);
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Range(starti, limiti, deltai), DataType.Int, ctx.backend.backendType) as TensorInt;
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Range(O, starti, deltai);
            }
            else
            {
                float startf = ctx.storage.GetFloat(inputs[0]);
                float limitf = ctx.storage.GetFloat(inputs[1]);
                float deltaf = ctx.storage.GetFloat(inputs[2]);
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Range(startf, limitf, deltaf), DataType.Float, ctx.backend.backendType) as TensorFloat;
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Range(O, startf, deltaf);
            }
        }

        internal override string profilerTag => "Range";
    }
}
