using System;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a `Shape` layer. This computes the shape of an input tensor as a 1D `TensorInt`.
    /// </summary>
    [Optimization.CPUFallback.NoDataDependencyInputs(0)]
    class Shape : Layer
    {
        public int start;
        public int end;

        public Shape(string output, string input, int start = 0, int end = TensorShape.maxRank)
            : base(new[] { output }, new[] { input })
        {
            this.start = start;
            this.end = end;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            if (start == end)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, new SymbolicTensorShape(SymbolicTensorDim.Zero)));
                return;
            }

            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;

            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, SymbolicTensorShape.UnknownOfRank(1)));
                return;
            }

            var startX = start < 0 ? start + shapeX.rank : start;
            var endX = end < 0 ? end + shapeX.rank : end;
            startX = Mathf.Clamp(startX, 0, shapeX.rank);
            endX = Mathf.Clamp(endX, 0, shapeX.rank);

            Logger.AssertIsTrue(endX >= startX, "PartialTensorFromSymbolicShape.InputError: start value cannot be greater than end value for shape slicing");

            var tensorOut = new PartialTensor(DataType.Int, new SymbolicTensorShape(endX - startX));
            for (var i = startX; i < endX; i++)
            {
                tensorOut[i - startX] = (PartialTensorElement)shapeX[i];
            }

            ctx.AddPartialTensor(outputs[0], tensorOut);
        }

        public override void Execute(ExecutionContext ctx)
        {
            var shapeX = ctx.storage.GetTensor(inputs[0]).shape;
            var startX = start < 0 ? start + shapeX.rank : start;
            var endX = end < 0 ? end + shapeX.rank : end;
            startX = Mathf.Clamp(startX, 0, shapeX.rank);
            endX = Mathf.Clamp(endX, 0, shapeX.rank);

            Logger.AssertIsTrue(endX >= startX, "Shape.InputError: start value cannot be greater than end value for shape slicing");
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(endX - startX), DataType.Int, BackendType.CPU) as TensorInt;
            BurstTensorData.Pin(O);
            for (var i = startX; i < endX; i++)
                O.SetItem<int>(i - startX, shapeX[i]);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, start: {start}, end: {end}";
        }

        internal override string profilerTag => "Shape";
    }

    /// <summary>
    /// Represents a `Size` layer. This computes the number of elements of an input tensor as a scalar `TensorInt`.
    /// </summary>
    [Optimization.CPUFallback.NoDataDependencyInputs(0)]
    class Size : Layer
    {
        public Size(string output, string input)
            : base(new[] { output }, new[] { input }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, new SymbolicTensorShape())
            {
                [0] = (PartialTensorElement)X.shape.Length()
            });
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(), DataType.Int, ctx.backend.backendType) as TensorInt;
            BurstTensorData.Pin(O);
            O[0] = X.shape.length;
        }

        internal override string profilerTag => "Size";
    }
}
