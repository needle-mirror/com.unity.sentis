using System;
using Unity.Profiling;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a `Shape` layer. This computes the shape of an input tensor as a 1D `Tensor<int>`.
    /// </summary>
    class Shape : Layer
    {
        static readonly string k_OpName = "Shape";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public int start;
        public int end;

        public Shape(int output, int input, int start = 0, int end = TensorShape.maxRank)
            : base(new[] { output }, new[] { input })
        {
            this.start = start;
            this.end = end;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            if (start == end)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, new DynamicTensorShape(DynamicTensorDim.Zero)));
                return;
            }

            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;

            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, DynamicTensorShape.DynamicOfRank(1)));
                return;
            }

            var startX = start < 0 ? start + shapeX.rank : start;
            var endX = end < 0 ? end + shapeX.rank : end;
            startX = Mathf.Clamp(startX, 0, shapeX.rank);
            endX = Mathf.Clamp(endX, 0, shapeX.rank);

            Logger.AssertIsTrue(endX >= startX, "PartialTensorFromSymbolicShape.InputError: start value cannot be greater than end value for shape slicing");

            var tensorOut = new PartialTensor(DataType.Int, new DynamicTensorShape(endX - startX));
            for (var i = startX; i < endX; i++)
            {
                tensorOut[i - startX] = (PartialTensorElement)shapeX[i];
            }

            ctx.AddPartialTensor(outputs[0], tensorOut);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var shapeX = ctx.storage.GetTensorShape(inputs[0]);
            var startX = start < 0 ? start + shapeX.rank : start;
            var endX = end < 0 ? end + shapeX.rank : end;
            startX = Mathf.Clamp(startX, 0, shapeX.rank);
            endX = Mathf.Clamp(endX, 0, shapeX.rank);

            Logger.AssertIsTrue(endX >= startX, "Shape.InputError: start value cannot be greater than end value for shape slicing");
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(endX - startX), DataType.Int, BackendType.CPU) as Tensor<int>;
            O.CompleteAllPendingOperations(); // TODO is the because allocator might return a pending tensor
            for (var i = startX; i < endX; i++)
                O.SetItem(i - startX, shapeX[i]);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, start: {start}, end: {end}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `Size` layer. This computes the number of elements of an input tensor as a scalar `Tensor<int>`.
    /// </summary>
    class Size : Layer
    {
        static readonly string k_OpName = "Size";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Size(int output, int input)
            : base(new[] { output }, new[] { input }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, new DynamicTensorShape())
            {
                [0] = (PartialTensorElement)X.shape.Length()
            });
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var shapeX = ctx.storage.GetTensorShape(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(), DataType.Int, BackendType.CPU) as Tensor<int>;
            O.CompleteAllPendingOperations(); // TODO is the because allocator might return a pending tensor
            O[0] = shapeX.length;
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }
}
