using System;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Options for the reduction operation to use in a scatter layer.
    /// </summary>
    public enum ScatterReductionMode
    {
        /// <summary>
        /// Use no reduction.
        /// </summary>
        None,
        /// <summary>
        /// Use the addition operator when reducing.
        /// </summary>
        Add,
        /// <summary>
        /// Use the multiplication operator when reducing.
        /// </summary>
        Mul,
    }

    /// <summary>
    /// Represents a reduction which calculates indices.
    /// </summary>
    abstract class ArgReduce : Layer
    {
        public int axis;
        public bool keepdims;
        public bool selectLastIndex;

        protected ArgReduce(string output, string input, int axis, bool keepdims = true, bool selectLastIndex = false)
            : base(new[] { output }, new[] { input })
        {
            this.axis = axis;
            this.keepdims = keepdims;
            this.selectLastIndex = selectLastIndex;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;
            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int));
                return;
            }

            var reducedShape = new SymbolicTensorShape(shapeX);

            // reducing on a zero axis will result in a zero rather than a one
            if (shapeX[axis].isValue)
                reducedShape[axis] = shapeX[axis].value == 0 ? SymbolicTensorDim.Zero : SymbolicTensorDim.One;
            else
                reducedShape[axis] = SymbolicTensorDim.Unknown;

            var shapeOut = !keepdims ? reducedShape.Squeeze(axis) : reducedShape;
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, shapeOut));
        }
    }

    /// <summary>
    /// Represents an `ArgMax` layer. This computes the indices of the maximum elements of the input tensor along a given axis.
    /// </summary>
    class ArgMax : ArgReduce
    {
        public ArgMax(string output, string input, int axis, bool keepdims = true, bool selectLastIndex = false)
            : base(output, input, axis, keepdims, selectLastIndex) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var shapeO = X.shape.Reduce(axis, keepdims);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shapeO, DataType.Int, ctx.backend.backendType) as TensorInt;
            if (O.shape.HasZeroDims())
                return;
            if (X is TensorInt)
                ctx.backend.ArgMax(X as TensorInt, O, axis, keepdims, selectLastIndex);
            else
                ctx.backend.ArgMax(X as TensorFloat, O, axis, keepdims, selectLastIndex);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}, keepdims: {keepdims}, selectLastIndex: {selectLastIndex}";
        }

        internal override string profilerTag => "ArgMax";
    }

    /// <summary>
    /// Represents an `ArgMin` layer. This computes the indices of the minimum elements of the input tensor along a given axis.
    /// </summary>
    class ArgMin : ArgReduce
    {
        public ArgMin(string output, string input, int axis, bool keepdims = true, bool selectLastIndex = false)
            : base(output, input, axis, keepdims, selectLastIndex) { }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var shapeO = X.shape.Reduce(axis, keepdims);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shapeO, DataType.Int, ctx.backend.backendType) as TensorInt;
            if (O.shape.HasZeroDims())
                return;
            if (X is TensorInt)
                ctx.backend.ArgMin(X as TensorInt, O, axis, keepdims, selectLastIndex);
            else
                ctx.backend.ArgMin(X as TensorFloat, O, axis, keepdims, selectLastIndex);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}, keepdims: {keepdims}, selectLastIndex: {selectLastIndex}";
        }

        internal override string profilerTag => "ArgMin";
    }

    /// <summary>
    /// Represents a `Gather` layer. This takes values from the input tensor indexed by the indices tensor along a given axis and concatenates them.
    /// </summary>
    class Gather : Layer
    {
        public int axis;

        public Gather(string output, string input, string indices, int axis)
            : base(new[] { output }, new[] { input, indices })
        {
            this.axis = axis;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var input = ctx.GetPartialTensor(inputs[0]);
            var dataType = input.dataType;
            var indices = ctx.GetPartialTensor(inputs[1]);
            if (dataType == DataType.Int && axis == 0 && input.isPartiallyKnown && indices.isPartiallyKnown && input.shape.rank == 1 && indices.shape.rank <= 1)
            {
                var tensorOut = new PartialTensor(dataType, indices.shape);
                for (var i = 0; i < indices.length; i++)
                {
                    var index = indices[i];
                    index = index < 0 ? index + input.length : index;
                    tensorOut[i] = input[index];
                }

                ctx.AddPartialTensor(outputs[0], tensorOut);
                return;
            }

            var shapeX = input.shape;
            var shapeIndices = indices.shape;
            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType));
                return;
            }

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            if (!shapeIndices.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType));
                return;
            }

            var axisX = shapeX.Axis(axis);

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank - 1 + shapeIndices.rank);

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (i < axisX)
                    shapeOut[i] = shapeX[i];
                else if (i < axisX + shapeIndices.rank)
                    shapeOut[i] = shapeIndices[i - axisX];
                else
                    shapeOut[i] = shapeX[i - shapeOut.rank];
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var indices = ctx.storage.GetTensor(inputs[1]) as TensorInt;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Gather(X.shape, indices.shape, axis), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Gather(X, indices, O, axis);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "Gather";
    }

    /// <summary>
    /// Represents a `GatherElements` layer. This takes values from the input tensor indexed by the `indices` tensor along a given axis.
    /// </summary>
    class GatherElements : Layer
    {
        public int axis;

        public GatherElements(string output, string input, string indices, int axis)
            : base(new[] { output }, new[] { input, indices })
        {
            this.axis = axis;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var shapeX = X.shape;
            var shapeIndices = ctx.GetPartialTensor(inputs[1]).shape;
            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            if (shapeX.hasRank)
            {
                shapeX.Axis(axis);
                shapeIndices.DeclareRank(shapeX.rank);
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, shapeIndices));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var indices = ctx.storage.GetTensor(inputs[1]) as TensorInt;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], indices.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.GatherElements(X, indices, O, axis);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "GatherElements";
    }

    /// <summary>
    /// Represents a `GatherND` layer. This takes slices of values from the batched input tensor indexed by the `indices` tensor.
    /// </summary>
    class GatherND : Layer
    {
        public int batchDims;

        public GatherND(string output, string input, string indices, int batchDims)
            : base(new[] { output }, new[] { input, indices })
        {
            this.batchDims = batchDims;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var indices = ctx.GetPartialTensor(inputs[1]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeIndices = indices.shape;
            // from https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= batchDims : true, "RankError: incorrect rank, expecting at least {0}, got {1}", batchDims, shapeX.rank);
            Logger.AssertIsTrue(shapeIndices.hasRank ? shapeIndices.rank >= batchDims : true, "RankError: incorrect rank, expecting at least {0}, got {1}", batchDims, shapeIndices.rank);

            if (!shapeX.hasRank || !shapeIndices.hasRank || !shapeIndices[-1].isValue)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType));
                return;
            }

            Logger.AssertIsTrue(batchDims + shapeIndices[-1].value <= shapeX.rank, "GatherND.InputError: last indices dim too large");
            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank + shapeIndices.rank - shapeIndices[-1].value - 1 - batchDims);
            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (i < batchDims)
                    shapeOut[i] = SymbolicTensorDim.MaxDefinedDim(shapeX[i], shapeIndices[i]);
                else if (i < shapeIndices.rank - 1)
                    shapeOut[i] = shapeIndices[i];
                else
                    shapeOut[i] = shapeX[i - shapeOut.rank];
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var indices = ctx.storage.GetTensor(inputs[1]) as TensorInt;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.GatherND(X.shape, indices.shape, batchDims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.GatherND(X, indices, O, batchDims);
        }

        internal override string profilerTag => "GatherND";
    }

    /// <summary>
    /// Represents a `NonZero` layer. This returns the indices of the elements of the input tensor that are not zero.
    /// </summary>
    class NonZero : Layer
    {
        public NonZero(string output, string input)
            : base(new[] { output }, new[] { input }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var shapeX = X.shape;
            var shape = !shapeX.hasRank ? SymbolicTensorShape.UnknownOfRank(2) : new SymbolicTensorShape(SymbolicTensorDim.Int(shapeX.rank), SymbolicTensorDim.Unknown);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            Logger.AssertIsFalse(ctx.backend is GPUCommandBufferBackend, "BackendTypeError: GPUCommandBuffer is not supported on the NonZero layer");

            var X = ctx.storage.GetTensor(inputs[0]);

            if (X is TensorInt)
            {
                BurstTensorData.Pin(X);
                int nbNonZeroIndices = 0;
                var end = X.shape.length;
                for (int i = 0; i < end; ++i)
                {
                    if (X.GetItem<int>(i) != 0)
                        nbNonZeroIndices += 1;
                }

                var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(X.shape.rank, nbNonZeroIndices), DataType.Int, ctx.backend.backendType) as TensorInt;

                BurstTensorData.Pin(O);
                int nonZeroIndicesIdx = 0;
                for (var it = new TensorNDIterator(X.shape); it.HasNext(); it.MoveNext())
                {
                    if (X.GetItem<int>(it.index) != 0)
                    {
                        for (int i = 0; i < X.shape.rank; i++)
                            O.SetItem<int>(i * nbNonZeroIndices + nonZeroIndicesIdx, it[i]);
                        nonZeroIndicesIdx++;
                    }
                }
            }
            else
            {
                BurstTensorData.Pin(X);
                int nbNonZeroIndices = 0;
                var end = X.shape.length;
                for (int i = 0; i < end; ++i)
                {
                    if (X.GetItem<float>(i) != 0.0f)
                        nbNonZeroIndices += 1;
                }

                var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(X.shape.rank, nbNonZeroIndices), DataType.Int, ctx.backend.backendType) as TensorInt;

                BurstTensorData.Pin(O);
                int nonZeroIndicesIdx = 0;
                for (var it = new TensorNDIterator(X.shape); it.HasNext(); it.MoveNext())
                {
                    if (X.GetItem<float>(it.index) != 0.0f)
                    {
                        for (int i = 0; i < X.shape.rank; i++)
                            O.SetItem<int>(i * nbNonZeroIndices + nonZeroIndicesIdx, it[i]);
                        nonZeroIndicesIdx++;
                    }
                }
            }
        }

        internal override string profilerTag => "NonZero";
    }

    /// <summary>
    /// Represents a `ScatterElements` layer. This copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor along a given axis.
    ///
    /// `ScatterElements` updates the values depending on the reduction mode used.
    /// </summary>
    class ScatterElements : Layer
    {
        public int axis;
        public ScatterReductionMode reduction;

        public ScatterElements(string output, string input, string indices, string updates, int axis, ScatterReductionMode reduction)
            : base(new[] { output }, new[] { input, indices, updates })
        {
            this.axis = axis;
            this.reduction = reduction;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeIndices = ctx.GetPartialTensor(inputs[1]).shape;
            var shapeUpdates = ctx.GetPartialTensor(inputs[2]).shape;

            if (!shapeX.hasRank && !shapeIndices.hasRank && !shapeUpdates.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType));
                return;
            }

            if (!shapeX.hasRank && shapeIndices.hasRank)
                shapeX = SymbolicTensorShape.UnknownOfRank(shapeIndices.rank);

            if (!shapeX.hasRank && shapeUpdates.hasRank)
                shapeX = SymbolicTensorShape.UnknownOfRank(shapeUpdates.rank);

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            shapeIndices.DeclareRank(shapeX.rank);
            shapeUpdates.DeclareRank(shapeX.rank);

            // throw error if axis incorrect
            shapeX.Axis(axis);

            // throw error if indices and updates don't match
            for (var i = 0; i < shapeIndices.rank; i++)
            {
                SymbolicTensorDim.MaxDefinedDim(shapeIndices[i], shapeUpdates[i]);
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeX));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var indices = ctx.storage.GetTensor(inputs[1]) as TensorInt;
            var updates = ctx.storage.GetTensor(inputs[2]);
            Logger.AssertIsTrue(indices.shape == updates.shape, "ScatterElements.InputError indices and updates must have same shape");
            if (indices.shape.HasZeroDims())
                ctx.backend.MemCopy(X, O);
            else
                ctx.backend.ScatterElements(X, indices, updates, O, axis, reduction);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}, reduction: {reduction}";
        }

        internal override string profilerTag => "ScatterElements";
    }

    /// <summary>
    /// Represents a `ScatterND` layer. This copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor.
    ///
    /// `ScatterND` updates the values depending on the reduction mode used.
    /// </summary>
    class ScatterND : Layer
    {
        public ScatterReductionMode reduction;

        public ScatterND(string output, string input, string indices, string updates, ScatterReductionMode reduction)
            : base(new[] { output }, new[] { input, indices, updates })
        {
            this.reduction = reduction;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeIndices = ctx.GetPartialTensor(inputs[1]).shape;
            var shapeUpdates = ctx.GetPartialTensor(inputs[2]).shape;

            Logger.AssertIsTrue(shapeIndices.hasRank ? shapeIndices.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", shapeIndices.rank, 1);

            if (shapeIndices.hasRank && shapeUpdates.hasRank && shapeIndices[-1].isValue)
                shapeX.DeclareRank(shapeUpdates.rank - (shapeIndices.rank - shapeIndices[-1].value - 1));

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeX));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var indices = ctx.storage.GetTensor(inputs[1]) as TensorInt;
            if (indices.shape.HasZeroDims())
            {
                ctx.backend.MemCopy(X, O);
                return;
            }

            if (X is TensorInt)
                ctx.backend.ScatterND(X as TensorInt, ctx.storage.GetTensor(inputs[1]) as TensorInt, ctx.storage.GetTensor(inputs[2]) as TensorInt, O as TensorInt, reduction);
            else
                ctx.backend.ScatterND(X as TensorFloat, ctx.storage.GetTensor(inputs[1]) as TensorInt, ctx.storage.GetTensor(inputs[2]) as TensorFloat, O as TensorFloat, reduction);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, reduction: {reduction}";
        }

        internal override string profilerTag => "ScatterND";
    }

    /// <summary>
    /// Represents a `TopK` layer. This calculates the top-K largest or smallest elements of an input tensor along a given axis.
    ///
    /// This layer calculates both the values tensor of the top-K elements and the indices tensor of the top-K elements as outputs.
    /// </summary>
    [Optimization.CPUFallback.CPUReadInputs(1)]
    class TopK : Layer
    {
        public int axis;
        public bool largest;
        public bool sorted;

        public TopK(string values, string indices, string input, string k, int axis, bool largest, bool sorted)
            : base(new[] { values, indices }, new[] { input, k })
        {
            this.axis = axis;
            this.largest = largest;
            this.sorted = sorted;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var K = ctx.GetPartialTensor(inputs[1]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[1], new PartialTensor(DataType.Int));
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType));
                return;
            }

            var shapeK = K.shape;
            shapeK.DeclareRank(1);
            Logger.AssertIsFalse(shapeK[0] != 1, "TopK.InputError: k must be a single value");

            var shapeOut = new SymbolicTensorShape(shapeX);

            var axisX = shapeX.Axis(axis);

            shapeOut[axisX] = (SymbolicTensorDim)K[0];

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
            ctx.AddPartialTensor(outputs[1], new PartialTensor(DataType.Int, shapeOut));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var k = ctx.storage.GetTensor(inputs[1]).ToReadOnlySpan<int>()[0];
            var outputShape = new TensorShape(X.shape);
            outputShape[axis] = k;

            var values = ctx.storage.AllocateTensorAndStore(outputs[0], outputShape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            var indices = ctx.storage.AllocateTensorAndStore(outputs[1], outputShape, DataType.Int, ctx.backend.backendType) as TensorInt;
            if (outputShape.HasZeroDims())
                return;
            ctx.backend.TopK(X as TensorFloat, values, indices, k, axis, largest);
        }

        internal override string profilerTag => "TopK";
    }
}
