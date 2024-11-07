using System;
using Unity.Collections;
using Unity.Profiling;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Options for the reduction operation to use in a scatter layer.
    /// </summary>
    enum ScatterReductionMode
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

        protected ArgReduce(int output, int input, int axis, bool keepdims = true, bool selectLastIndex = false)
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

            var reducedShape = new DynamicTensorShape(shapeX);

            // reducing on a zero axis will result in a zero rather than a one
            if (shapeX[axis].isValue)
                reducedShape[axis] = shapeX[axis].value == 0 ? DynamicTensorDim.Zero : DynamicTensorDim.One;
            else
                reducedShape[axis] = DynamicTensorDim.Unknown;

            var shapeOut = !keepdims ? reducedShape.Squeeze(axis) : reducedShape;
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, shapeOut));
        }
    }

    /// <summary>
    /// Represents an `ArgMax` layer. This computes the indices of the maximum elements of the input tensor along a given axis.
    /// </summary>
    class ArgMax : ArgReduce
    {
        static readonly string k_OpName = "ArgMax";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public ArgMax(int output, int input, int axis, bool keepdims = true, bool selectLastIndex = false)
            : base(output, input, axis, keepdims, selectLastIndex) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var shapeO = X.shape.Reduce(axis, keepdims);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shapeO, DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
                ctx.backend.ArgMax(X as Tensor<int>, O, axis, selectLastIndex);
            else
                ctx.backend.ArgMax(X as Tensor<float>, O, axis, selectLastIndex);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}, keepdims: {keepdims}, selectLastIndex: {selectLastIndex}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an `ArgMin` layer. This computes the indices of the minimum elements of the input tensor along a given axis.
    /// </summary>
    class ArgMin : ArgReduce
    {
        static readonly string k_OpName = "ArgMin";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public ArgMin(int output, int input, int axis, bool keepdims = true, bool selectLastIndex = false)
            : base(output, input, axis, keepdims, selectLastIndex) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var shapeO = X.shape.Reduce(axis, keepdims);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shapeO, DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
                ctx.backend.ArgMin(X as Tensor<int>, O, axis, selectLastIndex);
            else
                ctx.backend.ArgMin(X as Tensor<float>, O, axis, selectLastIndex);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}, keepdims: {keepdims}, selectLastIndex: {selectLastIndex}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `Gather` layer. This takes values from the input tensor indexed by the indices tensor along a given axis and concatenates them.
    /// </summary>
    class Gather : Layer
    {
        static readonly string k_OpName = "Gather";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public int axis;

        public Gather(int output, int input, int indices, int axis)
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

            var shapeOut = DynamicTensorShape.DynamicOfRank(shapeX.rank - 1 + shapeIndices.rank);

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

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var indices = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Gather(X.shape, indices.shape, axis), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Gather(X, indices, O, axis);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `GatherElements` layer. This takes values from the input tensor indexed by the `indices` tensor along a given axis.
    /// </summary>
    class GatherElements : Layer
    {
        static readonly string k_OpName = "GatherElements";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public int axis;

        public GatherElements(int output, int input, int indices, int axis)
            : base(new[] { output }, new[] { input, indices })
        {
            this.axis = axis;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var shapeX = X.shape;
            var shapeIndices = ctx.GetPartialTensor(inputs[1]).shape;
            if (shapeX.hasRank)
                Logger.AssertIsTrue(shapeX.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            if (shapeX.hasRank)
            {
                shapeX.Axis(axis);
                shapeIndices.DeclareRank(shapeX.rank);
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, shapeIndices));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var indices = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], indices.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.GatherElements(X, indices, O, axis);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `GatherND` layer. This takes slices of values from the batched input tensor indexed by the `indices` tensor.
    /// </summary>
    class GatherND : Layer
    {
        static readonly string k_OpName = "GatherND";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public int batchDims;

        public GatherND(int output, int input, int indices, int batchDims)
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
            if (shapeX.hasRank)
                Logger.AssertIsTrue(shapeX.rank >= batchDims, "RankError: incorrect rank, expecting at least {0}, got {1}", batchDims, shapeX.rank);
            if (shapeIndices.hasRank)
                Logger.AssertIsTrue(shapeIndices.rank >= batchDims, "RankError: incorrect rank, expecting at least {0}, got {1}", batchDims, shapeIndices.rank);

            if (!shapeX.hasRank || !shapeIndices.hasRank || !shapeIndices[-1].isValue)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType));
                return;
            }

            Logger.AssertIsTrue(batchDims + shapeIndices[-1].value <= shapeX.rank, "GatherND.InputError: last indices dim too large");
            var shapeOut = DynamicTensorShape.DynamicOfRank(shapeX.rank + shapeIndices.rank - shapeIndices[-1].value - 1 - batchDims);
            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (i < batchDims)
                    shapeOut[i] = DynamicTensorDim.MaxDefinedDim(shapeX[i], shapeIndices[i]);
                else if (i < shapeIndices.rank - 1)
                    shapeOut[i] = shapeIndices[i];
                else
                    shapeOut[i] = shapeX[i - shapeOut.rank];
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var indices = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.GatherND(X.shape, indices.shape, batchDims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.GatherND(X, indices, O, batchDims);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `NonZero` layer. This returns the indices of the elements of the input tensor that are not zero.
    /// </summary>
    class NonZero : Layer
    {
        static readonly string k_OpName = "NonZero";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public NonZero(int output, int input)
            : base(new[] { output }, new[] { input }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var shapeX = X.shape;
            var shape = !shapeX.hasRank ? DynamicTensorShape.DynamicOfRank(2) : new DynamicTensorShape(DynamicTensorDim.Int(shapeX.rank), DynamicTensorDim.Unknown);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            // need to download, if gpucompute need to execute commandbuffer and flush.
            if (ctx.backend is GPUComputeBackend gpuBackend)
                gpuBackend.ExecuteCommandBufferAndClear();

            // pixel we don't know which dim to pin
            var outputBackendType = ctx.backend.backendType;
            if (outputBackendType == BackendType.GPUPixel)
                outputBackendType = BackendType.CPU;

            if (X is Tensor<int>)
            {
                // reduce(notequal(X, 0)) to get nbNonZeroIndices
                var arrayX = (X as Tensor<int>).DownloadToNativeArray();
                int nbNonZeroIndices = 0;
                for (int i = 0; i < X.shape.length; ++i)
                {
                    if (arrayX[i] != 0)
                        nbNonZeroIndices += 1;
                }

                // compact with condition mask?
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(X.shape.rank, nbNonZeroIndices), DataType.Int, outputBackendType) as Tensor<int>;
                if (O.shape.HasZeroDims())
                    return;
                var arrayO = new NativeArray<int>(O.shape.length, Allocator.Temp);

                int nonZeroIndicesIdx = 0;
                for (var it = new TensorNDIterator(X.shape); it.HasNext(); it.MoveNext())
                {
                    if (arrayX[it.index] != 0)
                    {
                        for (int i = 0; i < X.shape.rank; i++)
                            arrayO[i * nbNonZeroIndices + nonZeroIndicesIdx] = it[i];
                        nonZeroIndicesIdx++;
                    }
                }
                O.dataOnBackend.Upload(arrayO, arrayO.Length);
            }
            else
            {
                // reduce(notequal(X, 0)) to get nbNonZeroIndices
                var arrayX = (X as Tensor<float>).DownloadToNativeArray();
                int nbNonZeroIndices = 0;
                for (int i = 0; i < X.shape.length; ++i)
                {
                    if (arrayX[i] != 0)
                        nbNonZeroIndices += 1;
                }

                // compact with condition mask?
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(X.shape.rank, nbNonZeroIndices), DataType.Int, outputBackendType) as Tensor<int>;
                if (O.shape.HasZeroDims())
                    return;
                var arrayO = new NativeArray<int>(O.shape.length, Allocator.Temp);

                int nonZeroIndicesIdx = 0;
                for (var it = new TensorNDIterator(X.shape); it.HasNext(); it.MoveNext())
                {
                    if (arrayX[it.index] != 0)
                    {
                        for (int i = 0; i < X.shape.rank; i++)
                            arrayO[i * nbNonZeroIndices + nonZeroIndicesIdx] = it[i];
                        nonZeroIndicesIdx++;
                    }
                }
                O.dataOnBackend.Upload(arrayO, arrayO.Length);
            }
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `ScatterElements` layer. This copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor along a given axis.
    ///
    /// `ScatterElements` updates the values depending on the reduction mode used.
    /// </summary>
    class ScatterElements : Layer
    {
        static readonly string k_OpName = "ScatterElements";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public int axis;
        public ScatterReductionMode reduction;

        public ScatterElements(int output, int input, int indices, int updates, int axis, ScatterReductionMode reduction)
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
                shapeX = DynamicTensorShape.DynamicOfRank(shapeIndices.rank);

            if (!shapeX.hasRank && shapeUpdates.hasRank)
                shapeX = DynamicTensorShape.DynamicOfRank(shapeUpdates.rank);

            if (shapeX.hasRank)
                Logger.AssertIsTrue(shapeX.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            shapeIndices.DeclareRank(shapeX.rank);
            shapeUpdates.DeclareRank(shapeX.rank);

            // throw error if axis incorrect
            shapeX.Axis(axis);

            // throw error if indices and updates don't match
            for (var i = 0; i < shapeIndices.rank; i++)
            {
                DynamicTensorDim.MaxDefinedDim(shapeIndices[i], shapeUpdates[i]);
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeX));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var indices = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
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

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `ScatterND` layer. This copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor.
    ///
    /// `ScatterND` updates the values depending on the reduction mode used.
    /// </summary>
    class ScatterND : Layer
    {
        static readonly string k_OpName = "ScatterND";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public ScatterReductionMode reduction;

        public ScatterND(int output, int input, int indices, int updates, ScatterReductionMode reduction)
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

            if (shapeIndices.hasRank)
                Logger.AssertIsTrue(shapeIndices.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", shapeIndices.rank, 1);

            if (shapeIndices.hasRank && shapeUpdates.hasRank && shapeIndices[-1].isValue)
                shapeX.DeclareRank(shapeUpdates.rank - (shapeIndices.rank - shapeIndices[-1].value - 1));

            if (shapeX.hasRank)
                Logger.AssertIsTrue(shapeX.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeX));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var indices = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            if (indices.shape.HasZeroDims())
            {
                ctx.backend.MemCopy(X, O);
                return;
            }

            if (X is Tensor<int>)
                ctx.backend.ScatterND(X as Tensor<int>, ctx.storage.GetTensor(inputs[1]) as Tensor<int>, ctx.storage.GetTensor(inputs[2]) as Tensor<int>, O as Tensor<int>, reduction);
            else
                ctx.backend.ScatterND(X as Tensor<float>, ctx.storage.GetTensor(inputs[1]) as Tensor<int>, ctx.storage.GetTensor(inputs[2]) as Tensor<float>, O as Tensor<float>, reduction);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, reduction: {reduction}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `TopK` layer. This calculates the top-K largest or smallest elements of an input tensor along a given axis.
    ///
    /// This layer calculates both the values tensor of the top-K elements and the indices tensor of the top-K elements as outputs.
    /// </summary>
    class TopK : Layer
    {
        static readonly string k_OpName = "TopK";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public int axis;
        public bool largest;
        public bool sorted;

        public TopK(int values, int indices, int input, int k, int axis, bool largest, bool sorted)
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

            var shapeOut = new DynamicTensorShape(shapeX);

            var axisX = shapeX.Axis(axis);

            shapeOut[axisX] = (DynamicTensorDim)K[0];

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
            ctx.AddPartialTensor(outputs[1], new PartialTensor(DataType.Int, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var k = ctx.storage.GetInt(inputs[1]);
            var outputShape = new TensorShape(X.shape);
            outputShape[axis] = k;

            var values = ctx.storage.AllocateTensorAndStore(outputs[0], outputShape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            var indices = ctx.storage.AllocateTensorAndStore(outputs[1], outputShape, DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (outputShape.HasZeroDims())
                return;
            ctx.backend.TopK(X as Tensor<float>, values, indices, k, axis, largest);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }
}
