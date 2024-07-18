using System;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Options for the padding values for `Pad`.
    /// </summary>
    public enum PadMode
    {
        /// <summary>
        /// Use a constant value for the padded data.
        /// </summary>
        Constant,
        /// <summary>
        /// Use the reflection of the values of the input tensor mirrored on the first and last values along the axis. The edge values appear once in the output tensor.
        /// </summary>
        Reflect,
        /// <summary>
        /// Use the edge values of the input tensor.
        /// </summary>
        Edge,
        /// <summary>
        /// Use the reflection of the values of the input tensor mirrored half a step outside the first and last values along the axis. The edge values appear twice in the output tensor.
        /// </summary>
        Symmetric,
        /// <summary>
        /// Wrap the values of the input tensor like a torus for the padded data.
        /// </summary>
        Wrap,
    }

    /// <summary>
    /// Options for the scaling mode to use for `Resize`.
    /// </summary>
    public enum ScaleMode
    {
        /// <summary>
        /// Use the size tensor directly for the shape of the output tensor.
        /// </summary>
        Sizes,
        /// <summary>
        /// Use the scales tensor to multiply the shape of the input tensor to calculate the shape of the output tensor.
        /// </summary>
        Scales
    }

    /// <summary>
    /// Options for the interpolation mode to use for `Resize`.
    /// </summary>
    public enum InterpolationMode
    {
        /// <summary>
        /// Use the nearest element to the calculated coordinate. The exact behaviour depends on `nearestMode`.
        /// </summary>
        Nearest,
        /// <summary>
        /// Use a linear sampling of the surrounding elements to the calculated coordinate.
        /// </summary>
        Linear,
        /// <summary>
        /// Use a cubic sampling of the surrounding elements to the calculated coordinate.
        /// </summary>
        Cubic
    }

    /// <summary>
    /// Options for how to sample the nearest element in `Resize` when using `InterpolationMode.NearestMode`.
    /// </summary>
    public enum NearestMode
    {
        /// <summary>
        /// Use rounding to the nearest integer coordinate. If the fractional part equals 0.5 then round down.
        /// </summary>
        RoundPreferFloor,
        /// <summary>
        /// Use rounding to the nearest integer coordinate. If the fractional part equals 0.5 then round up.
        /// </summary>
        RoundPreferCeil,
        /// <summary>
        /// Use rounding down to the next integer coordinate less than or equal to the input coordinate.
        /// </summary>
        Floor,
        /// <summary>
        /// Use rounding up to the next integer coordinate greater than or equal to the input coordinate.
        /// </summary>
        Ceil
    }

    /// <summary>
    /// Padding mode for outside grid values.
    /// </summary>
    public enum PaddingMode
    {
        /// <summary>
        /// Use 0 for out-of-bound grid locations.
        /// </summary>
        Zeros,
        /// <summary>
        /// Use border value for out-of-bound grid locations.
        /// </summary>
        Border,
        /// <summary>
        /// Use values at locations reflected by the border for out-of-bound grid locations. Distant values are reflected multiple times until in bounds.
        /// </summary>
        Reflection
    }

    /// <summary>
    /// Options for how to transform between the coordinate in the output tensor and the coordinate in the input tensor in `Resize`.
    /// </summary>
    public enum CoordTransformMode
    {
        /// <summary>
        /// Use shifting by half a pixel before and after scaling.
        /// </summary>
        HalfPixel,
        /// <summary>
        /// Use shifting by half a pixel before and after scaling if the output length is greater than 1, otherwise use 0.
        /// </summary>
        PytorchHalfPixel,
        /// <summary>
        /// Use scaling by `length - 1` so that corner pixels align.
        /// </summary>
        AlignCorners,
        /// <summary>
        /// Use direct scaling of coordinates by the scaling factor.
        /// </summary>
        Asymmetric,
    }

    /// <summary>
    /// Options for which part of the input matrix to retain in `Trilu`.
    /// </summary>
    public enum TriluMode
    {
        /// <summary>
        /// Use retaining of the lower part of the input matrix.
        /// </summary>
        Lower = 0,
        /// <summary>
        /// Use retaining of the upper part of the input matrix.
        /// </summary>
        Upper = 1,
    }

    /// <summary>
    /// Options for the ordering of the elements in `DepthToSpace`.
    /// </summary>
    public enum DepthToSpaceMode
    {
        /// <summary>
        /// Use depth, column, row ordering where the data is arranged (by * blocksize * channels) + (bx * channels) + c.
        /// </summary>
        DepthColumnRow,
        /// <summary>
        /// Use column, row, depth ordering where the data is arranged (c * blocksize * blocksize) + (by * blocksize) + bx.
        /// </summary>
        ColumnRowDepth,
    }

    /// <summary>
    /// Represents an element-wise `Cast` layer: f(x) = (float)x or f(x) = (int)x depending on the value of `toType`.
    /// </summary>
    class Cast : Layer
    {
        public DataType toType;

        public Cast(int output, int input, DataType toType)
            : base(new[] { output }, new[] { input })
        {
            this.toType = toType;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(outputs[0], new PartialTensor(toType, ctx.GetPartialTensor(inputs[0]).shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, toType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;

            if (X.dataType == O.dataType)
                ctx.backend.MemCopy(X, O);
            else if (X.dataType == DataType.Int && O.dataType == DataType.Float)
                ctx.backend.Cast(X as TensorInt, O as TensorFloat);
            else if (X.dataType == DataType.Float && O.dataType == DataType.Int)
                ctx.backend.Cast(X as TensorFloat, O as TensorInt);
            else if (X.dataType == DataType.Short && O.dataType == DataType.Float)
                ctx.backend.Cast(X as TensorShort, O as TensorFloat);
            else
                throw new NotImplementedException();
        }

        internal override string profilerTag => "Cast";
    }

    /// <summary>
    /// Represents an element-wise `CastLike` layer: f(x) = (float)x or f(x) = (int)x depending on the data type of the targetType tensor.
    /// </summary>
    class CastLike : Layer
    {
        public CastLike(int output, int input, int targetType)
            : base(new[] { output }, new[] { input, targetType }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var toType = ctx.GetPartialTensor(inputs[1]).dataType;
            ctx.AddPartialTensor(outputs[0], new PartialTensor(toType, ctx.GetPartialTensor(inputs[0]).shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var dataType = ctx.storage.GetDataType(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;

            if (X.dataType == dataType)
                ctx.backend.MemCopy(X, O);
            else if (X.dataType == DataType.Int && dataType == DataType.Float)
                ctx.backend.Cast(X as TensorInt, O as TensorFloat);
            else if (X.dataType == DataType.Float && dataType == DataType.Int)
                ctx.backend.Cast(X as TensorFloat, O as TensorInt);
            else if (X.dataType == DataType.Short && dataType == DataType.Float)
                ctx.backend.Cast(X as TensorShort, O as TensorFloat);
            else
                throw new NotImplementedException();
        }

        internal override string profilerTag => "CastLike";
    }

    /// <summary>
    /// Represents a `Concat` concatenation layer. The layer computes the output tensor by concatenating the input tensors along a given axis.
    /// </summary>
    class Concat : Layer
    {
        public int axis;

        public Concat(int output, int[] inputs, int axis)
            : base(new[] { output }, inputs)
        {
            this.axis = axis;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            Logger.AssertIsTrue(inputs.Length > 0, "Concat.InputError: can't broadcast shapes array of size 0");
            var inputTensors = ctx.GetPartialTensors(inputs);
            var dataType = inputTensors[0].dataType;

            var rank = SymbolicTensorDim.Unknown;
            foreach (var tensorInput in inputTensors)
            {
                if (tensorInput.shape.hasRank)
                    rank = SymbolicTensorDim.MaxDefinedDim(rank, SymbolicTensorDim.Int(tensorInput.shape.rank));
            }

            if (rank.isUnknown)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, SymbolicTensorShape.UnknownShape));
                return;
            }

            foreach (var tensorInput in inputTensors)
                tensorInput.shape.DeclareRank(rank.value);

            var shapeOut = SymbolicTensorShape.UnknownOfRank(rank.value);
            var axisOut = shapeOut.Axis(axis);

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (i == axisOut)
                {
                    shapeOut[i] = SymbolicTensorDim.Zero;
                    foreach (var tensorInput in inputTensors)
                    {
                        shapeOut[i] += tensorInput.shape[i];
                    }
                }
                else
                {
                    shapeOut[i] = SymbolicTensorDim.Unknown;
                    foreach (var tensorInput in inputTensors)
                    {
                        shapeOut[i] = SymbolicTensorDim.MaxDefinedDim(shapeOut[i], tensorInput.shape[i]);
                    }
                }
            }

            var tensorOut = new PartialTensor(dataType, shapeOut);

            if (shapeOut.rank != 1 || !tensorOut.isPartiallyKnown)
            {
                ctx.AddPartialTensor(outputs[0], tensorOut);
                return;
            }

            var index = 0;
            foreach (var X in inputTensors)
            {
                for (var i = 0; i < X.length; i++)
                {
                    tensorOut[index++] = X[i];
                }
            }

            ctx.AddPartialTensor(outputs[0], tensorOut);
        }

        public override void Execute(ExecutionContext ctx)
        {
            var shapeO = ctx.storage.GetTensorShape(inputs[0]);
            for (var i = 1; i < inputs.Length; i++)
            {
                var shape = ctx.storage.GetTensorShape(inputs[i]);
                shapeO = shapeO.Concat(shape, axis);
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shapeO, ctx.storage.GetDataType(inputs[0]), ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            // this is necessary for not propagating NaN values
            if (ctx.backend.backendType == BackendType.GPUPixel)
                ctx.backend.MemClear(O);
            var start = 0;
            for (var i = 0; i < inputs.Length; i++)
            {
                var X = ctx.storage.GetTensor(inputs[i]);
                var length = X.shape[axis];
                if (length == 0)
                    continue;
                ctx.backend.SliceSet(X, O, axis, start, 1);
                start += length;
            }
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "Concat";
    }

    /// <summary>
    /// Represents a `DepthToSpace` layer. The layer computes the output tensor by permuting data from depth into blocks of spatial data.
    /// </summary>
    class DepthToSpace : Layer
    {
        public int blocksize;
        public DepthToSpaceMode mode;

        public DepthToSpace(int output, int input, int blocksize, DepthToSpaceMode mode)
            : base(new[] { output }, new[] { input })
        {
            this.blocksize = blocksize;
            this.mode = mode;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var shapeX = X.shape;
            shapeX.DeclareRank(4);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, new SymbolicTensorShape(shapeX[0], shapeX[1] / (blocksize * blocksize), shapeX[2] * blocksize, shapeX[3] * blocksize)));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.DepthToSpace(X.shape, blocksize), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.DepthToSpace(X, O, blocksize, mode);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, blocksize: {string.Join(", ", blocksize)}, mode: {mode}";
        }

        internal override string profilerTag => "DepthToSpace";
    }

    /// <summary>
    /// Represents an `Expand` layer. The layer computes the output tensor by broadcasting the input tensor into a given shape.
    /// </summary>
    class Expand : Layer
    {
        public Expand(int output, int input, int shape)
            : base(new[] { output }, new[] { input, shape }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var shape = ctx.GetPartialTensor(inputs[1]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, shape.ToSymbolicTensorShape().Broadcast(X.shape)));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var shape = new TensorShape(ctx.storage.GetInts(inputs[1]));
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Broadcast(shape), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Expand(X, O);
        }

        internal override string profilerTag => "Expand";
    }

    /// <summary>
    /// Represents a `Flatten` layer. The layer computes the output tensor by reshaping the input tensor into a 2D matrix according to the given axis.
    /// </summary>
    class Flatten : Layer
    {
        public int axis;

        public Flatten(int output, int input, int axis = 1)
            : base(new[] { output }, new[] { input })
        {
            this.axis = axis;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var shapeX = X.shape;
            if (!shapeX.hasRank)
            {
                if (axis == 0)
                    ctx.AddPartialTensor(outputs[0], X.Reshape(new SymbolicTensorShape(SymbolicTensorDim.One, shapeX.Length())));
                else
                    ctx.AddPartialTensor(outputs[0], X.Reshape(SymbolicTensorShape.UnknownOfRank(2)));
                return;
            }

            var axisX = axis >= 0 ? axis : shapeX.rank + axis;

            var shapeOut = SymbolicTensorShape.Ones(2);
            for (var i = 0; i < axisX; i++)
            {
                shapeOut[0] *= shapeX[i];
            }
            for (var i = axisX; i < shapeX.rank; i++)
            {
                shapeOut[1] *= shapeX[i];
            }

            ctx.AddPartialTensor(outputs[0], X.Reshape(shapeOut));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var shape = X.shape.Flatten(axis);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reshape(X, O);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "Flatten";
    }

    /// <summary>
    /// Represents a `GridSample` layer. The layer computes the output tensor by sampling the input tensor with coordinates given by the grid tensor.
    /// </summary>
    class GridSample : Layer
    {
        public InterpolationMode mode;
        public PaddingMode paddingMode;
        public bool alignCorners;

        public GridSample(int output, int input, int grid, InterpolationMode mode, PaddingMode paddingMode, bool alignCorners)
            : base(new[] { output }, new[] { input, grid })
        {
            this.mode = mode;
            this.paddingMode = paddingMode;
            this.alignCorners = alignCorners;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var grid = ctx.GetPartialTensor(inputs[1]);

            var outShape = SymbolicTensorShape.UnknownShape;

            if (X.shape.hasRank)
                outShape.DeclareRank(X.shape.rank);
            if (grid.shape.hasRank)
                outShape.DeclareRank(grid.shape.rank);

            for (var i = 0; i < (outShape.hasRank ? outShape.rank : 0); i++)
            {
                outShape[i] = i switch
                {
                    0 => SymbolicTensorDim.MaxDefinedDim(X.shape[0], grid.shape[0]),
                    1 => X.shape[i],
                    _ => grid.shape[i - 1]
                };
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, outShape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var grid = ctx.storage.GetTensor(inputs[1]) as TensorFloat;
            var shapeO = TensorShape.Ones(X.shape.rank);
            shapeO[0] = X.shape[0];
            shapeO[1] = X.shape[1];
            for (var i = 2; i < shapeO.rank; i++)
                shapeO[i] = grid.shape[i - 1];
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shapeO, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.GridSample(X, grid, O, mode, paddingMode, alignCorners);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, mode: {mode}, paddingMode: {paddingMode}, alignCorners: {alignCorners}";
        }

        internal override string profilerTag => "GridSample";
    }

    /// <summary>
    /// Represents an `Identity` layer. The output tensor is a copy of the input tensor.
    /// </summary>
    class Identity : Layer
    {
        public Identity(int output, int input)
            : base(new[] { output }, new[] { input }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(outputs[0], ctx.GetPartialTensor(inputs[0]));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.MemCopy(X, O);
        }

        internal override string profilerTag => "Identity";
    }

    /// <summary>
    /// Represents a `MoveDim` layer. The layer computes the output tensor by moving the dimensions of input at the positions in source to the positions in destination.
    ///
    /// Other dimensions of input that are not explicitly moved remain in their original order and appear at the positions not specified in destination.
    /// </summary>
    class MoveDim : Layer
    {
        public int[] source;
        public int[] destination;

        public MoveDim(int output, int input, int[] source, int[] destination)
            : base(new[] { output }, new[] { input })
        {
            Logger.AssertIsTrue(source.Length == destination.Length, "MoveDim.ValueError: source and destination arrays must be same length");
            this.source = source;
            this.destination = destination;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;

            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(ctx.GetPartialTensor(inputs[0]).dataType));
                return;
            }

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank);

            // move given dims
            uint srcAxesBitMask = 0;
            uint dstAxesBitMask = 0;
            for (var i = 0; i < source.Length; i++)
            {
                var srcAxis = shapeX.Axis(source[i]);
                var dstAxis = shapeX.Axis(destination[i]);
                Logger.AssertIsTrue(((srcAxesBitMask >> srcAxis) & 1U) == 0, "MoveDim.ValueError: source dims may not repeat");
                Logger.AssertIsTrue(((dstAxesBitMask >> dstAxis) & 1U) == 0, "MoveDim.ValueError: destination dims may not repeat");
                srcAxesBitMask |= 1U << srcAxis;
                dstAxesBitMask |= 1U << dstAxis;
                shapeOut[dstAxis] = shapeX[srcAxis];
            }

            // fill remaining dims in order
            for (int srcAxis = 0, dstAxis = 0; srcAxis < shapeX.rank; srcAxis++)
            {
                if (((srcAxesBitMask >> srcAxis) & 1U) != 0)
                    continue;
                while (((dstAxesBitMask >> dstAxis) & 1U) != 0)
                    dstAxis++;
                srcAxesBitMask |= 1U << srcAxis;
                dstAxesBitMask |= 1U << dstAxis;
                shapeOut[dstAxis] = shapeX[srcAxis];
                dstAxis++;
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(ctx.GetPartialTensor(inputs[0]).dataType, shapeOut));
        }

        public override unsafe void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);

            Span<int> permutations = stackalloc int[X.shape.rank];
            // move given dims
            uint srcAxesBitMask = 0;
            uint dstAxesBitMask = 0;
            for (var i = 0; i < source.Length; i++)
            {
                var srcAxis = X.shape.Axis(source[i]);
                var dstAxis = X.shape.Axis(destination[i]);
                Logger.AssertIsTrue(((srcAxesBitMask >> srcAxis) & 1U) == 0, "MoveDim.ValueError: source dims may not repeat");
                Logger.AssertIsTrue(((dstAxesBitMask >> dstAxis) & 1U) == 0, "MoveDim.ValueError: destination dims may not repeat");
                srcAxesBitMask |= 1U << srcAxis;
                dstAxesBitMask |= 1U << dstAxis;
                permutations[dstAxis] = srcAxis;
            }

            // fill remaining dims in order
            for (int srcAxis = 0, dstAxis = 0; srcAxis < X.shape.rank; srcAxis++)
            {
                if (((srcAxesBitMask >> srcAxis) & 1U) != 0)
                    continue;
                while (((dstAxesBitMask >> dstAxis) & 1U) != 0)
                    dstAxis++;
                srcAxesBitMask |= 1U << srcAxis;
                dstAxesBitMask |= 1U << dstAxis;
                permutations[dstAxis] = srcAxis;
                dstAxis++;
            }

            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Transpose(permutations), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Transpose(X, O, permutations);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, source: [{string.Join(", ", source)}], destination: [{string.Join(", ", destination)}]";
        }

        internal override string profilerTag => "MoveDim";
    }

    /// <summary>
    /// Represents a `Narrow` layer. The layer calculates the output tensor by slicing the input tensor along a given dim with a given start and length.
    /// </summary>
    class Narrow : Layer
    {
        public Narrow(int output, int input, int dim, int start, int length)
            : base(new[] { output }, new[] { input, dim, start, length }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var dim = ctx.GetPartialTensor(inputs[1]);
            var start = ctx.GetPartialTensor(inputs[2]);
            var length = ctx.GetPartialTensor(inputs[3]);
            if (!dim.IsFullyKnown())
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, SymbolicTensorShape.UnknownOfRank(X.shape.rank)));
                return;
            }

            var dimValue = dim[0].intValue;

            if (X.isPartiallyKnown && X.shape.rank == 1 && start.IsFullyKnown() && length.IsFullyKnown())
            {
                var dimSize = X.shape[dimValue].value;
                var startValue = (start[0].intValue + dimSize) % dimSize;
                var end = Mathf.Min(startValue + length[0].intValue, dimSize);
                var lengthValue = end - startValue;
                var oShape = new SymbolicTensorShape(lengthValue);
                var tensorOut = new PartialTensor(X.dataType, oShape);
                for (var i = 0; i < lengthValue; i++)
                    tensorOut[i] = X[startValue + i];
                ctx.AddPartialTensor(outputs[0], tensorOut);
                return;
            }

            var outShape = X.shape;
            outShape[dimValue] = SymbolicTensorDim.Unknown;
            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, outShape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var dim = ctx.storage.GetInt(inputs[1]);
            var start = ctx.storage.GetInt(inputs[2]);
            var length = ctx.storage.GetInt(inputs[3]);
            dim = X.shape.Axis(dim);
            var dimSize = X.shape[dim];
            start = (start + dimSize) % dimSize;
            var end = Mathf.Min(start + length, dimSize);
            length = end - start;
            var oShape = X.shape;
            oShape[dim] = length;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], oShape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Split(X, O, dim, start);
        }

        internal override string profilerTag => "Narrow";
    }

    /// <summary>
    /// Represents a `Pad` layer. The layer calculates the output tensor by adding padding to the input tensor according to the given padding values and mode.
    /// </summary>
    class Pad : Layer
    {
        public PadMode padMode;

        public Pad(int output, int data, int pads, int constantValue = -1, int axes = -1, PadMode mode = PadMode.Constant)
            : base(new[] { output }, new[] { data, pads, constantValue, axes })
        {
            padMode = mode;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var pads = ctx.GetPartialTensor(inputs[1]);
            var shapeX = X.shape;
            var shapePads = pads.shape;
            if (shapePads.hasRank)
            {
                Logger.AssertIsTrue(shapePads.rank == 1, "Pad.ValueError: pads must be rank 1");
                Logger.AssertIsTrue(!shapePads[0].isValue || shapePads[0].value % 2 == 0, "Pad.ValueError: length of pads must divide by 2");
            }

            var axes = ctx.GetPartialTensor(inputs[3]);

            if (axes == null)
            {
                shapeX.DeclareRank(shapePads[0] / 2);
                axes = shapeX.hasRank ? PartialTensor.Range(0, shapeX.rank) : new PartialTensor(DataType.Int, SymbolicTensorShape.UnknownOfRank(1));
            }

            var shapeOut = SymbolicTensorShape.UnknownOfRankLike(shapeX);

            if (!axes.isPartiallyKnown)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, shapeOut));
                return;
            }

            Logger.AssertIsTrue(!shapePads[0].isValue || shapePads[0].value == axes.length * 2, "Pad.ValueError: length of pads must be twice the length of the axes");

            for (var i = 0; i < axes.length; i++)
            {
                if (!axes[i].isIntValue)
                    continue;
                var axis = axes[i].intValue;
                var dimPad = pads[i] + pads[i + axes.length];
                shapeOut[axis] = shapeX[axis] + (SymbolicTensorDim)dimPad;
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, shapeOut));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var pad = ctx.storage.GetInts(inputs[1]);
            var axes = ctx.storage.GetInts(inputs[3], null);

            Span<int> pads = stackalloc int[2 * X.shape.rank];
            if (axes != null)
            {
                for (var i = 0; i < axes.Length; i++)
                {
                    var axis = X.shape.Axis(axes[i]);
                    pads[axis] = pad[i];
                    pads[axis + X.shape.rank] = pad[i + axes.Length];
                }
            }
            else
            {
                pad.CopyTo(pads);
            }

            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Pad(pads), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (padMode != PadMode.Constant)
            {
                Assert.IsFalse(X.shape.HasZeroDims(), "ValueError: zero dimensions input for Pad operator is not supported");
                if (X.dataType == DataType.Float)
                    ctx.backend.Pad(X as TensorFloat, O as TensorFloat, pads, padMode, 0);
                else
                    ctx.backend.Pad(X as TensorInt, O as TensorInt, pads, padMode, 0);
                return;
            }

            if (X.dataType == DataType.Float)
            {
                var constantValue = ctx.storage.GetFloat(inputs[2]);
                if (X.shape.HasZeroDims())
                    ctx.backend.MemSet(O as TensorFloat, constantValue);
                else
                    ctx.backend.Pad(X as TensorFloat, O as TensorFloat, pads, padMode, constantValue);
            }
            else
            {
                var constantValue = ctx.storage.GetInt(inputs[2]);
                if (X.shape.HasZeroDims())
                    ctx.backend.MemSet(O as TensorInt, constantValue);
                else
                    ctx.backend.Pad(X as TensorInt, O as TensorInt, pads, padMode, constantValue);
            }
        }

        internal override string profilerTag => "Pad";
    }

    /// <summary>
    /// Represents a `Reshape` layer. The layer calculates the output tensor by copying the data from the input tensor and using a given shape. The data from the input tensor is unchanged.
    ///
    /// Only one of the elements of the shape can be -1. The layer infers the size of this dimension from the remaining dimensions and the length of the input tensor.
    /// </summary>
    class Reshape : Layer
    {
        public bool allowZero;

        public Reshape(int output, int input, int shape, bool allowZero = false)
            : base(new[] { output }, new[] { input, shape })
        {
            this.allowZero = allowZero;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var shape = ctx.GetPartialTensor(inputs[1]);
            var shapeX = X.shape;
            shape.shape.DeclareRank(1);

            if (!shape.isPartiallyKnown)
            {
                if (shape.shape[0].isValue)
                    ctx.AddPartialTensor(outputs[0], X.Reshape(SymbolicTensorShape.UnknownOfRank(shape.shape[0].value)));
                else
                    ctx.AddPartialTensor(outputs[0], X.Reshape(SymbolicTensorShape.UnknownShape));
                return;
            }

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shape.length);

            var containsMinusOne = false;

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (shape[i] == -1)
                    containsMinusOne = true;
            }

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (shape[i].isUnknown)
                    continue;

                var dim = (SymbolicTensorDim)shape[i];
                if (shape[i].isParam)
                {
                    if (allowZero || (shapeX.hasRank && i >= shapeX.rank) || shapeX[i] == dim)
                        shapeOut[i] = dim;
                    else if (containsMinusOne)
                    {
                        for (var j = 0; j < shapeX.rank; j++)
                        {
                            if (shapeX[j] == dim)
                            {
                                shapeOut[i] = dim;
                                break;
                            }
                        }
                    }
                    continue;
                }

                if (shape[i].intValue > 0)
                    shapeOut[i] = dim;
                else if (shape[i].intValue == 0)
                    shapeOut[i] = allowZero ? SymbolicTensorDim.Zero : shapeX[i];
            }

            ctx.AddPartialTensor(outputs[0], X.Reshape(shapeOut, !containsMinusOne));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var shape = X.shape.Reshape(ctx.storage.GetInts(inputs[1]), allowZero);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reshape(X, O);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, allowZero: {allowZero}";
        }

        internal override string profilerTag => "Reshape";
    }

    /// <summary>
    /// Represents a `Resize` layer. The layer calculates the output tensor by resampling the input tensor along the spatial dimensions to a given shape.
    /// </summary>
    class Resize : Layer
    {
        public ScaleMode scaleMode;
        public CoordTransformMode coordTransformMode;
        public InterpolationMode mode;
        public NearestMode nearestMode;
        public int[] axes;

        public Resize(int output, int input, int scalesOrSizes, ScaleMode scaleMode, InterpolationMode mode, CoordTransformMode coordTransformMode, NearestMode nearestMode, int[] axes)
            : base(new[] { output }, new[] { input, scalesOrSizes })
        {
            this.scaleMode = scaleMode;
            this.coordTransformMode = coordTransformMode;
            this.mode = mode;
            this.nearestMode = nearestMode;
            this.axes = axes;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var dataType = ctx.GetPartialTensor(inputs[0]).dataType;
            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;

            var sizesOrScales = ctx.GetPartialTensor(inputs[1]);
            sizesOrScales.shape.DeclareRank(1);
            if (axes == null)
                shapeX.DeclareRank(sizesOrScales.shape[0]);
            var shapeOut = new SymbolicTensorShape(shapeX);
            if (shapeOut.hasRank)
            {
                if (axes == null)
                {
                    for (var i = 0; i < shapeOut.rank; i++)
                        shapeOut[i] = scaleMode == ScaleMode.Sizes ? (SymbolicTensorDim)sizesOrScales[i] : shapeX[i].Resize(sizesOrScales[i]);
                }
                else
                {
                    for (var i = 0; i < axes.Length; i++)
                    {
                        var axis = shapeOut.Axis(axes[i]);
                        shapeOut[axis] = scaleMode == ScaleMode.Sizes ? (SymbolicTensorDim)sizesOrScales[i] : shapeX[axis].Resize(sizesOrScales[i]);
                    }
                }
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            Span<float> s = stackalloc float[X.shape.rank];
            for (var i = 0; i < s.Length; i++)
                s[i] = 1f;

            if (scaleMode == ScaleMode.Sizes)
            {
                var sizes = ctx.storage.GetInts(inputs[1]);

                if (axes != null)
                {
                    for (var i = 0; i < axes.Length; i++)
                    {
                        var axis = X.shape.Axis(axes[i]);
                        s[axis] = sizes[i] / (float)X.shape[axis];
                    }
                }
                else
                {
                    for (var i = 0; i < X.shape.rank; i++)
                        s[i] = sizes[i] / (float)X.shape[i];
                }

                var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Resize(X.shape, s), DataType.Float, ctx.backend.backendType) as TensorFloat;
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Resize(X, O, s, mode, nearestMode, coordTransformMode);
            }
            else
            {
                var scales = ctx.storage.GetFloats(inputs[1]);

                if (axes != null)
                {
                    for (var i = 0; i < axes.Length; i++)
                    {
                        var axis = X.shape.Axis(axes[i]);
                        s[axis] = scales[i];
                    }
                }
                else
                {
                    for (var i = 0; i < X.shape.rank; i++)
                        s[i] = scales[i];
                }

                var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Resize(X.shape, scales), DataType.Float, ctx.backend.backendType) as TensorFloat;
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Resize(X, O, scales, mode, nearestMode, coordTransformMode);
            }
        }

        public override string ToString()
        {
            return $"{base.ToString()}, mode: {mode}, coordTransformMode: {coordTransformMode}, nearestMode: {nearestMode}";
        }

        internal override string profilerTag => "Resize";
    }

    /// <summary>
    /// Represents a `Select` layer. The layer calculates the output tensor by slicing the input tensor along a given dim with a given index, the sliced dim is removed from the output.
    /// </summary>
    class Select : Layer
    {
        public Select(int output, int input, int dim, int selectIndex)
            : base(new[] { output }, new[] { input, dim, selectIndex }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var dim = ctx.GetPartialTensor(inputs[1]);
            var selectIndex = ctx.GetPartialTensor(inputs[2]);
            if (!dim.IsFullyKnown())
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, SymbolicTensorShape.UnknownOfRank(X.shape.rank - 1)));
                return;
            }

            var axis = dim[0].intValue;
            var outShape = X.shape;
            outShape[axis] = SymbolicTensorDim.One;
            outShape = outShape.Squeeze(axis);
            var tensorOut = new PartialTensor(X.dataType, outShape);

            if (axis == 0 && X.isPartiallyKnown && selectIndex.IsFullyKnown())
            {
                var index = selectIndex[0].intValue;
                index = index < 0 ? index + X.shape.Length().value : index;
                tensorOut[0] = X[index];
            }

            ctx.AddPartialTensor(outputs[0], tensorOut);
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var dim = ctx.storage.GetInt(inputs[1]);
            var selectIndex = ctx.storage.GetInt(inputs[2]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(dim, false), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var unsqueezed = ctx.storage.AllocateTensor(X.shape.Reduce(dim, true), X.dataType, ctx.backend.backendType);
            dim = X.shape.Axis(dim);
            selectIndex = selectIndex < 0 ? selectIndex + X.shape[dim] : selectIndex;
            ctx.backend.Split(X, unsqueezed, dim, selectIndex);
            ctx.backend.Reshape(unsqueezed, O);
            ctx.storage.Dispose(unsqueezed);
        }

        internal override string profilerTag => "Select";
    }

    /// <summary>
    /// Represents a `Slice` layer. The layer calculates the output tensor by slicing the input tensor along given axes with given starts, ends and steps.
    /// </summary>
    class Slice : Layer
    {
        public Slice(int output, int input, int starts, int ends, int axes = -1, int steps = -1)
            : base(new[] { output }, new[] { input, starts, ends, axes, steps }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var data = ctx.GetPartialTensor(inputs[0]);
            if (!data.shape.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(data.dataType));
                return;
            }

            var starts = ctx.GetPartialTensor(inputs[1]);
            var ends = ctx.GetPartialTensor(inputs[2]);
            var axes = ctx.GetPartialTensor(inputs[3]) ?? PartialTensor.Range(0, data.shape.rank);
            var steps = ctx.GetPartialTensor(inputs[4]) ?? PartialTensor.Ones(starts.shape);

            if (data.isPartiallyKnown && data.shape.rank == 1 && starts[0].isIntValue && ends[0].isIntValue && steps[0].isIntValue)
            {
                var dim = data.shape[0].value;
                var start = starts[0].intValue;
                var end = ends[0].intValue;
                var step = steps[0].intValue;

                var clampAdjustDirection = step < 0 ? -1 : 0;

                start = start < 0 ? dim + start : start;
                start = Mathf.Clamp(start, 0, dim + clampAdjustDirection);

                end = end < 0 ? dim + end : end;
                end = Mathf.Clamp(end, clampAdjustDirection, dim);

                var length = (int)Math.Ceiling((end - start) / (double)step);
                length = Mathf.Max(length, 0);

                var tensorOut = new PartialTensor(data.dataType, new SymbolicTensorShape(length));

                for (var i = 0; i < length; i++)
                {
                    tensorOut[i] = data[start + i * step];
                }

                ctx.AddPartialTensor(outputs[0], tensorOut);
                return;
            }

            if (!axes.isPartiallyKnown)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(data.dataType, SymbolicTensorShape.UnknownOfRank(data.shape.rank)));
                return;
            }

            var shapeOut = new SymbolicTensorShape(data.shape);

            for (var i = 0; i < axes.length; i++)
            {
                var axisElement = axes[i];
                if (!axisElement.isIntValue)
                {
                    shapeOut = SymbolicTensorShape.UnknownOfRank(data.shape.rank);
                    continue;
                }
                var axis = shapeOut.Axis(axisElement.intValue);
                shapeOut[axis] = data.shape[axis].Slice(starts[i], ends[i], steps[i]);
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(data.dataType, shapeOut));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var starts = ctx.storage.GetInts(inputs[1]);
            var ends = ctx.storage.GetInts(inputs[2]);
            var axes = ctx.storage.GetInts(inputs[3], null);
            var steps = ctx.storage.GetInts(inputs[4], null);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Slice(starts, ends, axes, steps), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Slice(X, O, starts, axes, steps);
        }

        internal override string profilerTag => "Slice";
    }

    class SliceSet : Layer
    {
        public SliceSet(int output, int input, int values, int starts, int ends, int axes = -1, int steps = -1)
            : base(new[] { output }, new[] { input, values, starts, ends, axes, steps }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, X.shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var values = ctx.storage.GetTensor(inputs[1]);
            var starts = ctx.storage.GetInts(inputs[2]);
            var ends = ctx.storage.GetInts(inputs[3]);
            var axes = ctx.storage.GetInts(inputs[4], null);
            var steps = ctx.storage.GetInts(inputs[5], null);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var slicedShape = X.shape.Slice(starts, ends, axes, steps);
            Logger.AssertIsTrue(slicedShape.Broadcast(values.shape) == slicedShape, "SliceSet.InputError: values shape must be broadcastable to sliced shape, {0} {1}", values.shape, slicedShape);
            if (slicedShape != values.shape)
            {
                // broadcast values
                var broadcastValues = ctx.storage.AllocateTensor(slicedShape, values.dataType, ctx.backend.backendType);
                ctx.backend.Expand(values, broadcastValues);
                ctx.backend.SliceSet(X, broadcastValues, O, starts, axes, steps);
                ctx.storage.Dispose(broadcastValues);
            }
            else
            {
                ctx.backend.SliceSet(X, values, O, starts, axes, steps);
            }
        }

        internal override string profilerTag => "SliceSet";
    }

    /// <summary>
    /// Represents a `SpaceToDepth` layer. The layer computes the output tensor by permuting data from blocks of spatial data into depth.
    /// </summary>
    class SpaceToDepth : Layer
    {
        public int blocksize;

        public SpaceToDepth(int output, int input, int blocksize)
            : base(new[] { output }, new[] { input })
        {
            this.blocksize = blocksize;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;
            shapeX.DeclareRank(4);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(ctx.GetPartialTensor(inputs[0]).dataType, new SymbolicTensorShape(shapeX[0], shapeX[1] * (blocksize * blocksize), shapeX[2] / blocksize, shapeX[3] / blocksize)));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.SpaceToDepth(X.shape, blocksize), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.SpaceToDepth(X, O, blocksize);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, blocksize: {blocksize}";
        }

        internal override string profilerTag => "SpaceToDepth";
    }

    /// <summary>
    /// Represents a `Split` layer. The layer computes the output tensors by splitting the input tensor along a single given axis.
    /// </summary>
    class Split : Layer
    {
        public int axis;
        public int numOutputs;

        public Split(int[] outputs, int input, int split = -1, int axis = 0, int numOutputs = 0)
            : base(outputs, new[] { input, split })
        {
            Logger.AssertIsTrue(outputs.Length >= 1, "Split.InputError: output array must have length at least 1");
            this.axis = axis;
            this.numOutputs = numOutputs;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var partialSplit = ctx.GetPartialTensor(inputs[1]);
            if (partialSplit == null)
            {
                partialSplit = new PartialTensor(DataType.Int, new SymbolicTensorShape(numOutputs));

                var dim = X.shape[axis];
                if (dim.isParam && numOutputs == 1)
                {
                    partialSplit[0] = PartialTensorElement.Param(dim.param);
                }
                else if (dim.isValue)
                {
                    var splitLength = Mathf.CeilToInt(dim.value / (float)numOutputs);
                    for (var i = 0; i < numOutputs - 1; i++)
                    {
                        partialSplit[i] = PartialTensorElement.IntValue(splitLength);
                    }

                    // final split length is the (possible smaller) remainder along the axis
                    var lastSplitLength = dim.value - (splitLength * (numOutputs - 1));
                    Logger.AssertIsTrue(lastSplitLength >= 0, "Split.InputError: split axis too small for numOutputs");
                    partialSplit[numOutputs - 1] = PartialTensorElement.IntValue(lastSplitLength);
                }
            }

            for (var i = 0; i < outputs.Length; i++)
            {
                var outputShape = new SymbolicTensorShape(X.shape);
                outputShape[axis] = (SymbolicTensorDim)partialSplit[i];
                ctx.AddPartialTensor(outputs[i], new PartialTensor(X.dataType, outputShape));
            }
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);

            var dim = X.shape[axis];
            var equalSplitLength = 0;
            var split = ctx.storage.GetInts(inputs[1], null);
            if (split == null)
            {
                // if splits are not given calculate even split length
                equalSplitLength = (int)Math.Ceiling(dim / (double)numOutputs);
            }
            var start = 0;
            for (var i = 0; i < outputs.Length; i++)
            {
                var end = start + (split != null ? split[i] : equalSplitLength);
                end = Math.Min(end, dim);
                var O = ctx.storage.AllocateTensorAndStore(outputs[i], X.shape.Split(axis, start, end), X.dataType, ctx.backend.backendType);
                if (!O.shape.HasZeroDims())
                    ctx.backend.Split(X, O, axis, start);
                start = end;
            }
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}, numOutputs: {numOutputs}";
        }

        internal override string profilerTag => "Split";
    }

    /// <summary>
    /// Represents a `Squeeze` layer. The layer computes the output tensor by reshaping the input tensor by removing dimensions of size 1.
    /// </summary>
    class Squeeze : Layer
    {
        public Squeeze(int output, int input, int axes = -1)
            : base(new[] { output }, new[] { input, axes }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var axes = ctx.GetPartialTensor(inputs[1]);
            if (axes != null)
            {
                if (!axes.isPartiallyKnown)
                    ctx.AddPartialTensor(outputs[0], X.Reshape(SymbolicTensorShape.UnknownShape));
                else if (!axes.IsFullyKnown())
                    ctx.AddPartialTensor(outputs[0], X.Reshape(SymbolicTensorShape.UnknownOfRank(X.shape.rank - axes.length)));
                else
                    ctx.AddPartialTensor(outputs[0], X.Reshape(X.shape.Squeeze(axes)));
                return;
            }

            ctx.AddPartialTensor(outputs[0], X.Reshape(X.shape.Squeeze()));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = ctx.storage.GetInts(inputs[1], null);
            var shape = axes != null ? X.shape.Squeeze(axes) : X.shape.Squeeze();
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reshape(X, O); // TODO<tensordata> refcount tensordata
        }

        internal override string profilerTag => "Squeeze";
    }

    /// <summary>
    /// Represents a `Tile` layer. The layer computes the output tensor by repeating the input layer a given number of times along each axis.
    /// </summary>
    class Tile : Layer
    {
        public Tile(int output, int input, int repeats)
            : base(new[] { output }, new[] { input, repeats }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var dataType = ctx.GetPartialTensor(inputs[0]).dataType;
            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;
            var repeats = ctx.GetPartialTensor(inputs[1]);
            repeats.shape.DeclareRank(1);

            if (!repeats.isPartiallyKnown)
            {
                if (repeats.shape[0].isValue && !shapeX.hasRank)
                    shapeX = SymbolicTensorShape.UnknownOfRank(repeats.shape[0].value);
                Logger.AssertIsFalse(repeats.shape[0] != shapeX.rank, "Tile.InputError: repeats value must be equal to input rank");
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, SymbolicTensorShape.UnknownOfRankLike(shapeX)));
                return;
            }

            shapeX.DeclareRank(repeats.length);

            var shapeOut = new SymbolicTensorShape(shapeX);
            for (var i = 0; i < shapeOut.rank; i++)
            {
                shapeOut[i] *= (SymbolicTensorDim)repeats[i];
            }
            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var repeats = ctx.storage.GetInts(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Tile(repeats), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Tile(X, O, repeats);
        }

        internal override string profilerTag => "Tile";
    }

    /// <summary>
    /// Represents a `Transpose` layer. The layer computes the output tensor by permuting the axes and data of the input tensor according to the given permutations.
    /// </summary>
    class Transpose : Layer
    {
        public int[] permutations;

        public Transpose(int output, int input, int[] permutations)
            : base(new[] { output }, new[] { input })
        {
            this.permutations = permutations;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;
            if (permutations != null)
                shapeX.DeclareRank(permutations.Length);

            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(ctx.GetPartialTensor(inputs[0]).dataType));
                return;
            }

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank);

            if (permutations == null || permutations.Length == 0)
            {
                // reverse axes
                for (var i = 0; i < shapeX.rank; i++)
                {
                    shapeOut[i] = shapeX[shapeX.rank - 1 - i];
                }
            }
            else
            {
                uint axesBitMask = 0;
                for (var i = 0; i < permutations.Length; i++)
                {
                    var axis = shapeX.Axis(permutations[i]);
                    Logger.AssertIsTrue(((axesBitMask >> axis) & 1U) == 0, "Transpose.ValueError: permutation must be a permutation of the axis (0, rank-1)");
                    axesBitMask |= 1U << axis;
                    shapeOut[i] = shapeX[axis];
                }
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(ctx.GetPartialTensor(inputs[0]).dataType, shapeOut));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            if (permutations == null)
            {
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Transpose(), X.dataType, ctx.backend.backendType);
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Transpose(X, O);
                return;
            }
            else
            {
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Transpose(permutations), X.dataType, ctx.backend.backendType);
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Transpose(X, O, permutations);
                return;
            }
        }

        public override string ToString()
        {
            if (permutations == null)
                return base.ToString();
            else
                return $"{base.ToString()}, permutations: [{string.Join(", ", permutations)}]";
        }

        internal override string profilerTag => "Transpose";
    }

    /// <summary>
    /// Represents a `Trilu` layer. The layer computes the output tensor by retaining the upper or lower triangular values from an input matrix or matrix batch and setting the other values to zero.
    /// </summary>
    class Trilu : Layer
    {
        public TriluMode mode;

        public Trilu(int output, int input, int k = -1, TriluMode mode = TriluMode.Upper)
            : base(new[] { output }, new[] { input, k })
        {
            this.mode = mode;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, X.shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var k = ctx.storage.GetInt(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (mode == TriluMode.Upper)
                ctx.backend.Triu(X, O, k);
            else
                ctx.backend.Tril(X, O, k);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, mode: {mode}";
        }

        internal override string profilerTag => "Trilu";
    }

    /// <summary>
    /// Represents an `Unsqueeze` layer. The layer computes the output tensor by reshaping the input tensor by adding dimensions of size 1 at the given axes.
    /// </summary>
    class Unsqueeze : Layer
    {
        public Unsqueeze(int output, int input, int axes)
            : base(new[] { output }, new[] { input, axes }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var shape = ctx.GetPartialTensor(inputs[1]);
            ctx.AddPartialTensor(outputs[0], X.Reshape(X.shape.Unsqueeze(shape)));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = ctx.storage.GetInts(inputs[1]);
            var shape = X.shape.Unsqueeze(axes);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reshape(X, O); // TODO<tensordata> refcount tensordata
        }

        internal override string profilerTag => "Unsqueeze";
    }
}
