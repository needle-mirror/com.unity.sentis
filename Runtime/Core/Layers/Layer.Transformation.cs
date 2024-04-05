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
    [Serializable]
    class Cast : Layer
    {
        /// <summary>
        /// The data type to cast to as a `DataType`.
        /// </summary>
        public DataType toType;

        /// <summary>
        /// Initializes and returns an instance of `Cast` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="toType">The data type to cast to as a `DataType`.</param>
        public Cast(string name, string input, DataType toType)
        {
            index = name;
            inputs = new[] { input };
            this.toType = toType;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(index, new PartialTensor(toType, ctx.GetPartialTensor(inputs[0]).shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, toType, ctx.backend.backendType);
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
    [Serializable]
    [Optimization.CPUFallback.NoDataDependencyInputs(1)]
    class CastLike : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `CastLike` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="targetType">The name to use for the targetType tensor of the layer. The layer does not use the values of this tensor in the computation.</param>
        public CastLike(string name, string input, string targetType)
        {
            index = name;
            inputs = new[] { input, targetType };
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var toType = ctx.GetPartialTensor(inputs[1]).dataType;
            ctx.AddPartialTensor(index, new PartialTensor(toType, ctx.GetPartialTensor(inputs[0]).shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, ctx.vars.GetTensor(inputs[1]).dataType, ctx.backend.backendType);
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

        internal override string profilerTag => "CastLike";
    }

    /// <summary>
    /// Represents a `Concat` concatenation layer. The layer computes the output tensor by concatenating the input tensors along a given axis.
    /// </summary>
    [Serializable]
    class Concat : Layer
    {
        /// <summary>
        /// The axis along which to concatenate the input tensors.
        /// </summary>
        public int axis;

        /// <summary>
        /// Initializes and returns an instance of `Concat` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The names to use for the input tensors of the layer.</param>
        /// <param name="axis">The axis along which to concatenate the input tensors.</param>
        public Concat(string name, string[] inputs, int axis)
        {
            index = name;
            this.inputs = inputs;
            this.axis = axis;
        }

        /// <inheritdoc/>
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
                ctx.AddPartialTensor(this.index, new PartialTensor(dataType, SymbolicTensorShape.UnknownShape));
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
                ctx.AddPartialTensor(this.index, tensorOut);
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

            ctx.AddPartialTensor(this.index, tensorOut);
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            Tensor[] tensors = new Tensor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                tensors[i] = ctx.vars.GetTensor(inputs[i]);
            }
            var O = ctx.vars.AllocateTensorAndStore(index, TensorShapeHelper.ConcatShape(tensors, axis), ctx.vars.GetTensor(inputs[0]).dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Concat(tensors, O, axis);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "Concat";
    }

    /// <summary>
    /// Represents a `DepthToSpace` layer. The layer computes the output tensor by permuting data from depth into blocks of spatial data.
    /// </summary>
    [Serializable]
    class DepthToSpace : Layer
    {
        /// <summary>
        /// The size of the blocks to move the depth data into.
        /// </summary>
        public int blocksize;
        /// <summary>
        /// The ordering of the data in the output tensor as a `DepthToSpaceMode`.
        /// </summary>
        public DepthToSpaceMode mode;

        /// <summary>
        /// Initializes and returns an instance of `DepthToSpace` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="blocksize">The size of the blocks to move the depth data into.</param>
        /// <param name="mode">The ordering of the data in the output tensor as a `DepthToSpaceMode`.</param>
        public DepthToSpace(string name, string input, int blocksize, DepthToSpaceMode mode)
        {
            index = name;
            inputs = new[] { input };
            this.blocksize = blocksize;
            this.mode = mode;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var shapeX = X.shape;
            shapeX.DeclareRank(4);
            ctx.AddPartialTensor(index, new PartialTensor(X.dataType, new SymbolicTensorShape(shapeX[0], shapeX[1] / (blocksize * blocksize), shapeX[2] * blocksize, shapeX[3] * blocksize)));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.vars.AllocateTensorAndStore(index, ShapeInference.DepthToSpace(X.shape, blocksize), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.DepthToSpace(X, O, blocksize, mode);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, blocksize: {string.Join(", ", blocksize)}, mode: {mode}";
        }

        internal override string profilerTag => "DepthToSpace";
    }

    /// <summary>
    /// Represents an `Expand` layer. The layer computes the output tensor by broadcasting the input tensor into a given shape.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    class Expand : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Expand` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="shape">The name to use for the 1D shape tensor of the layer.</param>
        public Expand(string name, string input, string shape)
        {
            index = name;
            inputs = new[] { input, shape };
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var shape = ctx.GetPartialTensor(inputs[1]);
            ctx.AddPartialTensor(index, new PartialTensor(X.dataType, shape.ToSymbolicTensorShape().Broadcast(X.shape)));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var shape = new TensorShape(ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>());
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Broadcast(shape), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Expand(X, O);
        }

        internal override string profilerTag => "Expand";
    }

    /// <summary>
    /// Represents a `Flatten` layer. The layer computes the output tensor by reshaping the input tensor into a 2D matrix according to the given axis.
    /// </summary>
    [Serializable]
    class Flatten : Layer
    {
        /// <summary>
        /// The axis up to which to flatten the input dimensions into the first dimension of the output tensor.
        /// </summary>
        public int axis;

        /// <summary>
        /// Initializes and returns an instance of `Flatten` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axis">The axis up to which to flatten the input dimensions into the first dimension of the output tensor. The default value is 1.</param>
        public Flatten(string name, string input, int axis = 1)
        {
            index = name;
            inputs = new[] { input };
            this.axis = axis;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var shapeX = X.shape;
            if (!shapeX.hasRank)
            {
                if (axis == 0)
                    ctx.AddPartialTensor(index, X.Reshape(new SymbolicTensorShape(SymbolicTensorDim.One, shapeX.Length())));
                else
                    ctx.AddPartialTensor(index, X.Reshape(SymbolicTensorShape.UnknownOfRank(2)));
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

            ctx.AddPartialTensor(index, X.Reshape(shapeOut));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var shape = X.shape.Flatten(axis);
            var O = ctx.vars.AllocateTensorAndStore(index, shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reshape(X, O);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "Flatten";
    }

    /// <summary>
    /// Represents an `Identity` layer. The output tensor is a copy of the input tensor.
    /// </summary>
    [Serializable]
    class Identity : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Identity` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Identity(string name, string input)
        {
            index = name;
            inputs = new[] { input };
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(index, ctx.GetPartialTensor(inputs[0]));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType);
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
    [Serializable]
    class MoveDim : Layer
    {
        /// <summary>
        /// Original positions of the dims to move. These must be unique.
        /// </summary>
        public int[] source;
        /// <summary>
        /// Destination positions for each of the original dims. These must be unique.
        /// </summary>
        public int[] destination;

        /// <summary>
        /// Initializes and returns an instance of `MoveDim` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="source">Original positions of the dims to move. These must be unique.</param>
        /// <param name="destination">Destination positions for each of the original dims. These must be unique.</param>
        public MoveDim(string name, string input, int[] source, int[] destination)
        {
            index = name;
            inputs = new[] { input };
            Logger.AssertIsTrue(source.Length == destination.Length, "MoveDim.ValueError: source and destination arrays must be same length");
            this.source = source;
            this.destination = destination;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;

            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(index, new PartialTensor(ctx.GetPartialTensor(inputs[0]).dataType));
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

            ctx.AddPartialTensor(index, new PartialTensor(ctx.GetPartialTensor(inputs[0]).dataType, shapeOut));
        }

        /// <inheritdoc/>
        public override unsafe void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);

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

            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Transpose(permutations), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Transpose(X, O, permutations);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, source: [{string.Join(", ", source)}], destination: [{string.Join(", ", destination)}]";
        }

        internal override string profilerTag => "MoveDim";
    }

    /// <summary>
    /// Represents a `Narrow` layer. The layer calculates the output tensor by slicing the input tensor along a given dim with a given start and length.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1, 2, 3)]
    class Narrow : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Narrow` layer with given dim, start and length.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="dim">The name to use for the scalar int dim tensor of the layer.</param>
        /// <param name="start">The name to use for the scalar int start tensor of the layer.</param>
        /// <param name="length">The name to use for the scalar int length tensor of the layer.</param>
        public Narrow(string name, string input, string dim, string start, string length)
        {
            index = name;
            inputs = new[] { input, dim, start, length };
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var dim = ctx.GetPartialTensor(inputs[1]);
            if (!dim.IsFullyKnown())
            {
                ctx.AddPartialTensor(index, new PartialTensor(X.dataType, SymbolicTensorShape.UnknownOfRank(X.shape.rank)));
                return;
            }

            var dimValue = dim[0].intValue;

            var outShape = X.shape;
            outShape[dimValue] = SymbolicTensorDim.Unknown;
            ctx.AddPartialTensor(index, new PartialTensor(X.dataType, outShape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var dim = ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>()[0];
            var start = ctx.vars.GetTensor(inputs[2]).ToReadOnlySpan<int>()[0];
            var length = ctx.vars.GetTensor(inputs[3]).ToReadOnlySpan<int>()[0];
            dim = X.shape.Axis(dim);
            var dimSize = X.shape[dim];
            start = (start + dimSize) % dimSize;
            var end = Mathf.Min(start + length, dimSize);
            length = end - start;
            var oShape = X.shape;
            oShape[dim] = length;
            var O = ctx.vars.AllocateTensorAndStore(index, oShape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Split(X, O, dim, start);
        }

        internal override string profilerTag => "Narrow";
    }

    /// <summary>
    /// Represents a `Pad` layer. The layer calculates the output tensor by adding padding to the input tensor according to the given padding values and mode.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1, 2, 3)]
    class Pad : Layer
    {
        /// <summary>
        /// The `PadMode` to use when padding.
        /// </summary>
        public PadMode padMode;

        /// <summary>
        /// Initializes and returns an instance of `Pad` layer without a constant value tensor.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="data">The name to use for the input tensor of the layer.</param>
        /// <param name="pads">The name to use for the 1D pad tensor of the layer.</param>
        /// <param name="mode">The `PadMode` to use when padding.</param>
        public Pad(string name, string data, string pads, PadMode mode)
        {
            index = name;
            inputs = new[] { data, pads };
            padMode = mode;
        }

        /// <summary>
        /// Initializes and returns an instance of `Pad` layer with a constant value tensor.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="data">The name to use for the input tensor of the layer.</param>
        /// <param name="pads">The name to use for the 1D pad tensor of the layer.</param>
        /// <param name="constantValue">The name to use for the scalar constant value tensor of the layer.</param>
        /// <param name="mode">The `PadMode` to use when padding.</param>
        public Pad(string name, string data, string pads, string constantValue, PadMode mode)
        {
            index = name;
            inputs = new[] { data, pads, constantValue };
            padMode = mode;
        }

        /// <summary>
        /// Initializes and returns an instance of `Pad` layer with a constant value tensor and axes tensor.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="data">The name to use for the input tensor of the layer.</param>
        /// <param name="pads">The name to use for the 1D pad tensor of the layer.</param>
        /// <param name="constantValue">The name to use for the scalar constant value tensor of the layer.</param>
        /// <param name="axes">The name to use for the scalar constant value tensor of the layer.</param>
        /// <param name="mode">The `PadMode` to use when padding.</param>
        public Pad(string name, string data, string pads, string constantValue, string axes, PadMode mode)
        {
            index = name;
            inputs = new[] { data, pads, constantValue, axes };
            padMode = mode;
        }

        /// <inheritdoc/>
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

            PartialTensor axes;

            if (inputs.Length > 3 && !string.IsNullOrEmpty(inputs[3]))
            {
                axes = ctx.GetPartialTensor(inputs[3]);
            }
            else
            {
                shapeX.DeclareRank(shapePads[0] / 2);
                axes = shapeX.hasRank ? PartialTensor.Range(0, shapeX.rank) : new PartialTensor(DataType.Int, SymbolicTensorShape.UnknownOfRank(1));
            }

            var shapeOut = SymbolicTensorShape.UnknownOfRankLike(shapeX);

            if (!axes.isPartiallyKnown)
            {
                ctx.AddPartialTensor(index, new PartialTensor(X.dataType, shapeOut));
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

            ctx.AddPartialTensor(index, new PartialTensor(X.dataType, shapeOut));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var pad = ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>();
            var axes = (inputs.Length > 3 && !string.IsNullOrEmpty(inputs[3])) ? ctx.vars.GetTensor(inputs[3]).ToReadOnlySpan<int>() : null;

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

            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Pad(pads), X.dataType, ctx.backend.backendType);
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
                var constantValue = inputs.Length > 2 && !string.IsNullOrEmpty(inputs[2]) ? ctx.vars.GetTensor(inputs[2]).ToReadOnlySpan<float>()[0] : 0f;
                if (X.shape.HasZeroDims())
                    ctx.backend.MemSet(O as TensorFloat, constantValue);
                else
                    ctx.backend.Pad(X as TensorFloat, O as TensorFloat, pads, padMode, constantValue);
            }
            else
            {
                var constantValue = inputs.Length > 2 && !string.IsNullOrEmpty(inputs[2]) ? ctx.vars.GetTensor(inputs[2]).ToReadOnlySpan<int>()[0] : 0;
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
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    class Reshape : Layer
    {
        /// <summary>
        /// Whether to handle zeros in the shape like numpy.
        ///
        /// If the shape has a dimension of size 0 and `allowZero` is `true`, the output tensor has a dimension of size zero in the same place.
        ///
        /// If the shape has a dimension of size 0 and if `allowZero` is `false`, the output tensor has the same dimension as the input tensor at this axis.
        /// </summary>
        public bool allowZero;

        /// <summary>
        /// Initializes and returns an instance of `Reshape` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="shape">The name to use for the 1D shape tensor of the layer.</param>
        /// <param name="allowZero">Whether to handle zeros in the shape like numpy.
        ///
        /// The default value is `false` and zero-sized dimensions in the shape take their value from the input tensor shape.</param>
        public Reshape(string name, string input, string shape, bool allowZero = false)
        {
            index = name;
            inputs = new[] { input, shape };
            this.allowZero = allowZero;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var shape = ctx.GetPartialTensor(inputs[1]);
            var shapeX = X.shape;
            shape.shape.DeclareRank(1);

            if (!shape.isPartiallyKnown)
            {
                if (shape.shape[0].isValue)
                    ctx.AddPartialTensor(index, X.Reshape(SymbolicTensorShape.UnknownOfRank(shape.shape[0].value)));
                else
                    ctx.AddPartialTensor(index, X.Reshape(SymbolicTensorShape.UnknownShape));
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

            ctx.AddPartialTensor(index, X.Reshape(shapeOut, !containsMinusOne));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var shape = X.shape.Reshape(ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>(), allowZero);
            var O = ctx.vars.AllocateTensorAndStore(index, shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reshape(X, O);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, allowZero: {allowZero}";
        }

        internal override string profilerTag => "Reshape";
    }

    /// <summary>
    /// Represents a `Resize` layer. The layer calculates the output tensor by resampling the input tensor along the spatial dimensions to a given shape.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    class Resize : Layer
    {
        /// <summary>
        /// The `ScaleMode` to use for the layer.
        /// </summary>
        public ScaleMode scaleMode;
        /// <summary>
        /// The `CoordTransformMode` to use for the layer.
        /// </summary>
        public CoordTransformMode coordTransformMode;
        /// <summary>
        /// The `InterpolationMode` to use for the layer.
        /// </summary>
        public InterpolationMode mode;
        /// <summary>
        /// The `NearestMode` to use for the layer when using `InterpolationMode.NearestMode`.
        /// </summary>
        public NearestMode nearestMode;
        /// <summary>
        /// The axes to resize.
        /// </summary>
        public int[] axes;

        /// <summary>
        /// Initializes and returns an instance of `Resize` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="scalesOrSizes">The name to use for the 1D scales or sizes tensor of the layer depending on the `scaleMode`.</param>
        /// <param name="scaleMode">The `ScaleMode` to use for the layer.</param>
        /// <param name="mode">The `InterpolationMode` to use for the layer.</param>
        /// <param name="coordTransformMode">The `CoordTransformMode` to use for the layer.</param>
        /// <param name="nearestMode">The `NearestMode` to use for the layer when using `InterpolationMode.NearestMode`.</param>
        /// <param name="axes">The axes to resize.</param>
        public Resize(string name, string input, string scalesOrSizes, ScaleMode scaleMode, InterpolationMode mode, CoordTransformMode coordTransformMode, NearestMode nearestMode, int[] axes)
        {
            index = name;
            inputs = new[] { input, scalesOrSizes };
            this.scaleMode = scaleMode;
            this.coordTransformMode = coordTransformMode;
            this.mode = mode;
            this.nearestMode = nearestMode;
            this.axes = axes;
        }

        /// <inheritdoc/>
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

            ctx.AddPartialTensor(index, new PartialTensor(dataType, shapeOut));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var scalesOrSizes = ctx.vars.GetTensor(inputs[1]);
            Span<float> s = stackalloc float[X.shape.rank];
            for (var i = 0; i < s.Length; i++)
                s[i] = 1f;

            if (scaleMode == ScaleMode.Sizes)
            {
                var sizes = scalesOrSizes.ToReadOnlySpan<int>();

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

                var O = ctx.vars.AllocateTensorAndStore(index, ShapeInference.Resize(X.shape, s), DataType.Float, ctx.backend.backendType) as TensorFloat;
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Resize(X, O, s, mode, nearestMode, coordTransformMode);
            }
            else
            {
                var scales = scalesOrSizes.ToReadOnlySpan<float>();

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

                var O = ctx.vars.AllocateTensorAndStore(index, ShapeInference.Resize(X.shape, scales), DataType.Float, ctx.backend.backendType) as TensorFloat;
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Resize(X, O, scales, mode, nearestMode, coordTransformMode);
            }
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, mode: {mode}, coordTransformMode: {coordTransformMode}, nearestMode: {nearestMode}";
        }

        internal override string profilerTag => "Resize";
    }

    /// <summary>
    /// Represents a `Select` layer. The layer calculates the output tensor by slicing the input tensor along a given dim with a given index, the sliced dim is removed from the output.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1, 2)]
    class Select : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Select` layer with given dim and index.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="dim">The name to use for the scalar int dim tensor of the layer.</param>
        /// <param name="index">The name to use for the scalar int index tensor of the layer.</param>
        public Select(string name, string input, string dim, string index)
        {
            this.index = name;
            inputs = new[] { input, dim, index };
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var dim = ctx.GetPartialTensor(inputs[1]);
            if (!dim.IsFullyKnown())
            {
                ctx.AddPartialTensor(index, new PartialTensor(X.dataType, SymbolicTensorShape.UnknownOfRank(X.shape.rank - 1)));
                return;
            }

            var axis = dim[0].intValue;
            var outShape = X.shape;
            outShape[axis] = SymbolicTensorDim.One;
            outShape = outShape.Squeeze(axis);
            ctx.AddPartialTensor(index, new PartialTensor(X.dataType, outShape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var dim = ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>()[0];
            var i = ctx.vars.GetTensor(inputs[2]).ToReadOnlySpan<int>()[0];
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Reduce(dim, false), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var unsqueezed = ctx.vars.AllocateTensor(X.shape.Reduce(dim, true), X.dataType, ctx.backend.backendType);
            dim = X.shape.Axis(dim);
            i = i < 0 ? i + X.shape[dim] : i;
            ctx.backend.Split(X, unsqueezed, dim, i);
            ctx.backend.Reshape(unsqueezed, O);
            ctx.vars.Dispose(unsqueezed);
        }

        internal override string profilerTag => "Select";
    }

    /// <summary>
    /// Represents a `Slice` layer. The layer calculates the output tensor by slicing the input tensor along given axes with given starts, ends and steps.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1, 2, 3, 4)]
    class Slice : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Slice` layer with given starts and ends. The layer slices the first axes of the input with step 1.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="starts">The name to use for the 1D starts tensor of the layer.</param>
        /// <param name="ends">The name to use for the 1D ends tensor of the layer.</param>
        public Slice(string name, string input, string starts, string ends)
        {
            index = name;
            inputs = new[] { input, starts, ends };
        }

        /// <summary>
        /// Initializes and returns an instance of `Slice` layer with given starts, ends and axes. The layer uses step 1.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="starts">The name to use for the 1D starts tensor of the layer.</param>
        /// <param name="ends">The name to use for the 1D ends tensor of the layer.</param>
        /// <param name="axes">The name to use for the 1D axes tensor of the layer.</param>
        public Slice(string name, string input, string starts, string ends, string axes)
        {
            index = name;
            inputs = new[] { input, starts, ends, axes };
        }

        /// <summary>
        /// Initializes and returns an instance of `Slice` layer with given starts, ends, axes, and steps.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="starts">The name to use for the 1D starts tensor of the layer.</param>
        /// <param name="ends">The name to use for the 1D ends tensor of the layer.</param>
        /// <param name="axes">The name to use for the 1D axes tensor of the layer.</param>
        /// <param name="steps">The name to use for the 1D steps tensor of the layer.</param>
        public Slice(string name, string input, string starts, string ends, string axes, string steps)
        {
            index = name;
            inputs = new[] { input, starts, ends, axes, steps };
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var data = ctx.GetPartialTensor(inputs[0]);
            if (!data.shape.hasRank)
            {
                ctx.AddPartialTensor(index, new PartialTensor(data.dataType));
                return;
            }

            var starts = ctx.GetPartialTensor(inputs[1]);
            var ends = ctx.GetPartialTensor(inputs[2]);
            var axes = (inputs.Length > 3 ? ctx.GetPartialTensor(inputs[3]) : null) ?? PartialTensor.Range(0, data.shape.rank);
            var steps = (inputs.Length > 4 ? ctx.GetPartialTensor(inputs[4]) : null) ?? PartialTensor.Ones(starts.shape);

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

                ctx.AddPartialTensor(index, tensorOut);
                return;
            }

            if (!axes.isPartiallyKnown)
            {
                ctx.AddPartialTensor(index, new PartialTensor(data.dataType, SymbolicTensorShape.UnknownOfRank(data.shape.rank)));
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

            ctx.AddPartialTensor(index, new PartialTensor(data.dataType, shapeOut));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var starts = ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>();
            var ends = ctx.vars.GetTensor(inputs[2]).ToReadOnlySpan<int>();
            var axes = inputs.Length > 3 && ctx.vars.GetTensor(inputs[3]) != null ? ctx.vars.GetTensor(inputs[3]).ToReadOnlySpan<int>() : null;
            var steps = inputs.Length > 4 && ctx.vars.GetTensor(inputs[4]) != null ? ctx.vars.GetTensor(inputs[4]).ToReadOnlySpan<int>() : null;
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Slice(starts, ends, axes, steps), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Slice(X, O, starts, axes, steps);
        }

        internal override string profilerTag => "Slice";
    }

    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(2, 3, 4, 5)]
    class SliceSet : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `SliceSet` layer with given starts and ends. The layer slices the first axes of the input with step 1.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="values">The name to use for the source tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="starts">The name to use for the 1D starts tensor of the layer.</param>
        /// <param name="ends">The name to use for the 1D ends tensor of the layer.</param>
        public SliceSet(string name, string input, string values, string starts, string ends)
        {
            index = name;
            inputs = new[] { input, values, starts, ends };
        }

        /// <summary>
        /// Initializes and returns an instance of `SliceSet` layer with given starts, ends and axes. The layer uses step 1.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="values">The name to use for the source tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="starts">The name to use for the 1D starts tensor of the layer.</param>
        /// <param name="ends">The name to use for the 1D ends tensor of the layer.</param>
        /// <param name="axes">The name to use for the 1D axes tensor of the layer.</param>
        public SliceSet(string name, string input, string values, string starts, string ends, string axes)
        {
            index = name;
            inputs = new[] { input, values, starts, ends, axes };
        }

        /// <summary>
        /// Initializes and returns an instance of `SliceSet` layer with given starts, ends, axes, and steps.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="values">The name to use for the source tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="starts">The name to use for the 1D starts tensor of the layer.</param>
        /// <param name="ends">The name to use for the 1D ends tensor of the layer.</param>
        /// <param name="axes">The name to use for the 1D axes tensor of the layer.</param>
        /// <param name="steps">The name to use for the 1D steps tensor of the layer.</param>
        public SliceSet(string name, string input, string values, string starts, string ends, string axes, string steps)
        {
            index = name;
            inputs = new[] { input, values, starts, ends, axes, steps };
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            ctx.AddPartialTensor(index, new PartialTensor(X.dataType, X.shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var values = ctx.vars.GetTensor(inputs[1]);
            var starts = ctx.vars.GetTensor(inputs[2]).ToReadOnlySpan<int>();
            var ends = ctx.vars.GetTensor(inputs[3]).ToReadOnlySpan<int>();
            var axes = inputs.Length > 4 && ctx.vars.GetTensor(inputs[4]) != null ? ctx.vars.GetTensor(inputs[4]).ToReadOnlySpan<int>() : null;
            var steps = inputs.Length > 5 && ctx.vars.GetTensor(inputs[5]) != null ? ctx.vars.GetTensor(inputs[5]).ToReadOnlySpan<int>() : null;
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var slicedShape = X.shape.Slice(starts, ends, axes, steps);
            Logger.AssertIsTrue(slicedShape.Broadcast(values.shape) == slicedShape, "SliceSet.InputError: values shape must be broadcastable to sliced shape, {0} {1}", values.shape, slicedShape);
            if (slicedShape != values.shape)
            {
                // broadcast values
                var broadcastValues = ctx.vars.AllocateTensor(slicedShape, values.dataType, ctx.backend.backendType);
                ctx.backend.Expand(values, broadcastValues);
                ctx.backend.SliceSet(X, broadcastValues, O, starts, axes, steps);
                ctx.vars.Dispose(broadcastValues);
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
    [Serializable]
    class SpaceToDepth : Layer
    {
        /// <summary>
        /// The size of the spatial blocks to move into depth.
        /// </summary>
        public int blocksize;

        /// <summary>
        /// Initializes and returns an instance of `SpaceToDepth` layer with given starts, ends, axes, and steps.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="blocksize">The size of the spatial blocks to move into depth.</param>
        public SpaceToDepth(string name, string input, int blocksize)
        {
            index = name;
            inputs = new[] { input };
            this.blocksize = blocksize;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;
            shapeX.DeclareRank(4);
            ctx.AddPartialTensor(index, new PartialTensor(ctx.GetPartialTensor(inputs[0]).dataType, new SymbolicTensorShape(shapeX[0], shapeX[1] * (blocksize * blocksize), shapeX[2] / blocksize, shapeX[3] / blocksize)));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.vars.AllocateTensorAndStore(index, ShapeInference.SpaceToDepth(X.shape, blocksize), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.SpaceToDepth(X, O, blocksize);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, blocksize: {blocksize}";
        }

        internal override string profilerTag => "SpaceToDepth";
    }

    /// <summary>
    /// Represents a `Split` layer. The layer computes the output tensors by splitting the input tensor along a single given axis.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    class Split : Layer
    {
        /// <summary>
        /// The axis along which to split.
        /// </summary>
        public int axis;
        /// <summary>
        /// The number of outputs along which to split the input tensor if no split tensor is used.
        /// </summary>
        public int numOutputs;

        /// <summary>
        /// Initializes and returns an instance of `Split` layer where the input tensor is split equally.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="outputs">The names to use for all of the output tensors of the layer.</param>
        /// <param name="axis">The axis along which to split.</param>
        /// <param name="numOutputs">The number of outputs to split the input tensor into.</param>
        public Split(string name, string input, string[] outputs, int axis, int numOutputs)
        {
            index = name;
            inputs = new[] { input };
            Logger.AssertIsTrue(outputs.Length >= 1, "Split.InputError: output array must have length at least 1");
            this.outputs = outputs;
            this.axis = axis;
            this.numOutputs = numOutputs;
            Logger.AssertIsTrue(numOutputs >= outputs.Length, "Split.InputError: numOutputs must be at least the length of output array");
        }

        /// <summary>
        /// Initializes and returns an instance of `Split` layer where the input tensor is split according to the split tensor.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="split">The name to use for the 1D split tensor of the layer.</param>
        /// <param name="outputs">The names to use for all of the output tensors of the layer.</param>
        /// <param name="axis">The axis along which to split.</param>
        public Split(string name, string input, string split, string[] outputs, int axis)
        {
            index = name;
            inputs = new[] { input, split };
            Logger.AssertIsTrue(outputs.Length >= 1, "Split.InputError: output array must have length at least 1");
            this.outputs = outputs;
            this.axis = axis;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            PartialTensor partialSplit;
            if (inputs.Length == 2)
            {
                partialSplit = ctx.GetPartialTensor(inputs[1]);
            }
            else
            {
                partialSplit = new PartialTensor(DataType.Int, new SymbolicTensorShape(numOutputs));

                var dim = ctx.GetPartialTensor(inputs[0]).shape[axis];
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
                var output = i == 0 ? index : outputs[i];
                if (string.IsNullOrEmpty(output))
                    continue;
                var outputShape = new SymbolicTensorShape(ctx.GetPartialTensor(inputs[0]).shape);
                outputShape[axis] = (SymbolicTensorDim)partialSplit[i];
                ctx.AddPartialTensor(output, new PartialTensor(ctx.GetPartialTensor(inputs[0]).dataType, outputShape));
            }
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);

            Tensor firstOutput = null;
            var dim = X.shape[axis];
            ReadOnlySpan<int> split = null;
            var equalSplitLength = 0;
            if (inputs.Length > 1 && ctx.vars.GetTensor(inputs[1]) != null)
            {
                split = ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>();
            }
            else
            {
                // if splits are not given calculate even split length
                equalSplitLength = (int)Math.Ceiling(dim / (double)numOutputs);
            }
            var start = 0;
            for (var i = 0; i < outputs.Length; i++)
            {
                var end = start + (split != null ? split[i] : equalSplitLength);
                end = Math.Min(end, dim);
                var O = ctx.vars.AllocateTensor(X.shape.Split(axis, start, end), X.dataType, ctx.backend.backendType);
                if (!O.shape.HasZeroDims())
                    ctx.backend.Split(X, O, axis, start);
                if (i == 0)
                    firstOutput = O;
                else
                    ctx.vars.Store(outputs[i], O);
                start = end;
            }
            ctx.vars.Store(index, firstOutput);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}, numOutputs: {numOutputs}";
        }

        internal override string profilerTag => "Split";
    }

    /// <summary>
    /// Represents a `Squeeze` layer. The layer computes the output tensor by reshaping the input tensor by removing dimensions of size 1.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    class Squeeze : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Squeeze` layer where the layer squeezes all the axes of size 1 from the input.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Squeeze(string name, string input)
        {
            index = name;
            inputs = new[] { input };
        }

        /// <summary>
        /// Initializes and returns an instance of `Squeeze` layer where the layer squeezes the specified axes of size 1 from the input.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axes">The name to use for the 1D axes tensor of the layer.</param>
        public Squeeze(string name, string input, string axes)
        {
            index = name;
            inputs = new[] { input, axes };
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            if (inputs.Length == 2)
            {
                var axes = ctx.GetPartialTensor(inputs[1]);
                if (!axes.isPartiallyKnown)
                    ctx.AddPartialTensor(index, X.Reshape(SymbolicTensorShape.UnknownShape));
                else if (!axes.IsFullyKnown())
                    ctx.AddPartialTensor(index, X.Reshape(SymbolicTensorShape.UnknownOfRank(X.shape.rank - axes.length)));
                else
                    ctx.AddPartialTensor(index, X.Reshape(X.shape.Squeeze(axes)));
                return;
            }

            ctx.AddPartialTensor(index, X.Reshape(X.shape.Squeeze()));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            TensorShape shape;
            if (inputs.Length > 1 && ctx.vars.GetTensor(inputs[1]) != null)
            {
                var axes = ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>();
                shape = X.shape.Squeeze(axes);
            }
            else
            {
                shape = X.shape.Squeeze();
            }
            var O = ctx.vars.AllocateTensorAndStore(index, shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reshape(X, O); // TODO<tensordata> refcount tensordata
        }

        internal override string profilerTag => "Squeeze";
    }

    /// <summary>
    /// Represents a `Tile` layer. The layer computes the output tensor by repeating the input layer a given number of times along each axis.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    class Tile : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Tile` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="repeats">The name to use for the 1D repeats tensor of the layer.</param>
        public Tile(string name, string input, string repeats)
        {
            index = name;
            inputs = new[] { input, repeats };
        }

        /// <inheritdoc/>
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
                ctx.AddPartialTensor(index, new PartialTensor(dataType, SymbolicTensorShape.UnknownOfRankLike(shapeX)));
                return;
            }

            shapeX.DeclareRank(repeats.length);

            var shapeOut = new SymbolicTensorShape(shapeX);
            for (var i = 0; i < shapeOut.rank; i++)
            {
                shapeOut[i] *= (SymbolicTensorDim)repeats[i];
            }
            ctx.AddPartialTensor(index, new PartialTensor(dataType, shapeOut));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var repeats = ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>();
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Tile(repeats), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Tile(X, O, repeats);
        }

        internal override string profilerTag => "Tile";
    }

    /// <summary>
    /// Represents a `Transpose` layer. The layer computes the output tensor by permuting the axes and data of the input tensor according to the given permutations.
    /// </summary>
    [Serializable]
    class Transpose : Layer
    {
        /// <summary>
        /// The axes to sample the output tensor from in the input tensor.
        ///
        /// If this is `null`, the layer reverses the dimensions of the input tensor in the output tensor.
        /// </summary>
        public int[] permutations;

        /// <summary>
        /// Initializes and returns an instance of `Transpose` layer with permutations as an array of integers.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="permutations">The axes to sample the output tensor from in the input tensor.</param>
        public Transpose(string name, string input, int[] permutations)
        {
            index = name;
            inputs = new[] { input };
            this.permutations = permutations;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;
            if (permutations != null)
                shapeX.DeclareRank(permutations.Length);

            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(index, new PartialTensor(ctx.GetPartialTensor(inputs[0]).dataType));
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

            ctx.AddPartialTensor(index, new PartialTensor(ctx.GetPartialTensor(inputs[0]).dataType, shapeOut));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            if (permutations == null)
            {
                var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Transpose(), X.dataType, ctx.backend.backendType);
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Transpose(X, O);
                return;
            }
            else
            {
                var O = ctx.vars.AllocateTensorAndStore(index, X.shape.Transpose(permutations), X.dataType, ctx.backend.backendType);
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Transpose(X, O, permutations);
                return;
            }
        }

        /// <inheritdoc/>
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
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    class Trilu : Layer
    {
        /// <summary>
        /// The lower or upper mode for the operation.
        /// </summary>
        public TriluMode mode;

        /// <summary>
        /// Initializes and returns an instance of `Trilu` layer with no k offset value.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="mode">The lower or upper mode for the operation.</param>
        public Trilu(string name, string input, TriluMode mode)
        {
            index = name;
            inputs = new[] { input };
            this.mode = mode;
        }

        /// <summary>
        /// Initializes and returns an instance of `Trilu` layer with k offset value.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="k">The name to use for the scalar k offset tensor of the layer.</param>
        /// <param name="mode">The lower or upper mode for the operation.</param>
        public Trilu(string name, string input, string k, TriluMode mode)
        {
            index = name;
            inputs = new[] { input, k };
            this.mode = mode;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(index, new PartialTensor(ctx.GetPartialTensor(inputs[0]).dataType, ctx.GetPartialTensor(inputs[0]).shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var k = inputs.Length > 1 && ctx.vars.GetTensor(inputs[1]) != null ? ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>()[0] : 0;
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (mode == TriluMode.Upper)
                ctx.backend.Triu(X, O, k);
            else
                ctx.backend.Tril(X, O, k);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, mode: {mode}";
        }

        internal override string profilerTag => "Trilu";
    }

    /// <summary>
    /// Represents an `Unsqueeze` layer. The layer computes the output tensor by reshaping the input tensor by adding dimensions of size 1 at the given axes.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    class Unsqueeze : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Unsqueeze` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axes">The name to use for the 1D axes tensor of the layer.</param>
        public Unsqueeze(string name, string input, string axes)
        {
            index = name;
            inputs = new[] { input, axes };
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var shape = ctx.GetPartialTensor(inputs[1]);
            ctx.AddPartialTensor(index, X.Reshape(X.shape.Unsqueeze(shape)));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var shape = X.shape.Unsqueeze(ctx.vars.GetTensor(inputs[1]).ToReadOnlySpan<int>());
            var O = ctx.vars.AllocateTensorAndStore(index, shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reshape(X, O); // TODO<tensordata> refcount tensordata
        }

        internal override string profilerTag => "Unsqueeze";
    }
}
