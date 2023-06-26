using System;
using UnityEngine;

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
    public class Cast : Layer
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
            this.name = name;
            inputs = new[] { input };
            this.toType = toType;
        }

        internal override PartialTensor InferPartialTensor(PartialTensor[] partialTensors, ShapeInferenceContext ctx)
        {
            if (toType == DataType.Int)
                return partialTensors[0];
            return PartialTensor.Unknown;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return inputShapes[0];
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.Cast(inputTensors[0], toType);
        }

        internal override string profilerTag => "Cast";
    }

    /// <summary>
    /// Represents an element-wise `CastLike` layer: f(x) = (float)x or f(x) = (int)x depending on the data type of the targetType tensor.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.NoDataDependencyInputs(1)]
    public class CastLike : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `CastLike` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="targetType">The name to use for the targetType tensor of the layer. The layer does not use the values of this tensor in the computation.</param>
        public CastLike(string name, string input, string targetType)
        {
            this.name = name;
            inputs = new[] { input, targetType };
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return inputShapes[0];
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.Cast(inputTensors[0], inputTensors[1].dataType);
        }

        internal override string profilerTag => "CastLike";
    }

    /// <summary>
    /// Represents a `Concat` concatenation layer. The layer computes the output tensor by concatenating the input tensors along a given axis.
    /// </summary>
    [Serializable]
    public class Concat : Layer
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
            this.name = name;
            this.inputs = inputs;
            this.axis = axis;
        }

        internal override PartialTensor InferPartialTensor(PartialTensor[] partialTensors, ShapeInferenceContext ctx)
        {
            return PartialInferenceHelper.Concat(partialTensors, axis);
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.Concat(inputShapes, axis);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.Concat(inputTensors, axis);
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
    public class DepthToSpace : Layer
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
            this.name = name;
            inputs = new[] { input };
            this.blocksize = blocksize;
            this.mode = mode;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.DepthToSpace(inputShapes[0], blocksize);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.DepthToSpace(inputTensors[0] as TensorFloat, blocksize, mode);
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
    public class Expand : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Expand` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="shape">The name to use for the 1D shape tensor of the layer.</param>
        public Expand(string name, string input, string shape)
        {
            this.name = name;
            this.inputs = new[] { input, shape };
        }

        internal override PartialTensor InferPartialTensor(PartialTensor[] partialTensors, ShapeInferenceContext ctx)
        {
            ctx.AddShape(name, SymbolicInference.Expand(ctx.GetSymbolicTensorShape(inputs[0]), partialTensors[1]));
            return base.InferPartialTensor(partialTensors, ctx);
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            var partialTensor = ctx.GetPartialTensor(inputs[1]);
            if (partialTensor.isPartiallyKnown)
                return SymbolicInference.Broadcast(new[] { inputShapes[0], partialTensor.ToSymbolicTensorShape() });

            return SymbolicInference.Expand(inputShapes[0], inputShapes[1]);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var shape = (inputTensors[1] as TensorInt).ToReadOnlyArray();
            return ctx.ops.Expand(inputTensors[0], new TensorShape(shape));
        }

        internal override string profilerTag => "Expand";
    }

    /// <summary>
    /// Represents a `Flatten` layer. The layer computes the output tensor by reshaping the input tensor into a 2D matrix according to the given axis.
    /// </summary>
    [Serializable]
    public class Flatten : Layer
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
            this.name = name;
            inputs = new[] { input };
            this.axis = axis;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.Flatten(inputShapes[0], axis);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            Tensor X = inputTensors[0];

            return ctx.ops.Reshape(X, X.shape.Flatten(axis));
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
    public class Identity : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Identity` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Identity(string name, string input)
        {
            this.name = name;
            inputs = new[] { input };
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return inputShapes[0];
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.Copy(inputTensors[0]);
        }

        internal override string profilerTag => "Identity";
    }

    /// <summary>
    /// Represents a `Pad` layer. The layer calculates the output tensor by adding padding to the input tensor according to the given padding values and mode.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1, 2)]
    public class Pad : Layer
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
            this.name = name;
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
            this.name = name;
            inputs = new[] { data, pads, constantValue };
            padMode = mode;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            var pads = ctx.GetPartialTensor(inputs[1]);
            return SymbolicInference.Pad(inputShapes[0], inputShapes[1], pads);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var pads = (inputTensors[1] as TensorInt).ToReadOnlyArray();
            var constantValue = inputTensors.Length > 2 && inputTensors[2] != null ? (inputTensors[2] as TensorFloat)[0] : 0f;
            return ctx.ops.Pad(inputTensors[0] as TensorFloat, pads, padMode, constantValue);
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
    public class Reshape : Layer
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
            this.name = name;
            inputs = new[] { input, shape };
            this.allowZero = allowZero;
        }

        internal override PartialTensor InferPartialTensor(PartialTensor[] partialTensors, ShapeInferenceContext ctx)
        {
            var shape = partialTensors[1];
            var outputShape = SymbolicInference.Reshape(ctx.GetSymbolicTensorShape(inputs[0]), ctx.GetSymbolicTensorShape(inputs[1]), ref shape, allowZero);
            ctx.AddShape(name, outputShape);

            if (shape.IsFullyKnown())
            {
                ctx.AddPartialTensor(inputs[1], shape);
            }

            return PartialTensor.Unknown;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            var shape = ctx.GetPartialTensor(inputs[1]);
            var outputShape = SymbolicInference.Reshape(inputShapes[0], inputShapes[1], ref shape, allowZero);
            if (shape.IsFullyKnown())
            {
                ctx.AddPartialTensor(inputs[1], shape);
            }
            return outputShape;
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var size = (inputTensors[1] as TensorInt).ToReadOnlyArray();
            return ctx.ops.Reshape(inputTensors[0], inputTensors[0].shape.Reshape(size, allowZero));
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
    public class Resize : Layer
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
        /// Initializes and returns an instance of `Resize` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="scalesOrSizes">The name to use for the 1D scales or sizes tensor of the layer depending on the `scaleMode`.</param>
        /// <param name="scaleMode">The `ScaleMode` to use for the layer.</param>
        /// <param name="mode">The `InterpolationMode` to use for the layer.</param>
        /// <param name="coordTransformMode">The `CoordTransformMode` to use for the layer.</param>
        /// <param name="nearestMode">The `NearestMode` to use for the layer when using `InterpolationMode.NearestMode`.</param>
        public Resize(string name, string input, string scalesOrSizes, ScaleMode scaleMode, InterpolationMode mode, CoordTransformMode coordTransformMode, NearestMode nearestMode)
        {
            this.name = name;
            inputs = new[] { input, scalesOrSizes };
            this.scaleMode = scaleMode;
            this.coordTransformMode = coordTransformMode;
            this.mode = mode;
            this.nearestMode = nearestMode;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            if (scaleMode == ScaleMode.Sizes)
                return SymbolicInference.ResizeSizes(inputShapes[0], inputShapes[1], ctx.GetPartialTensor(inputs[1]));

            return SymbolicInference.ResizeScales(inputShapes[0], inputShapes[1], (ctx.GetKnownTensor(inputs[1]) as TensorFloat)?.ToReadOnlyArray());
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            float[] scales;
            if (scaleMode == ScaleMode.Sizes)
            {
                var inputShape = inputTensors[0].shape;
                scales = new float[inputShape.rank];

                var sizes = (inputTensors[1] as TensorInt).ToReadOnlyArray();

                for (var i = 0; i < scales.Length; i++)
                {
                    scales[i] = sizes[i] / (float)inputShape[i];
                }
            }
            else
            {
                scales = (inputTensors[1] as TensorFloat).ToReadOnlyArray();
            }

            return ctx.ops.Resize(inputTensors[0] as TensorFloat, scales, mode, nearestMode, coordTransformMode);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, coordTransformMode: {coordTransformMode}, nearestMode: {nearestMode}";
        }

        internal override string profilerTag => "Resize";
    }

    /// <summary>
    /// Represents a `Slice` layer. The layer calculates the output tensor by slicing the input tensor along given axes with given starts, ends and steps.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1, 2, 3, 4)]
    public class Slice : Layer
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
            this.name = name;
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
            this.name = name;
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
            this.name = name;
            inputs = new[] { input, starts, ends, axes, steps };
        }

        internal override PartialTensor InferPartialTensor(PartialTensor[] partialTensors, ShapeInferenceContext ctx)
        {
            if (inputs.Length == 3)
                return PartialInferenceHelper.Slice(partialTensors[0], partialTensors[1], partialTensors[2]);
            if (inputs.Length == 4)
                return PartialInferenceHelper.Slice(partialTensors[0], partialTensors[1], partialTensors[2], partialTensors[3]);
            return PartialInferenceHelper.Slice(partialTensors[0], partialTensors[1], partialTensors[2], partialTensors[3], partialTensors[4]);
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            var starts = ctx.GetPartialTensor(inputs[1]);
            var ends = ctx.GetPartialTensor(inputs[2]);
            if (inputs.Length == 3)
                return SymbolicInference.Slice(inputShapes, starts, ends);
            var axes = ctx.GetPartialTensor(inputs[3]);
            if (inputs.Length == 4)
                return SymbolicInference.Slice(inputShapes, starts, ends, axes);
            var steps = ctx.GetPartialTensor(inputs[4]);
            return SymbolicInference.Slice(inputShapes, starts, ends, axes, steps);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var axes = inputTensors.Length > 3 && inputTensors[3] != null ? (inputTensors[3] as TensorInt).ToReadOnlyArray() : null;
            var steps = inputTensors.Length > 4 && inputTensors[4] != null ? (inputTensors[4] as TensorInt).ToReadOnlyArray() : null;
            return ctx.ops.Slice(inputTensors[0], (inputTensors[1] as TensorInt).ToReadOnlyArray(), (inputTensors[2] as TensorInt).ToReadOnlyArray(), axes, steps);
        }

        internal override string profilerTag => "Slice";
    }

    /// <summary>
    /// Represents a `SpaceToDepth` layer. The layer computes the output tensor by permuting data from blocks of spatial data into depth.
    /// </summary>
    [Serializable]
    public class SpaceToDepth : Layer
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
            this.name = name;
            inputs = new[] { input };
            this.blocksize = blocksize;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.SpaceToDepth(inputShapes[0], blocksize);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.SpaceToDepth(inputTensors[0] as TensorFloat, blocksize);
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
    public class Split : Layer
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
            this.name = name;
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
            this.name = name;
            inputs = new[] { input, split };
            Logger.AssertIsTrue(outputs.Length >= 1, "Split.InputError: output array must have length at least 1");
            this.outputs = outputs;
            this.axis = axis;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            // initialize split lengths along axis to Unknown
            PartialTensor symbolicSplit;
            if (inputs.Length == 2)
            {
                // set split lengths along axis to values from partial tensor
                symbolicSplit = ctx.GetPartialTensor(inputs[1]);
            }
            else
            {
                // set split lengths along axis to split dim length divided by numOutputs
                symbolicSplit = SymbolicInference.SplitDim(inputShapes[0][axis], numOutputs);
            }

            // calculate first output shape to return
            var retShape = new SymbolicTensorShape(inputShapes[0]);
            if (retShape.hasRank)
                retShape[axis] = symbolicSplit[0].ToSymbolicTensorDim();
            for (var i = 1; i < outputs.Length; i++)
            {
                // calculate and store other output shapes
                var outputShape = new SymbolicTensorShape(inputShapes[0]);
                if (outputShape.hasRank)
                    outputShape[axis] = symbolicSplit[i].ToSymbolicTensorDim();
                ctx.AddShape(outputs[i], outputShape);
            }

            return retShape;
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            Tensor firstOutput = null;
            var dim = inputTensors[0].shape[axis];
            // if splits are not given calculate even split length
            var equalSplitLength = inputTensors.Length == 2 ? 0 : (int)Math.Ceiling(dim / (double)numOutputs);
            var start = 0;
            for (var i = 0; i < outputs.Length; i++)
            {
                var end = start + (inputTensors.Length == 2 ? (inputTensors[1] as TensorInt)[i] : equalSplitLength);
                end = Math.Min(end, dim);
                var output = ctx.ops.Split(inputTensors[0], axis, start, end);
                if (i == 0)
                    firstOutput = output;
                else
                    ctx.vars.Store(outputs[i], output);
                start = end;
            }
            return firstOutput;
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
    public class Squeeze : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Squeeze` layer where the layer squeezes all the axes of size 1 from the input.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public Squeeze(string name, string input)
        {
            this.name = name;
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
            this.name = name;
            inputs = new[] { input, axes };
        }

        internal override PartialTensor InferPartialTensor(PartialTensor[] partialTensors, ShapeInferenceContext ctx)
        {
            if (inputs.Length == 2)
                return PartialInferenceHelper.Squeeze(partialTensors[0], partialTensors[1]);

            return PartialInferenceHelper.Squeeze(partialTensors[0]);
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            if (inputs.Length == 2)
            {
                var axes = ctx.GetPartialTensor(inputs[1]);
                return SymbolicInference.Squeeze(inputShapes[0], axes);
            }

            return inputShapes[0].Squeeze();
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var X = inputTensors[0];
            if (inputTensors.Length == 2)
            {
                var axes = (inputTensors[1] as TensorInt).ToReadOnlyArray();
                return ctx.ops.Reshape(X, X.shape.Squeeze(axes));
            }

            return ctx.ops.Reshape(X, X.shape.Squeeze());
        }

        internal override string profilerTag => "Squeeze";
    }

    /// <summary>
    /// Represents a `Tile` layer. The layer computes the output tensor by repeating the input layer a given number of times along each axis.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    public class Tile : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Tile` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="repeats">The name to use for the 1D repeats tensor of the layer.</param>
        public Tile(string name, string input, string repeats)
        {
            this.name = name;
            inputs = new[] { input, repeats };
        }

        internal override PartialTensor InferPartialTensor(PartialTensor[] partialTensors, ShapeInferenceContext ctx)
        {
            var outputShape = SymbolicInference.Tile(ctx.GetSymbolicTensorShape(inputs[0]), ctx.GetSymbolicTensorShape(inputs[1]), partialTensors[1]);
            ctx.AddShape(name, outputShape);
            return PartialTensor.Unknown;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.Tile(inputShapes[0], inputShapes[1], ctx.GetPartialTensor(inputs[1]));
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var repeats = (inputTensors[1] as TensorInt).ToReadOnlyArray();
            return ctx.ops.Tile(inputTensors[0], repeats);
        }

        internal override string profilerTag => "Tile";
    }

    /// <summary>
    /// Represents a `Transpose` layer. The layer computes the output tensor by permuting the axes and data of the input tensor according to the given permutations.
    /// </summary>
    [Serializable]
    public class Transpose : Layer
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
            this.name = name;
            inputs = new[] { input };
            this.permutations = permutations;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.Transpose(inputShapes[0], permutations);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (permutations == null)
                return ctx.ops.Transpose(inputTensors[0]);
            else
                return ctx.ops.Transpose(inputTensors[0], permutations);
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
    public class Trilu : Layer
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
            this.name = name;
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
            this.name = name;
            inputs = new[] { input, k };
            this.mode = mode;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return inputShapes[0];
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            int k = inputTensors.Length == 1 ? 0 : (inputTensors[1] as TensorInt)[0];
            if (mode == TriluMode.Upper)
                return ctx.ops.Triu(inputTensors[0], k);
            else
                return ctx.ops.Tril(inputTensors[0], k);
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
    public class Unsqueeze : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Unsqueeze` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axes">The name to use for the 1D axes tensor of the layer.</param>
        public Unsqueeze(string name, string input, string axes)
        {
            this.name = name;
            inputs = new[] { input, axes };
        }

        internal override PartialTensor InferPartialTensor(PartialTensor[] partialTensors, ShapeInferenceContext ctx)
        {
            return PartialInferenceHelper.Unsqueeze(partialTensors[0], partialTensors[1]);
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            var axes = ctx.GetPartialTensor(inputs[1]);
            return SymbolicInference.Unsqueeze(inputShapes[0], axes);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            Tensor X = inputTensors[0];
            var axes = (inputTensors[1] as TensorInt).ToReadOnlyArray();
            return ctx.ops.Reshape(X, X.shape.Unsqueeze(axes));
        }

        internal override string profilerTag => "Unsqueeze";
    }
}
