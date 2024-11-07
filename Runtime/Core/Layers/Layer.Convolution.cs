using System;
using Unity.Profiling;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Options for auto padding in image layers.
    /// </summary>
    enum AutoPad
    {
        /// <summary>
        /// Use explicit padding.
        /// </summary>
        NotSet,
        /// <summary>
        /// Use no padding.
        /// </summary>
        Valid,
        /// <summary>
        /// Use equal or almost equal padding on both sides. When the padding is odd, add the extra value to the end.
        /// </summary>
        SameUpper,
        /// <summary>
        /// Use equal or almost equal padding on both sides. When the padding is odd, add the extra value to the start.
        /// </summary>
        SameLower,
    }

    /// <summary>
    /// Represents a `Conv` convolution layer, which applies a convolution filter to an input tensor.
    /// </summary>
    class Conv : FusedActivation
    {
        static readonly string k_OpName = "Conv";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public AutoPad autoPad;
        public int[] dilations;
        public int group;
        public int[] pads;
        public int[] strides;
        public int[] kernelShape;

        public Conv(int output, int X, int W, int B = -1, int group = 1, int[] strides = null, int[] pads = null, int[] dilations = null, AutoPad autoPad = AutoPad.NotSet, int[] kernelShape = null, FusableActivation fusedActivation = FusableActivation.None)
            : base(new[] { output }, new[] { X, W, B })
        {
            this.dilations = dilations ?? new[] { 1, 1, 1, 1, 1, 1 };
            this.group = group;
            this.strides = strides ?? new[] { 1, 1, 1, 1, 1, 1 };
            this.pads = pads ?? new int[12];
            this.autoPad = autoPad;
            this.kernelShape = kernelShape;
            this.fusedActivation = fusedActivation;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var W = ctx.GetPartialTensor(inputs[1]);
            var shapeX = X.shape;
            var shapeKernel = W.shape;
            for (var i = 0; kernelShape != null && i < kernelShape.Length; i++)
            {
                shapeKernel[i + 2] = DynamicTensorDim.MaxDefinedDim(shapeKernel[i + 2], DynamicTensorDim.Int(kernelShape[i]));
            }

            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float));
                return;
            }

            shapeKernel.DeclareRank(shapeX.rank);

            var shapeOut = DynamicTensorShape.DynamicOfRank(shapeX.rank);

            shapeOut[0] = shapeX[0];
            shapeOut[1] = shapeKernel[0];

            var shapeBias = ctx.GetPartialTensor(inputs[2])?.shape ?? DynamicTensorShape.DynamicRank;
            shapeBias.DeclareRank(1);
            shapeOut[1] = DynamicTensorDim.MaxDefinedDim(shapeOut[1], shapeBias[0]);

            for (var i = 2; i < shapeOut.rank; i++)
            {
                var stride = strides == null ? 1 : strides[i - 2];
                var pad = pads == null || autoPad != AutoPad.NotSet ? 0 : pads[i - 2] + pads[i - 2 + (shapeX.rank - 2)];
                var dilation = dilations == null ? 1 : dilations[i - 2];
                var dimX = shapeX[i];
                var dimKernel = shapeKernel[i];
                if (dimKernel.isValue)
                    shapeOut[i] = dimX.Pool(dimKernel.value, stride, pad, dilation, false, autoPad);
                else if (dimKernel.isParam && (autoPad is AutoPad.SameLower || autoPad is AutoPad.SameUpper))
                    shapeOut[i] = dimX.Pool(0, stride, pad, dilation, false, autoPad);
                else
                    shapeOut[i] = DynamicTensorDim.Unknown;
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var W = ctx.storage.GetTensor(inputs[1]) as Tensor<float>;
            var B = ctx.storage.GetTensor(inputs[2]) as Tensor<float>;

            var numSpatialDims = X.shape.rank - 2;
            var stridesSpan = strides.AsSpan(0, numSpatialDims);
            var padsSpan = pads.AsSpan(0, 2 * numSpatialDims);
            var dilationsSpan = dilations.AsSpan(0, numSpatialDims);
            ShapeInference.UpdatePadForConvAutoPadding(X.shape, W.shape, stridesSpan, dilationsSpan, autoPad, padsSpan);
            var shapeO = ShapeInference.Conv(X.shape, W.shape, group, stridesSpan, padsSpan, dilationsSpan);

            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shapeO, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;

            ctx.backend.Conv(X, W, B, O, group, stridesSpan, padsSpan, dilationsSpan, fusedActivation);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, group: {group}, strides: [{(string.Join(", ", strides))}], pads: [{(string.Join(", ", pads))}], dilations: [{(string.Join(", ", dilations))}], autoPad: {autoPad}, kernelShape: [{(kernelShape == null ? "null" : string.Join(", ", kernelShape))}], fusedActivation: {fusedActivation}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `ConvTranspose` transpose convolution layer, which applies a convolution filter to an input tensor.
    /// </summary>
    class ConvTranspose : FusedActivation
    {
        static readonly string k_OpName = "ConvTranspose";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public AutoPad autoPad;
        public int[] outputPadding;
        public int[] pads;
        public int[] strides;
        public int[] kernelShape;

        public ConvTranspose(int output, int input, int kernel, int bias = -1, int[] strides = null, int[] pads = null, AutoPad autoPad = AutoPad.NotSet, int[] outputPadding = null, int[] kernelShape = null, FusableActivation fusedActivation = FusableActivation.None)
            : base(new[] { output }, new[] { input, kernel, bias })
        {
            this.autoPad = autoPad;
            this.outputPadding = outputPadding ?? new int[6];
            this.pads = pads ?? new int[12];
            this.strides = strides ?? new[] { 1, 1, 1, 1, 1, 1 };
            this.kernelShape = kernelShape;
            this.fusedActivation = fusedActivation;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var W = ctx.GetPartialTensor(inputs[1]);
            var shapeX = X.shape;
            var shapeKernel = W.shape;
            for (var i = 0; kernelShape != null && i < kernelShape.Length; i++)
            {
                shapeKernel[i + 2] = DynamicTensorDim.MaxDefinedDim(shapeKernel[i + 2], DynamicTensorDim.Int(kernelShape[i]));
            }

            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float));
                return;
            }

            shapeKernel.DeclareRank(shapeX.rank);

            var shapeOut = DynamicTensorShape.Ones(shapeX.rank);

            shapeOut[0] = shapeX[0];
            shapeOut[1] = shapeKernel[1];

            var shapeBias = ctx.GetPartialTensor(inputs[2])?.shape ?? DynamicTensorShape.DynamicRank;
            shapeBias.DeclareRank(1);
            shapeOut[1] = DynamicTensorDim.MaxDefinedDim(shapeOut[1], shapeBias[0]);

            for (var i = 2; i < shapeOut.rank; i++)
            {
                var stride = strides == null ? 1 : strides[i - 2];
                var pad = pads == null || autoPad != AutoPad.NotSet ? 0 : pads[i - 2] + pads[i - 2 + (shapeX.rank - 2)];
                var dilation = 1;
                var outputPad = outputPadding == null ? 0 : outputPadding[i - 2];
                var dimX = shapeX[i];
                var dimKernel = shapeKernel[i];
                if (autoPad == AutoPad.NotSet)
                    shapeOut[i] = stride * (dimX - 1) + outputPad + (dimKernel - 1) * dilation + 1 - pad;
                else
                    shapeOut[i] = dimX * stride;
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var W = ctx.storage.GetTensor(inputs[1]) as Tensor<float>;
            var B = ctx.storage.GetTensor(inputs[2]) as Tensor<float>;

            var numSpatialDims = X.shape.rank - 2;
            var stridesSpan = strides.AsSpan(0, numSpatialDims);
            var padsSpan = pads.AsSpan(0, 2 * numSpatialDims);
            var outputPaddingSpan = outputPadding.AsSpan(0, numSpatialDims);
            ShapeInference.UpdatePadForConvTransAutoPadding(X.shape, W.shape, stridesSpan, autoPad, outputPaddingSpan, padsSpan);
            var shapeO = ShapeInference.ConvTranspose(X.shape, W.shape, stridesSpan, padsSpan, outputPaddingSpan);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shapeO, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.ConvTranspose(X, W, B, O, stridesSpan, padsSpan, outputPaddingSpan, fusedActivation);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, strides: [{(string.Join(", ", strides))}], pads: [{(string.Join(", ", pads))}], outputPadding: [{(string.Join(", ", outputPadding))}], autoPad, {autoPad}, kernelShape: [{(kernelShape == null ? "null" : string.Join(", ", kernelShape))}], fusedActivation: {fusedActivation}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }
}
