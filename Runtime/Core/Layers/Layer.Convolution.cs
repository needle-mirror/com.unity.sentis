using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Options for auto padding in image layers.
    /// </summary>
    public enum AutoPad
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
    [Serializable]
    public class Conv : FusedActivation
    {
        /// <summary>
        /// The auto padding mode of the convolution as an `AutoPad`.
        /// </summary>
        public AutoPad autoPad;
        /// <summary>
        /// The dilation value of each spatial dimension of the filter.
        ///
        /// If this is `null` the layer uses a default of [1, 1, ..., 1].
        /// </summary>
        public int[] dilations;
        /// <summary>
        /// The number of groups that input channels and output channels are divided into.
        /// </summary>
        public int group;
        /// <summary>
        /// The lower and upper padding values for each spatial dimension of the filter, [pad_left, pad_right] for 1D, [pad_top, pad_bottom, pad_left, pad_right] for 2D, etc.
        ///
        /// If this is `null` the layer uses a default of [0, 0, ..., 0].
        /// </summary>
        public int[] pads;
        /// <summary>
        /// The stride value for each spatial dimension of the filter.
        ///
        /// If this is `null` the layer uses a default of [1, 1, ..., 1].
        /// </summary>
        public int[] strides;

        /// <summary>
        /// Initializes and returns an instance of `Conv` convolution layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="X">The name to use for the input tensor of the layer.</param>
        /// <param name="W">The name to use for the filter tensor of the layer.</param>
        /// <param name="B">The name to use for the optional bias tensor of the layer.</param>
        /// <param name="group">The number of groups that input channels and output channels are divided into.</param>
        /// <param name="strides">The optional stride value for each spatial dimension of the filter.</param>
        /// <param name="pads">The optional lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="dilations">The optional dilation value of each spatial dimension of the filter.</param>
        /// <param name="autoPad">The auto padding mode of the convolution as an `AutoPad`.</param>
        /// <param name="fusedActivation">The fused activation type to apply after the convolution. The default value is `None`.</param>
        public Conv(string name, string X, string W, string B, int group, int[] strides, int[] pads, int[] dilations, AutoPad autoPad = AutoPad.NotSet, FusableActivation fusedActivation = FusableActivation.None)
        {
            this.name = name;
            inputs = new[] { X, W, B };
            this.autoPad = autoPad;
            this.dilations = dilations;
            this.group = group;
            this.pads = pads;
            this.strides = strides;
            this.fusedActivation = fusedActivation;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.Conv(inputShapes[0], inputShapes[1], inputShapes[2], strides, pads, dilations, autoPad);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            ShapeInference.UpdatePadForConvAutoPadding(inputTensors[0].shape, inputTensors[1].shape, strides, dilations, autoPad, pads);
            return ctx.ops.Conv(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, inputTensors[2] as TensorFloat, group, strides, pads, dilations, fusedActivation);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, group: {group}, strides: [{string.Join(", ", strides)}], pads: [{string.Join(", ", pads)}], dilations: [{string.Join(", ", dilations)}], autoPad: {autoPad}, fusedActivation: {fusedActivation}";
        }

        internal override string profilerTag => "Conv";
    }

    /// <summary>
    /// Represents a `ConvTranspose` transpose convolution layer, which applies a convolution filter to an input tensor.
    /// </summary>
    [Serializable]
    public class Conv2DTrans : FusedActivation
    {
        /// <summary>
        /// The auto padding mode of the transpose convolution.
        /// </summary>
        public AutoPad autoPad;
        /// <summary>
        /// The output padding value for each spatial dimension in the filter.
        ///
        /// The layer adds the output padding to the side with higher coordinate indices in the output tensor.
        ///
        /// If this is `null` the layer uses a default of [0, 0, ..., 0].
        /// </summary>
        public int[] outputPadding;
        /// <summary>
        /// The lower and upper padding values for each spatial dimension of the filter. For example [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.
        ///
        /// If this is `null` the layer uses a default of [0, 0, ..., 0].
        /// </summary>
        public int[] pads;
        /// <summary>
        /// The stride value for each spatial dimension of the filter.
        ///
        /// If this is `null` the layer uses a default of [1, 1, ..., 1].
        /// </summary>
        public int[] strides;

        /// <summary>
        /// Initializes and returns an instance of `ConvTranspose` convolution layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="kernel">The name to use for the filter tensor of the layer.</param>
        /// <param name="bias">The name to use for the optional bias tensor of the layer.</param>
        /// <param name="strides">The optional stride value for each spatial dimension of the filter.</param>
        /// <param name="pads">The optional lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="autoPad">The auto padding mode of the convolution.</param>
        /// <param name="outputPadding">The output padding value for each spatial dimension in the filter.</param>
        /// <param name="fusedActivation">The fused activation type to apply after the convolution. The default value is `None`.</param>
        public Conv2DTrans(string name, string input, string kernel, string bias, int[] strides, int[] pads, AutoPad autoPad, int[] outputPadding, FusableActivation fusedActivation = FusableActivation.None)
        {
            this.name = name;
            inputs = new[] { input, kernel, bias };
            this.autoPad = autoPad;
            this.outputPadding = outputPadding;
            this.pads = pads;
            this.strides = strides;
            this.fusedActivation = fusedActivation;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.ConvTranspose(inputShapes[0], inputShapes[1], inputShapes[2], strides, pads, autoPad, outputPadding);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            ShapeInference.UpdatePadForConvTransAutoPadding(inputTensors[0].shape, inputTensors[1].shape, strides, autoPad, outputPadding, pads);
            return ctx.ops.Conv2DTrans(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, inputTensors[2] as TensorFloat, strides, pads, outputPadding, fusedActivation);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, strides: [{string.Join(", ", strides)}], pads: [{string.Join(", ", pads)}], outputPadding: [{string.Join(", ", outputPadding)}], autoPad, {autoPad}, fusedActivation: {fusedActivation}";
        }

        internal override string profilerTag => "ConvTranspose";
    }
}
