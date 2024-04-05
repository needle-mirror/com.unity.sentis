using System;

namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns the elements of the input tensor rearranged from a (∗,C×r^2,H,W) tensor to a (∗,C,H×r,W×r) tensor where r is the upscale factor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="upscaleFactor">The upscale factor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor PixelShuffle(FunctionalTensor input, int upscaleFactor)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.DepthToSpace(null, null, upscaleFactor, Layers.DepthToSpaceMode.DepthColumnRow), input.DataType, input);
        }

        /// <summary>
        /// Returns the elements of the input tensor rearranged from a (∗,C,H×r,W×r) tensor to a (∗,C×r^2,H,W) tensor where r is the downscale factor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="downscaleFactor">The downscale factor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor PixelUnshuffle(FunctionalTensor input, int downscaleFactor)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.SpaceToDepth(null, null, downscaleFactor), input.DataType, input);
        }

        /// <summary>
        /// Returns the input tensor with the spatial dimensions downsampled or upsampled to a size or by a scale factor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="size">The optional output size.</param>
        /// <param name="scaleFactor">The optional output scale factors.</param>
        /// <param name="mode">The mode used for interpolating, can be 'nearest', 'linear', or 'bicubic'.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Interpolate(FunctionalTensor input, int[] size = null, float[] scaleFactor = null, string mode = "nearest")
        {
            // TODO add recompute_scale_factor, antialias, single value size, scaleFactor
            input = input.Float();
            var interpolationMode = mode switch
            {
                "nearest" => Layers.InterpolationMode.Nearest,
                "linear" => Layers.InterpolationMode.Linear,
                "bicubic" => Layers.InterpolationMode.Cubic,
                _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, null)
            };
            var numAxes = size?.Length ?? scaleFactor.Length;
            var axes = new int[numAxes];
            for (var i = 0; i < numAxes; i++)
                axes[i] = 2 + i;
            if (size != null)
                return FunctionalTensor.FromLayer(new Layers.Resize(null, null, null, Layers.ScaleMode.Sizes, interpolationMode, Layers.CoordTransformMode.PytorchHalfPixel, Layers.NearestMode.RoundPreferFloor, axes), input.DataType, new[] { input, Tensor(size) });
            return FunctionalTensor.FromLayer(new Layers.Resize(null, null, null, Layers.ScaleMode.Scales, interpolationMode, Layers.CoordTransformMode.PytorchHalfPixel, Layers.NearestMode.RoundPreferFloor, axes), input.DataType, new[] { input, Tensor(scaleFactor) });
        }
    }
}
