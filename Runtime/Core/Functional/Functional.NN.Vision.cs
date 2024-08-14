using System;
using UnityEngine;

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
            var output = FromLayer(new Layers.DepthToSpace(-1, -1, upscaleFactor, Layers.DepthToSpaceMode.DepthColumnRow), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(ShapeInference.DepthToSpace(input.shape, upscaleFactor));
            return output;
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
            var output = FromLayer(new Layers.SpaceToDepth(-1, -1, downscaleFactor), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(ShapeInference.SpaceToDepth(input.shape, downscaleFactor));
            return output;
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

            FunctionalTensor output;
            if (size != null)
            {
                output = FromLayer(new Layers.Resize(-1, -1, -1, Layers.ScaleMode.Sizes, interpolationMode, Layers.CoordTransformMode.PytorchHalfPixel, Layers.NearestMode.RoundPreferFloor, axes), input.dataType, new[] { input, Constant(size) });
                if (input.isShapeKnown)
                {
                    var outputShape = new TensorShape(input.shape);
                    for (var i = 0; i < numAxes; i++)
                        outputShape[2 + i] = size[i];
                    output.SetShape(outputShape);
                }
            }
            else
            {
                output = FromLayer(new Layers.Resize(-1, -1, -1, Layers.ScaleMode.Scales, interpolationMode, Layers.CoordTransformMode.PytorchHalfPixel, Layers.NearestMode.RoundPreferFloor, axes), input.dataType, new[] { input, Constant(scaleFactor) });
                if (input.isShapeKnown)
                {
                    output.SetShape(ShapeInference.Resize(input.shape, scaleFactor));
                }
            }
            return output;
        }

        /// <summary>
        /// Returns the input tensor by sampled by coordinates given by the grid tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="grid">The grid tensor containing the spatial coordinates per output pixel.</param>
        /// <param name="mode">The mode used for interpolating, can be 'nearest', 'bilinear', or 'bicubic'.</param>
        /// <param name="paddingMode">The mode to use for sampling out-of-bounds coordinates, can be 'zeros', 'border', or 'reflection'.</param>
        /// <param name="alignCorners">Whether to map the extreme values in the coordinates 0 and 1 to the centre of the corner pixels rather than the outer corners.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor GridSample(FunctionalTensor input, FunctionalTensor grid, string mode = "bilinear", string paddingMode = "zeros", bool alignCorners = false)
        {
            input = input.Float();
            grid = grid.Float();
            var interpolationMode = mode switch
            {
                "nearest" => Layers.InterpolationMode.Nearest,
                "bilinear" => Layers.InterpolationMode.Linear,
                "bicubic" => Layers.InterpolationMode.Cubic,
                _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, null)
            };
            var padMode = paddingMode switch
            {
                "zeros" => Layers.PaddingMode.Zeros,
                "border" => Layers.PaddingMode.Border,
                "reflection" => Layers.PaddingMode.Reflection,
                _ => throw new ArgumentOutOfRangeException(nameof(paddingMode), paddingMode, null)
            };
            var output = FromLayer(new Layers.GridSample(-1, -1, -1, interpolationMode, padMode, alignCorners), input.dataType, new[] { input, grid });
            if (input.isShapeKnown && grid.isShapeKnown)
                output.SetShape(ShapeInference.GridSample(input.shape, grid.shape));
            return output;
        }
    }
}
