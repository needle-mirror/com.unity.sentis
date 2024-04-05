using System;

namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns the result of a 1D average pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor AvgPool1D(FunctionalTensor input, int kernelSize, int? stride = null, int padding = 0)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            input = input.Float();
            var s = stride ?? kernelSize;
            return FunctionalTensor.FromLayer(new Layers.AveragePool(null, null, new[] { kernelSize }, new[] { s }, new[] { padding, padding }), input.DataType, input);
        }

        /// <summary>
        /// Returns the result of a 2D average pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor AvgPool2D(FunctionalTensor input, int kernelSize, int? stride = null, int padding = 0)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            input = input.Float();
            var s = stride ?? kernelSize;
            return FunctionalTensor.FromLayer(new Layers.AveragePool(null, null, new[] { kernelSize, kernelSize }, new[] { s, s }, new[] { padding, padding, padding, padding }), input.DataType, input);
        }

        /// <summary>
        /// Returns the result of a 2D average pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor AvgPool2D(FunctionalTensor input, (int, int) kernelSize, (int, int)? stride = null, (int, int)? padding = null)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            input = input.Float();
            var strideArray = new[] { stride?.Item1 ?? kernelSize.Item1, stride?.Item2 ?? kernelSize.Item2 };
            var paddingArray = new[] { padding?.Item1 ?? 0, padding?.Item2 ?? 0, padding?.Item1 ?? 0, padding?.Item2 ?? 0 };
            return FunctionalTensor.FromLayer(new Layers.AveragePool(null, null, new[] { kernelSize.Item1, kernelSize.Item2 }, strideArray, paddingArray), input.DataType, input);
        }

        /// <summary>
        /// Returns the result of a 3D average pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor AvgPool3D(FunctionalTensor input, int kernelSize, int? stride = null, int padding = 0)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            input = input.Float();
            var s = stride ?? kernelSize;
            return FunctionalTensor.FromLayer(new Layers.AveragePool(null, null, new[] { kernelSize, kernelSize, kernelSize }, new[] { s, s, s }, new[] { padding, padding, padding, padding, padding, padding }), input.DataType, input);
        }

        /// <summary>
        /// Returns the result of a 3D average pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor AvgPool3D(FunctionalTensor input, (int, int, int) kernelSize, (int, int, int)? stride = null, (int, int, int)? padding = null)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            input = input.Float();
            var strideArray = new[] { stride?.Item1 ?? kernelSize.Item1, stride?.Item2 ?? kernelSize.Item2, stride?.Item3 ?? kernelSize.Item3 };
            var paddingArray = new[] { padding?.Item1 ?? 0, padding?.Item2 ?? 0, padding?.Item3 ?? 0, padding?.Item1 ?? 0, padding?.Item2 ?? 0, padding?.Item3 ?? 0 };
            return FunctionalTensor.FromLayer(new Layers.AveragePool(null, null, new[] { kernelSize.Item1, kernelSize.Item2, kernelSize.Item3 }, strideArray, paddingArray), input.DataType, input);
        }

        /// <summary>
        /// Returns the result of a 1D maximum pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MaxPool1D(FunctionalTensor input, int kernelSize, int? stride = null, int padding = 0)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            input = input.Float();
            var s = stride ?? kernelSize;
            return FunctionalTensor.FromLayer(new Layers.MaxPool(null, null, new[] { kernelSize }, new[] { s }, new[] { padding, padding }), input.DataType, input);
        }

        /// <summary>
        /// Returns the result of a 2D maximum pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MaxPool2D(FunctionalTensor input, int kernelSize, int? stride = null, int padding = 0)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            input = input.Float();
            var s = stride ?? kernelSize;
            return FunctionalTensor.FromLayer(new Layers.MaxPool(null, null, new[] { kernelSize, kernelSize }, new[] { s, s }, new[] { padding, padding, padding, padding }), input.DataType, input);
        }

        /// <summary>
        /// Returns the result of a 2D maximum pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MaxPool2D(FunctionalTensor input, (int, int) kernelSize, (int, int)? stride = null, (int, int)? padding = null)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            input = input.Float();
            var strideArray = new[] { stride?.Item1 ?? kernelSize.Item1, stride?.Item2 ?? kernelSize.Item2 };
            var paddingArray = new[] { padding?.Item1 ?? 0, padding?.Item2 ?? 0, padding?.Item1 ?? 0, padding?.Item2 ?? 0 };
            return FunctionalTensor.FromLayer(new Layers.MaxPool(null, null, new[] { kernelSize.Item1, kernelSize.Item2 }, strideArray, paddingArray), input.DataType, input);
        }

        /// <summary>
        /// Returns the result of a 3D maximum pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MaxPool3D(FunctionalTensor input, int kernelSize, int? stride = null, int padding = 0)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            input = input.Float();
            var s = stride ?? kernelSize;
            return FunctionalTensor.FromLayer(new Layers.MaxPool(null, null, new[] { kernelSize, kernelSize, kernelSize }, new[] { s, s, s }, new[] { padding, padding, padding, padding, padding, padding }), input.DataType, input);
        }

        /// <summary>
        /// Returns the result of a 3D maximum pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MaxPool3D(FunctionalTensor input, (int, int, int) kernelSize, (int, int, int)? stride = null, (int, int, int)? padding = null)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            input = input.Float();
            var strideArray = new[] { stride?.Item1 ?? kernelSize.Item1, stride?.Item2 ?? kernelSize.Item2, stride?.Item3 ?? kernelSize.Item3 };
            var paddingArray = new[] { padding?.Item1 ?? 0, padding?.Item2 ?? 0, padding?.Item3 ?? 0, padding?.Item1 ?? 0, padding?.Item2 ?? 0, padding?.Item3 ?? 0 };
            return FunctionalTensor.FromLayer(new Layers.MaxPool(null, null, new[] { kernelSize.Item1, kernelSize.Item2, kernelSize.Item3 }, strideArray, paddingArray), input.DataType, input);
        }
    }
}
