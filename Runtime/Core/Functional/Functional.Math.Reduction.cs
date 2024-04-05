using System;

namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns the indices of the maximum value of the elements of the input along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ArgMax(FunctionalTensor input, int dim = 0, bool keepdim = false)
        {
            return FunctionalTensor.FromLayer(new Layers.ArgMax(null, null, dim, keepdim), DataType.Int, input);
        }

        /// <summary>
        /// Returns the indices of the minimum value of the elements of the input along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ArgMin(FunctionalTensor input, int dim = 0, bool keepdim = false)
        {
            return FunctionalTensor.FromLayer(new Layers.ArgMin(null, null, dim, keepdim), DataType.Int, input);
        }

        /// <summary>
        /// Returns the maximum value of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceMax(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            return FunctionalTensor.FromLayer(new Layers.ReduceMax(null, new string[2], keepdim), input.DataType, new[] { input, Tensor(dim) });
        }

        /// <summary>
        /// Returns the maximum value of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceMax(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceMax(input, new[] { dim }, keepdim);
        }

        /// <summary>
        /// Returns the minimum value of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceMin(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            return FunctionalTensor.FromLayer(new Layers.ReduceMin(null, new string[2], keepdim), input.DataType, new[] { input, Tensor(dim) });
        }

        /// <summary>
        /// Returns the minimum value of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceMin(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceMin(input, new[] { dim }, keepdim);
        }

        /// <summary>
        /// Returns the log of summed exponentials of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceLogSumExp(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.ReduceLogSumExp(null, new string[2], keepdim), input.DataType, new[] { input, Tensor(dim) });
        }

        /// <summary>
        /// Returns the log of summed exponentials of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceLogSumExp(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceLogSumExp(input, new[] { dim }, keepdim);
        }

        /// <summary>
        /// Returns the mean of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceMean(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.ReduceMean(null, new string[2], keepdim), input.DataType, new[] { input, Tensor(dim) });
        }

        /// <summary>
        /// Returns the mean  of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceMean(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceMean(input, new[] { dim }, keepdim);
        }

        /// <summary>
        /// Returns the product of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceProd(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            return FunctionalTensor.FromLayer(new Layers.ReduceProd(null, new string[2], keepdim), input.DataType, new[] { input, Tensor(dim) });
        }

        /// <summary>
        /// Returns the product of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceProd(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceProd(input, new[] { dim }, keepdim);
        }

        /// <summary>
        /// Returns the sum of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceSum(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            return FunctionalTensor.FromLayer(new Layers.ReduceSum(null, new string[2], keepdim), input.DataType, new[] { input, Tensor(dim) });
        }

        /// <summary>
        /// Returns the sum of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceSum(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceSum(input, new[] { dim }, keepdim);
        }
    }
}
