using System;

namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns an array where each input tensor with rank less than 1 is expanded to rank 1.
        /// </summary>
        /// <param name="tensors">The input tensor array.</param>
        /// <returns>The output tensor array.</returns>
        public static FunctionalTensor[] AtLeast1D(params FunctionalTensor[] tensors)
        {
            var outputs = new FunctionalTensor[tensors.Length];
            for (var i = 0; i < outputs.Length; i++)
                outputs[i] = BroadcastTo(tensors[i], new[] { 1 });
            return outputs;
        }

        /// <summary>
        /// Returns an array where each input tensor with rank less than 2 is expanded to rank 2.
        /// </summary>
        /// <param name="tensors">The input tensor array.</param>
        /// <returns>The output tensor array.</returns>
        public static FunctionalTensor[] AtLeast2D(params FunctionalTensor[] tensors)
        {
            var outputs = new FunctionalTensor[tensors.Length];
            for (var i = 0; i < outputs.Length; i++)
                outputs[i] = BroadcastTo(tensors[i], new[] { 1, 1 });
            return outputs;
        }

        /// <summary>
        /// Returns an array where each input tensor with rank less than 3 is expanded to rank 3.
        /// </summary>
        /// <param name="tensors">The input tensor array.</param>
        /// <returns>The output tensor array.</returns>
        public static FunctionalTensor[] AtLeast3D(params FunctionalTensor[] tensors)
        {
            var outputs = new FunctionalTensor[tensors.Length];
            for (var i = 0; i < outputs.Length; i++)
                outputs[i] = BroadcastTo(tensors[i], new[] { 1, 1, 1 });
            return outputs;
        }

        /// <summary>
        /// Returns the input tensor broadcasted to a shape using the numpy broadcasting rules.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="shape">The shape to broadcast to.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor BroadcastTo(this FunctionalTensor input, int[] shape)
        {
            return FunctionalTensor.FromLayer(new Layers.Expand(null, null, null), input.DataType, new[] { input, Tensor(shape) });
        }

        /// <summary>
        /// Returns a copy of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Clone(this FunctionalTensor input)
        {
            return FunctionalTensor.FromLayer(new Layers.Identity(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns the cumulative sum of the elements of the input in a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension in which to sum.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor CumSum(FunctionalTensor input, int dim)
        {
            return FunctionalTensor.FromLayer(new Layers.CumSum(null, null, null, false, false), input.DataType, new[] { input, Tensor(dim) });
        }

        /// <summary>
        /// Returns the sums the product of the elements of the input tensors along dimensions specified using a notation based on the Einstein summation convention.
        /// </summary>
        /// <param name="equation">The equation of the Einstein summation as a comma-separated list of subscript labels.</param>
        /// <param name="operands">The input tensors.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Einsum(string equation, params FunctionalTensor[] operands)
        {
            return FunctionalTensor.FromLayer(new Layers.Einsum(null, new string[operands.Length], equation), CommonType(operands), operands);
        }

        /// <summary>
        /// Returns the input tensor with its elements reversed on some dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dims">The dimensions on which to reverse the elements, values may not repeat.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Flip(this FunctionalTensor input, int[] dims)
        {
            //Slice(x, starts = [-1], ends = [INT_MIN], steps = [-1])
            var starts = new int[dims.Length];
            var ends = new int[dims.Length];
            var steps = new int[dims.Length];
            for (var i = 0; i < dims.Length; i++)
            {
                starts[i] = -1;
                ends[i] = int.MinValue;
                steps[i] = -1;
            }

            return FunctionalTensor.FromLayer(new Layers.Slice(null, null, null, null, null, null), input.DataType, new[] { input, Tensor(starts), Tensor(ends), Tensor(dims), Tensor(steps) });
        }

        /// <summary>
        /// Returns the input tensor with its elements reversed on the second dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor FlipLR(this FunctionalTensor input)
        {
            return Flip(input, new[] { 1 });
        }

        /// <summary>
        /// Returns the input tensor with its elements reversed on the first dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor FlipUD(this FunctionalTensor input)
        {
            return Flip(input, new[] { 0 });
        }

        /// <summary>
        /// Returns the input tensor with its elements flattened to a single dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Ravel(this FunctionalTensor input)
        {
            return Reshape(input, new[] { -1 });
        }

        /// <summary>
        /// Retains the lower triangular values of an input matrix (batch). THe other values are zeroed.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="diagonal">The integer offset of the diagonal.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor TriL(FunctionalTensor input, int diagonal = 0)
        {
            return FunctionalTensor.FromLayer(new Layers.Trilu(null, null, null, Layers.TriluMode.Lower), input.DataType, new[] { input, Tensor(diagonal) });
        }

        /// <summary>
        /// Retains the upper triangular values of an input matrix (batch). THe other values are zeroed.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="diagonal">The integer offset of the diagonal.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor TriU(FunctionalTensor input, int diagonal = 0)
        {
            return FunctionalTensor.FromLayer(new Layers.Trilu(null, null, null, Layers.TriluMode.Upper), input.DataType, new[] { input, Tensor(diagonal) });
        }
    }
}
