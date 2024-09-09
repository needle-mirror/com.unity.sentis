using System;

namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns the matrix product input @ other.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MatMul(FunctionalTensor input, FunctionalTensor other)
        {
            input = input.Float();
            other = other.Float();
            var output = FromLayer(new Layers.MatMul(-1, -1, -1), CommonType(input, other), new[] { input, other });
            if (input.isShapeKnown && other.isShapeKnown)
                output.SetShape(input.shape.MatMul(other.shape));
            return output;
        }

        /// <summary>
        /// Performs a batch matrix-matrix product of matrices : y = x @ a + b.
        /// B : (N)
        /// A : (K, N)
        /// X : (..., M, K)
        /// O : (..., M, K)
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="weight">The second input tensor.</param>
        /// <param name="bias">The bias input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor AddBMM(FunctionalTensor input, FunctionalTensor weight, FunctionalTensor bias)
        {
            input = input.Float();
            weight = weight.Float();
            bias = bias.Float();
            var output = FromLayer(new Layers.Dense(-1, -1, -1, -1, Layers.FusableActivation.None), DataType.Float, new[] { input, weight, bias });
            if (input.isShapeKnown && weight.isShapeKnown)
                output.SetShape(input.shape.MatMul(weight.shape));
            return output;
        }
    }
}
