using System;

namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns the matrix product input x other.
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
    }
}
