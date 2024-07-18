namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns a scalar integer tensor.
        /// </summary>
        /// <param name="value">The value of the element.</param>
        /// <returns>The tensor.</returns>
        public static FunctionalTensor Tensor(int value)
        {
            return FunctionalTensor.FromConstant(new Layers.Constant(-1, new TensorShape(), new[] { value }));
        }

        /// <summary>
        /// Returns a 1D integer tensor.
        /// </summary>
        /// <param name="values">The values of the elements.</param>
        /// <returns>The tensor.</returns>
        public static FunctionalTensor Tensor(int[] values)
        {
            return FunctionalTensor.FromConstant(new Layers.Constant(-1, new TensorShape(values.Length), values));
        }

        /// <summary>
        /// Returns a scalar float tensor.
        /// </summary>
        /// <param name="value">The value of the element.</param>
        /// <returns>The tensor.</returns>
        public static FunctionalTensor Tensor(float value)
        {
            return FunctionalTensor.FromConstant(new Layers.Constant(-1, new TensorShape(), new[] { value }));
        }

        /// <summary>
        /// Returns a 1D float tensor.
        /// </summary>
        /// <param name="values">The values of the elements.</param>
        /// <returns>The tensor.</returns>
        public static FunctionalTensor Tensor(float[] values)
        {
            return FunctionalTensor.FromConstant(new Layers.Constant(-1, new TensorShape(values.Length), values));
        }
    }
}
