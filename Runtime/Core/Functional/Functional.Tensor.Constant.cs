namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns a functional tensor with shape and data taken from a tensor.
        /// </summary>
        /// <param name="tensor">The tensor to use as the source.</param>
        /// <returns>The functional tensor.</returns>
        public static FunctionalTensor Constant(Tensor tensor)
        {
            return FunctionalTensor.FromTensor(tensor);
        }

        /// <summary>
        /// Returns an integer tensor.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="values">The values of the element.</param>
        /// <returns>The tensor.</returns>
        public static FunctionalTensor Constant(TensorShape shape, int[] values)
        {
            return FunctionalTensor.FromConstant(new Constant(-1, shape, values));
        }

        /// <summary>
        /// Returns a scalar integer tensor.
        /// </summary>
        /// <param name="value">The value of the element.</param>
        /// <returns>The tensor.</returns>
        public static FunctionalTensor Constant(int value)
        {
            return Constant(new TensorShape(), new[] { value });
        }

        /// <summary>
        /// Returns a 1D integer tensor.
        /// </summary>
        /// <param name="values">The values of the elements.</param>
        /// <returns>The tensor.</returns>
        public static FunctionalTensor Constant(int[] values)
        {
            return Constant(new TensorShape(values.Length), values);
        }

        /// <summary>
        /// Returns an float tensor.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="values">The values of the element.</param>
        /// <returns>The tensor.</returns>
        public static FunctionalTensor Constant(TensorShape shape, float[] values)
        {
            return FunctionalTensor.FromConstant(new Constant(-1, shape, values));
        }

        /// <summary>
        /// Returns a scalar float tensor.
        /// </summary>
        /// <param name="value">The value of the element.</param>
        /// <returns>The tensor.</returns>
        public static FunctionalTensor Constant(float value)
        {
            return Constant(new TensorShape(), new[] { value });
        }

        /// <summary>
        /// Returns a 1D float tensor.
        /// </summary>
        /// <param name="values">The values of the elements.</param>
        /// <returns>The tensor.</returns>
        public static FunctionalTensor Constant(float[] values)
        {
            return Constant(new TensorShape(values.Length), values);
        }
    }
}
