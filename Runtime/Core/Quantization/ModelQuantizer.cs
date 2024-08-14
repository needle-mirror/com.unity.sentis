namespace Unity.Sentis
{
    /// <summary>
    /// Type of quantization for float weights.
    /// </summary>
    public enum QuantizationType
    {
        /// <summary>
        /// Cast to 16 bit floats.
        /// </summary>
        Float16,
        /// <summary>
        /// Quantized to 8 bit fixed point values with scale and zero point.
        /// </summary>
        Uint8,
    }

    /// <summary>
    /// Provides methods for quantizing models.
    /// </summary>
    public static class ModelQuantizer
    {
        /// <summary>
        /// Quantize the weights of a model to a
        /// </summary>
        /// <param name="quantizationType">Data type to quantize to.</param>
        /// <param name="model">The model to quantize.</param>
        public static void QuantizeWeights(QuantizationType quantizationType, ref Model model)
        {
            var pass = new QuantizeConstantsPass(quantizationType);
            pass.Run(ref model);
        }
    }
}
