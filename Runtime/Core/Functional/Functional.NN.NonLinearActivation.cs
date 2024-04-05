using System;

namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns relu(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Relu(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Relu(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns hardswish(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor HardSwish(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.HardSwish(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns relu6(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Relu6(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Relu6(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns elu(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="alpha">The alpha value for the elu.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Elu(FunctionalTensor input, float alpha = 1.0f)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Elu(null, null, alpha), input.DataType, input);
        }

        /// <summary>
        /// Returns selu(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Selu(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Selu(null, null, 1.67326319217681884765625f, 1.05070102214813232421875f), input.DataType, input);
        }

        /// <summary>
        /// Returns celu(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="alpha">The alpha value for the celu.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Celu(FunctionalTensor input, float alpha = 1.0f)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Celu(null, null, alpha), input.DataType, input);
        }

        /// <summary>
        /// Returns leaky_relu(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="negativeSlope">The negative slope value for the leaky relu.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LeakyRelu(FunctionalTensor input, float negativeSlope = 0.01f)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.LeakyRelu(null, null, negativeSlope), input.DataType, input);
        }

        /// <summary>
        /// Returns PRelu(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor PRelu(FunctionalTensor input, FunctionalTensor weight)
        {
            input = input.Float();
            weight = weight.Float();
            return FunctionalTensor.FromLayer(new Layers.PRelu(null, null, null), input.DataType, new[] { input, weight });
        }

        /// <summary>
        /// Returns gelu(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Gelu(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Gelu(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns softsign(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Softsign(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Softsign(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns softplus(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Softplus(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Softplus(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns softmax(input) element-wise along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to calculate the softmax.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Softmax(FunctionalTensor input, int dim = -1)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Softmax(null, null, dim), input.DataType, input);
        }

        /// <summary>
        /// Returns log(softmax(input)) element-wise along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to calculate the softmax.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogSoftmax(FunctionalTensor input, int dim = -1)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.LogSoftmax(null, null, dim), input.DataType, input);
        }

        /// <summary>
        /// Returns sigmoid(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sigmoid(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Sigmoid(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns hard_sigmoid(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor HardSigmoid(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.HardSigmoid(null, null, 1 / 6f), input.DataType, input);
        }

        /// <summary>
        /// Returns the result of computing the mean variance on the second dimension of the input tensor and normalizes it according to the weight and bias.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="runningMean">The mean values tensor.</param>
        /// <param name="runningVar">The variance values tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="bias">The bias tensor.</param>
        /// <param name="eps">The epsilon value used to avoid division by zero.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor BatchNorm(FunctionalTensor input, FunctionalTensor runningMean, FunctionalTensor runningVar, FunctionalTensor weight, FunctionalTensor bias, float eps = 1e-5f)
        {
            input = input.Float();
            runningMean = runningMean.Float();
            runningVar = runningVar.Float();
            weight = weight.Float();
            bias = bias.Float();
            return FunctionalTensor.FromLayer(new Layers.BatchNormalization(null, null, null, null, null, null, eps), input.DataType, new[] { input, weight, bias, runningMean, runningVar });
        }

        /// <summary>
        /// Returns the result of computing the mean variance on the spatial dimensions of the input tensor and normalizes it according to the weight and bias.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="bias">The bias tensor.</param>
        /// <param name="eps">The epsilon value used to avoid division by zero.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor InstanceNorm(FunctionalTensor input, FunctionalTensor weight, FunctionalTensor bias, float eps = 1e-5f)
        {
            input = input.Float();
            weight = weight.Float();
            bias = bias.Float();
            return FunctionalTensor.FromLayer(new Layers.InstanceNormalization(null, null, null, null, eps), input.DataType, new[] { input, weight, bias });
        }

        /// <summary>
        /// Returns the result of normalizing the input tensor over local input regions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="size">The size of the regions used for normalization.</param>
        /// <param name="alpha">The multiplicative factor in the normalization.</param>
        /// <param name="beta">The exponent in the normalization.</param>
        /// <param name="k">The additive factor in the normalization.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LocalResponseNorm(FunctionalTensor input, int size, float alpha = 0.0001f, float beta = 0.75f, float k = 1.0f)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.LRN(null, null, alpha, beta, k, size), input.DataType, input);
        }
    }
}
