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
            var output = FromLayer(new Layers.Relu(-1, -1), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }

        /// <summary>
        /// Returns hardswish(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor HardSwish(FunctionalTensor input)
        {
            input = input.Float();
            var output = FromLayer(new Layers.HardSwish(-1, -1), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }

        /// <summary>
        /// Returns relu6(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Relu6(FunctionalTensor input)
        {
            input = input.Float();
            var output = FromLayer(new Layers.Relu6(-1, -1), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
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
            var output = FromLayer(new Layers.Elu(-1, -1, alpha), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }

        /// <summary>
        /// Returns selu(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Selu(FunctionalTensor input)
        {
            input = input.Float();
            var output = FromLayer(new Layers.Selu(-1, -1, 1.67326319217681884765625f, 1.05070102214813232421875f), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
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
            var output = FromLayer(new Layers.Celu(-1, -1, alpha), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
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
            var output = FromLayer(new Layers.LeakyRelu(-1, -1, negativeSlope), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
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
            var output = FromLayer(new Layers.PRelu(-1, -1, -1), input.dataType, new[] { input, weight });
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }

        /// <summary>
        /// Returns gelu(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Gelu(FunctionalTensor input)
        {
            input = input.Float();
            var output = FromLayer(new Layers.Gelu(-1, -1), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }

        /// <summary>
        /// Returns softsign(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Softsign(FunctionalTensor input)
        {
            input = input.Float();
            var output = FromLayer(new Layers.Softsign(-1, -1), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }

        /// <summary>
        /// Returns softplus(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Softplus(FunctionalTensor input)
        {
            input = input.Float();
            var output = FromLayer(new Layers.Softplus(-1, -1), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
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
            var output = FromLayer(new Layers.Softmax(-1, -1, dim), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
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
            var output = FromLayer(new Layers.LogSoftmax(-1, -1, dim), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }

        /// <summary>
        /// Returns sigmoid(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sigmoid(FunctionalTensor input)
        {
            input = input.Float();
            var output = FromLayer(new Layers.Sigmoid(-1, -1), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }

        /// <summary>
        /// Returns hard_sigmoid(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor HardSigmoid(FunctionalTensor input)
        {
            input = input.Float();
            var output = FromLayer(new Layers.HardSigmoid(-1, -1, 1 / 6f), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
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
            var output = FromLayer(new Layers.BatchNormalization(-1, -1, -1, -1, -1, -1, eps), input.dataType, new[] { input, weight, bias, runningMean, runningVar });
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
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
            var output = FromLayer(new Layers.InstanceNormalization(-1, -1, -1, -1, eps), input.dataType, new[] { input, weight, bias });
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }

        /// <summary>
        /// Returns the result of computing Layer Normalization over a mini-batch of inputs.
        /// see paper: https://arxiv.org/abs/1607.06450
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="bias">The bias tensor.</param>
        /// <param name="eps">The epsilon value used to avoid division by zero.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LayerNorm(FunctionalTensor input, FunctionalTensor weight, FunctionalTensor bias, float eps = 1e-5f)
        {
            input = input.Float();
            weight = weight.Float();
            bias = bias.Float();
            var output = FromLayer(new Layers.LayerNormalization(-1, -1, -1, -1, eps), DataType.Float, new[] { input, weight, bias });
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
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
            var output = FromLayer(new Layers.LRN(-1, -1, alpha, beta, k, size), input.dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }
    }
}
