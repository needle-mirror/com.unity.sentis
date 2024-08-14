using System;

namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns a tensor filled with zeros with a shape and a data type.
        /// </summary>
        /// <param name="size">The shape of the tensor.</param>
        /// <param name="dataType">The data type of the tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Zeros(int[] size, DataType dataType = DataType.Int)
        {
            return dataType switch
            {
                DataType.Float => Full(size, 0f),
                DataType.Int => Full(size, 0),
                _ => throw new ArgumentOutOfRangeException(nameof(dataType), dataType, null)
            };
        }

        /// <summary>
        /// Returns a tensor filled with zeros with the shape of input and a data type.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dataType">The data type of the tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ZerosLike(FunctionalTensor input, DataType dataType = DataType.Int)
        {
            return dataType switch
            {
                DataType.Float => FullLike(input, 0f),
                DataType.Int => FullLike(input, 0),
                _ => throw new ArgumentOutOfRangeException(nameof(dataType), dataType, null)
            };
        }

        /// <summary>
        /// Returns a tensor filled with ones with a shape and a data type.
        /// </summary>
        /// <param name="size">The shape of the tensor.</param>
        /// <param name="dataType">The data type of the tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Ones(int[] size, DataType dataType = DataType.Int)
        {
            return dataType switch
            {
                DataType.Float => Full(size, 1f),
                DataType.Int => Full(size, 1),
                _ => throw new ArgumentOutOfRangeException(nameof(dataType), dataType, null)
            };
        }

        /// <summary>
        /// Returns a tensor filled with ones with the shape of input and a data type.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dataType">The data type of the tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor OnesLike(FunctionalTensor input, DataType dataType = DataType.Int)
        {
            return dataType switch
            {
                DataType.Float => FullLike(input, 1f),
                DataType.Int => FullLike(input, 1),
                _ => throw new ArgumentOutOfRangeException(nameof(dataType), dataType, null)
            };
        }

        /// <summary>
        /// Returns a 1D tensor of size ⌈(end − start) / step⌉ with values from the interval [start, end) with a step beginning from start.
        /// </summary>
        /// <param name="start">The value of the first element.</param>
        /// <param name="end">The upper end of the interval.</param>
        /// <param name="step">The delta between each element.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ARange(int start, int end, int step = 1)
        {
            var output = FromLayer(new Layers.Range(-1, -1, -1, -1), DataType.Int, new[] { Constant(start), Constant(end), Constant(step) });
            output.SetShape(ShapeInference.Range(start, end, step));
            return output;
        }

        /// <summary>
        /// Returns a 1D tensor of size ⌈end / step⌉ with values from the interval [0, end) with a step 1 beginning from start.
        /// </summary>
        /// <param name="end">The upper end of the interval.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ARange(int end)
        {
            return ARange(0, end);
        }

        /// <summary>
        /// Returns a 1D tensor of size ⌈(end − start) / step⌉ with values from the interval [start, end) with a step beginning from start.
        /// </summary>
        /// <param name="start">The value of the first element.</param>
        /// <param name="end">The upper end of the interval.</param>
        /// <param name="step">The delta between each element.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ARange(float start, float end, float step = 1)
        {
            var output = FromLayer(new Layers.Range(-1, -1, -1, -1), DataType.Float, new[] { Constant(start), Constant(end), Constant(step) });
            output.SetShape(ShapeInference.Range(start, end, step));
            return output;
        }

        /// <summary>
        /// Returns a 1D tensor of size ⌈end / step⌉ with values from the interval [0, end) with a step 1 beginning from start.
        /// </summary>
        /// <param name="end">The upper end of the interval.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ARange(float end)
        {
            return ARange(0, end);
        }

        /// <summary>
        /// Returns a 1D tensor of size steps with values evenly spaced from the interval [start, end].
        /// </summary>
        /// <param name="start">The value of the first element.</param>
        /// <param name="end">The value of the last element.</param>
        /// <param name="steps">The number of elements.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LinSpace(float start, float end, int steps)
        {
            Logger.AssertIsTrue(steps >= 0, "LinSpace.InputError steps must be non-negative");
            if (steps == 0)
                return Constant(Array.Empty<float>());
            if (steps == 1)
                return Constant(new[] { start });
            var delta = (end - start) / (steps - 1);
            var starts = start;
            var ends = end + 0.5f * delta;
            var output = FromLayer(new Layers.Range(-1, -1, -1, -1), DataType.Float, new[] { Constant(starts), Constant(ends), Constant(delta) });
            output.SetShape(ShapeInference.Range(starts, ends, delta));
            return output;
        }

        // Creates a one-dimensional tensor of size steps whose values are evenly spaced from base ^ start to base ^ end, inclusive, on a logarithmic scale with base base.

        /// <summary>
        /// Returns a 1D tensor of size steps with values evenly spaced from the interval [logBase^start, logBase^end] on a logarithmic scale.
        /// </summary>
        /// <param name="start">The value of the first exponent.</param>
        /// <param name="end">The value of the last exponent.</param>
        /// <param name="steps">The number of elements.</param>
        /// <param name="logBase">The base of the logarithm.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogSpace(float start, float end, int steps, float logBase = 10)
        {
            return Pow(Constant(logBase), LinSpace(start, end, steps));
        }

        /// <summary>
        /// Returns a tensor filled with a constant value with a shape.
        /// </summary>
        /// <param name="size">The shape of the tensor.</param>
        /// <param name="fillValue">The fill value of the tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Full(int[] size, int fillValue)
        {
            var output = FromLayer(new Layers.ConstantOfShape(-1, -1, fillValue), DataType.Int, Constant(size));
            output.SetShape(new TensorShape(size));
            return output;
        }

        /// <summary>
        /// Returns a tensor filled with a constant value with a shape.
        /// </summary>
        /// <param name="size">The shape of the tensor.</param>
        /// <param name="fillValue">The fill value of the tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Full(int[] size, float fillValue)
        {
            var output = FromLayer(new Layers.ConstantOfShape(-1, -1, fillValue), DataType.Float, Constant(size));
            output.SetShape(new TensorShape(size));
            return output;
        }

        /// <summary>
        /// Returns a tensor filled with a constant value with the same shape as the input tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="fillValue">The fill value of the tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor FullLike(FunctionalTensor input, int fillValue)
        {
            var shape = FromLayer(new Layers.Shape(-1, -1), DataType.Int, input);
            var output = FromLayer(new Layers.ConstantOfShape(-1, -1, fillValue), DataType.Int, shape);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }

        /// <summary>
        /// Returns a tensor filled with a constant value with the same shape as the input tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="fillValue">The fill value of the tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor FullLike(FunctionalTensor input, float fillValue)
        {
            var shape = FromLayer(new Layers.Shape(-1, -1), DataType.Int, input);
            var output = FromLayer(new Layers.ConstantOfShape(-1, -1, fillValue), DataType.Float, shape);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }
    }
}
