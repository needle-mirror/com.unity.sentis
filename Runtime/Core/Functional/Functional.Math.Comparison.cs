using System;

namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns input == other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Equals(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            var output = FromLayer(new Layers.Equal(-1, -1, -1), DataType.Int, new[] { input, other });
            if (input.isShapeKnown && other.isShapeKnown)
                output.SetShape(input.shape.Broadcast(other.shape));
            return output;
        }

        /// <summary>
        /// Returns input == value element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="value">The integer value.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Equals(FunctionalTensor input, int value)
        {
            return input.dataType == DataType.Float ? Equals(input, (float)value) : Equals(input, Constant(value));
        }

        /// <summary>
        /// Returns input == value element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="value">The float value.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Equals(FunctionalTensor input, float value)
        {
            return Equals(input, Constant(value));
        }

        /// <summary>
        /// Returns input ≥ other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor GreaterEqual(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            var output = FromLayer(new Layers.GreaterOrEqual(-1, -1, -1), DataType.Int, new[] { input, other });
            if (input.isShapeKnown && other.isShapeKnown)
                output.SetShape(input.shape.Broadcast(other.shape));
            return output;
        }

        /// <summary>
        /// Returns input > other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Greater(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            var output = FromLayer(new Layers.Greater(-1, -1, -1), DataType.Int, new[] { input, other });
            if (input.isShapeKnown && other.isShapeKnown)
                output.SetShape(input.shape.Broadcast(other.shape));
            return output;
        }

        /// <summary>
        /// Returns an integer tensor with elements representing if each element of the input is finite.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor IsFinite(FunctionalTensor input)
        {
            // TODO add to backend and layers
            if (input.dataType == DataType.Int)
                return OnesLike(input);
            return LogicalNot(LogicalOr(IsInf(input), IsNaN(input)));
        }

        /// <summary>
        /// Returns an integer tensor with elements representing if each element of the input is positive or negative infinity.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor IsInf(FunctionalTensor input)
        {
            if (input.dataType == DataType.Int)
                return ZerosLike(input);
            var output = FromLayer(new Layers.IsInf(-1, -1, true, true), DataType.Int, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }

        /// <summary>
        /// Returns an integer tensor with elements representing if each element of the input is NaN.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor IsNaN(FunctionalTensor input)
        {
            if (input.dataType == DataType.Int)
                return ZerosLike(input);
            var output = FromLayer(new Layers.IsNaN(-1, -1), DataType.Int, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }

        /// <summary>
        /// Returns input ≤ other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LessEqual(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            var output = FromLayer(new Layers.LessOrEqual(-1, -1, -1), DataType.Int, new[] { input, other });
            if (input.isShapeKnown && other.isShapeKnown)
                output.SetShape(input.shape.Broadcast(other.shape));
            return output;
        }

        /// <summary>
        /// Returns input &lt; other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Less(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            var output = FromLayer(new Layers.Less(-1, -1, -1), DataType.Int, new[] { input, other });
            if (input.isShapeKnown && other.isShapeKnown)
                output.SetShape(input.shape.Broadcast(other.shape));
            return output;
        }

        /// <summary>
        /// Returns the element-wise maximum of input and other.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Max(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            var output = FromLayer(new Layers.Max(-1, -1, -1), CommonType(input, other), new[] { input, other });
            if (input.isShapeKnown && other.isShapeKnown)
                output.SetShape(input.shape.Broadcast(other.shape));
            return output;
        }

        /// <summary>
        /// Returns the element-wise minimum of input and other.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Min(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            var output = FromLayer(new Layers.Min(-1, -1, -1), CommonType(input, other), new[] { input, other });
            if (input.isShapeKnown && other.isShapeKnown)
                output.SetShape(input.shape.Broadcast(other.shape));
            return output;
        }

        /// <summary>
        /// Returns input ≠ other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor NotEqual(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            // TODO implement backend and layer
            return LogicalNot(Equals(input, other));
        }

        /// <summary>
        /// Returns the k largest elements of the input tensor along a given dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="k">The number of elements to calculate.</param>
        /// <param name="dim">The axis along which to perform the top-K operation.</param>
        /// <param name="largest">Whether to calculate the top-K largest elements. If this is `false` the layer calculates the top-K smallest elements.</param>
        /// <param name="sorted">Whether to return the elements in sorted order in the output tensor.</param>
        /// <returns>The output values and indices tensors in an array.</returns>
        public static FunctionalTensor[] TopK(FunctionalTensor input, int k, int dim = -1, bool largest = true, bool sorted = true)
        {
            var outputs = FromLayer(new Layers.TopK(-1, -1, -1, -1, dim, largest, sorted), new[] { DataType.Float, DataType.Int }, new[] { input, Constant(new[] { k }) });
            if (input.isShapeKnown)
            {
                var outputShape = new TensorShape(input.shape);
                outputShape[dim] = k;
                outputs[0].SetShape(outputShape);
                outputs[1].SetShape(outputShape);
            }
            return outputs;
        }
    }
}
