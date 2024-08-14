using System;

namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns the input cast to the data type element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dataType">The data type.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Type(this FunctionalTensor input, DataType dataType)
        {
            if (input.dataType == dataType)
                return input;
            var output = FromLayer(new Layers.Cast(-1, -1, dataType), dataType, input);
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }

        /// <summary>
        /// Returns the input cast to integers element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Int(this FunctionalTensor input)
        {
            return input.Type(DataType.Int);
        }

        /// <summary>
        /// Returns the input cast to floats element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Float(this FunctionalTensor input)
        {
            return input.Type(DataType.Float);
        }

        // Promotes a and b to the same type that is the lowest type compatible with both.
        static (FunctionalTensor, FunctionalTensor) PromoteTypes(FunctionalTensor a, FunctionalTensor b)
        {
            return a.dataType == b.dataType ? (a, b) : (a.Float(), b.Float());
        }

        // Returns the common type of all of the input tensors, asserts if any pair of input tensors have different types.
        internal static DataType CommonType(params FunctionalTensor[] tensors)
        {
            var type = tensors[0].dataType;
            for (var i = 1; i < tensors.Length; i++)
                Logger.AssertIsTrue(type == tensors[i].dataType, "FunctionalTensors must have same type.");
            return type;
        }

        // Asserts if any of the input tensors have a type different to a type.
        static void DeclareType(DataType dataType, params FunctionalTensor[] tensors)
        {
            for (var i = 0; i < tensors.Length; i++)
                Logger.AssertIsTrue(tensors[i].dataType == dataType, "FunctionalTensor has incorrect type.");
        }

        static void DeclareRank(FunctionalTensor tensor, int rank)
        {
            if (tensor.isShapeKnown)
                Logger.AssertIsTrue(tensor.shape.rank == rank, "FunctionalTensor has incorrect rank, received {0} expected {1}.", tensor.shape.rank, rank);
        }
    }
}
