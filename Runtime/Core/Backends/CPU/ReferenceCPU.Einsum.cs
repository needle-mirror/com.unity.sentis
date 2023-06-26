using Unity.Sentis;

namespace Unity.Sentis
{
    public partial class CPUOps
    {
        TensorFloat EinsumND(string equation, params TensorFloat[] operands)
        {
            var tensorShapes = new TensorShape[operands.Length];
            for (var i = 0; i < tensorShapes.Length; i++)
            {
                tensorShapes[i] = operands[i].shape;
            }
            var operandIndices = new TensorIndex[operands.Length];
            EinsumHelper.ParseEquationString(equation, tensorShapes, ref operandIndices, out var outputIndices, out var outputShape, out var sumIndices, out var sumShape, out var numIndices);

            var output = NewOutputTensorFloat(outputShape);
            var outSize = output.shape.length;
            var sumSize = sumShape.length;

            var position = new int[outputIndices.rank + sumIndices.rank];

            for (var outIndex = 0; outIndex < outSize; outIndex++)
            {
                SetPositionFromIndex(position, outputIndices, outputShape, outIndex);
                float sum = 0;
                for (var sumIndex = 0; sumIndex < sumSize; sumIndex++)
                {
                    SetPositionFromIndex(position, sumIndices, sumShape, sumIndex);
                    float product = 1f;
                    for (var i = 0; i < operands.Length; i++)
                    {
                        var operandIndex = GetIndexFromPosition(position, operandIndices[i], operands[i].shape);
                        product *= operands[i][operandIndex];
                    }

                    sum += product;
                }

                output[outIndex] = sum;
            }

            return output;
        }

        static void SetPositionFromIndex(int[] position, TensorIndex indices, TensorShape shape, int index)
        {
            for (var i = shape.rank - 1; i >= 0; i--)
            {
                position[indices[i]] = index % shape[i];
                index /= shape[i];
            }
        }

        static int GetIndexFromPosition(int[] position, TensorIndex indices, TensorShape shape)
        {
            var index = 0;
            var stride = 1;
            for (var i = shape.rank - 1; i >= 0; i--)
            {
                index += stride * position[indices[i]];
                stride *= shape[i];
            }
            return index;
        }
    }
}
