using System;
using UnityEngine;

namespace Unity.Sentis
{
    static class PartialInferenceHelper
    {
        public static PartialTensor Squeeze(PartialTensor tensorInput)
        {
            var shapeOut = tensorInput.symbolicShape.Squeeze();
            return tensorInput.Reshape(shapeOut);
        }

        public static PartialTensor Squeeze(PartialTensor tensorInput, PartialTensor axes)
        {
            if (!axes.IsFullyKnown())
                return PartialTensor.Unknown;
            var shapeOut = tensorInput.symbolicShape.Squeeze(axes.ToIntArray());
            return tensorInput.Reshape(shapeOut);
        }

        public static PartialTensor Unsqueeze(PartialTensor tensorInput, PartialTensor axes)
        {
            if (!axes.IsFullyKnown())
                return PartialTensor.Unknown;
            var shapeOut = tensorInput.symbolicShape.Unsqueeze(axes.ToIntArray());
            return tensorInput.Reshape(shapeOut);
        }

        public static PartialTensor Concat(PartialTensor[] tensors, int axis)
        {
            if (axis != 0)
                return PartialTensor.Unknown;

            var length = 0;
            foreach (var tensor in tensors)
            {
                if (!tensor.isPartiallyKnown || tensor.shape.rank != 1)
                    return PartialTensor.Unknown;
                length += tensor.shape.length;
            }

            var tensorOut = new PartialTensor(new TensorShape(length));
            var index = 0;
            foreach (var tensor in tensors)
            {
                for (var i = 0; i < tensor.shape.length; i++)
                {
                    tensorOut[index++] = tensor[i];
                }
            }

            return tensorOut;
        }

        public static PartialTensor Slice(PartialTensor data, PartialTensor starts, PartialTensor ends, PartialTensor? axesOptional = null, PartialTensor? stepsOptional = null)
        {
            if (!data.isPartiallyKnown || data.shape.rank != 1)
                return PartialTensor.Unknown;

            var steps = stepsOptional ?? PartialTensor.ConstantOfShape(starts.shape, 1);

            var dim = data.shape[0];

            var length = SymbolicInference.SliceDim(new SymbolicTensorDim(dim), starts[0], ends[0], steps[0]);
            if (!length.isValue)
                return PartialTensor.Unknown;

            var tensorOut = new PartialTensor(new TensorShape(length.value));

            if (!starts[0].isValue || !steps[0].isValue)
                return tensorOut;

            var start = starts[0].value;
            var step = steps[0].value;

            var clampAdjustDirection = step < 0 ? -1 : 0;

            start = start < 0 ? dim + start : start;
            start = Mathf.Clamp(start, 0, dim + clampAdjustDirection);

            for (var i = 0; i < length.value; i++)
            {
                tensorOut[i] = data[start + i * step];
            }

            return tensorOut;
        }

        public static PartialTensor Gather(PartialTensor input, PartialTensor indices, int axis)
        {
            if (!input.isPartiallyKnown || !indices.isPartiallyKnown)
                return PartialTensor.Unknown;
            if (input.shape.rank != 1 || indices.shape.rank > 1 || axis != 0)
                return PartialTensor.Unknown;

            var tensorOut = new PartialTensor(indices.shape);
            for (var i = 0; i < tensorOut.shape.length; i++)
            {
                if (indices[i].isValue)
                {
                    var index = indices[i].value;
                    index = index < 0 ? index + input.shape.length : index;
                    tensorOut[i] = input[index];
                }
                else
                {
                    tensorOut[i] = PartialTensorElement.Unknown;
                }
            }

            return tensorOut;
        }

        public static PartialTensor PartialTensorFromSymbolicShape(SymbolicTensorShape shape, int start, int end)
        {
            if (start == end)
                return new PartialTensor(new TensorShape(0));

            if (!shape.hasRank)
                return PartialTensor.Unknown;

            start = start < 0 ? start + shape.rank : start;
            end = end < 0 ? end + shape.rank : end;
            start = Mathf.Clamp(start, 0, shape.rank);
            end = Mathf.Clamp(end, 0, shape.rank);

            Logger.AssertIsTrue(end >= start, "PartialTensorFromSymbolicShape.InputError: start value cannot be greater than end value for shape slicing");

            var tensorOut = new PartialTensor(new TensorShape(end - start));
            for (var i = start; i < end; i++)
            {
                if (shape[i].isParam)
                    tensorOut[i - start] = new PartialTensorElement(shape[i].param);
                else if (shape[i].isValue)
                    tensorOut[i - start] = new PartialTensorElement(shape[i].value);
            }

            return tensorOut;
        }
    }
}
