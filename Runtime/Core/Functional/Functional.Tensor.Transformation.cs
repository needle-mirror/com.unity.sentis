using System;

namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns the input tensors concatenated along a dimension.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <param name="dim">The dimension along which to concatenate.</param>
        /// <returns></returns>
        public static FunctionalTensor Concat(FunctionalTensor[] tensors, int dim = 0)
        {
            return FunctionalTensor.FromLayer(new Layers.Concat(null, new string[tensors.Length], dim), CommonType(tensors), tensors);
        }

        /// <summary>
        /// Returns the input tensor gathered along a dimension with indices.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to gather.</param>
        /// <param name="index">The indices tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Gather(this FunctionalTensor input, int dim, FunctionalTensor index)
        {
            DeclareType(DataType.Int, index);
            return FunctionalTensor.FromLayer(new Layers.GatherElements(null, null, null, dim), input.DataType, new[] { input, index });
        }

        /// <summary>
        /// Returns the input tensor with a dimension moved from source to destination.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="source">The dimension in the input tensor to move.</param>
        /// <param name="destination">The moved dimension in the output tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MoveDim(this FunctionalTensor input, int source, int destination)
        {
            return FunctionalTensor.FromLayer(new Layers.MoveDim(null, null, new[] { source }, new[] { destination }), input.DataType, input);
        }

        /// <summary>
        /// Returns the input tensor with multiple dimensions moved from source to destination.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="source">The dimensions in the input tensor to move.</param>
        /// <param name="destination">The moved dimensions in the output tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MoveDim(this FunctionalTensor input, int[] source, int[] destination)
        {
            return FunctionalTensor.FromLayer(new Layers.MoveDim(null, null, source, destination), input.DataType, input);
        }

        /// <summary>
        /// Returns the input tensor narrowed along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to narrow.</param>
        /// <param name="start">The start index along the dimension.</param>
        /// <param name="length">The number of elements along the dimension.</param>
        /// <returns></returns>
        public static FunctionalTensor Narrow(this FunctionalTensor input, int dim, int start, int length)
        {
            return FunctionalTensor.FromLayer(new Layers.Narrow(null, null, null, null, null), input.DataType, new[] { input, Tensor(dim), Tensor(start), Tensor(length) });
        }

        /// <summary>
        /// Returns the input tensor narrowed along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to narrow.</param>
        /// <param name="start">The functional start index along the dimension.</param>
        /// <param name="length">The functional number of elements along the dimension.</param>
        /// <returns></returns>
        public static FunctionalTensor Narrow(this FunctionalTensor input, int dim, FunctionalTensor start, FunctionalTensor length)
        {
            DeclareType(DataType.Int, start, length);
            return FunctionalTensor.FromLayer(new Layers.Narrow(null, null, null, null, null), input.DataType, new[] { input, Tensor(dim), start, length });
        }

        /// <summary>
        /// Returns the indices of the input tensor with values not equal to zero.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor NonZero(FunctionalTensor input)
        {
            // TODO support asTuple
            return Transpose(FunctionalTensor.FromLayer(new Layers.NonZero(null, null), DataType.Int, input), 0, 1);
        }

        /// <summary>
        /// Returns the input tensor padded with size determined by the pad array and values determined by the mode.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="pad">The padding lower and upper sizes starting from the final dimension (pad_w_lower, pad_w_upper, pad_h_lower, pad_h_upper, ...), not all dimensions need to be padded.</param>
        /// <param name="mode">The mode to use for sampling values, should be `constant`, `reflect`, `replicate` or `circular`, for constant padding with non zero values use one of the other `Pad` methods.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Pad(this FunctionalTensor input, int[] pad, string mode)
        {
            var padMode = mode switch
            {
                "constant" => Layers.PadMode.Constant,
                "reflect" => Layers.PadMode.Reflect,
                "replicate" => Layers.PadMode.Edge,
                "circular" => Layers.PadMode.Wrap,
                _ => Layers.PadMode.Constant
            };
            var axes = new int[pad.Length / 2];
            var pads = new int[pad.Length];
            for (var i = 0; i < axes.Length; i++)
            {
                axes[i] = -i - 1;
                pads[i] = pad[2 * i];
                pads[i + axes.Length] = pad[2 * i + 1];
            }
            return FunctionalTensor.FromLayer(new Layers.Pad(null, null, null, null, null, padMode), input.DataType, new[] { input, Tensor(pads), null, Tensor(axes) });
        }

        /// <summary>
        /// Returns the input tensor padded with size determined by the pad array and a constant value.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="pad">The padding lower and upper sizes starting from the final dimension (pad_w_lower, pad_w_upper, pad_h_lower, pad_h_upper, ...), not all dimensions need to be padded.</param>
        /// <param name="value">The constant value to use for padding.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Pad(this FunctionalTensor input, int[] pad, int value)
        {
            if (input.DataType == DataType.Float)
                return Pad(input, pad, (float)value);
            var axes = new int[pad.Length / 2];
            var pads = new int[pad.Length];
            for (var i = 0; i < axes.Length; i++)
            {
                axes[i] = -i - 1;
                pads[i] = pad[2 * i];
                pads[i + axes.Length] = pad[2 * i + 1];
            }
            return FunctionalTensor.FromLayer(new Layers.Pad(null, null, null, null, null, Layers.PadMode.Constant), input.DataType, new[] { input, Tensor(pads), Tensor(value), Tensor(axes) });
        }

        /// <summary>
        /// Returns the input tensor padded with size determined by the pad array and a constant value.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="pad">The padding lower and upper sizes starting from the final dimension (pad_w_lower, pad_w_upper, pad_h_lower, pad_h_upper, ...), not all dimensions need to be padded.</param>
        /// <param name="value">The constant value to use for padding.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Pad(this FunctionalTensor input, int[] pad, float value)
        {
            DeclareType(DataType.Float, input);
            var axes = new int[pad.Length / 2];
            var pads = new int[pad.Length];
            for (var i = 0; i < axes.Length; i++)
            {
                axes[i] = -i - 1;
                pads[i] = pad[2 * i];
                pads[i + axes.Length] = pad[2 * i + 1];
            }
            return FunctionalTensor.FromLayer(new Layers.Pad(null, null, null, null, null, Layers.PadMode.Constant), input.DataType, new[] { input, Tensor(pads), Tensor(value), Tensor(axes) });
        }

        /// <summary>
        /// Returns the input tensor with permuted dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dims">The dimensions of the input tensor to use in the permuted output tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Permute(this FunctionalTensor input, int[] dims)
        {
            return FunctionalTensor.FromLayer(new Layers.Transpose(null, null, dims), input.DataType, input);
        }

        /// <summary>
        /// Returns the input tensor elements reshaped.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="shape">The shape of the output tensor. A negative value is inferred from the others.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Reshape(this FunctionalTensor input, int[] shape)
        {
            return FunctionalTensor.FromLayer(new Layers.Reshape(null, null, null), input.DataType, new[] { input, Tensor(shape) });
        }

        /// <summary>
        /// Returns the input tensor sliced along a dimension at an index.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to select.</param>
        /// <param name="index">The index along the dimension to select.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Select(this FunctionalTensor input, int dim, int index)
        {
            return FunctionalTensor.FromLayer(new Layers.Select(null, null, null, null), input.DataType, new[] { input, Tensor(dim), Tensor(index) });
        }

        /// <summary>
        /// Returns the input tensor sliced along a dimension at an index.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to select.</param>
        /// <param name="index">The functional index along the dimension to select.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Select(this FunctionalTensor input, int dim, FunctionalTensor index)
        {
            DeclareType(DataType.Int, index);
            return FunctionalTensor.FromLayer(new Layers.Select(null, null, null, null), input.DataType, new[] { input, Tensor(dim), index });
        }

        /// <summary>
        /// Returns a copy of the input with the elements replaced by those from source given by the index along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to scatter.</param>
        /// <param name="index">The index tensor.</param>
        /// <param name="src">The source tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Scatter(FunctionalTensor input, int dim, FunctionalTensor index, FunctionalTensor src)
        {
            // TODO add reduction
            DeclareType(DataType.Int, index);
            return FunctionalTensor.FromLayer(new Layers.ScatterElements(null, null, null, null, dim, Layers.ScatterReductionMode.None), CommonType(input, src), new[] { input, index, src });
        }

        // Embeds the values of the src tensor into input at the given index.
        /// <summary>
        /// Returns a copy of the input with the elements replaced by those from source at a dimension and index.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="src">The source tensor.</param>
        /// <param name="dim">The dimension along which to scatter.</param>
        /// <param name="index">The index at which to scatter along the dimension.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor SelectScatter(FunctionalTensor input, FunctionalTensor src, int dim, int index)
        {
            return FunctionalTensor.FromLayer(new Layers.SliceSet(null, null, null, null, null, null), CommonType(input, src), new[] { input, Unsqueeze(src, dim), Tensor(new[] { index }), Tensor(new[] { index + 1 }), Tensor(new[] { dim }) });
        }

        /// <summary>
        /// Returns a copy of the input with the elements replaced by those from source along a dimension with start, end and step.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="src">The source tensor.</param>
        /// <param name="dim">The dimension along which to scatter.</param>
        /// <param name="start">The index of the first element to replace along the dimension.</param>
        /// <param name="end">The end index of the scatter along the dimension.</param>
        /// <param name="step">The step between the indices along the dimension.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor SliceScatter(FunctionalTensor input, FunctionalTensor src, int dim = 0, int start = 0, int end = int.MaxValue, int step = 1)
        {
            return FunctionalTensor.FromLayer(new Layers.SliceSet(null, null, null, null, null, null, null), CommonType(input, src), new[] { input, src, Tensor(new[] { start }), Tensor(new[] { end }), Tensor(new[] { dim }), Tensor(new[] { step }) });
        }

        /// <summary>
        /// Returns a copy of the input with the elements updated by adding by those from source given by the index along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to scatter.</param>
        /// <param name="index">The index tensor.</param>
        /// <param name="src">The source tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ScatterAdd(FunctionalTensor input, int dim, FunctionalTensor index, FunctionalTensor src)
        {
            return FunctionalTensor.FromLayer(new Layers.ScatterElements(null, null, null, null, dim, Layers.ScatterReductionMode.Add), CommonType(input, src), new[] { input, index, src });
        }

        /// <summary>
        /// Returns an array of tensors by splitting the input into sections along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="sections">The length of each section along the dimension.</param>
        /// <param name="dim">The dimension along which to split.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor[] Split(this FunctionalTensor input, int[] sections, int dim = 0)
        {
            var dataTypes = new DataType[sections.Length];
            for (var i = 0; i < dataTypes.Length; i++)
                dataTypes[i] = input.DataType;
            return FunctionalTensor.FromLayerMultiOutputs(new Layers.Split(null, null, null, new string[sections.Length], dim), dataTypes, new[] { input, Tensor(sections) });
        }

        /// <summary>
        /// Returns the input tensor with all dimensions of size 1 removed.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Squeeze(this FunctionalTensor input)
        {
            return FunctionalTensor.FromLayer(new Layers.Squeeze(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns the input tensor with all specified dimensions of size 1 removed.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions of size 1 to remove.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Squeeze(this FunctionalTensor input, int[] dim)
        {
            return FunctionalTensor.FromLayer(new Layers.Squeeze(null, null, null), input.DataType, new[] { input, Tensor(dim) });
        }

        /// <summary>
        /// Returns the input tensors concatenated along a new dimension.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <param name="dim">The dimension along which to stack.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Stack(FunctionalTensor[] tensors, int dim = 0)
        {
            // TODO add properly
            var unsqueezedTensors = new FunctionalTensor[tensors.Length];
            for (var i = 0; i < unsqueezedTensors.Length; i++)
                unsqueezedTensors[i] = Unsqueeze(tensors[i], dim);
            return Concat(unsqueezedTensors, dim);
        }

        /// <summary>
        /// Returns a tensor with the elements of input at indices.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="index">The index tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Take(this FunctionalTensor input, FunctionalTensor index)
        {
            return Gather(Ravel(input), 0, index);
        }

        /// <summary>
        /// Returns the input tensor repeated on the dims.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dims">The number of times to repeat the input tensor along each dim.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Tile(this FunctionalTensor input, int[] dims)
        {
            // TODO deal with cases where dims.length != input.shape.rank
            return FunctionalTensor.FromLayer(new Layers.Tile(null, null, null), input.DataType, new[] { input, Tensor(dims) });
        }

        /// <summary>
        /// Returns the input tensor with two dimensions swapped.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim0">The first dimension to swap.</param>
        /// <param name="dim1">The second dimension to swap.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Transpose(this FunctionalTensor input, int dim0, int dim1)
        {
            return MoveDim(input, new[] { dim0, dim1 }, new[] { dim1, dim0 });
        }

        /// <summary>
        /// Returns the input tensor with a new dimension of size 1 inserted.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension at which to insert a size 1 dimension in the output tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Unsqueeze(this FunctionalTensor input, int dim)
        {
            return FunctionalTensor.FromLayer(new Layers.Unsqueeze(null, null, null), input.DataType, new[] { input, Tensor(new[] { dim }) });
        }

        /// <summary>
        /// Returns condition ? input : other element-wise.
        /// </summary>
        /// <param name="condition">The condition tensor.</param>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Where(FunctionalTensor condition, FunctionalTensor input, FunctionalTensor other)
        {
            DeclareType(DataType.Int, condition);
            return FunctionalTensor.FromLayer(new Layers.Where(null, null, null, null), CommonType(input, other), new[] { condition, input, other });
        }
    }
}
