using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a `Shape` layer. This computes the shape of an input tensor as a 1D `TensorInt`.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.NoDataDependencyInputs(0)]
    public class Shape : Layer
    {
        /// <summary>
        /// The inclusive start axis for slicing the shape of the input tensor.
        ///
        /// If this is negative then the axes of the tensor are counted from the back.
        /// </summary>
        public int start;
        /// <summary>
        /// The exclusive end axis for slicing the shape of the input tensor.
        ///
        /// If this is negative then the dimensions of the tensor are counted from the back.
        /// </summary>
        public int end;

        /// <summary>
        /// Initializes and returns an instance of `Shape` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer. The layer does not use the values of this tensor in the computation.</param>
        /// <param name="start">The inclusive start axis for slicing the shape of the input tensor. The default value is 0.</param>
        /// <param name="end">The exclusive end axis for slicing the shape of the input tensor. The default value is 8.</param>
        public Shape(string name, string input, int start = 0, int end = TensorShape.maxRank)
        {
            this.name = name;
            inputs = new[] { input };
            this.start = start;
            this.end = end;
        }

        internal override PartialTensor InferPartialTensor(PartialTensor[] partialTensors, ShapeInferenceContext ctx)
        {
            if (partialTensors[0].isPartiallyKnown)
                return PartialInferenceHelper.PartialTensorFromSymbolicShape(partialTensors[0].symbolicShape, start, end);
            return PartialInferenceHelper.PartialTensorFromSymbolicShape(ctx.GetSymbolicTensorShape(inputs[0]), start, end);
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.Shape(inputShapes[0], start, end);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.Shape(inputTensors[0], start, end);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, start: {start}, end: {end}";
        }

        internal override string profilerTag => "Shape";
    }

    /// <summary>
    /// Represents a `Size` layer. This computes the number of elements of an input tensor as a scalar `TensorInt`.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.NoDataDependencyInputs(0)]
    public class Size : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `Size` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer. The layer does not use the values of this tensor in the computation.</param>
        public Size(string name, string input)
        {
            this.name = name;
            inputs = new[] { input };
        }

        internal override PartialTensor InferPartialTensor(PartialTensor[] partialTensors, ShapeInferenceContext ctx)
        {
            var symbolicShape = partialTensors[0].isPartiallyKnown ? partialTensors[0].symbolicShape : ctx.GetSymbolicTensorShape(inputs[0]);
            var sizeTensor = new PartialTensor(new TensorShape());
            sizeTensor[0] = PartialTensorElement.FromSymbolicTensorDim(symbolicShape.Length());
            return sizeTensor;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return new SymbolicTensorShape();
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.Size(inputTensors[0].shape);
        }

        internal override string profilerTag => "Size";
    }
}
