using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Options for the reduction operation to use in a scatter layer.
    /// </summary>
    public enum ScatterReductionMode
    {
        /// <summary>
        /// Use no reduction.
        /// </summary>
        None,
        /// <summary>
        /// Use the addition operator when reducing.
        /// </summary>
        Add,
        /// <summary>
        /// Use the multiplication operator when reducing.
        /// </summary>
        Mul,
    }

    /// <summary>
    /// Represents an `ArgMax` layer. This computes the indices of the maximum elements of the input tensor along a given axis.
    /// </summary>
    [Serializable]
    public class ArgMax : Layer
    {
        /// <summary>
        /// The axis along which to perform the operation.
        /// </summary>
        public int axis;
        /// <summary>
        /// Whether to keep the axis dimension in the output tensor.
        /// </summary>
        public bool keepdims;
        /// <summary>
        /// Whether to perform the operation from the back of the axis.
        /// </summary>
        public bool selectLastIndex;

        /// <summary>
        /// Initializes and returns an instance of `ArgMax` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axis">The axis along which to perform the operation.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis. The default value is `false`.</param>
        public ArgMax(string name, string input, int axis, bool keepdims = true, bool selectLastIndex = false)
        {
            this.name = name;
            this.inputs = new[] { input };
            this.axis = axis;
            this.keepdims = keepdims;
            this.selectLastIndex = selectLastIndex;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.Reduce(inputShapes[0], axis, keepdims);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (inputTensors[0] is TensorInt)
                return ctx.ops.ArgMax(inputTensors[0] as TensorInt, axis, keepdims, selectLastIndex);
            else
                return ctx.ops.ArgMax(inputTensors[0] as TensorFloat, axis, keepdims, selectLastIndex);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}, keepdims: {keepdims}, selectLastIndex: {selectLastIndex}";
        }

        internal override string profilerTag => "ArgMax";
    }

    /// <summary>
    /// Represents an `ArgMin` layer. This computes the indices of the minimum elements of the input tensor along a given axis.
    /// </summary>
    [Serializable]
    public class ArgMin : Layer
    {
        /// <summary>
        /// The axis along which to perform the operation.
        /// </summary>
        public int axis;
        /// <summary>
        /// Whether to keep the axis dimension in the output tensor.
        /// </summary>
        public bool keepdims;
        /// <summary>
        /// Whether to perform the operation from the back of the axis.
        /// </summary>
        public bool selectLastIndex;

        /// <summary>
        /// Initializes and returns an instance of `ArgMin` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="axis">The axis along which to perform the operation.</param>
        /// <param name="keepdims">Whether to keep the axis dimension in the output tensor. The default value is `true`.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis. The default value is `false`.</param>
        public ArgMin(string name, string input, int axis, bool keepdims = true, bool selectLastIndex = false)
        {
            this.name = name;
            this.inputs = new[] { input };
            this.axis = axis;
            this.keepdims = keepdims;
            this.selectLastIndex = selectLastIndex;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.Reduce(inputShapes[0], axis, keepdims);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (inputTensors[0] is TensorInt)
                return ctx.ops.ArgMin(inputTensors[0] as TensorInt, axis, keepdims, selectLastIndex);
            else
                return ctx.ops.ArgMin(inputTensors[0] as TensorFloat, axis, keepdims, selectLastIndex);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}, keepdims: {keepdims}, selectLastIndex: {selectLastIndex}";
        }

        internal override string profilerTag => "ArgMin";
    }

    /// <summary>
    /// Represents a `Gather` layer. This takes values from the input tensor indexed by the indices tensor along a given axis and concatenates them.
    /// </summary>
    [Serializable]
    public class Gather : Layer
    {
        /// <summary>
        /// The axis along which to perform the gather.
        /// </summary>
        public int axis;

        /// <summary>
        /// Initializes and returns an instance of `Gather` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="indices">The name to use for the indices tensor of the layer.</param>
        /// <param name="axis">The axis along which to perform the gather.</param>
        public Gather(string name, string input, string indices, int axis)
        {
            this.name = name;
            inputs = new[] { input, indices };
            this.axis = axis;
        }

        internal override PartialTensor InferPartialTensor(PartialTensor[] partialTensors, ShapeInferenceContext ctx)
        {
            return PartialInferenceHelper.Gather(partialTensors[0], partialTensors[1], axis);
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.Gather(inputShapes[0], inputShapes[1], axis);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.Gather(inputTensors[0], inputTensors[1] as TensorInt, axis);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "Gather";
    }

    /// <summary>
    /// Represents a `GatherElements` layer. This takes values from the input tensor indexed by the `indices` tensor along a given axis.
    /// </summary>
    [Serializable]
    public class GatherElements : Layer
    {
        /// <summary>
        /// The axis along which to perform the gather.
        /// </summary>
        public int axis;

        /// <summary>
        /// Initializes and returns an instance of `GatherElements` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="indices">The name to use for the indices tensor of the layer.</param>
        /// <param name="axis">The axis along which to perform the gather.</param>
        public GatherElements(string name, string input, string indices, int axis)
        {
            this.name = name;
            inputs = new[] { input, indices };
            this.axis = axis;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.GatherElements(inputShapes[0], inputShapes[1], axis);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.GatherElements(inputTensors[0], inputTensors[1] as TensorInt, axis);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        internal override string profilerTag => "GatherElements";
    }

    /// <summary>
    /// Represents a `GatherND` layer. This takes slices of values from the batched input tensor indexed by the `indices` tensor.
    /// </summary>
    [Serializable]
    public class GatherND : Layer
    {
        /// <summary>
        /// The number of batch dimensions of the input tensor. The gather begins at the next dimension.
        /// </summary>
        public int batchDims;

        /// <summary>
        /// Initializes and returns an instance of `GatherND` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="indices">The name to use for the indices tensor of the layer.</param>
        /// <param name="batchDims">The number of batch dimensions of the input tensor, the gather begins at the next dimension.</param>
        public GatherND(string name, string input, string indices, int batchDims)
        {
            this.name = name;
            inputs = new[] { input, indices };
            this.batchDims = batchDims;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.GatherND(inputShapes[0], inputShapes[1], batchDims);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.GatherND(inputTensors[0], inputTensors[1] as TensorInt, batchDims);
        }

        internal override string profilerTag => "GatherND";
    }

    /// <summary>
    /// Represents a `NonZero` layer. This returns the indices of the elements of the input tensor that are not zero.
    /// </summary>
    [Serializable]
    public class NonZero : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of `NonZero` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        public NonZero(string name, string input)
        {
            this.name = name;
            inputs = new[] { input };
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            if (!inputShapes[0].hasRank)
                return SymbolicTensorShape.UnknownOfRank(2);
            return new SymbolicTensorShape(new SymbolicTensorDim(inputShapes[0].rank), SymbolicTensorDim.Unknown);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (inputTensors[0] is TensorInt)
                return ctx.ops.NonZero(inputTensors[0] as TensorInt);
            else
                return ctx.ops.NonZero(inputTensors[0] as TensorFloat);
        }

        internal override string profilerTag => "NonZero";
    }

    /// <summary>
    /// Represents a `ScatterElements` layer. This copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor along a given axis.
    ///
    /// `ScatterElements` updates the values depending on the reduction mode used.
    /// </summary>
    [Serializable]
    public class ScatterElements : Layer
    {
        /// <summary>
        /// The axis on which to perform the scatter.
        /// </summary>
        public int axis;
        /// <summary>
        /// The reduction mode used to update the values as a `ScatterReductionMode`.
        /// </summary>
        public ScatterReductionMode reduction;

        /// <summary>
        /// Initializes and returns an instance of `ScatterElements` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="indices">The name to use for the indices tensor of the layer.</param>
        /// <param name="updates">The name to use for the updates tensor of the layer.</param>
        /// <param name="axis">The axis on which to perform the scatter.</param>
        /// <param name="reduction">The reduction mode used to update the values as a `ScatterReductionMode`.</param>
        public ScatterElements(string name, string input, string indices, string updates, int axis, ScatterReductionMode reduction)
        {
            this.name = name;
            inputs = new[] { input, indices, updates };
            this.axis = axis;
            this.reduction = reduction;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.ScatterElements(inputShapes[0], inputShapes[1], inputShapes[2], axis, reduction);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.ScatterElements(inputTensors[0], inputTensors[1] as TensorInt, inputTensors[2], axis, reduction);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}, reduction: {reduction}";
        }

        internal override string profilerTag => "ScatterElements";
    }

    /// <summary>
    /// Represents a `ScatterND` layer. This copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor.
    ///
    /// `ScatterND` updates the values depending on the reduction mode used.
    /// </summary>
    [Serializable]
    public class ScatterND : Layer
    {
        /// <summary>
        /// The reduction mode used to update the values as a `ScatterReductionMode`.
        /// </summary>
        public ScatterReductionMode reduction;

        /// <summary>
        /// Initializes and returns an instance of `ScatterND` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="indices">The name to use for the indices tensor of the layer.</param>
        /// <param name="updates">The name to use for the updates tensor of the layer.</param>
        /// <param name="reduction">The reduction mode used to update the values as a `ScatterReductionMode`.</param>
        public ScatterND(string name, string input, string indices, string updates, ScatterReductionMode reduction)
        {
            this.name = name;
            inputs = new[] { input, indices, updates };
            this.reduction = reduction;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.ScatterND(inputShapes[0], inputShapes[1], inputShapes[2], reduction);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            if (inputTensors[0] is TensorInt)
                return ctx.ops.ScatterND(inputTensors[0] as TensorInt, inputTensors[1] as TensorInt, inputTensors[2] as TensorInt, reduction);
            else
                return ctx.ops.ScatterND(inputTensors[0] as TensorFloat, inputTensors[1] as TensorInt, inputTensors[2] as TensorFloat, reduction);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, reduction: {reduction}";
        }

        internal override string profilerTag => "ScatterND";
    }

    /// <summary>
    /// Represents a `TopK` layer. This calculates the top-K largest or smallest elements of an input tensor along a given axis.
    ///
    /// This layer calculates both the values tensor of the top-K elements and the indices tensor of the top-K elements as outputs.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(1)]
    public class TopK : Layer
    {
        /// <summary>
        /// The axis along which to perform the top-K operation.
        /// </summary>
        public int axis;
        /// <summary>
        /// Whether to calculate the top-K largest elements. If this is `false` the layer calculates the top-K smallest elements.
        /// </summary>
        public bool largest;
        /// <summary>
        /// Whether to return the elements in sorted order in the output tensor.
        /// </summary>
        public bool sorted;

        /// <summary>
        /// Initializes and returns an instance of `TopK` layer.
        /// </summary>
        /// <param name="name">The name to use for the values tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="k">The name to use for the single value 1D tensor containing the number of elements to calculate.</param>
        /// <param name="axis">The axis along which to perform the top-K operation.</param>
        /// <param name="largest">Whether to calculate the top-K largest elements. If this is `false` the layer calculates the top-K smallest elements.</param>
        /// <param name="sorted">Whether to return the elements in sorted order in the output tensor.</param>
        /// <param name="outputNames">A two-element array containing the names to use for the values and indices output tensors of the layer respectively.</param>
        public TopK(string name, string input, string k, int axis, bool largest, bool sorted, string[] outputNames)
        {
            this.name = name;
            this.inputs = new[] { input, k };
            this.axis = axis;
            this.largest = largest;
            this.sorted = sorted;
            this.outputs = outputNames;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            SymbolicInference.TopK(inputShapes[0], inputShapes[1], axis, out var shapeValues, out var shapeIndices);

            ctx.AddShape(outputs[1], shapeIndices);

            return shapeValues;
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            Tensor[] Y = ctx.ops.TopK(inputTensors[0] as TensorFloat, (inputTensors[1] as TensorInt)[0], axis, largest, sorted);

            ctx.vars.Store(outputs[1], Y[1]);

            return Y[0];
        }

        internal override string profilerTag => "TopK";
    }
}
