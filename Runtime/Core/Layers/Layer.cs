using System;
using System.Reflection;
using Unity.Profiling;
using Unity.Sentis.Layers;
using UnityEngine;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents the base class for all model layers.
    /// </summary>
    public abstract class Layer
    {
        internal const string k_ProfilerMarkerPrefix = "Sentis.Layer.";

        /// <summary>
        /// The indices to use for the input tensors for a layer.
        /// </summary>
        public int[] inputs;

        /// <summary>
        /// The indices to use for all of the output tensors for a layer.
        /// </summary>
        public int[] outputs;

        /// <summary>
        /// ProfilerMarker for this layer
        /// </summary>
        public abstract ProfilerMarker profilerMarker { get; }

        /// <summary>
        /// Initializes and returns a `Layer` from given arrays of input and output indices
        /// </summary>
        /// <param name="outputs">The indices array representing the outputs of this layer.</param>
        /// <param name="inputs">The indices array representing the inputs of this layer.</param>
        protected Layer(int[] outputs, int[] inputs)
        {
            this.outputs = outputs;
            this.inputs = inputs;
        }

        /// <summary>
        /// Infer the output partial tensors.
        ///
        /// Output partial tensors are saved to 'ctx'.
        /// </summary>
        internal abstract void InferPartial(PartialInferenceContext ctx);

        /// <summary>
        /// Executes the layer using the operations and variables from the `ExecutionContext`.
        /// </summary>
        /// <param name="ctx">The execution context with the backend and variables for the execution.</param>
        internal abstract void Execute(ExecutionContext ctx);

        /// <summary>
        /// Returns a string that represents the operation of the `Layer`.
        /// </summary>
        public abstract string opName { get; }

        /// <summary>
        /// Returns a string that represents the `Layer`.
        /// </summary>
        /// <returns>The string representation of the `Layer`.</returns>
        public override string ToString()
        {
            return $"{opName} - index: {outputs[0]}, inputs: [{string.Join(", ", inputs)}]";
        }
    }
}

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Options for applying an activation at the end of executing a `FusedActivation` layer.
    /// </summary>
    enum FusableActivation
    {
        /// <summary>
        /// Use no activation function.
        /// </summary>
        None,
        /// <summary>
        /// Use `Relu` activation function: f(x) = max(0, x).
        /// </summary>
        Relu
    }

    /// <summary>
    /// Represents a base class for layers with an optional fused activation at the end of the execution.
    /// </summary>
    abstract class FusedActivation : Layer
    {
        public FusableActivation fusedActivation;

        protected FusedActivation(int[] outputs, int[] inputs)
            : base(outputs, inputs) { }

        public override string ToString()
        {
            return $"{base.ToString()}, fusedActivation: {fusedActivation}";
        }
    }

    /// <summary>
    /// Represents a base class for layers that apply an operation to input tensors using numpy-style broadcasting.
    /// </summary>
    abstract class Broadcast : Layer
    {
        protected Broadcast(int output, params int[] inputs)
            : base(new[] { output }, inputs) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            Logger.AssertIsTrue(inputs.Length > 0, "Broadcast.InputError: can't broadcast shapes array of size 0");
            var inputTensors = ctx.GetPartialTensors(inputs);
            var dataType = InferPartialDataType(inputTensors);

            if (inputTensors.Length == 1)
            {
                ctx.AddPartialTensor(outputs[0], inputTensors[0]);
                return;
            }

            DynamicTensorShape shapeOut;

            if (inputTensors.Length == 2)
            {
                shapeOut = inputTensors[0].shape.Broadcast(inputTensors[1].shape);
                var tensorOut = new PartialTensor(dataType, shapeOut);
                var op = InferPartialOp;

                if (op != null && shapeOut.IsStatic() && shapeOut.rank <= 1 && inputTensors[0].isPartiallyKnown && inputTensors[1].isPartiallyKnown)
                {
                    for (var i = 0; i < tensorOut.length; i++)
                    {
                        tensorOut[i] = op(inputTensors[0][inputTensors[0].length > 1 ? i : 0], inputTensors[1][inputTensors[1].length > 1 ? i : 0]);
                    }
                }

                ctx.AddPartialTensor(outputs[0], tensorOut);
                return;
            }

            var outRank = 0;
            foreach (var input in inputTensors)
            {
                if (!input.shape.hasRank)
                {
                    ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType));
                    return;
                }

                outRank = Mathf.Max(outRank, input.shape.rank);
            }

            shapeOut = DynamicTensorShape.Ones(outRank);

            foreach (var tensorInput in inputTensors)
            {
                for (var j = 0; j < tensorInput.shape.rank; j++)
                {
                    shapeOut[shapeOut.rank - tensorInput.shape.rank + j] = DynamicTensorDim.Broadcast(shapeOut[shapeOut.rank - tensorInput.shape.rank + j], tensorInput.shape[j]);
                }
            }

            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
        }

        /// <summary>
        /// Returns the data type of the output partial tensor.
        /// </summary>
        internal virtual DataType InferPartialDataType(PartialTensor[] inputTensors)
        {
            return inputTensors[0].dataType;
        }

        /// <summary>
        /// Returns the optional function that calculates an output partial tensor element from input partial tensor elements.
        /// </summary>
        internal virtual Func<PartialTensorElement, PartialTensorElement, PartialTensorElement> InferPartialOp => null;
    }
}
