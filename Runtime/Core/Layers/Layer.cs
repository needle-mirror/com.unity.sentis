using System;
using System.Reflection;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Options for the flags of a layer.
    /// </summary>
    [Flags]
    public enum Flags
    {
        /// <summary>
        /// Use no layer flags.
        /// </summary>
        None = 0,

        /// <summary>
        /// Use layer preservation and don't edit or remove the layer in an optimization pass.
        /// </summary>
        Preserve = 1 << 1,
    }

    /// <summary>
    /// Represents the base class for all model layers.
    /// </summary>
    [Serializable]
    public abstract class Layer
    {
        /// <summary>
        /// The names to use for the input tensors for a layer.
        /// </summary>
        public string[] inputs;

        /// <summary>
        /// The name to use for the first output tensor for a layer.
        /// </summary>
        public string index;

        /// <summary>
        /// The names to use for all of the output tensors for a layer. This is `null` if a layer has only one output.
        /// </summary>
        public string[] outputs;

        /// <summary>
        /// The flags set on the layer.
        /// </summary>
        [NonSerialized]
        public Flags flags;

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
        public abstract void Execute(ExecutionContext ctx);

        internal virtual string profilerTag => MethodBase.GetCurrentMethod()?.DeclaringType?.Name;

        /// <summary>
        /// Returns a string that represents the `Layer`.
        /// </summary>
        /// <returns>The string representation of the `Layer`.</returns>
        public override string ToString()
        {
            return $"{profilerTag} - index: {index}, inputs: [{string.Join(", ", inputs)}]";
        }
    }

    /// <summary>
    /// Options for applying an activation at the end of executing a `FusedActivation` layer.
    /// </summary>
    public enum FusableActivation
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
    [Serializable]
    abstract class FusedActivation : Layer
    {
        /// <summary>
        /// The fused activation to apply at the end of the execution as a `FusableActivation`.
        /// </summary>
        public FusableActivation fusedActivation;

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, fusedActivation: {fusedActivation}";
        }
    }

    /// <summary>
    /// Represents a base class for layers that apply an operation to input tensors using numpy-style broadcasting.
    /// </summary>
    [Serializable]
    abstract class Broadcast : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of broadcast layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The names to use for the input tensors of the layer.</param>
        protected Broadcast(string name, params string[] inputs)
        {
            this.index = name;
            this.inputs = inputs;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            Logger.AssertIsTrue(inputs.Length > 0, "Broadcast.InputError: can't broadcast shapes array of size 0");
            var inputTensors = ctx.GetPartialTensors(inputs);
            var dataType = InferPartialDataType(inputTensors);

            if (inputTensors.Length == 1)
            {
                ctx.AddPartialTensor(index, inputTensors[0]);
                return;
            }

            SymbolicTensorShape shapeOut;

            if (inputTensors.Length == 2)
            {
                shapeOut = inputTensors[0].shape.Broadcast(inputTensors[1].shape);
                var tensorOut = new PartialTensor(dataType, shapeOut);
                var op = InferPartialOp;

                if (op != null && shapeOut.IsFullyKnown() && shapeOut.rank <= 1 && inputTensors[0].isPartiallyKnown && inputTensors[1].isPartiallyKnown)
                {
                    for (var i = 0; i < tensorOut.length; i++)
                    {
                        tensorOut[i] = op(inputTensors[0][inputTensors[0].length > 1 ? i : 0], inputTensors[1][inputTensors[1].length > 1 ? i : 0]);
                    }
                }

                ctx.AddPartialTensor(index, tensorOut);
                return;
            }

            var outRank = 0;
            foreach (var input in inputTensors)
            {
                if (!input.shape.hasRank)
                {
                    ctx.AddPartialTensor(index, new PartialTensor(dataType));
                    return;
                }

                outRank = Mathf.Max(outRank, input.shape.rank);
            }

            shapeOut = SymbolicTensorShape.Ones(outRank);

            foreach (var tensorInput in inputTensors)
            {
                for (var j = 0; j < tensorInput.shape.rank; j++)
                {
                    shapeOut[shapeOut.rank - tensorInput.shape.rank + j] = SymbolicTensorDim.Broadcast(shapeOut[shapeOut.rank - tensorInput.shape.rank + j], tensorInput.shape[j]);
                }
            }

            ctx.AddPartialTensor(index, new PartialTensor(dataType, shapeOut));
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
