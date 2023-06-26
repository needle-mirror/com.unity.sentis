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
        public string name;

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
        /// Executes the layer using the operations and variables from the `ExecutionContext` and returns the output tensor.
        ///
        /// If the layer has more than one output, output tensors are saved to variables.
        /// </summary>
        public abstract Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx);

        internal virtual PartialTensor InferPartialTensor(PartialTensor[] partialTensors, ShapeInferenceContext ctx)
        {
            return PartialTensor.Unknown;
        }

        internal virtual SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicTensorShape.UnknownShape;
        }

        internal virtual string profilerTag => MethodBase.GetCurrentMethod()?.DeclaringType?.Name;

        /// <summary>
        /// Returns a string that represents the `Layer`.
        /// </summary>
        public override string ToString()
        {
            return $"{profilerTag} - name: {name}, inputs: [{string.Join(", ", inputs)}]";
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
    public abstract class FusedActivation : Layer
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
    public abstract class Broadcast : Layer
    {
        /// <summary>
        /// Initializes and returns an instance of broadcast layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="inputs">The names to use for the input tensors of the layer.</param>
        protected Broadcast(string name, params string[] inputs)
        {
            this.name = name;
            this.inputs = inputs;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.Broadcast(inputShapes);
        }
    }
}
