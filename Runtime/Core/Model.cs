using System;
using System.Linq; // Select
using System.Collections.Generic;

namespace Unity.Sentis {

/// <summary>
/// Represents a Sentis neural network.
/// </summary>
public class Model
{
    /// <summary>
    /// The version of the model. The value increments each time the data structure changes.
    /// </summary>
    public const int Version = 30;
    internal const int WeightsAlignment = 16;

    /// <summary>
    /// Represents an input to a model.
    /// </summary>
    [Serializable]
    public struct Input
    {
        /// <summary>
        /// The name of the input.
        /// </summary>
        public string name;

        /// <summary>
        /// The index of the input.
        /// </summary>
        public string index;

        /// <summary>
        /// The data type of the input data.
        /// </summary>
        public DataType dataType;

        /// <summary>
        /// The shape of the input, as `SymbolicTensorShape`.
        /// </summary>
        public SymbolicTensorShape shape;
    }

    /// <summary>
    /// Represents an output to a model.
    /// </summary>
    [Serializable]
    public struct Output
    {
        /// <summary>
        /// The name of the output.
        /// </summary>
        public string name;

        /// <summary>
        /// The index of the output.
        /// </summary>
        public string index;
    }

    /// <summary>
    /// The inputs of the model.
    /// </summary>
    public List<Input> inputs = new List<Input>();

    /// <summary>
    /// The outputs of the model.
    /// </summary>
    public List<Output> outputs = new List<Output>();

    /// <summary>
    /// The layers of the model.
    /// </summary>
    public List<Layers.Layer> layers = new List<Layers.Layer>();

    /// <summary>
    /// The constants of the model.
    /// </summary>
    public List<Layers.Constant> constants = new List<Layers.Constant>();

    /// <summary>
    /// The producer of the model, as a string.
    /// </summary>
    public string ProducerName = "Script";

    /// <summary>
    /// The opset version number of the ONNX model, for the default domain.
    /// </summary>
    public long DefaultOpsetVersion;

    /// <summary>
    /// stores which layers should fallback to CPU for execution
    /// </summary>
    internal HashSet<string> LayerCPUFallback = new HashSet<string>();

    /// <summary>
    /// Returns a string that represents the `Model`.
    /// </summary>
    /// <returns>String representation of model.</returns>
    public override string ToString()
    {
        // weights are not loaded for UI, recompute size
        var totalUniqueWeights = 0;
        return $"inputs: [{string.Join(", ", inputs.Select(i => $"{i.index} {i.shape} [{i.dataType}]"))}], " +
            $"outputs: [{string.Join(", ", outputs)}] " +
            $"\n{layers.Count} layers, {totalUniqueWeights:n0} weights: \n{string.Join("\n", layers.Select(i => $"{i.GetType()} ({i})"))}";
    }

    /// <summary>
    /// Returns a string index not yet used in the model inputs, constants or layer outputs
    /// based on a given index, the index may be suffixed with "_0", "_1" etc. if required
    /// </summary>
    internal string GetUniqueIndex(string index)
    {
        if (!ContainsIndex(index))
            return index;

        for (var i = 0;; i++)
        {
            var currentIndex = index + "_" + i;
            if (!ContainsIndex(currentIndex))
                return currentIndex;
        }
    }

    /// <summary>
    /// Checks if `index` is used in any model inputs, constants or layer outputs.
    /// </summary>
    bool ContainsIndex(string index)
    {
        if (constants.Any(constant => constant.index == index))
            return true;
        if (inputs.Any(input => input.index == index))
            return true;
        foreach (var layer in layers)
        {
            if (layer.index == index)
                return true;
            if (layer.outputs == null)
                continue;
            foreach (var output in layer.outputs)
            {
                if (output == index)
                    return true;
            }
        }

        foreach (var output in outputs)
        {
            if (output.index == index)
                return true;
        }

        return false;
    }

    internal void ValidateInputTensorShape(Input input, TensorShape shape)
    {
        if (shape.rank != input.shape.rank)
        {
            D.LogWarning($"Given input shape: {shape} is not compatible with model input shape: {input.shape} for input: {input.index}");
            return;
        }

        for (var i = 0; i < shape.rank; i++)
        {
            if (input.shape[i] != shape[i])
                D.LogWarning($"Given input shape: {shape} has different dimension from model input shape: {input.shape} for input: {input.index} at axis: {i}");
        }
    }

    /// <summary>
    /// Adds an input to the model with a symbolic tensor shape.
    /// </summary>
    /// <param name="name">The name of the input.</param>
    /// <param name="index">The index of the input.</param>
    /// <param name="dataType">The data type of the input.</param>
    /// <param name="shape">The `SymbolicTensorShape` of the input.</param>
    internal void AddInput(string name, string index, DataType dataType, SymbolicTensorShape shape)
    {
        inputs.Add(new Input { name = name, index = index, dataType = dataType, shape = shape });
    }

    /// <summary>
    /// Adds an input to the model with a symbolic tensor shape.
    /// </summary>
    /// <param name="name">The name of the input.</param>
    /// <param name="dataType">The data type of the input.</param>
    /// <param name="shape">The `SymbolicTensorShape` of the input.</param>
    internal void AddInput(string name, DataType dataType, SymbolicTensorShape shape)
    {
        inputs.Add(new Input { name = name, index = name, dataType = dataType, shape = shape });
    }

    /// <summary>
    /// Adds an input to the model with a tensor shape.
    /// </summary>
    /// <param name="name">The name of the input.</param>
    /// <param name="dataType">The data type of the input.</param>
    /// <param name="shape">The `TensorShape` of the input.</param>
    internal void AddInput(string name, DataType dataType, TensorShape shape)
    {
        inputs.Add(new Input { name = name, index = name, dataType = dataType, shape = new SymbolicTensorShape(shape) });
    }

    /// <summary>
    /// Adds an output called `name` to the model.
    /// </summary>
    /// <param name="name">The name of the input.</param>
    /// <param name="index">The index of the output.</param>
    public void AddOutput(string name, string index)
    {
        outputs.Add(new Output { name = name, index = index });
    }

    /// <summary>
    /// Adds an output called `name` to the model.
    /// </summary>
    /// <param name="name">The name of the input.</param>
    internal void AddOutput(string name)
    {
        outputs.Add(new Output { name = name, index = name });
    }

    /// <summary>
    /// Appends a `layer` to the model.
    /// </summary>
    /// <param name="layer">The layer to append.</param>
    internal void AddLayer(Layers.Layer layer)
    {
        layers.Add(layer);
    }

    /// <summary>
    /// Adds a `constant` to the model.
    /// </summary>
    /// <param name="constant">The constant to add.</param>
    internal void AddConstant(Layers.Constant constant)
    {
        constants.Add(constant);
    }

    internal void DisposeWeights()
    {
        foreach (var constant in constants)
            constant.weights?.Dispose();
    }
}
} // namespace Unity.Sentis
