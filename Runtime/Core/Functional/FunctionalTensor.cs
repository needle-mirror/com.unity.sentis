using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents a tensor that is a result of tensor operations.
    /// </summary>
    public partial class FunctionalTensor
    {
        DataType m_DataType;
        FunctionalNode m_Source;
        int m_OutputIndex;

        internal DataType DataType => m_DataType;
        internal FunctionalNode Source => m_Source;
        internal int OutputIndex => m_OutputIndex;
        internal string Name => m_Source.OutputNames[m_OutputIndex];

        internal FunctionalTensor(DataType dataType, FunctionalNode source, int outputIndex)
        {
            m_DataType = dataType;
            m_Source = source;
            m_OutputIndex = outputIndex;
        }

        /// <summary>
        /// Creates and returns an instance of `FunctionalTensor` from an existing tensor.
        /// </summary>
        /// <param name="tensor">The tensor to use as the source.</param>
        /// <returns>The functional tensor.</returns>
        public static FunctionalTensor FromTensor(Tensor tensor)
        {
            var constant = new Layers.Constant(null, tensor);
            var constantNode = new FunctionalConstant(constant);
            return new FunctionalTensor(constant.dataType, constantNode, 0);
        }

        /// <summary>
        /// Creates and returns an instance of `FunctionalTensor` from an existing constant.
        /// </summary>
        /// <param name="constant">The constant to use as the source.</param>
        /// <returns>The functional tensor.</returns>
        public static FunctionalTensor FromConstant(Layers.Constant constant)
        {
            var constantNode = new FunctionalConstant(constant);
            return new FunctionalTensor(constant.dataType, constantNode, 0);
        }

        internal static FunctionalTensor FromInput(Model.Input input)
        {
            var inputNode = new FunctionalInput(input);
            return new FunctionalTensor(input.dataType, inputNode, 0);
        }

        internal static FunctionalTensor[] FromLayerMultiOutputs(Layers.Layer layer, DataType[] dataTypes, FunctionalTensor[] inputs)
        {
            Assert.AreEqual(layer.inputs.Length, inputs.Length);
            var layerNode = new FunctionalLayer(inputs, dataTypes, layer);
            return layerNode.CreateOutputs();
        }

        internal static FunctionalTensor FromLayer(Layers.Layer layer, DataType dataType, FunctionalTensor[] inputs)
        {
            return FromLayerMultiOutputs(layer, new[] { dataType }, inputs)[0];
        }

        internal static FunctionalTensor FromLayer(Layers.Layer layer, DataType dataType, FunctionalTensor input)
        {
            return FromLayerMultiOutputs(layer, new[] { dataType }, new[] { input })[0];
        }

        /// <summary>
        /// Creates and returns an array of `FunctionalTensor` as the outputs of an existing model.
        /// </summary>
        /// <param name="model">The model to use as the source.</param>
        /// <param name="inputs">The functional tensors to use as the inputs to the model.</param>
        /// <param name="withCopy">Whether to do a deep copy of the model. When `false` Sentis will make destructive edits of the source model.</param>
        /// <returns>The functional tensor array.</returns>
        public static FunctionalTensor[] FromModel(Model model, FunctionalTensor[] inputs, bool withCopy = false)
        {
            if (withCopy)
                model = model.DeepCopy();
            Logger.AssertIsTrue(inputs.Length == model.inputs.Count, "ModelOutputs.ValueError: inputs length does not equal model input count {0}, {1}", inputs.Length, model.inputs.Count);
            var expressions = new Dictionary<string, FunctionalTensor>();

            for (var i = 0; i < inputs.Length; i++)
                expressions[model.inputs[i].index] = inputs[i];

            foreach (var constant in model.constants)
            {
                var node = new FunctionalConstant(constant);
                expressions[constant.index] = new FunctionalTensor(constant.dataType, node, 0);
            }

            var ctx = new PartialInferenceContext();
            foreach (var kvp in expressions)
                ctx.AddPartialTensor(kvp.Key, new PartialTensor(kvp.Value.DataType));

            foreach (var layer in model.layers)
            {
                layer.inputs = (string[])layer.inputs.Clone();
                if (layer.outputs != null)
                    layer.outputs = (string[])layer.outputs.Clone();
                var layerInputs = new FunctionalTensor[layer.inputs.Length];
                for (var i = 0; i < layerInputs.Length; i++)
                {
                    if (string.IsNullOrEmpty(layer.inputs[i]))
                        continue;
                    layerInputs[i] = expressions[layer.inputs[i]];
                }

                // infer data types
                layer.InferPartial(ctx);
                var outputDataTypes = new DataType[layer.outputs?.Length ?? 1];
                for (var i = 0; i < outputDataTypes.Length; i++)
                {
                    var name = i == 0 ? layer.index : layer.outputs?[i];
                    if (string.IsNullOrEmpty(name))
                        continue;
                    outputDataTypes[i] = ctx.GetPartialTensor(name).dataType;
                }

                var node = new FunctionalLayer(layerInputs, outputDataTypes, layer);
                var layerOutputs = node.CreateOutputs();
                expressions[layer.index] = layerOutputs[0];
                if (layer.outputs is null)
                    continue;
                for (var i = 1; i < layer.outputs.Length; i++)
                {
                    if (string.IsNullOrEmpty(layer.outputs[i]))
                        continue;
                    expressions[layer.outputs[i]] = layerOutputs[i];
                }
            }

            var outputs = new FunctionalTensor[model.outputs.Count];
            for (var i = 0; i < model.outputs.Count; i++)
                outputs[i] = expressions[model.outputs[i].index];
            return outputs;
        }
    }
}
