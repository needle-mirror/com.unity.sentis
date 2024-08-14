using System;
using System.Collections.Generic;

namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Creates and returns an array of `FunctionalTensor` as the output of the forward pass of an existing model.
        ///
        /// Sentis will make destructive edits of the source model.
        /// </summary>
        /// <param name="model">The model to use as the source.</param>
        /// <param name="inputs">The functional tensors to use as the inputs to the model.</param>
        /// <returns>The functional tensor array.</returns>
        public static FunctionalTensor[] Forward(Model model, params FunctionalTensor[] inputs)
        {
            Logger.AssertIsTrue(inputs.Length == model.inputs.Count, "ModelOutputs.ValueError: inputs length does not equal model input count {0}, {1}", inputs.Length, model.inputs.Count);
            var expressions = new Dictionary<int, FunctionalTensor>();

            for (var i = 0; i < inputs.Length; i++)
                expressions[model.inputs[i].index] = inputs[i];

            foreach (var constant in model.constants)
            {
                var node = new ConstantNode(constant);
                expressions[constant.index] = new FunctionalTensor(constant.dataType, node, 0);
            }

            var ctx = new PartialInferenceContext();
            foreach (var kvp in expressions)
                ctx.AddPartialTensor(kvp.Key, new PartialTensor(kvp.Value.dataType));

            foreach (var layer in model.layers)
            {
                layer.inputs = (int[])layer.inputs.Clone();
                layer.outputs = (int[])layer.outputs.Clone();
                var layerInputs = new FunctionalTensor[layer.inputs.Length];
                for (var i = 0; i < layerInputs.Length; i++)
                {
                    if (layer.inputs[i] == -1)
                        continue;
                    layerInputs[i] = expressions[layer.inputs[i]];
                }

                // infer data types
                layer.InferPartial(ctx);
                var outputDataTypes = new DataType[layer.outputs.Length];
                var isOutputShapes = new bool[layer.outputs.Length];
                var outputShapes = new TensorShape[layer.outputs.Length];
                for (var i = 0; i < outputDataTypes.Length; i++)
                {
                    if (layer.outputs[i] == -1)
                        continue;
                    var outputPartialTensor = ctx.GetPartialTensor(layer.outputs[i]);
                    outputDataTypes[i] = outputPartialTensor.dataType;
                    if (outputPartialTensor.shape.IsStatic())
                    {
                        isOutputShapes[i] = true;
                        outputShapes[i] = outputPartialTensor.shape.ToTensorShape();
                    }
                }

                var node = new LayerNode(layerInputs, outputDataTypes, layer);
                var layerOutputs = node.CreateOutputs();
                for (var i = 0; i < layer.outputs.Length; i++)
                {
                    if (isOutputShapes[i])
                        layerOutputs[i].SetShape(outputShapes[i]);
                    if (layer.outputs[i] == -1)
                        continue;
                    expressions[layer.outputs[i]] = layerOutputs[i];
                }
            }

            var outputs = new FunctionalTensor[model.outputs.Count];
            for (var i = 0; i < model.outputs.Count; i++)
                outputs[i] = expressions[model.outputs[i].index];
            return outputs;
        }

        /// <summary>
        /// Creates and returns an array of `FunctionalTensor` as the output of the forward pass of an existing model.
        ///
        /// Sentis will copy the source model and not make edits to it.
        /// </summary>
        /// <param name="model">The model to use as the source.</param>
        /// <param name="inputs">The functional tensors to use as the inputs to the model.</param>
        /// <returns>The functional tensor array.</returns>
        public static FunctionalTensor[] ForwardWithCopy(Model model, params FunctionalTensor[] inputs)
        {
            model = model.DeepCopy();
            return Forward(model, inputs);
        }
    }
}
