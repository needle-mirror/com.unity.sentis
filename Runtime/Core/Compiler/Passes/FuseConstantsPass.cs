using System;
using System.Collections.Generic;
using System.Reflection;
using Unity.Sentis.Layers;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class FuseConstantsPass : IModelPass
    {
        public void Run(ref Model model)
        {
            FuseConstants(ref model);
        }

        static void FuseConstants(ref Model model)
        {
            var ctx = new PartialInferenceContext();
            var backend = new CPUBackend();
            var vars = new ModelStorage();
            var executionContext = new ExecutionContext
            {
                backend = backend,
                vars = vars
            };

            var constantTensors = new Dictionary<string, Tensor>();
            var calculatedTensors = new Dictionary<string, Tensor>();

            // model constants
            // TODO refactor pass to have constants in vars.m_TensorByName to avoid copy/access to layer.Execute
            foreach (var constant in model.constants)
            {
                var constantTensor = constant.WeightsToTensor();
                constantTensors.Add(constant.index, constantTensor);
                ctx.AddPartialTensor(constant.index, PartialTensor.FromTensor(constantTensor));
            }

            // model inputs
            foreach (var input in model.inputs)
            {
                ctx.AddPartialTensor(input.index, new PartialTensor(input.dataType, input.shape));
            }

            // iterate through layers executing if layer inputs are all known
            foreach (var layer in model.layers)
            {
                var isDeterministic = !layer.GetType().IsDefined(typeof(NonDeterministicOutput));
                for (var i = 0; i < layer.inputs.Length && isDeterministic; i++)
                {
                    isDeterministic &= string.IsNullOrEmpty(layer.inputs[i]) || calculatedTensors.ContainsKey(layer.inputs[i]) || constantTensors.ContainsKey(layer.inputs[i]);
                }

                if (!isDeterministic)
                {
                    // partial tensor inference
                    layer.InferPartial(ctx);
                    var outputPartialTensor = ctx.GetPartialTensor(layer.index);
                    if (outputPartialTensor.IsFullyKnown())
                        calculatedTensors.Add(layer.index, outputPartialTensor.ToTensor());
                    for (var i = 1; i < (layer.outputs?.Length ?? 0); i++)
                    {
                        outputPartialTensor = ctx.GetPartialTensor(layer.outputs[i]);
                        if (outputPartialTensor.IsFullyKnown())
                            calculatedTensors.Add(layer.outputs[i], outputPartialTensor.ToTensor());
                    }
                    continue;
                }

                for (var i = 0; i < layer.inputs.Length; i++)
                {
                    Tensor tensor = null;
                    if (string.IsNullOrEmpty(layer.inputs[i]))
                        continue;
                    if (calculatedTensors.TryGetValue(layer.inputs[i], out var calculatedInputTensor))
                        tensor = calculatedInputTensor;
                    else
                        tensor = constantTensors[layer.inputs[i]];

                    executionContext.vars.SetInput(layer.inputs[i], tensor);
                }

                // full inference
                layer.Execute(executionContext);
                var outputTensor = executionContext.vars.GetTensor(layer.index);

                calculatedTensors.Add(layer.index, outputTensor);
                ctx.AddPartialTensor(layer.index, PartialTensor.FromTensor(outputTensor));

                if (layer.outputs == null)
                    continue;

                for (var i = 1; i < layer.outputs.Length; i++)
                {
                    outputTensor = vars.PeekTensor(layer.outputs[i]);
                    calculatedTensors.Add(layer.outputs[i], outputTensor);
                    ctx.AddPartialTensor(layer.outputs[i], PartialTensor.FromTensor(outputTensor));
                }
            }

            // remove precalculated layers
            model.layers.RemoveAll(x => calculatedTensors.ContainsKey(x.index));

            // add precalculated constants
            foreach (var kvp in calculatedTensors)
            {
                model.constants.Add(new Constant(kvp.Key, kvp.Value));
                kvp.Value.Dispose();
            }

            // remove unused constants
            var removeUnusedPass = new Cleanup.RemoveUnusedPass();
            removeUnusedPass.Run(ref model);

            foreach (var constantTensor in constantTensors.Values)
            {
                constantTensor.Dispose();
            }

            backend.Dispose();
            vars.Dispose();
        }
    }
}
