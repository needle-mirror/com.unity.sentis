using System;
using System.Collections.Generic;
using System.Reflection;
using Unity.Sentis.Layers;
using Unity.Sentis;
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
            var allocator = new DefaultTensorAllocator();
            var computedLayerTensors = ComputeKnownLayerTensors(model, allocator);

            // remove precalculated layers
            model.layers.RemoveAll(x => computedLayerTensors.ContainsKey(x.name));

            // add precalculated constants
            foreach (var kvp in computedLayerTensors)
            {
                model.constants.Add(new Constant(kvp.Key, kvp.Value));
            }

            // dispose tensors
            allocator.Dispose();

            // remove unused constants
            var removeUnusedPass = new Cleanup.RemoveUnusedPass();
            removeUnusedPass.Run(ref model);
        }

        static Dictionary<string, Tensor> ComputeKnownLayerTensors(Model model, ITensorAllocator allocator)
        {
            var ctx = new ShapeInferenceContext();
            var op = new CPUOps(allocator);
            var vars = new DefaultVars();
            var executionContext = new ExecutionContext
            {
                ops = op,
                vars = vars
            };

            var constantNames = new HashSet<string>();

            // model constants
            foreach (var constant in model.constants)
            {
                constantNames.Add(constant.name);
                ctx.AddKnownTensor(constant.name, constant.DataSetToTensor());
            }

            // model inputs
            foreach (var input in model.inputs)
            {
                ctx.AddShape(input.name, input.shape);
            }

            // iterate through layers executing if layer inputs are all known
            foreach (var layer in model.layers)
            {
                if (ctx.TryGetKnownTensors(layer.inputs, out var inputs) && !layer.GetType().IsDefined(typeof(NonDeterministicOutput)))
                {
                    // full inference
                    var outputTensor = layer.Execute(inputs, executionContext);
                    ctx.AddKnownTensor(layer.name, outputTensor);

                    if (layer.outputs == null)
                        continue;

                    for (var i = 1; i < layer.outputs.Length; i++)
                    {
                        ctx.AddKnownTensor(layer.outputs[i], vars.PeekOutput(layer.outputs[i]));
                    }
                }
                else
                {
                    // shape inference
                    var layerInputShapes = ctx.GetShapes(layer.inputs);
                    var layerOutputShape = layer.InferOutputShape(layerInputShapes, ctx);
                    ctx.AddShape(layer.name, layerOutputShape);

                    // partial tensor inference
                    var layerInputPartialTensors = ctx.GetPartialTensors(layer.inputs);
                    var layerOutputPartialTensor = layer.InferPartialTensor(layerInputPartialTensors, ctx);
                    ctx.AddPartialTensor(layer.name, layerOutputPartialTensor);
                }
            }

            // create return dictionary without constants
            var calculatedTensors = new Dictionary<string, Tensor>();

            foreach (var kvp in ctx.KnownTensors)
            {
                if (!constantNames.Contains(kvp.Key))
                    calculatedTensors[kvp.Key] = kvp.Value.DeepCopy();
            }

            vars.Dispose();
            ctx.Dispose();

            return calculatedTensors;
        }
    }
}
