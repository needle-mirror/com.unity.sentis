using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;

namespace Unity.Sentis.Compiler.Analyser
{
    static class PartialInferenceAnalysis
    {
        public static PartialInferenceContext InferModelPartialTensors(Model model, bool useConstantWeights, IDictionary<string, TensorShape> inputShapes = null)
        {
            ProfilerMarkers.InferModelSymbolicTensors.Begin();

            var ctx = new PartialInferenceContext();

            foreach (var constant in model.constants)
            {
                if (useConstantWeights)
                    ctx.AddPartialTensor(constant.index, PartialTensor.FromTensor(constant.WeightsToTensor()));
                else
                    ctx.AddPartialTensor(constant.index, new PartialTensor(constant.dataType, new SymbolicTensorShape(constant.shape)));
            }

            // model inputs
            foreach (var input in model.inputs)
            {
                if (inputShapes != null && inputShapes.TryGetValue(input.index, out var inputShape))
                    ctx.AddPartialTensor(input.index, new PartialTensor(input.dataType, new SymbolicTensorShape(inputShape)));
                else
                    ctx.AddPartialTensor(input.index, new PartialTensor(input.dataType, input.shape));
            }

            // Partial tensor inference
            foreach (var layer in model.layers)
                layer.InferPartial(ctx);

            ProfilerMarkers.InferModelSymbolicTensors.End();

            return ctx;
        }
    }
}

