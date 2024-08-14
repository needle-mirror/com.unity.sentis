using System;
using System.Collections.Generic;
using UnityEngine;

namespace Unity.Sentis.Compiler.Analyser
{
    static class MemoryFootprintAnalysis
    {
        public static HashSet<Layer> FindLayersThatRequireStorage(Model model)
        {
            var allInputsExceptFromPreviousLayer = new HashSet<int>();
            Layer prevLayer = null;
            foreach (var layer in model.layers)
            {
                foreach (var input in layer.inputs)
                {
                    if (input == -1)
                        continue;
                    if (prevLayer != null && input != prevLayer.outputs[0])
                        allInputsExceptFromPreviousLayer.Add(input);
                }
                prevLayer = layer;
            }

            var allOutputs = new HashSet<int>();
            foreach (var output in model.outputs)
                allOutputs.Add(output.index);

            var requireStorage = new HashSet<Layer>();
            foreach (var layer in model.layers)
            {
                if (allInputsExceptFromPreviousLayer.Contains(layer.outputs[0]) ||
                    allOutputs.Contains(layer.outputs[0]))
                    requireStorage.Add(layer);
            }

            return requireStorage;
        }
    }
}

