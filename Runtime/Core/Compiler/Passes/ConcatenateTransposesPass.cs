using System;
using System.Collections.Generic;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class ConcatenateTransposesPass : IModelPass
    {
        public void Run(ref Model model)
        {
            var preserve = new HashSet<int>();
            var removeLayers = new HashSet<int>();
            var transposeReferences = new Dictionary<int, int>();
            var layerDownstreamCounts = new Dictionary<int, int>();
            foreach (var o in model.outputs)
                preserve.Add(o.index);

            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layer layer = model.layers[l];

                layerDownstreamCounts[layer.outputs[0]] = 0;

                foreach (var input in layer.inputs)
                {
                    if (input == -1)
                        continue;
                    if (layerDownstreamCounts.ContainsKey(input))
                        layerDownstreamCounts[input] += 1;
                }

                if (!(layer is Layers.Transpose))
                    continue;

                transposeReferences[layer.outputs[0]] = l;
            }

            for (int l = 0; l < model.layers.Count; ++l)
            {
                if (!(model.layers[l] is Layers.Transpose))
                    continue;
                Layers.Transpose layer = model.layers[l] as Layers.Transpose;

                int input = layer.inputs[0];

                if (!transposeReferences.ContainsKey(input))
                    continue;

                Layers.Transpose previousLayer = model.layers[transposeReferences[input]] as Layers.Transpose;

                // previous layer is a transpose and current layer is the only downstream layer
                var permutations = MergeTranspose(previousLayer.permutations, layer.permutations);

                model.layers[l] = new Layers.Transpose(layer.outputs[0], previousLayer.inputs[0], permutations);

                if (!preserve.Contains(input) && (layerDownstreamCounts[input] == 1))
                    removeLayers.Add(input);
            }

            Passes.PassesUtils.RemoveAndRemap(ref model, removeLayers, new Dictionary<int, int>());
        }

        int[] MergeTranspose(int[] transpose0, int[] transpose1)
        {
            return (new TensorShape(transpose0)).Transpose(transpose1).ToArray();
        }
    }
}
