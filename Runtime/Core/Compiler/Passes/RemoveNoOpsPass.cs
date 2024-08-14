using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Sentis.Compiler.Passes.Cleanup
{
    // TODO remove useless patterns:
    // Reduce keepdim 0 -> * -> Reshape
    class RemoveNoOpsPass : IModelPass
    {
        public void Run(ref Model model)
        {
            var noopLayers = new HashSet<int>();
            var remap = new Dictionary<int, int>();
            var preserve = new HashSet<int>();
            foreach (var o in model.outputs)
                preserve.Add(o.index);

            // algorithm:
            // - if input is pointing to a noop, we need to remap it to upstream layer
            // - if layer is a noop, store its link to upstream layer
            // layers are in order of appearance, so if layer_N has layer_M as input, we'd have treated layer_M before
            for (int l = 0; l < model.layers.Count; ++l)
            {
                var layer = model.layers[l];

                // replace removed layers with their upstream inputs
                for (int i = 0; i < layer.inputs.Length; ++i)
                {
                    var input = layer.inputs[i];

                    if (remap.ContainsKey(input))
                    {
                        Assert.IsTrue(noopLayers.Contains(input));
                        model.layers[l].inputs[i] = remap[input];
                    }
                    else
                    {
                        Assert.IsFalse(noopLayers.Contains(input));
                    }
                }

                if (layer.inputs.Length == 0) // const
                    continue;

                // if layer is noop = nop, identity or flatten
                if (layer is Layers.Identity)
                {
                    remap[layer.outputs[0]] = layer.inputs[0];
                    noopLayers.Add(layer.outputs[0]);
                }
            }

            model.layers.RemoveAll(x => noopLayers.Contains(x.outputs[0]) && !preserve.Contains(x.outputs[0]));
        }

        static bool IsLayerNoop(Model model, Layer layer)
        {
            if (layer is Layers.Identity)
                return true;
            else
                return false;
        }
    }
}
