using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis.Compiler.Analyser;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class FuseLinearLayersPass : IModelPass
    {
        public void Run(ref Model model)
        {
            using LinearLayerFusing linearLayerFusing = new LinearLayerFusing();
            Dictionary<int, Constant> constTensors = new Dictionary<int, Constant>();
            Dictionary<int, bool> sharedConstants = new Dictionary<int, bool>();
            foreach (var constant in model.constants)
                constTensors.Add(constant.index, constant);

            var indexToLayerIndex = new Dictionary<int, int>();
            int idx = 0;
            foreach (var l in model.layers)
            {
                foreach (var i in l.inputs)
                {
                    if (!constTensors.ContainsKey(i))
                        continue;

                    if (sharedConstants.ContainsKey(i))
                        sharedConstants[i] = true;
                    else
                        sharedConstants.Add(i, false);
                }

                indexToLayerIndex.Add(l.outputs[0], idx);
                idx++;
            }

            var preserve = new HashSet<int>();
            var remap = new Dictionary<int, int>();
            var mergedLayers = new HashSet<int>();
            foreach (var o in model.outputs)
                preserve.Add(o.index);

            for (int l = 0; l < model.layers.Count; ++l)
            {
                var layer = model.layers[l];

                bool isLayerLinear = LinearLayerFusing.IsLayerLinear(layer, constTensors);
                bool isLayerPreserved = preserve.Contains(layer.outputs[0]);
                bool layerHasActivation = IsLayerFusedActivation(layer);

                if (!isLayerLinear)
                    continue;

                // if layer has an activation, we fuse it, but treat it as non linear for future children
                if (!layerHasActivation)
                {
                    remap[layer.outputs[0]] = layer.outputs[0];
                }

                // Multi input nodes can only fuse constants and same inputs
                // only merge constants. @TODO: fuse equal input nodes
                var nonLinearInputs = layer.inputs.Where(x => !remap.ContainsKey(x) && !constTensors.ContainsKey(x)).ToList();
                var linearInputs = layer.inputs.Where(x => remap.ContainsKey(x)).ToList();

                // merge layer with one linearInput and eventual constants
                if (nonLinearInputs.Count > 0 || linearInputs.Count > 1)
                    continue;

                var input = linearInputs[0];

                // input is a linear layer, fuse it
                int inputLayerIndex = model.layers.FindIndex(x => x.outputs[0] == remap[input]);
                Layer inputLayer = model.layers[inputLayerIndex];

                if (!AreLayersFusable(inputLayer, layer, constTensors, sharedConstants, linearLayerFusing))
                    continue;

                //  if input has more than 1 child, we're going to duplicate layers, not worth merging
                if (GraphLogicAnalysis.GetDownStreamLayersCount(model, input) != 1)
                    continue;

                // convention: layer will be fused into inputLayer
                // => fused layer will have the same inputs as inputLayer
                Layer fusedLayer = linearLayerFusing.FuseLayers(inputLayer, layer, constTensors);

                if (layerHasActivation)
                {
                    (fusedLayer as Layers.FusedActivation).fusedActivation = (layer as Layers.FusedActivation).fusedActivation;
                }

                // preserve layer if output/memory
                if (isLayerPreserved)
                {
                    // cannot merge layer into input:
                    // remove input, no need to remap as inputs == input.inputs
                    fusedLayer.outputs = (int[])layer.outputs.Clone();
                    model.layers[l] = fusedLayer;

                    if (!preserve.Contains(inputLayer.outputs[0]))
                        mergedLayers.Add(inputLayer.outputs[0]);
                }
                else
                {
                    // merge layer into input
                    // remove current and remap input indexes
                    mergedLayers.Add(layer.outputs[0]);
                    remap[layer.outputs[0]] = fusedLayer.outputs[0];
                    model.layers[inputLayerIndex] = fusedLayer;
                }
            }

            // remove merged layers
            Passes.PassesUtils.RemoveAndRemap(ref model, mergedLayers, remap);

            model.constants = constTensors.Values.ToList();

            // unpack maths ops const inputs into new const layer
            // remove unused constants
            var removeUnusedPass = new Cleanup.RemoveUnusedPass();
            removeUnusedPass.Run(ref model);
        }

        static bool IsLayerFusedActivation(Layer layer)
        {
            if (layer is Layers.FusedActivation)
                return (layer as Layers.FusedActivation).fusedActivation != Layers.FusableActivation.None;
            return false;
        }

        static bool AreLayersFusable(Layer l0, Layer l1, Dictionary<int, Constant> constTensors, Dictionary<int, bool> sharedConstants, LinearLayerFusing linearLayerFuser)
        {
            if (l0.inputs.Any(i => sharedConstants.ContainsKey(i) && sharedConstants[i]))
                return false;
            if (l1.inputs.Any(i => sharedConstants.ContainsKey(i) && sharedConstants[i]))
                return false;
            // can't fuse if input has a fused activation or if fusing code not implemented
            return !IsLayerFusedActivation(l0) && linearLayerFuser.AreLayersFusable(l0, l1, constTensors);
        }
    }
}
