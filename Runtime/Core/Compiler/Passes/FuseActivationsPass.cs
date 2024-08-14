using System;
using System.Linq;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class FuseActivationPass : IModelPass
    {
        public void Run(ref Model model)
        {
            //Fused activation
            var fusableActivations = model.layers.Where(l => IsActivationFusable(l)).ToList();
            // Fused activation
            foreach (var activationLayer in fusableActivations)
            {
                if (activationLayer.inputs.Length != 1)
                    continue;

                var mainLayer = model.layers.Find(l => l.outputs[0] == activationLayer.inputs[0]);
                if (mainLayer == null)
                    continue;

                if (!(mainLayer is Layers.FusedActivation))
                    continue;
                if ((mainLayer as Layers.FusedActivation).fusedActivation != Layers.FusableActivation.None)
                    continue;

                if (model.outputs.Aggregate(false, (current, o) => current | o.index == mainLayer.outputs[0]))
                    continue;

                //Need to check that no other layers uses mainLayer directly.
                //Activation in the graph below can not be fused because (concat) layer needs raw output of (conv) layer
                //conv -> relu -----.
                //    \             v
                //     `---------> concat
                if (model.layers.Exists(l => l != activationLayer && l.inputs.Contains(mainLayer.outputs[0])))
                    continue;

                FuseActivation(ref model, mainLayer, activationLayer);
            }
        }

        public static bool IsActivationFusable(Layer layer)
        {
            return (layer is Layers.Relu);
        }

        public static Layers.FusableActivation LayerToActivation(Layer layer)
        {
            if (layer is Layers.Relu)
                return Layers.FusableActivation.Relu;
            else
                return Layers.FusableActivation.None;
        }

        static private void FuseActivation(ref Model model, Layer mainLayer, Layer activationToFuse)
        {
            // patch `mainLayer`
            if (mainLayer is Layers.FusedActivation)
                (mainLayer as Layers.FusedActivation).fusedActivation = LayerToActivation(activationToFuse);

            // patch all layers depending on `activationToFuse`
            foreach (var l in model.layers)
            {
                for (var i = 0; i < l.inputs.Length; ++i)
                {
                    if (l.inputs[i] == activationToFuse.outputs[0])
                        l.inputs[i] = mainLayer.outputs[0];
                }
            }

            // patch outputs
            for (var i = 0; i < model.outputs.Count; i++)
            {
                if (model.outputs[i].index == activationToFuse.outputs[0])
                    model.outputs[i] = new Model.Output { name = model.outputs[i].name, index = mainLayer.outputs[0] };
            }

            model.layers.Remove(activationToFuse);
        }
    }
}
