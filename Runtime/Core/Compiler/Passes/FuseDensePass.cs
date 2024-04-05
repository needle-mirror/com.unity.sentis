using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis.Compiler.Analyser;
using Unity.Sentis;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class FuseDensePass : IModelPass
    {
        public void Run(ref Model model)
        {
            using var ops = new CPUOps();

            var preserve = new HashSet<string>();
            foreach (var o in model.outputs)
                preserve.Add(o.index);

            var inputs = new HashSet<string>();
            foreach (var input in model.inputs)
                inputs.Add(input.index);

            Dictionary<string, Tensor> constTensors = new Dictionary<string, Tensor>();
            foreach (var constant in model.constants)
                constTensors.Add(constant.index, constant.WeightsToTensor());

            var layerDownstream = new Dictionary<string, List<Layers.Layer>>();
            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layers.Layer layer = model.layers[l];
                layerDownstream.Add(layer.index, new List<Layers.Layer>());

                foreach (var input in layer.inputs)
                {
                    if (string.IsNullOrEmpty(input) || inputs.Contains(input) || constTensors.ContainsKey(input))
                        continue;
                    layerDownstream[input].Add(layer);
                }

                if (layer.outputs == null)
                    continue;

                foreach (var output in layer.outputs)
                {
                    if (string.IsNullOrEmpty(output))
                        continue;
                    if (!layerDownstream.ContainsKey(output))
                        layerDownstream.Add(output, new List<Layers.Layer>());
                }
            }

            var removeLayers = new HashSet<string>();
            var remap = new Dictionary<string, string>();

            for (int l = 0; l < model.layers.Count - 1; ++l)
            {
                Layers.Layer layer = model.layers[l];
                if (!(layer is Layers.MatMul || (layer is Layers.MatMul2D && (layer as Layers.MatMul2D).transposeA != true)))
                    continue;

                // const weights of rank 2
                string weightsIndex = layer.inputs[1];
                if (!(constTensors.ContainsKey(weightsIndex) && constTensors[weightsIndex].shape.rank == 2))
                    continue;

                // const bias of rank 1
                List<Layers.Layer> downStreamLayers = layerDownstream[layer.index];
                if (!downStreamLayers.Any(x => x is Layers.Add) && !downStreamLayers.Any(x => x is Layers.ScalarMad))
                    continue;

                bool transposeWeights = layer is Layers.MatMul2D && (layer as Layers.MatMul2D).transposeB;

                Layers.Layer bias;
                string biasIndex;
                if (downStreamLayers.Any(x => x is Layers.ScalarMad))
                {
                    bias = downStreamLayers.Find(x => x is Layers.ScalarMad);
                    var biasMad = bias as Layers.ScalarMad;
                    if (biasMad.dataType == DataType.Int || biasMad.sFloat != 1)
                        continue;
                    var biasS = biasMad.bFloat;
                    using TensorFloat biasT = ops.ConstantOfShape(new TensorShape(constTensors[weightsIndex].shape[transposeWeights ? -2 : -1]), biasS);
                    biasIndex = bias.index + "_Bias";
                    constTensors.Add(biasIndex, biasT);
                    model.constants.Add(new Layers.Constant(biasIndex, biasT));
                }
                else
                {
                    bias = downStreamLayers.Find(x => x is Layers.Add);
                    var biasInputsConst = bias.inputs.Where(x =>
                        x != layer.index && constTensors.ContainsKey(x) && constTensors[x].shape.rank == 1).ToList();
                    if (biasInputsConst.Count != 1)
                        continue;
                    biasIndex = biasInputsConst[0];
                }

                if (preserve.Contains(bias.index))
                    continue;

                TensorFloat weightT = constTensors[weightsIndex] as TensorFloat;

                removeLayers.Add(weightsIndex);
                removeLayers.Add(bias.index);

                if (transposeWeights)
                {
                    weightsIndex = model.GetUniqueIndex(weightsIndex + "_t_for" + layer.index);
                    model.constants.Add(new Layers.Constant(weightsIndex, ops.Transpose(weightT)));
                }

                model.layers[l] = new Layers.Dense(layer.index, layer.inputs[0], weightsIndex, biasIndex);
                remap[bias.index] = layer.index;
            }

            Passes.PassesUtils.RemoveAndRemap(ref model, removeLayers, remap);

            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layers.Layer layer = model.layers[l];
                for (int i = 0; i < layer.inputs.Length; i++)
                {
                    var input = layer.inputs[i];
                    if (remap.ContainsKey(input) && layer.index != remap[input])
                        model.layers[l].inputs[i] = remap[input];
                }
            }

            foreach (var t in constTensors.Values)
                t.Dispose();

            // remove unused constants
            var removeUnusedPass = new Cleanup.RemoveUnusedPass();
            removeUnusedPass.Run(ref model);
        }
    }
}
