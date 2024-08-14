using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class FuseDensePass : IModelPass
    {
        public void Run(ref Model model)
        {
            using var ops = new CPUOps();

            var preserve = new HashSet<int>();
            foreach (var o in model.outputs)
                preserve.Add(o.index);

            var inputs = new HashSet<int>();
            foreach (var input in model.inputs)
                inputs.Add(input.index);

            Dictionary<int, Tensor> constTensors = new Dictionary<int, Tensor>();
            foreach (var constant in model.constants)
                constTensors.Add(constant.index, constant.WeightsToTensor());

            var layerDownstream = new Dictionary<int, List<Layer>>();
            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layer layer = model.layers[l];

                foreach (var input in layer.inputs)
                {
                    if ((input == -1) || inputs.Contains(input) || constTensors.ContainsKey(input))
                        continue;
                    layerDownstream[input].Add(layer);
                }

                foreach (var output in layer.outputs)
                {
                    if (output == -1)
                        continue;
                    if (!layerDownstream.ContainsKey(output))
                        layerDownstream.Add(output, new List<Layer>());
                }
            }

            var removeLayers = new HashSet<int>();
            var remap = new Dictionary<int, int>();

            for (int l = 0; l < model.layers.Count - 1; ++l)
            {
                Layer layer = model.layers[l];
                if (!(layer is Layers.MatMul || (layer is Layers.MatMul2D && (layer as Layers.MatMul2D).transposeA != true)))
                    continue;

                // const weights of rank 2
                var weightsIndex = layer.inputs[1];
                if (!(constTensors.ContainsKey(weightsIndex) && constTensors[weightsIndex].shape.rank == 2))
                    continue;

                // const bias of rank 1
                List<Layer> downStreamLayers = layerDownstream[layer.outputs[0]];
                if (!downStreamLayers.Any(x => x is Layers.Add) && !downStreamLayers.Any(x => x is Layers.ScalarMad))
                    continue;

                bool transposeWeights = layer is Layers.MatMul2D && (layer as Layers.MatMul2D).transposeB;

                Layer bias;
                int biasIndex;
                if (downStreamLayers.Any(x => x is Layers.ScalarMad))
                {
                    bias = downStreamLayers.Find(x => x is Layers.ScalarMad);
                    var biasMad = bias as Layers.ScalarMad;
                    if (biasMad.dataType == DataType.Int || biasMad.sFloat != 1)
                        continue;
                    var biasS = biasMad.bFloat;
                    using Tensor<float> biasT = ops.ConstantOfShape(new TensorShape(constTensors[weightsIndex].shape[transposeWeights ? -2 : -1]), biasS);
                    biasIndex = model.GetUniqueIndex();
                    constTensors.Add(biasIndex, biasT);
                    model.constants.Add(new Constant(biasIndex, biasT.shape, biasT.DownloadToArray()));
                }
                else
                {
                    bias = downStreamLayers.Find(x => x is Layers.Add);
                    var biasInputsConst = bias.inputs.Where(x =>
                        x != layer.outputs[0] && constTensors.ContainsKey(x) && constTensors[x].shape.rank == 1).ToList();
                    if (biasInputsConst.Count != 1)
                        continue;
                    biasIndex = biasInputsConst[0];
                }

                if (preserve.Contains(bias.outputs[0]))
                    continue;

                Tensor<float> weightT = constTensors[weightsIndex] as Tensor<float>;

                removeLayers.Add(weightsIndex);
                removeLayers.Add(bias.outputs[0]);

                if (transposeWeights)
                {
                    weightsIndex = model.GetUniqueIndex();
                    using var transposed = ops.Transpose(weightT);
                    model.constants.Add(new Constant(weightsIndex, transposed.shape, transposed.DownloadToArray()));
                }

                model.layers[l] = new Layers.Dense(layer.outputs[0], layer.inputs[0], weightsIndex, biasIndex);
                remap[bias.outputs[0]] = layer.outputs[0];
            }

            Passes.PassesUtils.RemoveAndRemap(ref model, removeLayers, remap);

            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layer layer = model.layers[l];
                for (int i = 0; i < layer.inputs.Length; i++)
                {
                    var input = layer.inputs[i];
                    if (remap.ContainsKey(input) && layer.outputs[0] != remap[input])
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
