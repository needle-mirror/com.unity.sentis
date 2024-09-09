using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis.Compiler.Analyser;
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

            var layerDownstream = new Dictionary<int, List<int>>();
            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layer layer = model.layers[l];

                foreach (var input in layer.inputs)
                {
                    if ((input == -1) || inputs.Contains(input) || constTensors.ContainsKey(input))
                        continue;
                    layerDownstream[input].Add(l);
                }

                foreach (var output in layer.outputs)
                {
                    if (output == -1)
                        continue;
                    if (!layerDownstream.ContainsKey(output))
                        layerDownstream.Add(output, new List<int>());
                }
            }

            var ctx = PartialInferenceAnalysis.InferModelPartialTensors(model);

            var removeLayers = new HashSet<int>();
            var remap = new Dictionary<int, int>();

            for (int l = 0; l < model.layers.Count - 1; ++l)
            {
                Layer layer = model.layers[l];
                var downStreamLayers = layerDownstream[layer.outputs[0]];
                if (downStreamLayers.Count != 1)
                    continue;
                var addLayerIndex = downStreamLayers[0];
                var addLayer = model.layers[addLayerIndex];
                if ((layer is Layers.MatMul || (layer is Layers.MatMul2D && (layer as Layers.MatMul2D).transposeA != true)))
                {
                    // const weights of rank 2
                    var weightsIndex = layer.inputs[1];
                    if (!(constTensors.ContainsKey(weightsIndex)))
                        continue;

                    var shapeX = ctx.GetPartialTensor(layer.inputs[0]).shape;
                    var shapeW = ctx.GetPartialTensor(layer.inputs[1]).shape;

                    if (shapeW.rank > 2 && layer is Layers.MatMul && addLayer is Layers.Add)
                    {
                        var biasIndex = addLayer.inputs[0] == layer.outputs[0] ? addLayer.inputs[1] : addLayer.inputs[0];
                        var shapeB = ctx.GetPartialTensor(biasIndex).shape;
                        if (shapeX.rank != shapeW.rank || shapeX.rank != shapeB.rank)
                            continue;

                        var allEqual = true;
                        for (int i = 0; i < shapeX.rank - 2; i++)
                        {
                            allEqual = allEqual && (shapeX[i] == shapeW[i]) && (shapeX[i] == shapeB[i]);
                        }
                        if (allEqual)
                        {
                            model.layers[addLayerIndex] = new Layers.DenseBatched(addLayer.outputs[0], layer.inputs[0], layer.inputs[1], biasIndex);
                            removeLayers.Add(layer.outputs[0]);
                        }
                    }
                    else
                    {
                        // const bias of rank 1
                        if (!(addLayer is Layers.Add) && !(addLayer is Layers.ScalarMad))
                            continue;

                        bool transposeWeights = layer is Layers.MatMul2D && (layer as Layers.MatMul2D).transposeB;

                        int biasIndex;
                        if (addLayer is Layers.ScalarMad)
                        {
                            var biasMad = addLayer as Layers.ScalarMad;
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
                            var biasInputsConst = addLayer.inputs.Where(x =>
                                x != layer.outputs[0] && constTensors.ContainsKey(x) && constTensors[x].shape.rank == 1).ToList();
                            if (biasInputsConst.Count != 1)
                                continue;
                            biasIndex = biasInputsConst[0];
                        }

                        Tensor<float> weightT = constTensors[weightsIndex] as Tensor<float>;

                        removeLayers.Add(weightsIndex);
                        removeLayers.Add(addLayer.outputs[0]);

                        if (transposeWeights)
                        {
                            weightsIndex = model.GetUniqueIndex();
                            using var transposed = ops.Transpose(weightT);
                            model.constants.Add(new Constant(weightsIndex, transposed.shape, transposed.DownloadToArray()));
                        }

                        model.layers[l] = new Layers.Dense(layer.outputs[0], layer.inputs[0], weightsIndex, biasIndex);
                        remap[addLayer.outputs[0]] = layer.outputs[0];
                    }
                }
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
