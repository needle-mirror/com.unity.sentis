using System.Linq;
using Unity.Sentis.Compiler.Analyser;
using Unity.Sentis.Layers;
using Unity.Sentis;
using System.Collections.Generic;
using System;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class ContractToSimplerLayerPass : IModelPass
    {
        public void Run(ref Model model)
        {
            var ctx = new ShapeInferenceContext();
            ShapeInferenceAnalysis.InferModelShapes(model, ctx);

            var modelConstants = new Dictionary<string, Constant>();
            foreach (var c in model.constants)
                modelConstants.Add(c.name, c);

            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layer layer = model.layers[l];

                if (layer is Concat concatLayer)
                {
                    var shape = ctx.GetSymbolicTensorShape(concatLayer.name);
                    if (!shape.hasRank || concatLayer.inputs.Any(o => o != layer.inputs[0]))
                        continue;

                    var tileShape = TensorShape.Ones(shape.rank);
                    tileShape[concatLayer.axis] = concatLayer.inputs.Length;

                    var repeatsConstant = new Constant(model.GetUniqueName(concatLayer.name + "_Repeats"), new TensorInt(new TensorShape(tileShape.rank), tileShape.ToArray()));
                    model.AddConstant(repeatsConstant);
                    model.layers[l] = new Tile(concatLayer.name, concatLayer.inputs[0], repeatsConstant.name);
                }
                if (layer is Layers.Transpose transposeLayer)
                {
                    if (transposeLayer.permutations == null)
                        continue;

                    bool nopTranspose = true;
                    for (int i = 0; i < transposeLayer.permutations.Length; ++i)
                    {
                        if (transposeLayer.permutations[i] == i)
                            continue;
                        nopTranspose = false;
                        break;
                    }

                    if (!nopTranspose)
                        continue;

                    model.layers[l] = new Identity(transposeLayer.name, transposeLayer.inputs[0]);
                }
            }
        }
    }
}
