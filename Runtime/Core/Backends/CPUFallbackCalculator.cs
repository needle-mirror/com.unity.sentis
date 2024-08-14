using System;
using System.Collections.Generic;

namespace Unity.Sentis
{
    static class CPUFallbackCalculator
    {
        public static HashSet<int> Calculate(Model model, BackendType backendType)
        {
            // Algorithm:
            // start to gather all CPU seeds:
            //  - all layers that needs a given input to be on the CPU (ie read-back)
            //  - they set their respective inputs to need to run on the CPU
            // foreach layers (starting from the bottom's-up)
            //  if a layer is flagged to need to run on the CPU, all inputs also should run on CPU
            //  exception is holes nodes that operate regardless of their input's data
            // Ex:
            //               c = add   d = concat
            //       ...         \    /
            //        |   s = div(c, d)
            //         \  |
            // t = tile(a, s)
            //      \
            //       mul ...
            // * s is set to need to run on cpu = cpu seed
            // * bottoms up:
            //      - mul -> no cpu skip
            //      - tile -> no cpu skip
            //      - a -> no cpu skip
            //      - s -> is cpu, all inputs (a, d) needs to run on cpu
            //   + continue propagating up to start of graph
            var layerCPUFallback = new HashSet<int>();
            if (backendType == BackendType.CPU)
                return layerCPUFallback;

            for (var i = 0; i < model.layers.Count; i++)
            {
                var layer = model.layers[i];

                for (var j = 0; j < layer.inputs.Length; j++)
                {
                    var input = layer.inputs[j];
                    if (input == -1)
                        continue;

                    if (IsInputCPURead(layer, j))
                        layerCPUFallback.Add(input);
                }
            }

            for (var i = model.layers.Count - 1; i >= 0; i--)
            {
                var layer = model.layers[i];

                var isLayerCPU = false;

                foreach (var output in layer.outputs)
                {
                    isLayerCPU |= layerCPUFallback.Contains(output);
                }

                if (!isLayerCPU)
                    continue;

                for (var j = 0; j < layer.inputs.Length; j++)
                {
                    var input = layer.inputs[j];
                    if (input == -1)
                        continue;

                    if (IsInputDataDependency(layer, j))
                        layerCPUFallback.Add(input);
                }
            }

            return layerCPUFallback;
        }

        static bool IsInputCPURead(Layer layer, int inputIndex)
        {
            return layer switch
            {
                Layers.ConstantOfShape => inputIndex is 0,
                Layers.OneHot => inputIndex is 1 or 2,
                Layers.Range => inputIndex is 0 or 1 or 2,
                Layers.TopK => inputIndex is 1,
                Layers.Clip => inputIndex is 1 or 2,
                Layers.CumSum => inputIndex is 1,
                Layers.Reduce => inputIndex is 1,
                Layers.NonMaxSuppression => inputIndex is 2 or 3 or 4,
                Layers.Expand => inputIndex is 1,
                Layers.Narrow => inputIndex is 1 or 2 or 3,
                Layers.Pad => inputIndex is 1 or 2 or 3,
                Layers.Reshape => inputIndex is 1,
                Layers.Resize => inputIndex is 1,
                Layers.Select => inputIndex is 1 or 2,
                Layers.Slice => inputIndex is 1 or 2 or 3 or 4,
                Layers.SliceSet => inputIndex is 2 or 3 or 4 or 5,
                Layers.Split => inputIndex is 1,
                Layers.Squeeze => inputIndex is 1,
                Layers.Tile => inputIndex is 1,
                Layers.Trilu => inputIndex is 1,
                Layers.Unsqueeze => inputIndex is 1,
                _ => false
            };
        }

        static bool IsInputDataDependency(Layer layer, int inputIndex)
        {
            return layer switch
            {
                Layers.Shape => false,
                Layers.Size => false,
                Layers.RandomNormalLike => false,
                Layers.RandomUniformLike => false,
                Layers.CastLike => inputIndex is 0,
                _ => true
            };
        }
    }
}
