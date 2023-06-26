using System;
using System.Collections.Generic;
using CPUReadInputs = Unity.Sentis.Optimization.CPUFallback.CPUReadInputs;
using NoDataDependencyInputs = Unity.Sentis.Optimization.CPUFallback.NoDataDependencyInputs;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class CPUFallbackPass : IModelPass
    {
        public void Run(ref Model model)
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
            HashSet<string> layersOnCPU = model.LayerCPUFallback;
            layersOnCPU.Clear();
            foreach (var layer in model.layers)
            {
                CPUReadInputs attribute = (CPUReadInputs)Attribute.GetCustomAttribute(layer.GetType(), typeof(CPUReadInputs));
                if (attribute == null)
                    continue;

                foreach (var i in attribute.InputsOnCPU)
                {
                    if (i >= layer.inputs.Length)
                        continue;

                    string input = layer.inputs[i];
                    if (string.IsNullOrEmpty(input))
                        continue;

                    layersOnCPU.Add(input);
                }
            }

            for (int layerIndex = (model.layers.Count - 1); layerIndex >= 0; layerIndex--)
            {
                var layer = model.layers[layerIndex];

                if (!layersOnCPU.Contains(layer.name))
                    continue;

                NoDataDependencyInputs attribute = (NoDataDependencyInputs)Attribute.GetCustomAttribute(layer.GetType(), typeof(NoDataDependencyInputs));
                if (attribute != null)
                {
                    HashSet<int> inputsNoDataDependecy = new HashSet<int>(attribute.InputsNoDataDependency);
                    for (int i = 0; i < layer.inputs.Length; i++)
                    {
                        string input = layer.inputs[i];
                        if (string.IsNullOrEmpty(input))
                            continue;

                        if (!inputsNoDataDependecy.Contains(i))
                            layersOnCPU.Add(input);
                    }

                    continue;
                }

                // if layer needs to be run on the CPU, all of its inputs needs to be on the CPU as well
                foreach (var input in layer.inputs)
                {
                    if (string.IsNullOrEmpty(input))
                        continue;
                    layersOnCPU.Add(input);
                }
            }
        }
    }
}
