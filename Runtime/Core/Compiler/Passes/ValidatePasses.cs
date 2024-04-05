using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;
using Unity.Sentis.Compiler.Validation;

namespace Unity.Sentis.Compiler.Passes
{
    class ValidateBrokenLinks : IValidationPass
    {
        public void Run(Model model)
        {
            var knownInputs = new HashSet<string>();
            foreach (var i in model.inputs)
                knownInputs.Add(i.index);

            var globalOutputs = new Dictionary<string, bool>();
            foreach (var o in model.outputs)
                globalOutputs.Add(o.index, false);

            foreach (var c in model.constants)
            {
                knownInputs.Add(c.index);
                if (globalOutputs.ContainsKey(c.index))
                    globalOutputs[c.index] = true;
            }

            List<string> unconnectedLinks = new List<string>();
            foreach (var layer in model.layers)
            {
                foreach (var input in layer.inputs)
                {
                    if (!string.IsNullOrEmpty(input) && !knownInputs.Contains(input))
                    {
                        unconnectedLinks.Add(layer.index);
                        break;
                    }
                }
                knownInputs.Add(layer.index);

                if (globalOutputs.ContainsKey(layer.index))
                    globalOutputs[layer.index] = true;

                if (layer.outputs == null)
                    continue;

                foreach (var output in layer.outputs)
                {
                    if (string.IsNullOrEmpty(output))
                        continue;
                    if (globalOutputs.ContainsKey(output))
                        globalOutputs[output] = true;
                    knownInputs.Add(output);
                }
            }

            Logger.AssertAreEqual(unconnectedLinks.Count, 0, "unexpected broken links: {0}", unconnectedLinks);

            List<string> unconnectedOutput = new List<string>();
            foreach (var gO in globalOutputs)
            {
                if (!gO.Value)
                    unconnectedOutput.Add(gO.Key);
            }

            Logger.AssertAreEqual(unconnectedOutput.Count, 0, "unexpected broken links: {0}", unconnectedOutput);
        }
    }

    class ValidateUniqueOutputs : IValidationPass
    {
        public void Run(Model model)
        {
            // validate, all model outputs are unique
            // https://stackoverflow.com/questions/18547354/c-sharp-linq-find-duplicates-in-list
            var duplicateOutputs = model.outputs.GroupBy(x => x)
                .Where(g => g.Count() > 1)
                .Select(y => y.Key).ToList();
            Logger.AssertAreEqual(duplicateOutputs.Count, 0, "Output is specified more than once in the model: {0}", duplicateOutputs);
        }
    }

    class ValidateUnconectedLayers : IValidationPass
    {
        public void Run(Model model)
        {
            var globalOutputs = new Dictionary<string, bool>();
            foreach (var o in model.outputs)
                globalOutputs.Add(o.index, false);

            foreach (var c in model.constants)
            {
                if (globalOutputs.ContainsKey(c.index))
                    globalOutputs[c.index] = true;
            }

            foreach (var layer in model.layers)
            {
                if (globalOutputs.ContainsKey(layer.index))
                    globalOutputs[layer.index] = true;

                if (layer.outputs == null)
                    continue;

                foreach (var o in layer.outputs)
                {
                    if (globalOutputs.ContainsKey(o))
                        globalOutputs[o] = true;
                }
            }

            List<string> unconnectedOutput = new List<string>();
            foreach (var gO in globalOutputs)
            {
                if (!gO.Value)
                    unconnectedOutput.Add(gO.Key);
            }

            Assert.IsTrue(unconnectedOutput.Count == 0, $"unexpected broken links: {String.Join(",", unconnectedOutput)}");
        }
    }
}
