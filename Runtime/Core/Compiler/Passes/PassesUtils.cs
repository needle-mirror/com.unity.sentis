using System.Collections.Generic;

namespace Unity.Sentis.Compiler.Passes
{
    static class PassesUtils
    {
        /// <summary>
        /// removes specified layers and remap inputs accordingly
        /// </summary>
        public static void RemoveAndRemap(ref Model model, HashSet<int> removeLayers, Dictionary<int, int> remap)
        {
            model.layers.RemoveAll(l => removeLayers.Contains(l.outputs[0]));
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
            for (int i = 0; i < model.outputs.Count; i++)
            {
                var output = model.outputs[i];
                if (remap.TryGetValue(output.index, out int newIndex))
                {
                    output.index = newIndex;
                    model.outputs[i] = output;
                }
            }
        }
    }
}
