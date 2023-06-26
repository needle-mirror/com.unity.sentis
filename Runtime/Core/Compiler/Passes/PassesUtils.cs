using System.Collections.Generic;

namespace Unity.Sentis.Compiler.Passes
{
static class PassesUtils
{
    /// <summary>
    /// removes specified layers and remap inputs accordingly
    /// </summary>
    public static void RemoveAndRemap(ref Model model, HashSet<string> removeLayers, Dictionary<string, string> remap)
    {
        model.layers.RemoveAll(l => removeLayers.Contains(l.name));
        for (int l = 0; l < model.layers.Count; ++l)
        {
            Layers.Layer layer = model.layers[l];
            for (int i = 0; i < layer.inputs.Length; i++)
            {
                var input = layer.inputs[i];
                if (remap.ContainsKey(input) && layer.name != remap[input])
                    model.layers[l].inputs[i] = remap[input];
            }
        }
    }

    /// <summary>
    /// returns an Tensor given input name if found in model constants, otherwise null
    /// </summary>
    public static Tensor GetConstantInputAsTensor(Model model, string name)
    {
        foreach (var constant in model.constants)
        {
            if (constant.name == name)
                return constant.DataSetToTensor();
        }

        return null;
    }
}
} // namespace Unity.Sentis
