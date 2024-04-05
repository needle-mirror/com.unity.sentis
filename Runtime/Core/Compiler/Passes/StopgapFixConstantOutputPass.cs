using System.Collections.Generic;
using System.Linq;
using Unity.Sentis.Layers;

namespace Unity.Sentis.Compiler.Passes.Cleanup
{
    // TODO: work out why serialization breaks with constant leading directly to output and remove this pass
    class StopgapFixConstantOutputPass : IModelPass
    {
        /// <summary>
        /// This adds an identity layer if a model output comes directly from a model constant, to fix serialization issues.
        /// </summary>
        public void Run(ref Model model)
        {
            var constants = new HashSet<string>();
            foreach (var constant in model.constants)
                constants.Add(constant.index);
            for (var i = 0; i < model.outputs.Count; i++)
            {
                var outputIndex = model.outputs[i].index;
                if (constants.Contains(outputIndex))
                {
                    var newOutput = model.GetUniqueIndex(outputIndex + "_identity");
                    model.layers.Add(new Identity(newOutput, outputIndex));
                    model.outputs[i] = new Model.Output() { name = model.outputs[i].name, index = newOutput };
                }
            }
        }
    }
}
