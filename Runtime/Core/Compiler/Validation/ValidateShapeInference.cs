using Unity.Sentis.Compiler.Analyser;

namespace Unity.Sentis.Compiler.Validation
{
    struct ValidateShapeInference : IValidationPass
    {
        public void Run(Model model)
        {
            MemoryFootprintAnalysis.FindLayersThatRequireStorage(model);
        }
    }
}
