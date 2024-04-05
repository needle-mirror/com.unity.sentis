using System;
using System.Linq; // ToArray(), ToDictionary()
using Unity.Sentis.Compiler.Passes;
using Unity.Sentis.Compiler.Validation;

namespace Unity.Sentis
{

static class ModelValidator
{
    internal static Model ValidateModel(Model model)
    {
        var validationPasses = new IValidationPass[] {
            new ValidateBrokenLinks(),
            new ValidateUnconectedLayers(),
            new ValidateUniqueOutputs() };

        foreach (var pass in validationPasses)
        {
            pass.Run(model);
        }

        return model;
    }
}

} // namespace Unity.Sentis
