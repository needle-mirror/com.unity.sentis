using System;
using System.Linq;
using System.Runtime.CompilerServices; // ToArray(), ToDictionary()
using Unity.Sentis.Compiler.Passes;
using Unity.Sentis.Compiler.Passes.Cleanup;
using Unity.Sentis.Compiler.Passes.Optimization;

[assembly: InternalsVisibleTo("Unity.Sentis.ONNX")]
[assembly: InternalsVisibleTo("Unity.Sentis.Editor")]

namespace Unity.Sentis
{

static class ModelOptimizer
{
    internal static Model OptimizeModel(Model model)
    {
        var optimizationPasses = new IModelPass[]
        {
            new EinsumToMatMulPass(),
            new FuseConstantsPass(),
            new RemoveNoOpsPass(),
            new RemoveUnusedPass(),
            new ContractSubExpressionPass(),
            new ConcatenateTransposesPass(),
            new ContractToSimplerLayerPass(),
            new SimplifyReshapeInputPass(),
            new FuseDensePass(),
            new FuseLinearLayersPass(),
            new FuseActivationPass(),
            new RemoveDuplicatesPass(),
            new RemoveNoOpsPass(),
            // // Good to do those passes at the very end
            new RemoveUnusedPass(),
            new RoundDenormalWeightsPass(),
            new CPUFallbackPass(),
        };

        foreach (var pass in optimizationPasses)
        {
            try
            {
                pass.Run(ref model);
            }
            catch (Exception e)
            {
                model.Warnings.Add(new Model.ImporterWarning($"Optimization Error: {pass.GetType().Name}", Model.WarningType.Error, e.Message));
                Debug.LogError(model.Warnings.Last().Message);
            }
        }

        return model;
    }
}

} // namespace Unity.Sentis
