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
    public static bool IsLayerSupportingActivationFusing(Layers.Layer layer)
    {
        return layer is Layers.FusedActivation;
    }

    internal static Model OptimizeModel(Model model)
    {
        var optimizationPasses = new IModelPass[] {
            new EinsumToMatMulPass(),
            new RemoveNoOpsPass(),
            new RemoveUnusedPass(),
            new FuseConstantsPass(),
            new ConcatenateTransposesPass(),
            new ContractToSimplerLayerPass(),
            new ContractSubExpressionPass(),
            new FuseDensePass(),
            new FuseLinearLayersPass(),
            new FuseActivationPass(),
            new RemoveDuplicatesPass(),
            // Good to do those passes at the very end
            new RemoveUnusedPass(),
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
