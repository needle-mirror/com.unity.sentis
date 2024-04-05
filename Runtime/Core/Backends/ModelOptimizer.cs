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
    static void RunPasses(ref Model model, IModelPass[] passes)
    {
        foreach (var pass in passes)
        {
            pass.Run(ref model);
        }
    }

    internal static void OptimizeModel(ref Model model)
    {
        var optimizationPasses = new IModelPass[]
        {
            new EinsumToMatMulPass(),
            new FuseConstantsPass(),
            new RemoveNoOpsPass(),
            new RemoveUnusedPass(),
            new ConcatenateTransposesPass(),
            new ContractToSimplerLayerPass(),
            new SimplifyReshapeInputPass(),
            new ContractSubExpressionPass(),
            new FuseDensePass(),
            new FuseLinearLayersPass(),
            new FuseActivationPass(),
            new RemoveDuplicatesPass(),
            new RemoveNoOpsPass(),
            // // Good to do those passes at the very end
            new RemoveUnusedPass(),
            new StopgapFixConstantOutputPass(),
            new RoundDenormalWeightsPass(),
        };

        RunPasses(ref model, optimizationPasses);
    }

    internal static void RunCPUFallbackPass(ref Model model)
    {
        var optimizationPasses = new IModelPass[]
        {
            new CPUFallbackPass(),
        };

        RunPasses(ref model, optimizationPasses);
    }
}

} // namespace Unity.Sentis
