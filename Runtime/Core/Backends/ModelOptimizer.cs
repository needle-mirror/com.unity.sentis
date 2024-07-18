using System.Runtime.CompilerServices; // ToArray(), ToDictionary()
using Unity.Sentis.Compiler.Passes;
using Unity.Sentis.Compiler.Passes.Cleanup;
using Unity.Sentis.Compiler.Passes.Optimization;

[assembly: InternalsVisibleTo("Unity.Sentis.ONNX")]
[assembly: InternalsVisibleTo("Unity.Sentis.RuntimeTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

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
                new RemoveNoOpsPass(),
                new SimplifyReshapeInputPass(),
                new ContractSubExpressionPass(),
                new FuseDensePass(),
                new FuseLinearLayersPass(),
                new FuseActivationPass(),
                new RemoveDuplicatesPass(),
                new RemoveNoOpsPass(),
                // Good to do those passes at the very end
                new RemoveUnusedPass(),
                new RoundDenormalWeightsPass(),
            };

            RunPasses(ref model, optimizationPasses);
        }
    }
}
