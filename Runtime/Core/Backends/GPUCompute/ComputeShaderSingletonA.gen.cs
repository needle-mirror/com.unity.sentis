// This is auto-generated -- do not modify directly

namespace Unity.Sentis
{
    public partial class ComputeShaderSingleton
    {
        void RegisterGeneratedKernelsA()
        {
            RegisterKernels("Sentis/ComputeShaders/Compute.Shaders.Broadcast.gen", new[]
            {
                "ScalarBroadcastPowFloat", "BroadcastPowFloat", "ElementwisePowFloat",
                "ScalarBroadcastPowInt", "BroadcastPowInt", "ElementwisePowInt",
                "ScalarBroadcastAddFloat", "BroadcastAddFloat", "ElementwiseAddFloat",
                "ScalarBroadcastSubFloat", "BroadcastSubFloat", "ElementwiseSubFloat",
                "ScalarBroadcastMulFloat", "BroadcastMulFloat", "ElementwiseMulFloat",
                "ScalarBroadcastDivFloat", "BroadcastDivFloat", "ElementwiseDivFloat",
                "ScalarBroadcastMinFloat", "BroadcastMinFloat", "ElementwiseMinFloat",
                "ScalarBroadcastMaxFloat", "BroadcastMaxFloat", "ElementwiseMaxFloat",
                "ScalarBroadcastMeanFloat", "BroadcastMeanFloat", "ElementwiseMeanFloat",
                "ScalarBroadcastFModFloat", "BroadcastFModFloat", "ElementwiseFModFloat",
                "ScalarBroadcastAddInt", "BroadcastAddInt", "ElementwiseAddInt",
                "ScalarBroadcastSubInt", "BroadcastSubInt", "ElementwiseSubInt",
                "ScalarBroadcastMulInt", "BroadcastMulInt", "ElementwiseMulInt",
                "ScalarBroadcastDivInt", "BroadcastDivInt", "ElementwiseDivInt",
                "ScalarBroadcastMinInt", "BroadcastMinInt", "ElementwiseMinInt",
                "ScalarBroadcastMaxInt", "BroadcastMaxInt", "ElementwiseMaxInt",
                "ScalarBroadcastModInt", "BroadcastModInt", "ElementwiseModInt",
                "ScalarBroadcastFModInt", "BroadcastFModInt", "ElementwiseFModInt",
            });

            RegisterKernels("Sentis/ComputeShaders/Compute.Shaders.Conv.gen", new[]
            {
              "Conv2D_KxK",
              "Conv2D_1x1",
              "Conv1D_KxK",
              "Conv1D_1x1",
            });

            RegisterKernels("Sentis/ComputeShaders/Compute.Shaders.ConvTranspose.gen", new[]
            {
              "ConvTranspose2D_KxK",
            });

            RegisterKernels("Sentis/ComputeShaders/Compute.Shaders.Reduction.gen", new[]
            {
                "ReduceMaxFloat",
                "ReduceMinFloat",
                "ReduceSumFloat",
                "ReduceSumSquareFloat",
                "ReduceMeanFloat",
                "ReduceProdFloat",
                "ReduceL1Float",
                "ReduceL2Float",
                "ReduceSqrtFloat",
                "ReduceLogSumFloat",
                "ReduceMaxInt",
                "ReduceMinInt",
                "ReduceSumInt",
                "ReduceSumSquareInt",
                "ReduceProdInt",
                "ReduceL1Int",
            });
        }
    }
}
