// This is auto-generated -- do not modify directly

namespace Unity.Sentis
{
    public partial class ComputeShaderSingleton
    {
        private void RegisterGeneratedKernels()
        {
            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl.GenericA.gen", new[] {
                "Transpose",
                "InstanceNormalizationTail",
            });
            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl.PadA.gen", new[] {
                "PadBorderND",
                "PadReflectND",
                "PadSymmetricND",
                "PadEdgeND",
                "PadWrapND",
            });
            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl.PoolA.gen", new[] {
                "MaxPool2D",
                "AveragePool2D",
                "MaxPool1D",
                "AveragePool1D",
            });
            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl.Einsum.gen", new[] {
                "EinsumOne",
                "EinsumTwo",
            });
            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl.IndexingOpsA.gen", new[] {
                "Tile",
                "Gather",
                "GatherElementsFast",
                "GatherElements",
                "ScatterElementsFast",
                "ScatterElements",
                "Expand",
                "Slice",
            });
            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl.Logical.gen", new[] {
                "Where",
            });
        }
    }
}