using UnityEngine;
using UnityEngine.Profiling;
using System.Runtime.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;
using static Unity.Sentis.ShaderPropertyID;

[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis {

struct ComputeFunc
{
    // dispatch dimension limitation coming from D3D11
    public const uint SafeDispatchLimit = 65535;

    public readonly ComputeShader shader;
    public readonly string kernelName;
    public readonly int kernelIndex;
    public readonly uint threadGroupSizeX;
    public readonly uint threadGroupSizeY;
    public readonly uint threadGroupSizeZ;
    public uint threadGroupSize { get { return threadGroupSizeX * threadGroupSizeY * threadGroupSizeZ; } }

    // ---------------------------------------------------------------------------------
    public ComputeFunc(string kn)
    {
        shader = ComputeShaderSingleton.Instance.FindComputeShader(kn);
        kernelName = kn;
        kernelIndex = ComputeShaderSingleton.Instance.GetKernelIndex(kn);
        ComputeShaderSingleton.Instance.GetKernelThreadGroupSizes(kn, out threadGroupSizeX, out threadGroupSizeY, out threadGroupSizeZ);
    }
}
} // namespace Unity.Sentis
