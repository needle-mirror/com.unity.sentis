using UnityEngine;

namespace Unity.Sentis
{
    public partial struct GPUPrefixSum
    {
        private static class ShaderIDs
        {
            public static readonly int _InputBuffer                   = Shader.PropertyToID("_InputBuffer");
            public static readonly int _NextInputBuffer               = Shader.PropertyToID("_NextInputBuffer");
            public static readonly int _OutputBuffer                  = Shader.PropertyToID("_OutputBuffer");
            public static readonly int _InputCountsBuffer             = Shader.PropertyToID("_InputCountsBuffer");
            public static readonly int _TotalLevelsBuffer             = Shader.PropertyToID("_TotalLevelsBuffer");
            public static readonly int _OutputTotalLevelsBuffer       = Shader.PropertyToID("_OutputTotalLevelsBuffer");
            public static readonly int _OutputDispatchLevelArgsBuffer = Shader.PropertyToID("_OutputDispatchLevelArgsBuffer");
            public static readonly int _LevelsOffsetsBuffer           = Shader.PropertyToID("_LevelsOffsetsBuffer");
            public static readonly int _OutputLevelsOffsetsBuffer     = Shader.PropertyToID("_OutputLevelsOffsetsBuffer");
            public static readonly int _PrefixSumIntArgs              = Shader.PropertyToID("_PrefixSumIntArgs");
            public static readonly int _PrefixSumIntArgs2             = Shader.PropertyToID("_PrefixSumIntArgs2");
        }
    }
}
