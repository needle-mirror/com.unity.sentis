using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;

namespace Unity.Sentis
{
    static class BurstJobsQuantizeTensor
    {
        [BurstCompile(CompileSynchronously = true)]
        internal unsafe struct CastFloatToHalfJob : IJobParallelFor
        {
            [NoAlias] [NativeDisableUnsafePtrRestriction] [ReadOnly] public float* src;
            [NoAlias] [NativeDisableUnsafePtrRestriction] public ushort* dst;

            public void Execute(int index)
            {
                float v = src[index];
                dst[index] = float.IsSubnormal(v) ? (ushort)0 : Mathf.FloatToHalf(Mathf.Clamp(v, half.MinValue, half.MaxValue));
            }
        }

        [BurstCompile(CompileSynchronously = true)]
        internal unsafe struct QuantizeUint8Job : IJobParallelFor
        {
            [NoAlias] [NativeDisableUnsafePtrRestriction] [ReadOnly] public float* src;
            [NoAlias] [NativeDisableUnsafePtrRestriction] public byte* dst;
            public float scale;
            public int zeroPoint;

            public void Execute(int index)
            {
                dst[index] = (byte)Mathf.Clamp(Mathf.Round(src[index] / scale) + zeroPoint, 0, 255);
            }
        }
    }
}
