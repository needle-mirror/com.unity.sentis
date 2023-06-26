using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace Unity.Sentis
{
    public partial class CPUOps
    {
        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
        unsafe struct AxisNormalizationTailJob : IJobParallelFor, IJobResourceDeclarationXSBWO
        {
            public float epsilon;
            public int axisDim;
            public int outerLength;
            public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
            public ReadOnlyMemResource S { get; set; } float* Sptr => (float*)S.ptr;
            public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
            public ReadOnlyMemResource W { get; set; } float* Wptr => (float*)W.ptr;
            public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

            const int k_InnerLoopLength = 32;

            [SkipLocalsInit]
            public void Execute(int outerIndex)
            {
                var Xp = Xptr + outerIndex * axisDim;
                var Sp = Sptr;
                var Bp = Bptr;
                var Op = Optr + outerIndex * axisDim;

                float mean = Wptr[outerIndex * 2 + 0];
                float variance = Wptr[outerIndex * 2 + 1];

                var it = stackalloc float[k_InnerLoopLength];

                for (var start = 0; start < axisDim; start += k_InnerLoopLength)
                {
                    var count = math.min(k_InnerLoopLength, axisDim - start);
                    int i;

                    for (i = 0; i < count; i++)
                    {
                        float scale = Sp[i];
                        float bias = Bp[i];
                        float v = Xp[i];

                        v = (v - mean) / math.sqrt(variance + epsilon);
                        v = scale * v + bias;

                        it[i] = v;
                    }

                    UnsafeUtility.MemCpy(Op, it, sizeof(float) * count);

                    Xp += count;
                    Sp += count;
                    Bp += count;
                    Op += count;
                }
            }
        }
    }
}
