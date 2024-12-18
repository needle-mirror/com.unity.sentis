// This is auto-generated -- do not modify directly
using UnityEngine;
using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Burst.Intrinsics;
using static Unity.Burst.Intrinsics.X86.Avx;
using static Unity.Burst.Intrinsics.X86.Avx2;
using static Unity.Burst.Intrinsics.X86.Fma;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Unity.Sentis {
partial class CPUBackend
{
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct LeakyReluJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return alpha * v + beta * abs(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct SwishJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return v / (1.0f + exp(-v));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct ReluJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return 0.5f * (v + abs(v));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct Relu6Job : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return 0.5f * (-abs(v - 6.0f) + abs(v) + 6.0f);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct TanhJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return tanh(clamp(v, -16.0f, 16.0f));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct SigmoidJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return 1.0f / (1.0f + exp(-v));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct GeluFastJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return (v * 0.5f) * (tanh(clamp((v + (pow(v, 3.0f) * 0.044714998453855515f)) * 0.7978845834732056f, -16.0f, 16.0f)) + 1);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct HardSigmoidJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return max(0.0f, min(1.0f, alpha * v + beta));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct GeluJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {

            float vv = v / sqrt(2);
            // Abramowitz/Stegun approximations
            // erf(x) = -erf(-x)
            float x = abs(vv);

            float p = 0.3275911f;
            float a1 = 0.254829592f; float a2 = -0.284496736f; float a3 = 1.421413741f;
            float a4 = -1.453152027f; float a5 = 1.061405429f;

            float t = 1.0f / (1.0f + p * x);
            float t2 = t * t;
            float t3 = t2 * t;
            float t4 = t3 * t;
            float t5 = t4 * t;

            float erf = sign(v) * (1 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * exp(-x * x));

            return (erf + 1) * v * 0.5f;

        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct ErfJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {

            // Abramowitz/Stegun approximations
            // erf(x) = -erf(-x)
            float x = abs(v);

            float p = 0.3275911f;
            float a1 = 0.254829592f; float a2 = -0.284496736f; float a3 = 1.421413741f;
            float a4 = -1.453152027f; float a5 = 1.061405429f;

            float t = 1.0f / (1.0f + p * x);
            float t2 = t * t;
            float t3 = t2 * t;
            float t4 = t3 * t;
            float t5 = t4 * t;

            return sign(v) * (1 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * exp(-x * x));

        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct CeluJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return max(0.0f, v) + min(0.0f, alpha * (exp(v / alpha) - 1.0f));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct ShrinkJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {

            float y = 0.0f;
            if (v < -beta)
                y = v + alpha;
            else if (v > beta)
                y = v - alpha;
            return y;

        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct ThresholdedReluJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {

            float y = 0.0f;
            if (v > alpha)
                y = v;
            return y;

        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct EluJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return v <= 0.0f ? alpha * (exp(v) - 1.0f) : v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct SeluJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return v <= 0.0f ? gamma * (alpha * exp(v) - alpha) : gamma * v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct SoftplusJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return log(1 + exp(-abs(v))) + max(v, 0);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct CeilJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return ceil(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct FloorJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return floor(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct RoundJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return round(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct ReciprocalJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return 1.0f / v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct ExpJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return exp(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct LogJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return log(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct SqrtJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return sqrt(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct AcosJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return acos(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct AcoshJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return log(v + sqrt(v*v - 1.0f));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct AsinJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return asin(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct AsinhJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return log(v + sqrt(v*v + 1.0f));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct AtanJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return atan(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct AtanhJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return 0.5f * log((1.0f + v)/(1.0f - v));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct CosJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return cos(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct CoshJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return 0.5f * (exp(v) + exp(-v));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct SinJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return sin(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct SinhJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return 0.5f * (exp(v) - exp(-v));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct TanJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return tan(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct SoftsignJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return v / (1.0f + abs(v));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct HardSwishJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return v * max(0, min(1, 0.16666667f * v + 0.5f));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct AbsIntJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;
            int* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    int x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public int Opperation(int v)
        {
            return math.abs(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct AbsFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return math.abs(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct NegIntJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;
            int* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    int x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public int Opperation(int v)
        {
            return -v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct NegFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return -v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct SquareIntJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;
            int* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    int x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public int Opperation(int v)
        {
            return v * v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct SquareFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return v * v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct IsNaNJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public int Opperation(float v)
        {
            return math.isnan(v) ? 1 : 0;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct CastIntToFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            int* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    int x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(int v)
        {
            return (float)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct CastFloatToIntJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public int Opperation(float v)
        {
            return (int)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct SignFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return math.sign(v);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct SignIntJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;
            int* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    int x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public int Opperation(int v)
        {
            return v == 0 ? 0 : (v > 0 ? 1 : -1);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct NotJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;
            int* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    int x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public int Opperation(int v)
        {
            return (v == 0) ? 1 : 0;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct ClipFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return math.min(beta, math.max(v, alpha));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct ClipIntJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;
            int* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    int x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public int Opperation(int v)
        {
            return math.min(betai, math.max(v, alphai));
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct ScalarMadFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;
            float* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    float x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public float Opperation(float v)
        {
            return alpha * v + beta;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct ScalarMadIntJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;
            int* Xp = Xptr + i;

            int lengthRemaining = length - i;

            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    int x = Xp[k];
                    Op[k] = Opperation(x);
                }

                Op += spanCount;
                Xp += spanCount;
            }
        }

        public int Opperation(int v)
        {
            return alphai * v + betai;
        }
    }



    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct RangeFloatJob : IParallelForBatch, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;

            int lengthRemaining = length - i;

            int index = i;
            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    Op[k] = Apply(index);
                    index++;
                }

                Op += spanCount;
            }
        }

        public float Apply(int i)
        {
            return alpha + (i * beta);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
    internal unsafe struct RangeIntJob : IParallelForBatch, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int length;
        public float alpha, beta, gamma;
        public int alphai, betai, gammai;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            int lengthRemaining = length - i;

            int index = i;
            while (count > 0)
            {
                int spanCount = math.min(count, lengthRemaining);
                count -= spanCount;

                for (int k = 0; k < spanCount; k++)
                {
                    Op[k] = Apply(index);
                    index++;
                }

                Op += spanCount;
            }
        }

        public int Apply(int i)
        {
            return alphai + (i * betai);
        }
    }

}
}
