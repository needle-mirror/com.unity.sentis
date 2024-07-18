using System;
using System.Runtime.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using static Unity.Sentis.ShaderPropertyID;

[assembly: InternalsVisibleTo("Unity.Sentis.RuntimeTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis
{
    static class ComputeHelper
    {
        public const uint SafeDispatchLimit = 65535;

        //See https://docs.unity3d.com/2020.2/Documentation/ScriptReference/ComputeShader.SetInts.html
        //SetInts API need CPU side to be padded
        static readonly int[] s_scratchPadInt16 = new int[16 * 4];
        static readonly int[] s_scratchPadInt8 = new int[8 * 4];
        static readonly int[] s_scratchPadInt6 = new int[6 * 4];
        static readonly int[] s_scratchPadInt4 = new int[4];
        static readonly int[] s_scratchPadInt2 = new int[2];

        public static int IDivC(int v, int div)
        {
            return (v + div - 1) / div;
        }

        /// <summary>
        /// Policy to decide the (log2) width of a texture given the number of pixels required
        /// </summary>
        public static int CalculateWidthShift(int numPixels)
        {
            // aim for square, this seems to be faster than using very thin textures (width/height 1 or 2)
            // variant of base 2 bit counting from https://stackoverflow.com/questions/8970101/whats-the-quickest-way-to-compute-log2-of-an-integer-in-c
            var n = numPixels;
            var shift = 0;

            if (n > 0x7FFF)
            {
                n >>= 16;
                shift = 0x8;
            }

            if (n > 0x7F)
            {
                n >>= 8;
                shift |= 0x4;
            }

            if (n > 0x7)
            {
                n >>= 4;
                shift |= 0x2;
            }

            if (n > 0x1)
            {
                shift |= 0x1;
            }

            return shift;
        }

        public static void SetTexture(this ComputeFunction fn, int nameID, Texture tex)
        {
            fn.shader.SetTexture(fn.kernelIndex, nameID, tex);
        }
        public static void SetFloat(this ComputeFunction fn, int nameID, float data)
        {
            fn.shader.SetFloat(nameID, data);
        }
        public static void SetInt(this ComputeFunction fn, int nameID, int data)
        {
            fn.shader.SetInt(nameID, data);
        }
        public static void SetBool(this ComputeFunction fn, int nameID, bool data)
        {
            fn.shader.SetBool(nameID, data);
        }
        public static void SetVector(this ComputeFunction fn, int nameID, Vector4 data)
        {
            fn.shader.SetVector(nameID, data);
        }
        public static void EnableKeyword(this ComputeFunction fn, string keyword)
        {
            fn.shader.EnableKeyword(keyword);
        }
        public static void DisableKeyword(this ComputeFunction fn, string keyword)
        {
            fn.shader.DisableKeyword(keyword);
        }
        public static void SetTensorAsBuffer(this ComputeFunction fn, int bufferID, ComputeTensorData tensorData)
        {
            fn.shader.SetBuffer(fn.kernelIndex, bufferID, tensorData.buffer);
        }
        public static void UnrolledDispatch(this ComputeFunction fn, int numThread)
        {
            if (numThread == 0)
                return;

            int threadPerTG = (int)(fn.threadGroupSizeX * fn.threadGroupSizeY * fn.threadGroupSizeZ);
            int neededTG = ComputeHelper.IDivC(numThread, threadPerTG);
            int threadGroupZ = 1;
            int threadGroupY = ComputeHelper.IDivC(neededTG, (int)ComputeHelper.SafeDispatchLimit);
            int threadGroupX = ComputeHelper.IDivC(neededTG, threadGroupY);
            s_scratchPadInt2[0] = threadGroupX * threadPerTG;
            s_scratchPadInt2[1] = numThread;
            fn.shader.SetInts(k_ID_unrolledDispatchArgs, s_scratchPadInt2);

            int workItemsZ = (int)(threadGroupZ * fn.threadGroupSizeZ);
            int workItemsY = (int)(threadGroupY * fn.threadGroupSizeY);
            int workItemsX = (int)(threadGroupX * fn.threadGroupSizeX);
            fn.shader.Dispatch(fn.kernelIndex, threadGroupX, threadGroupY, threadGroupZ);
        }
        public static void UnrolledDispatchFast(this ComputeFunction fn, int numThread)
        {
            if (numThread == 0)
                return;

            int threadPerTG = 256 * 1 * 1;
            int neededTG = ComputeHelper.IDivC(numThread, threadPerTG);
            int threadGroupZ = 1;
            int threadGroupY = ComputeHelper.IDivC(neededTG, (int)ComputeHelper.SafeDispatchLimit);
            int threadGroupX = ComputeHelper.IDivC(neededTG, threadGroupY);
            s_scratchPadInt2[0] = threadGroupX;
            s_scratchPadInt2[1] = numThread;
            fn.shader.SetInts(k_ID_unrolledDispatchArgs, s_scratchPadInt2);

            // some GFX APIs / GPU hw/drivers have limitation of 65535 per dimension
            if (threadGroupX > ComputeHelper.SafeDispatchLimit || threadGroupY > ComputeHelper.SafeDispatchLimit || threadGroupZ > ComputeHelper.SafeDispatchLimit)
                D.LogWarning($"Exceeded safe compute dispatch group count limit per dimension [{threadGroupX}, {threadGroupY}, {threadGroupZ}] for {fn.shader.ToString()}");

            fn.shader.Dispatch(fn.kernelIndex, threadGroupX, threadGroupY, threadGroupZ);
        }
        public static void Dispatch(this ComputeFunction fn, int workItemsX, int workItemsY, int workItemsZ)
        {
#if SENTIS_DEBUG
        Profiler.BeginSample(fn.kernelName);
#endif
            var x = IDivC(workItemsX, (int)fn.threadGroupSizeX);
            var y = IDivC(workItemsY, (int)fn.threadGroupSizeY);
            var z = IDivC(workItemsZ, (int)fn.threadGroupSizeZ);

            // some GFX APIs / GPU hw/drivers have limitation of 65535 per dimension
            if (x > ComputeHelper.SafeDispatchLimit || y > ComputeHelper.SafeDispatchLimit || z > ComputeHelper.SafeDispatchLimit)
                D.LogWarning($"Exceeded safe compute dispatch group count limit per dimension [{x}, {y}, {z}] for {fn.shader.ToString()}");

            fn.shader.Dispatch(fn.kernelIndex, x, y, z);
#if SENTIS_DEBUG
        Profiler.EndSample();
#endif
        }
        public static void SetInt16(this ComputeFunction fn, int nameID, ReadOnlySpan<int> ptr)
        {
            Logger.AssertIsTrue(ptr.Length <= 16, "cannot pin array > 16, got {0}", ptr.Length);
            for (int i = 0; i < ptr.Length; i++)
                s_scratchPadInt16[4 * i] = ptr[i];

            fn.shader.SetInts(nameID, s_scratchPadInt16);
        }
        public static unsafe void SetInt8(this ComputeFunction fn, int nameID, ReadOnlySpan<int> ptr)
        {
            Logger.AssertIsTrue(ptr.Length <= 8, "cannot pin array > 8, got {0}", ptr.Length);
            for (int i = 0; i < ptr.Length; i++)
                s_scratchPadInt8[4 * i] = ptr[i];

            fn.shader.SetInts(nameID, s_scratchPadInt8);
        }
        public static unsafe void SetInt8(this ComputeFunction fn, int nameID, int* ptr, int numElements = 8)
        {
            fixed (int* dst = &s_scratchPadInt8[0])
            {
                UnsafeUtility.MemCpyStride(dst, 4 * sizeof(int), ptr, 1 * sizeof(int), sizeof(int), numElements);
            }
            fn.shader.SetInts(nameID, s_scratchPadInt8);
        }
        public static unsafe void SetInt6(this ComputeFunction fn, int nameID, int* ptr)
        {
            fixed (int* dst = &s_scratchPadInt6[0])
            {
                UnsafeUtility.MemCpyStride(dst, 4 * sizeof(int), ptr, 1 * sizeof(int), sizeof(int), 6);
            }
            fn.shader.SetInts(nameID, s_scratchPadInt6);
        }
        public static void SetInt4(this ComputeFunction fn, int nameID, Span<int> ptr)
        {
            for (int i = 0; i < ptr.Length && i < 4; i++)
                s_scratchPadInt4[i] = ptr[i];

            fn.shader.SetInts(nameID, s_scratchPadInt4);
        }
        public static unsafe void SetTensorShapeStrides(this ComputeFunction fn, int shapeNameID, int strideNameID, TensorShape shape)
        {
            var pShape = stackalloc int[TensorShape.maxRank];
            var pStrides = stackalloc int[TensorShape.maxRank];
            OpsUtils.PinTensorShapeStrides(shape, pShape, pStrides);

            fn.SetInt8(shapeNameID, pShape);
            fn.SetInt8(strideNameID, pStrides);
        }
        // SetTensorShape and/or Strides above always sets 8-sized arrays on the GPU, and these arrays are valid
        // for rank elements starting from the end of the array - ie from the highest address - ie Shape[maxRank-1],
        // up to (inclusively) Shape[maxRank - 1 - rank + 1] and all other heading elements are invalid (garbage).
        // With the *CompactedAtHead versions, dimension numbers from 0 to rank-1 can directly be used to index
        // these shape and strides arrays.
        public static unsafe void SetTensorStridesCompactedAtHead(this ComputeFunction fn, int strideNameID, TensorShape shape)
        {
            int* pStrides = stackalloc int[shape.rank];
            OpsUtils.PinTensorStridesCompact(shape, pStrides);
            fn.SetInt8(strideNameID, pStrides, numElements: shape.rank);
        }
    }
}
