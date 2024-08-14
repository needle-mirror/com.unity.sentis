using UnityEngine;
using System.Runtime.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;
using static Unity.Sentis.ShaderPropertyID;
using UnityEngine.Rendering;
using System;

[assembly: InternalsVisibleTo("Unity.Sentis.RuntimeTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis
{
    static class CommandBufferHelper
    {
        //See https://docs.unity3d.com/2020.2/Documentation/ScriptReference/ComputeShader.SetInts.html
        //SetInts API need CPU side to be padded
        static readonly int[] k_ScratchPadInt16 = new int[16 * 4];
        static readonly int[] k_ScratchPadInt8 = new int[8 * 4];
        static readonly int[] k_ScratchPadInt6 = new int[6 * 4];
        static readonly int[] k_ScratchPadInt4 = new int[4];
        static readonly int[] k_ScratchPadInt2 = new int[2];

        public static void Dispatch(this CommandBuffer cb, ComputeFunction fn, int workItemsX, int workItemsY, int workItemsZ)
        {
            cb.BeginSample(fn.profilerMarker);

            var x = ComputeHelper.IDivC(workItemsX, (int)fn.threadGroupSizeX);
            var y = ComputeHelper.IDivC(workItemsY, (int)fn.threadGroupSizeY);
            var z = ComputeHelper.IDivC(workItemsZ, (int)fn.threadGroupSizeZ);

            // some GFX APIs / GPU hw/drivers have limitation of 65535 per dimension
            if (x > ComputeHelper.SafeDispatchLimit || y > ComputeHelper.SafeDispatchLimit || z > ComputeHelper.SafeDispatchLimit)
                D.LogWarning($"Exceeded safe compute dispatch group count limit per dimension [{x}, {y}, {z}] for {fn.shader.ToString()}");

            cb.DispatchCompute(fn.shader, fn.kernelIndex, x, y, z);

            cb.EndSample(fn.profilerMarker);
        }

        public static void UnrolledDispatch(this CommandBuffer cb, ComputeFunction fn, int numThread)
        {
            if (numThread == 0)
                return;

            int threadPerTG = (int)(fn.threadGroupSizeX * fn.threadGroupSizeY * fn.threadGroupSizeZ);
            int neededTG = ComputeHelper.IDivC(numThread, threadPerTG);
            int threadGroupZ = 1;
            int threadGroupY = ComputeHelper.IDivC(neededTG, (int)ComputeHelper.SafeDispatchLimit);
            int threadGroupX = ComputeHelper.IDivC(neededTG, threadGroupY);
            k_ScratchPadInt2[0] = threadGroupX * threadPerTG;
            k_ScratchPadInt2[1] = numThread;
            cb.SetComputeIntParams(fn.shader, k_ID_unrolledDispatchArgs, k_ScratchPadInt2);

            int workItemsZ = (int)(threadGroupZ * fn.threadGroupSizeZ);
            int workItemsY = (int)(threadGroupY * fn.threadGroupSizeY);
            int workItemsX = (int)(threadGroupX * fn.threadGroupSizeX);
            cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);
        }

        public static void UnrolledDispatchFast(this CommandBuffer cb, ComputeFunction fn, int numThread)
        {
            if (numThread == 0)
                return;

            cb.BeginSample(fn.profilerMarker);

            int threadPerTG = 256 * 1 * 1;
            int neededTG = ComputeHelper.IDivC(numThread, threadPerTG);
            int threadGroupZ = 1;
            int threadGroupY = ComputeHelper.IDivC(neededTG, (int)ComputeHelper.SafeDispatchLimit);
            int threadGroupX = ComputeHelper.IDivC(neededTG, threadGroupY);
            k_ScratchPadInt2[0] = threadGroupX;
            k_ScratchPadInt2[1] = numThread;
            cb.SetComputeIntParams(fn.shader, k_ID_unrolledDispatchArgs, k_ScratchPadInt2);

            // some GFX APIs / GPU hw/drivers have limitation of 65535 per dimension
            if (threadGroupX > ComputeHelper.SafeDispatchLimit || threadGroupY > ComputeHelper.SafeDispatchLimit || threadGroupZ > ComputeHelper.SafeDispatchLimit)
                D.LogWarning($"Exceeded safe compute dispatch group count limit per dimension [{threadGroupX}, {threadGroupY}, {threadGroupZ}] for {fn.shader.ToString()}");

            cb.DispatchCompute(fn.shader, fn.kernelIndex, threadGroupX, threadGroupY, threadGroupZ);

            cb.EndSample(fn.profilerMarker);
        }

        public static void SetBool(this CommandBuffer cb, ComputeFunction fn, int nameID, bool data)
        {
            cb.SetComputeIntParam(fn.shader, nameID, data ? 1 : 0);
        }

        public static unsafe void SetTensorShapeStrides(this CommandBuffer cb, ComputeFunction fn, int shapeNameID, int strideNameID, TensorShape shape)
        {
            int* pShape = stackalloc int[TensorShape.maxRank];
            int* pStrides = stackalloc int[TensorShape.maxRank];
            OpsUtils.PinTensorShapeStrides(shape, pShape, pStrides);

            cb.SetInt8(fn, shapeNameID, pShape);
            cb.SetInt8(fn, strideNameID, pStrides);
        }

        public static unsafe void SetTensorStrides(this CommandBuffer cb, ComputeFunction fn, int strideNameID, TensorShape shape)
        {
            int* pShape = stackalloc int[TensorShape.maxRank];
            int* pStrides = stackalloc int[TensorShape.maxRank];
            OpsUtils.PinTensorShapeStrides(shape, pShape, pStrides);

            cb.SetInt8(fn, strideNameID, pStrides);
        }

        // SetTensorShape and/or Strides above always sets 8-sized arrays on the GPU, and these arrays are valid
        // for rank elements starting from the end of the array - ie from the highest address - ie Shape[maxRank-1],
        // up to (inclusively) Shape[maxRank - 1 - rank + 1] and all other heading elements are invalid (garbage).
        // With the *CompactedAtHead versions, dimension numbers from 0 to rank-1 can directly be used to index
        // these shape and strides arrays.
        public static unsafe void SetTensorStridesCompactedAtHead(this CommandBuffer cb, ComputeFunction fn, int strideNameID, TensorShape shape)
        {
            int* pStrides = stackalloc int[shape.rank];
            OpsUtils.PinTensorStridesCompact(shape, pStrides);

            cb.SetInt8(fn, strideNameID, pStrides, numElements: shape.rank);
        }

        public static void SetInt16(this CommandBuffer cb, ComputeFunction fn, int nameID, ReadOnlySpan<int> ptr)
        {
            Logger.AssertIsTrue(ptr.Length <= 16, "cannot pin array > 16, got {0}", ptr.Length);
            for (int i = 0; i < ptr.Length; i++)
                k_ScratchPadInt16[4 * i] = ptr[i];

            cb.SetComputeIntParams(fn.shader, nameID, k_ScratchPadInt16);
        }

        public static void SetInt8(this CommandBuffer cb, ComputeFunction fn, int nameID, ReadOnlySpan<int> ptr)
        {
            Logger.AssertIsTrue(ptr.Length <= 8, "cannot pin array > 8, got {0}", ptr.Length);
            for (int i = 0; i < ptr.Length; i++)
                k_ScratchPadInt8[4 * i] = ptr[i];

            cb.SetComputeIntParams(fn.shader, nameID, k_ScratchPadInt8);
        }

        public static unsafe void SetInt8(this CommandBuffer cb, ComputeFunction fn, int nameID, int* ptr, int numElements = 8)
        {
            fixed (int* dst = &k_ScratchPadInt8[0])
            {
                UnsafeUtility.MemCpyStride(dst, 4 * sizeof(int), ptr, 1 * sizeof(int), sizeof(int), numElements);
            }

            cb.SetComputeIntParams(fn.shader, nameID, k_ScratchPadInt8);
        }

        public static unsafe void SetInt6(this CommandBuffer cb, ComputeFunction fn, int nameID, int* ptr)
        {
            fixed (int* dst = &k_ScratchPadInt6[0])
            {
                UnsafeUtility.MemCpyStride(dst, 4 * sizeof(int), ptr, 1 * sizeof(int), sizeof(int), 6);
            }

            cb.SetComputeIntParams(fn.shader, nameID, k_ScratchPadInt6);
        }

        public static void SetInt4(this CommandBuffer cb, ComputeFunction fn, int nameID, Span<int> ptr)
        {
            for (int i = 0; i < ptr.Length && i < 4; i++)
                k_ScratchPadInt4[i] = ptr[i];

            cb.SetComputeIntParams(fn.shader, nameID, k_ScratchPadInt4);
        }

        public static void SetTensorAsBuffer(this CommandBuffer cb, ComputeFunction fn, int bufferID, ComputeTensorData tensorData)
        {
            cb.SetComputeBufferParam(fn.shader, fn.kernelIndex, bufferID, tensorData.buffer);
        }
    }
}
