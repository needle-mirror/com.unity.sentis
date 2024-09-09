using System;
using System.Threading;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Assertions;
using static Unity.Sentis.CPUBackend;

namespace Unity.Sentis
{
    /// <summary>
    /// An interface that provides methods for converting custom tensor data to `CPUTensorData`.
    /// </summary>
    interface IConvertibleToCPUTensorData
    {
        /// <summary>
        /// Implement this method to convert to `CPUTensorData`.
        /// </summary>
        /// <param name="dstCount">The number of elements.</param>
        /// <returns>Converted `CPUTensorData`.</returns>
        CPUTensorData ConvertToCPUTensorData(int dstCount);
    }

    /// <summary>
    /// An interface that provides Job system dependency fences for the memory resource.
    /// </summary>
    interface IDependableMemoryResource
    {
        /// <summary>
        /// A read fence job handle. You can use `fence` as a `dependsOn` argument when you schedule a job that reads data. The job will start when the tensor data is ready for read access.
        /// </summary>
        Unity.Jobs.JobHandle fence { get; set; }
        /// <summary>
        /// A write fence job handle. You can use `reuse` as a `dependsOn` argument when you schedule a job that reads data. The job will start when the tensor data is ready for write access.
        /// </summary>
        Unity.Jobs.JobHandle reuse { get; set; }
        /// <summary>
        /// The raw memory pointer for the resource.
        /// </summary>
        unsafe void* rawPtr { get; }
    }

    /// <summary>
    /// Represents Burst-specific internal data storage for a `Tensor`.
    /// </summary>
    public class CPUTensorData : ITensorData, IDependableMemoryResource, IConvertibleToComputeTensorData
    {
        bool m_IsDisposed;
        JobHandle m_ReadFence;
        JobHandle m_WriteFence;
        NativeTensorArray m_Array;
        int m_Count;
        bool m_SafeToDispose = true;

        /// <inheritdoc/>
        public BackendType backendType => BackendType.CPU;
        /// <inheritdoc/>
        public int maxCapacity => m_Count;
        /// <summary>
        /// The `NativeTensorArray` managed array containing the `Tensor` data.
        /// </summary>
        public NativeTensorArray array => m_Array;

        /// <inheritdoc/>
        public JobHandle fence { get { return m_ReadFence; } set { m_ReadFence = value; m_WriteFence = value; m_SafeToDispose = false; } }
        /// <inheritdoc/>
        public JobHandle reuse { get { return m_WriteFence; } set { m_WriteFence = JobHandle.CombineDependencies(value, m_WriteFence); m_SafeToDispose = false; } }

        /// <inheritdoc/>
        public unsafe void* rawPtr => m_Array.AddressAt<float>(0);

        /// <summary>
        /// Initializes and returns an instance of `CPUTensorData`, and allocates storage for a tensor with the shape of `shape`.
        /// </summary>
        /// <param name="count">The number of elements.</param>
        /// <param name="clearOnInit">Whether to zero the data on allocation. The default value is `false`.</param>
        public CPUTensorData(int count, bool clearOnInit = false)
        {
            m_IsDisposed = false;
            m_Count = count;
            if (m_Count == 0)
                return;
            m_Array = new NativeTensorArray(m_Count, clearOnInit);
        }

        /// <summary>
        /// Initializes and returns an instance of `CPUTensorData` from a `NativeTensorArray`.
        /// </summary>
        /// <param name="data">The elements of the tensor data as a `NativeTensorArray`.</param>
        public CPUTensorData(NativeTensorArray data)
        {
            m_IsDisposed = false;
            if (data == null)
            {
                m_Count = 0; m_Array = null;
                return;
            }
            m_Count = data.Length;
            m_Array = data;
        }

        /// <summary>
        /// Finalizes the `CPUTensorData`.
        /// </summary>
        ~CPUTensorData()
        {
            if (m_Array == null || m_Array is NativeTensorArrayFromManagedArray)
                return;
            if (m_IsDisposed)
                return;

            D.LogWarning($"Found unreferenced, but undisposed CPUTensorData which might lead to CPU resource leak");
        }

        /// <summary>
        /// Disposes of the `CPUTensorData` and any associated memory.
        /// </summary>
        public void Dispose()
        {
            if (!m_SafeToDispose)
                CompleteAllPendingOperations();

            if (!m_IsDisposed)
            {
                m_Array?.Dispose();
                m_Array = null;
            }

            m_IsDisposed = true;
        }

        /// <inheritdoc/>
        public void CompleteAllPendingOperations()
        {
            fence.Complete();
            reuse.Complete();
            m_SafeToDispose = true;
        }

        /// <inheritdoc/>
        public void Upload<T>(NativeArray<T> data, int srcCount) where T : unmanaged
        {
            var job = new CopyJob<T>();
            job.srcIndex = 0;
            job.dstIndex = 0;
            job.length = srcCount;
            unsafe
            {
                job.X = new ReadOnlyMemResource() { ptr = data.GetUnsafeReadOnlyPtr<T>() };
                job.O = new ReadWriteMemResource() { ptr = m_Array.RawPtr };
            }
            this.fence = job.Schedule(this.reuse);
        }

        /// <inheritdoc/>
        public NativeArray<T> Download<T>(int dstCount) where T : unmanaged
        {
            if (dstCount == 0)
                return new NativeArray<T>();

            // Download() as optimization gives direct access to the internal buffer
            // thus need to prepare internal buffer for potential writes
            CompleteAllPendingOperations();
            var dest = new NativeArray<T>(dstCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            NativeTensorArray.Copy(m_Array, 0, dest, 0, dstCount);
            return dest;
        }

        #if UNITY_2023_2_OR_NEWER
        /// <inheritdoc/>
        public async Awaitable<NativeArray<T>> DownloadAsync<T>(int dstCount) where T : unmanaged
        {
            if (dstCount == 0)
                return new NativeArray<T>();

            while (!fence.IsCompleted)
            {
                await Awaitable.NextFrameAsync();
            }
            // Download() as optimization gives direct access to the internal buffer
            // thus need to prepare internal buffer for potential writes
            CompleteAllPendingOperations();
            var dest = new NativeArray<T>(dstCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            NativeTensorArray.Copy(m_Array, 0, dest, 0, dstCount);
            return dest;
        }
        #endif

        /// <inheritdoc/>
        public ComputeTensorData ConvertToComputeTensorData(int count)
        {
            CompleteAllPendingOperations();

            var output = new ComputeTensorData(count);
            if (count == 0)
                return output;

            output.buffer.SetData(array.GetNativeArrayHandle<float>(), 0, 0, count);

            return output;
        }

        /// <inheritdoc/>
        public bool IsReadbackRequestDone()
        {
            if (!fence.IsCompleted)
                return false;
            CompleteAllPendingOperations();
            return true;
        }

        /// <inheritdoc/>
        public void ReadbackRequest() {}

        /// <summary>
        /// Returns a string that represents the `CPUTensorData`.
        /// </summary>
        /// <returns>The string summary of the `CPUTensorData`.</returns>
        public override string ToString()
        {
            return string.Format("(CPU burst: [{0}], uploaded: {1})", m_Array?.Length, m_Count);
        }

        /// <summary>
        /// Moves a tensor into memory on the CPU backend device.
        /// </summary>
        /// <param name="X">The `Tensor` to move to the CPU.</param>
        /// <param name="clearOnInit">Whether to initialize the backend data. The default value is `true`.</param>
        /// <returns>The pinned `CPUTensorData`.</returns>
        public static CPUTensorData Pin(Tensor X, bool clearOnInit = false)
        {
            var onDevice = X.dataOnBackend;
            if (onDevice == null)
            {
                X.AdoptTensorData(new CPUTensorData(X.count, clearOnInit));
                return X.dataOnBackend as CPUTensorData;
            }

            if (onDevice is CPUTensorData)
                return onDevice as CPUTensorData;
            CPUTensorData dataOnBackend;
            if (onDevice is IConvertibleToCPUTensorData asConvertible)
            {
                dataOnBackend = asConvertible.ConvertToCPUTensorData(X.count);
            }
            else
            {
                dataOnBackend = new CPUTensorData(X.count, clearOnInit: false);
                dataOnBackend.Upload<int>(onDevice.Download<int>(X.count), X.count);
            }
            X.AdoptTensorData(dataOnBackend);

            return X.dataOnBackend as CPUTensorData;
        }
    }
}
