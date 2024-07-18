using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;
using System;
using Unity.Collections;
using System.Threading.Tasks;

namespace Unity.Sentis
{
    /// <summary>
    /// An interface that provides methods for converting custom tensor data to `ComputeTensorData`.
    /// </summary>
    public interface IConvertibleToComputeTensorData
    {
        /// <summary>
        /// Implement this method to convert to `ComputeTensorData`.
        /// </summary>
        /// <param name="dstCount">The number of elements.</param>
        /// <returns>Converted `ComputeTensorData`.</returns>
        ComputeTensorData ConvertToComputeTensorData(int dstCount);
    }

    /// <summary>
    /// Represents data storage for a `Tensor` as a compute buffer, for GPUCompute backend.
    /// </summary>
    public class ComputeTensorData : ITensorData, IConvertibleToBurstTensorData
    {
        bool m_IsDisposed;
        ComputeBuffer m_Buffer;
        int m_Count;

        /// <inheritdoc/>
        public int maxCapacity => m_Count;

        /// <inheritdoc/>
        public BackendType backendType => BackendType.GPUCompute;

        /// <summary>
        /// The data storage as a compute buffer.
        /// </summary>
        public ComputeBuffer buffer => m_Buffer;

        /// <summary>
        /// Initializes and returns an instance of `ComputeTensorData`, and allocates storage for a tensor with the shape of `shape`.
        /// </summary>
        /// <param name="count">The number of elements.</param>
        /// <param name="clearOnInit">Whether to zero the data on allocation. The default value is `false`.</param>
        public ComputeTensorData(int count, bool clearOnInit = false)
        {
            m_Count = count;
            m_IsDisposed = false;

            if (m_Count == 0)
                return;

            ProfilerMarkers.ComputeTensorDataNewEmpty.Begin();
            m_Buffer = new ComputeBuffer(count, sizeof(float));

            // @TODO: consider zero initialization only for "debug" mode
            if (clearOnInit)
            {
                var empty = new NativeArray<float>(count, Allocator.Temp, NativeArrayOptions.ClearMemory);
                m_Buffer.SetData(empty);
                empty.Dispose();
            }

            ProfilerMarkers.ComputeTensorDataNewEmpty.End();
        }

        /// <inheritdoc/>
        public ITensorData Clone()
        {
            var copy = new ComputeTensorData(m_Count);

            int length = m_Buffer.count;

            var fn = ComputeFunctions.k_MemCopy;
            fn.SetTensorAsBuffer(ShaderPropertyID.k_ID_Xptr, this);
            fn.SetTensorAsBuffer(ShaderPropertyID.k_ID_Optr, copy);
            fn.SetInt(ShaderPropertyID.k_ID_offsetX, 0);
            fn.SetInt(ShaderPropertyID.k_ID_offsetO, 0);
            fn.SetInt(ShaderPropertyID.k_ID_count, length);
            fn.Dispatch(ComputeHelper.IDivC(length, 4), 1, 1);

            return copy;
        }

        /// <summary>
        /// Finalizes the `ComputeTensorData`.
        /// </summary>
        ~ComputeTensorData()
        {
            if (m_Buffer == null)
                return;
            if (m_IsDisposed)
                return;

            D.LogWarning($"Found unreferenced, but undisposed ComputeTensorData which might lead to GPU resource leak");
        }

        /// <summary>
        /// Disposes of the `ComputeTensorData` and any associated memory.
        /// </summary>
        public void Dispose()
        {
            if (!m_IsDisposed)
            {
                m_Buffer?.Dispose();
                m_Buffer = null;
            }

            m_IsDisposed = true;
        }

        /// <inheritdoc/>
        public void Upload<T>(NativeArray<T> data, int srcCount) where T : unmanaged
        {
            var numItemToCopy = srcCount;
            var numItemAvailableInData = data.Length;

            Assert.IsTrue(numItemToCopy <= numItemAvailableInData);
            m_Buffer.SetData(data, 0, 0, numItemToCopy);

            m_AsyncDownloadRequested = false;
        }

        bool m_AsyncDownloadRequested = false;
        AsyncGPUReadbackRequest m_AsyncDownloadRequest;

        /// <inheritdoc/>
        public bool IsReadbackRequestDone()
        {
            return m_AsyncDownloadRequest.done;
        }

        /// <inheritdoc/>
        public void ReadbackRequest()
        {
            if (m_Count == 0)
                return;
            m_AsyncDownloadRequest = AsyncGPUReadback.Request(m_Buffer, m_Buffer.count * sizeof(float), 0 * sizeof(float));
            m_AsyncDownloadRequested = true;
        }

        #if UNITY_2023_2_OR_NEWER
        /// <inheritdoc/>
        public async Awaitable<NativeArray<T>> DownloadAsync<T>(int dstCount) where T : unmanaged
        {
            if (m_Count == 0)
                return new NativeArray<T>();
            var request = await AsyncGPUReadback.RequestAsync(m_Buffer, dstCount * sizeof(float), 0);
            return request.GetData<T>().GetSubArray(0, dstCount);
        }
        #endif

        /// <inheritdoc/>
        public BurstTensorData ConvertToBurstTensorData(int dstCount)
        {
            BurstTensorData output = new BurstTensorData(dstCount);
            if (dstCount == 0)
                return output;

            var array = output.array.GetNativeArrayHandle<float>();

            if (m_AsyncDownloadRequested)
            {
                m_AsyncDownloadRequested = false;
                if (!m_AsyncDownloadRequest.done)
                    m_AsyncDownloadRequest.WaitForCompletion();

                var reqData = m_AsyncDownloadRequest.GetData<float>();
                ProfilerMarkers.ComputeTensorDataDownload.End();
                NativeArray<float>.Copy(reqData, 0, array, 0, dstCount);
                return output;
            }

            m_AsyncDownloadRequest = AsyncGPUReadback.RequestIntoNativeArray<float>(ref array, m_Buffer, dstCount * sizeof(float), 0);
            m_AsyncDownloadRequest.WaitForCompletion();
            return output;
        }

        /// <inheritdoc/>
        public NativeArray<T> Download<T>(int dstCount) where T : unmanaged
        {
            var count = dstCount;

            Assert.IsTrue(maxCapacity >= count);
            count = Math.Min(maxCapacity, count);

            if (count == 0)
                return new NativeArray<T>();

            ProfilerMarkers.ComputeTensorDataDownload.Begin();

            if (m_AsyncDownloadRequested)
            {
                m_AsyncDownloadRequested = false;
                if (!m_AsyncDownloadRequest.done)
                    m_AsyncDownloadRequest.WaitForCompletion();

                var reqData = m_AsyncDownloadRequest.GetData<T>();
                ProfilerMarkers.ComputeTensorDataDownload.End();
                return reqData.GetSubArray(0, dstCount);
            }

            m_AsyncDownloadRequest = AsyncGPUReadback.Request(m_Buffer, dstCount * sizeof(float), 0);
            m_AsyncDownloadRequest.WaitForCompletion();

            var data = m_AsyncDownloadRequest.GetData<T>();

            ProfilerMarkers.ComputeTensorDataDownload.End();

            return data.GetSubArray(0, dstCount);
        }

        /// <inheritdoc/>
        public void CompleteAllPendingOperations()
        {
            if (m_AsyncDownloadRequested)
            {
                if (!m_AsyncDownloadRequest.done)
                    m_AsyncDownloadRequest.WaitForCompletion();
                return;
            }

            m_AsyncDownloadRequest = AsyncGPUReadback.Request(m_Buffer, m_Buffer.count * sizeof(float), 0 * sizeof(float));
            m_AsyncDownloadRequest.WaitForCompletion();
            m_AsyncDownloadRequested = false;
        }

        /// <summary>
        /// Returns a string that represents the `ComputeTensorData`.
        /// </summary>
        /// <returns>The string summary of the `ComputeTensorData`.</returns>
        public override string ToString()
        {
            return string.Format("GPU<ComputeTensorData>:{0} buffer: {1}", m_Count, m_Buffer);
        }

        /// <summary>
        /// Moves the tensor into GPU memory on the GPUCompute backend device.
        /// </summary>
        /// <param name="X">The tensor to move to the compute backend.</param>
        /// <param name="clearOnInit">Whether to zero the data on pinning. The default value is `false`.</param>
        /// <returns>The pinned `ComputeTensorData`.</returns>
        public static ComputeTensorData Pin(Tensor X, bool clearOnInit = false)
        {
            var onDevice = X.dataOnBackend;
            if (onDevice == null)
            {
                X.AttachToDevice(new ComputeTensorData(X.count, clearOnInit));
                return X.dataOnBackend as ComputeTensorData;
            }

            if (onDevice is ComputeTensorData)
                return onDevice as ComputeTensorData;

            ComputeTensorData dataOnBackend;
            if (onDevice is IConvertibleToComputeTensorData asConvertible)
            {
                dataOnBackend = asConvertible.ConvertToComputeTensorData(X.count);
            }
            else
            {
                dataOnBackend = new ComputeTensorData(X.count, clearOnInit: false);
                dataOnBackend.Upload<int>(onDevice.Download<int>(X.count), X.count);
            }
            X.AttachToDevice(dataOnBackend);

            return X.dataOnBackend as ComputeTensorData;
        }
    }
}
