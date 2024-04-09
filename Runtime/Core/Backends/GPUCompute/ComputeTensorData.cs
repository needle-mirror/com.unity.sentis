using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
using System;
using System.Threading;
using Unity.Collections;
using System.Threading.Tasks;

namespace Unity.Sentis {
/// <summary>
/// An interface that provides methods for converting custom tensor data to `ComputeTensorData`.
/// </summary>
public interface IConvertibleToComputeTensorData
{
    /// <summary>
    /// Implement this method to convert to `ComputeTensorData`.
    /// </summary>
    /// <param name="count"></param>
    /// <returns>Converted `ComputeTensorData`.</returns>
    ComputeTensorData ConvertToComputeTensorData(int count);
}

/// <summary>
/// Represents data storage for a `Tensor` as a compute buffer, for GPUCompute backend.
/// </summary>
public class ComputeTensorData : ITensorData
{
    bool m_DisposeBufferAfterUse;
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
        ProfilerMarkers.ComputeTensorDataNewEmpty.Begin();
        m_Count = count;

        // Minimum size of 1 to handle 0-dim tensors.
        m_Buffer = new ComputeBuffer(Math.Max(1, count), sizeof(float));

        // @TODO: consider zero initialization only for "debug" mode
        if (clearOnInit)
        {
            var empty = new NativeArray<float>(count, Allocator.Temp, NativeArrayOptions.ClearMemory);
            m_Buffer.SetData(empty);
            empty.Dispose();
        }

        m_DisposeBufferAfterUse = true;
        ProfilerMarkers.ComputeTensorDataNewEmpty.End();
    }

    /// <inheritdoc/>
    public ITensorData Clone()
    {
        var copy = new ComputeTensorData(m_Count);

        int length = m_Buffer.count;

        var fn = ComputeFuncSingleton.Instance.Get("MemCopy");
        fn.SetTensorAsBuffer(ShaderPropertyID.k_ID_Xptr, this);
        fn.SetTensorAsBuffer(ShaderPropertyID.k_ID_Optr, copy);
        fn.SetInt(ShaderPropertyID.k_ID_offsetX, 0);
        fn.SetInt(ShaderPropertyID.k_ID_offsetO, 0);
        fn.SetInt(ShaderPropertyID.k_ID_count, length);
        fn.Dispatch(ComputeHelper.IDivC(length, 4), 1, 1);

        return copy;
    }

    /// <summary>
    /// Initializes and returns an instance of `ComputeTensorData` with given data and offset.
    /// </summary>
    /// <param name="count">The number of elements.</param>
    /// <param name="array">The allocated data to use as backing data.</param>
    /// <param name="offset">The integer offset from the start of the backing array. The default value is 0.</param>
    public ComputeTensorData(int count, NativeTensorArray array, int offset = 0)
    {
        ProfilerMarkers.ComputeTensorDataNewArray.Begin();
        m_Count = count;

        // Minimum size of 1 to handle 0-dim tensors.
        m_Buffer = new ComputeBuffer(Math.Max(1, count), sizeof(float));
        if (count != 0)
            m_Buffer.SetData(array.GetNativeArrayHandle<float>(), offset, 0, count);

        m_DisposeBufferAfterUse = true;
        ProfilerMarkers.ComputeTensorDataNewArray.End();
    }

    /// <summary>
    /// Finalizes the `ComputeTensorData`.
    /// </summary>
    ~ComputeTensorData()
    {
        if (m_Buffer == null)
            return;
        if (!m_DisposeBufferAfterUse)
            return;

        D.LogWarning($"Found unreferenced, but undisposed ComputeTensorData which might lead to GPU resource leak");
    }

    /// <summary>
    /// Disposes of the `ComputeTensorData` and any associated memory.
    /// </summary>
    public void Dispose()
    {
        if (m_DisposeBufferAfterUse)
        {
            m_Buffer.Dispose();
            m_Buffer = null;
        }

        m_DisposeBufferAfterUse = false;
    }

    /// <inheritdoc/>
    public void Upload<T>(NativeArray<T> data, int srcCount, int srcOffset = 0) where T : unmanaged
    {
        var numItemToCopy = srcCount;
        var numItemAvailableInData = data.Length - srcOffset;

        Assert.IsTrue(srcOffset >= 0);
        Assert.IsTrue(numItemToCopy <= numItemAvailableInData);
        m_Buffer.SetData(data, srcOffset, 0, numItemToCopy);

        m_AsyncDownloadRequested = false;
    }

    bool m_AsyncDownloadRequested = false;
    AsyncGPUReadbackRequest m_AsyncDownloadRequest;

    /// <inheritdoc/>
    public bool IsReadbackRequestDone()
    {
        if (m_AsyncDownloadRequested)
        {
            if (m_AsyncDownloadRequest.hasError)
                m_AsyncDownloadRequested = false;
            else
                m_AsyncDownloadRequest.Update();
        }

        return m_AsyncDownloadRequest.done;
    }

    /// <inheritdoc/>
    public void ReadbackRequest(Action<bool> callback = null)
    {
        if (!SystemInfo.supportsAsyncGPUReadback)
        {
            callback?.Invoke(false);
            return;
        }

        Action<AsyncGPUReadbackRequest> task = request =>
        {
            callback?.Invoke(!request.hasError);
        };
        m_AsyncDownloadRequest = AsyncGPUReadback.Request(m_Buffer, m_Buffer.count * sizeof(float), 0 * sizeof(float), task);
        m_AsyncDownloadRequested = true;
    }

    /// <inheritdoc/>
    public async Task<bool> ReadbackRequestAsync()
    {
        var task = new TaskCompletionSource<bool>();
        Action<bool> callback = (bool success) =>
        {
            task.TrySetResult(success);
        };
        ReadbackRequest(callback);
        return await task.Task;
    }

    /// <inheritdoc/>
    public NativeArray<T> Download<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        var count = dstCount;

        Assert.IsTrue(maxCapacity >= count);
        count = Math.Min(maxCapacity, count);

        if (count == 0)
            return new NativeArray<T>(0, Allocator.Temp);

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

        if (!SystemInfo.supportsAsyncGPUReadback)
        {
            var dataArray = new T[count];
            m_Buffer.GetData(dataArray, 0, srcOffset, count);
            return new NativeArray<T>(dataArray, Allocator.Temp);
        }

        m_AsyncDownloadRequest = AsyncGPUReadback.Request(m_Buffer, dstCount * sizeof(float), srcOffset * sizeof(float));
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

        if (onDevice is IConvertibleToComputeTensorData asConvertible)
            X.AttachToDevice(asConvertible.ConvertToComputeTensorData(X.count));
        else
            X.UploadToDevice(new ComputeTensorData(X.count, clearOnInit: false)); // device is not compatible, create new array and upload

        return X.dataOnBackend as ComputeTensorData;
    }
}
} // namespace Unity.Sentis
