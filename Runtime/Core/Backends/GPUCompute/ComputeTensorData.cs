using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
using System;
using System.Threading;

namespace Unity.Sentis {
/// <summary>
/// An interface that provides methods for converting custom tensor data to `ComputeTensorData`.
/// </summary>
public interface IConvertibleToComputeTensorData
{
    /// <summary>
    /// Implement this method to convert to `ComputeTensorData`.
    /// </summary>
    ComputeTensorData ConvertToComputeTensorData(TensorShape shape);
}

/// <summary>
/// Represents data storage for a `Tensor` as a compute buffer, for GPUCompute backend.
/// </summary>
public class ComputeTensorData : ITensorData
{
    bool m_DisposeBufferAfterUse;
    ComputeBuffer m_Buffer;
    TensorShape m_Shape;

    /// <inheritdoc/>
    public int maxCapacity => m_Shape.length;

    /// <inheritdoc/>
    public DeviceType deviceType => DeviceType.GPU;

    /// <summary>
    /// The shape of the tensor using this data as a `TensorShape`.
    /// </summary>
    public TensorShape shape => m_Shape;

    /// <summary>
    /// The data storage as a compute buffer.
    /// </summary>
    public ComputeBuffer buffer => m_Buffer;

    /// <summary>
    /// Initializes and returns an instance of `ComputeTensorData`, and allocates storage for a tensor with the shape of `shape`.
    /// </summary>
    /// <param name="shape">The shape of the tensor data to allocate.</param>
    /// <param name="clearOnInit">Whether to zero the data on allocation. The default value is `true`.</param>
    public ComputeTensorData(TensorShape shape, bool clearOnInit = true)
    {
        // Minimum size of 1 to handle 0-dim tensors.
        m_Buffer = new ComputeBuffer(Math.Max(1, shape.length), sizeof(float));

        // @TODO: consider zero initialization only for "debug" mode
        if (clearOnInit)
            m_Buffer.SetData(new float[shape.length]);

        m_Shape = shape;

        m_DisposeBufferAfterUse = true;
    }

    /// <summary>
    /// Initializes and returns an instance of `ComputeTensorData` with given data and offset.
    /// </summary>
    /// <param name="shape">The shape of the tensor data.</param>
    /// <param name="array">The allocated data to use as backing data.</param>
    /// <param name="offset">The integer offset from the start of the backing array. The default value is 0.</param>
    public ComputeTensorData(TensorShape shape, NativeTensorArray array, int offset = 0)
    {
        // Minimum size of 1 to handle 0-dim tensors.
        m_Buffer = new ComputeBuffer(Math.Max(1, shape.length), sizeof(float));
        if (shape.length != 0)
            m_Buffer.SetData(array.GetNativeArrayHandle<float>(), offset, 0, shape.length);

        m_Shape = shape;

        m_DisposeBufferAfterUse = true;
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
        // It isn't safe to Release RT from a finalizer thread
        if (Thread.CurrentThread == CPUOps.MainThread)
        {
            if (m_DisposeBufferAfterUse)
            {
                m_Buffer.Dispose();
                m_Buffer = null;
            }

            m_DisposeBufferAfterUse = false;
        }
    }

    /// <inheritdoc/>
    public void Reserve(int count)
    {
        if (count > maxCapacity)
        {
            m_Buffer.Dispose();
            m_Buffer = new ComputeBuffer(count, sizeof(float));
        }
    }

    /// <inheritdoc/>
    public void Upload<T>(T[] data, int srcCount, int srcOffset = 0) where T : unmanaged
    {
        var numItemToCopy = srcCount;
        var numItemAvailableInData = data.Length - srcOffset;

        Assert.IsTrue(srcOffset >= 0);
        Assert.IsTrue(numItemToCopy <= numItemAvailableInData);
        m_Buffer.SetData(data, srcOffset, 0, numItemToCopy);

        m_AsyncDownloadRequested = false;
    }

    /// <inheritdoc/>
    public bool ScheduleAsyncDownload()
    {
        if (SystemInfo.supportsAsyncGPUReadback)
            return WaitForAsyncReadback();

        return false;
    }

    bool m_AsyncDownloadRequested = false;
    AsyncGPUReadbackRequest m_AsyncDownloadRequest;
    private bool WaitForAsyncReadback()
    {
        if (m_AsyncDownloadRequested)
        {
            if (m_AsyncDownloadRequest.hasError)
                m_AsyncDownloadRequested = false;
            else
                m_AsyncDownloadRequest.Update();
        }

        if (!m_AsyncDownloadRequested)
        {
            m_AsyncDownloadRequest = AsyncGPUReadback.Request(m_Buffer, m_Buffer.count * sizeof(float), 0 * sizeof(float));
            m_AsyncDownloadRequested = true;
        }

        return m_AsyncDownloadRequest.done;
    }

    /// <inheritdoc/>
    public T[] Download<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        var count = dstCount;

        Profiler.BeginSample("Sentis.ComputeTensorData.DownloadDataFromGPU");
        Assert.IsTrue(maxCapacity >= count);
        count = Math.Min(maxCapacity, count);

        if (m_AsyncDownloadRequested)
        {
            m_AsyncDownloadRequested = false;
            if (!m_AsyncDownloadRequest.done)
                m_AsyncDownloadRequest.WaitForCompletion();
            if (!m_AsyncDownloadRequest.hasError)
            {
                var reqData = m_AsyncDownloadRequest.GetData<T>().ToArray();
                if (reqData.Length >= count)
                { // if we have retrieved enough data
                    Profiler.EndSample();
                    return reqData;
                }
            }
        }

        var data = new T[count];
        m_Buffer.GetData(data, 0, srcOffset, count);

        Profiler.EndSample();

        return data;
    }

    /// <summary>
    /// Returns a string that represents the `ComputeTensorData`.
    /// </summary>
    public override string ToString()
    {
        return string.Format("GPU<ComputeTensorData>:{0} buffer: {1}", m_Shape, m_Buffer);
    }

    /// <summary>
    /// Moves the tensor into GPU memory on the GPUCompute backend device.
    /// </summary>
    /// <param name="uploadCache">Whether to also move the existing tensor data to the GPU. The default value is `true`.</param>
    public static ComputeTensorData Pin(Tensor X, bool uploadCache = true)
    {
        X.FlushCache(uploadCache);

        var onDevice = X.tensorOnDevice;
        if (onDevice is ComputeTensorData)
            return onDevice as ComputeTensorData;

        if (onDevice is IConvertibleToComputeTensorData asConvertible)
        {
            X.AttachToDevice(asConvertible.ConvertToComputeTensorData(X.shape));
        }
        else
        {
            if (uploadCache)
                X.UploadToDevice(new ComputeTensorData(X.shape)); // device is not compatible, create new array and upload
            else
                X.AllocateOnDevice(new ComputeTensorData(X.shape, false)); // device is not compatible, create new array but do not upload nor 0-fill
        }

        return X.tensorOnDevice as ComputeTensorData;
    }
}
} // namespace Unity.Sentis
