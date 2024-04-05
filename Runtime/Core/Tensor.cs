using UnityEngine.Assertions;
using System;
using Unity.Collections;
using System.Threading.Tasks;

namespace Unity.Sentis
{
/// <summary>
/// Represents data in a multidimensional array-like structure.
///
/// Ownership and lifetime:
/// * Disposed needs to be called on the main thread.
/// * Ownership is always to the owner of the object.
///
/// Data Representation:
/// * TensorShape represents the data layout of the tensor
/// * Data is held by a tensorData (ITensorData) which can be on a given backend
/// * Data is stored in a flattened row major format
/// * Data can be pending (ie computation is being done in parallel)
///      - call CompleteAllPendingOperations for a blocking call to finish computing the tensor's data
///      - call CompleteOperationsAndDownloadAsync/ReadbackRequest for a non blocking call to finish computing the tensor's data
///   Data can be in a non readable type (GPU/NPU)
///      - Call CompleteOperationsAndDownload to make the tensor readable (this will fetch the data on the backend and convert it to a readable format)
///      - CompleteOperationsAndDownload is a blocking call if called on its own.
///        for a non blocking call, make sure that CompleteOperationsAndDownloadAsync/ReadbackRequest have been called previously
///        you can check that with IsReadbackRequestDone
///
/// Data manipulation
/// * ToReadOnlyArray returns a copy of the tensor's data
/// * dataOnBackend can be manipulated directly to avoid a unnecessary copy
///   see ComputeTensorData/BurstTensorData for info
/// </summary>
public abstract class Tensor : IDisposable
{
    private protected ITensorData m_DataOnBackend;
    private protected TensorShape m_Shape;
    private protected bool m_Disposed = false;

    /// <summary>
    /// The data type of the elements of the tensor.
    /// </summary>
    public abstract DataType dataType { get; }

    /// <summary>
    /// The length of the tensor.
    /// </summary>
    public abstract int count { get; }

    /// <summary>
    /// The shape of the tensor, as a `TensorShape`.
    /// </summary>
    public TensorShape shape
    {
        get => m_Shape;
        internal set => m_Shape = value;
    }

    /// <summary>
    /// The device-specific internal representation of the tensor data.
    /// </summary>
    public ITensorData dataOnBackend
    {
        get => m_DataOnBackend;
        internal set => m_DataOnBackend = value;
    }

    /// <summary>
    /// The backend type where the tensor data is currently stored.
    /// </summary>
    public BackendType backendType
    {
        get => m_DataOnBackend.backendType;
    }

    internal bool disposed => m_Disposed;

    /// <summary>
    /// Dispose of the tensor and any associated memory.
    /// </summary>
    ~Tensor()
    {
        Dispose();
    }

    /// <summary>
    /// Change the shape of a tensor without changing the backing data.
    ///
    /// The new shape must be the same length as the current shape, and the data cannot be on the GPUPixel backend.
    /// </summary>
    /// <param name="shape">The new shape for the tensor.</param>
    public void Reshape(TensorShape shape)
    {
        Logger.AssertIsTrue(dataOnBackend is not TextureTensorData, "Tensor.Reshape: Sentis can only reshape when the dataOnBackend is not a TextureTensorData");
        Logger.AssertIsTrue(shape.length == m_Shape.length, "Tensor.Reshape: Sentis can only reshape when the new length is the same as the current length, got {0}, expected {1}", shape.length, m_Shape.length);
        m_Shape = shape;
    }

    private protected void PinToDevice(ITensorData onDevice, bool disposeUnpinned = true)
    {
        Logger.AssertIsTrue(onDevice?.maxCapacity >= count || onDevice == null, "Tensor.PinToDevice: not enough capacity on device to pin tensor or device null");

        if (disposeUnpinned)
            m_DataOnBackend?.Dispose();

        m_DataOnBackend = onDevice;
    }

    /// <summary>
    /// Associates a tensor with the block of data on a device. Sentis downloads from `source` on first access.
    ///
    /// Make sure `source` contains initialized and valid data that represents tensor values.
    /// </summary>
    /// <param name="source">The data on device to associate to the tensor.</param>
    public void AttachToDevice(ITensorData source)
    {
        if (m_DataOnBackend == source)
            return;

        PinToDevice(source, disposeUnpinned: true);
    }

    /// <summary>
    /// Uploads the tensor data to the destination data location on device.
    /// </summary>
    /// <param name="destination">The data on device to upload the tensor data to.</param>
    public abstract void UploadToDevice(ITensorData destination);

    /// <summary>
    /// Synchronizes the tensor data with the data on the device, then remove the tensor from the device.
    /// </summary>
    /// <param name="disposeDeviceData">Whether to free the space on device after detaching.</param>
    /// <returns>The detached tensor data.</returns>
    public ITensorData DetachFromDevice(bool disposeDeviceData = true)
    {
        ITensorData unpinned = (disposeDeviceData) ? null : m_DataOnBackend;
        PinToDevice(null, disposeDeviceData);
        return unpinned;
    }

    /// <summary>
    /// Blocking call to make tensor data read/write.
    ///
    /// Issues a blocking download of the internal data. And converts tensorData to BurstTensorData
    /// </summary>
    public void CompleteOperationsAndDownload()
    {
        m_DataOnBackend?.CompleteAllPendingOperations();
        if (m_DataOnBackend is IReadableTensorData)
            return;
        BurstTensorData.Pin(this);
    }

    /// <summary>
    /// Non blocking call to make tensor data read/write.
    ///
    /// Issues a non blocking download of the internal data. And converts tensorData to BurstTensorData
    /// </summary>
    /// <returns>The async task.</returns>
    public async Task<bool> CompleteOperationsAndDownloadAsync()
    {
        await ReadbackRequestAsync();
        CompleteOperationsAndDownload();
        return true;
    }

    /// <summary>
    /// Checks if asynchronous readback request it done.
    ///
    /// Returns true if async readback is successful.
    /// </summary>
    /// <returns>Whether the async readback request is successful.</returns>
    public bool IsReadbackRequestDone()
    {
        if (m_DataOnBackend == null)
            return false;

        return m_DataOnBackend.IsReadbackRequestDone();
    }

    /// <summary>
    /// Schedules asynchronous download of the internal data.
    /// </summary>
    /// <param name="callback">Callback invoked when async readback is finished. Return value indicates if async readback is successful.</param>
    public void ReadbackRequest(Action<bool> callback = null)
    {
        if (m_DataOnBackend == null)
        {
            callback?.Invoke(false);
            return;
        }

        m_DataOnBackend.ReadbackRequest(callback);
    }

    /// <summary>
    /// Schedules asynchronous download task of the internal data.
    ///
    /// Boolean indicates if async readback is successful.
    /// </summary>
    /// <returns>awaitable task for when the readback request is successful.</returns>
    public async Task<bool> ReadbackRequestAsync()
    {
        return await m_DataOnBackend.ReadbackRequestAsync();
    }

    /// <summary>
    /// Completes all scheduled tensor operations on device.
    /// </summary>
    public void CompleteAllPendingOperations()
    {
        m_DataOnBackend.CompleteAllPendingOperations();
    }

    /// puts Tensor in the ready for reuse state.
    internal ITensorData Invalidate()
    {
        ITensorData unpinned = m_DataOnBackend;
        PinToDevice(null, false);
        Assert.AreEqual(m_DataOnBackend, null, "Tensor.Invalidate: tensorOnDevice not null");
        m_DataOnBackend = null;
        return unpinned;
    }

    /// <summary>
    /// Disposes of the tensor and any associated memory.
    /// </summary>
    public void Dispose()
    {
        if (m_DataOnBackend != null)
        {
            m_DataOnBackend.Dispose();
        }

        m_DataOnBackend = null;
        m_Disposed = true;
    }

    /// <summary>
    /// Returns a string that represents the `Tensor`.
    /// </summary>
    /// <returns>String representation of tensor.</returns>
    public override string ToString()
    {
        return $"Tensor{dataType}{shape}";
    }

    internal T[] ToReadOnlyArray<T>() where T : unmanaged
    {
        if (shape.length == 0)
            return Array.Empty<T>();
        if (m_DataOnBackend is IReadableTensorData rwData)
            return rwData.ToArray<T>(count);
        else
            throw new InvalidOperationException("Tensor data cannot be read from, use .CompleteOperationsAndDownload() to allow reading from tensor.");
    }
    internal NativeArray<T>.ReadOnly ToReadOnlyNativeArray<T>() where T : unmanaged
    {
        if (shape.length == 0)
            return new NativeArray<T>.ReadOnly();
        if (m_DataOnBackend is IReadableTensorData rwData)
            return rwData.GetReadOnlyNativeArrayHandle<T>(count);
        else
            throw new InvalidOperationException("Tensor data cannot be read from, use .CompleteOperationsAndDownload() to allow reading from tensor.");
    }
    internal ReadOnlySpan<T> ToReadOnlySpan<T>() where T : unmanaged
    {
        if (shape.length == 0)
            return ReadOnlySpan<T>.Empty;
        if (m_DataOnBackend is IReadableTensorData rwData)
            return rwData.ToReadOnlySpan<T>(count);
        else
            throw new InvalidOperationException("Tensor data cannot be read from, use .CompleteOperationsAndDownload() to allow reading from tensor.");
    }
    internal T GetItem<T>(int d0) where T : unmanaged
    {
        if (m_DataOnBackend is IReadableTensorData rwData)
            return rwData.Get<T>(d0);
        else
            throw new InvalidOperationException("Tensor data cannot be read from, use .CompleteOperationsAndDownload() to allow reading from tensor.");
    }
    internal void SetItem<T>(int d0, T value) where T : unmanaged
    {
        if (m_DataOnBackend is IReadableTensorData rwData)
            rwData.Set<T>(d0, value);
        else
            throw new InvalidOperationException("Tensor data cannot be written to, use .CompleteOperationsAndDownload() to allow writing to tensor.");
    }
}
} // namespace Unity.Sentis
