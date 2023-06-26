using UnityEngine.Assertions;
using System;
using UnityEngine;

namespace Unity.Sentis {

/// <summary>
/// Represents data in a multidimensional array-like structure.
/// </summary>
public abstract class Tensor : IDisposable
{
    protected ITensorData m_TensorOnDevice;
    protected ITensorAllocator m_TensorAllocator;
    protected TensorShape m_Shape;
    protected bool m_CacheIsDirty;
    protected bool m_Disposed = false;

    protected abstract bool isCacheNull { get; }

    /// <summary>
    /// The data type of the elements of the tensor.
    /// </summary>
    public abstract DataType dataType { get; }

    /// <summary>
    /// The shape of the tensor, as a `TensorShape`.
    /// </summary>
    public TensorShape shape
    {
        get => m_Shape;
        protected set => m_Shape = value;
    }

    /// <summary>
    /// The device-specific internal representation of the tensor data.
    /// </summary>
    public ITensorData tensorOnDevice
    {
        get => m_TensorOnDevice;
        protected set => m_TensorOnDevice = value;
    }

    /// <summary>
    /// The allocator for the tensor. Refer to <see cref="ITensorAllocator"/>.
    /// </summary>
    public ITensorAllocator allocator => m_TensorAllocator;

    internal bool disposed => m_Disposed;

    /// <summary>
    /// Create a Tensor with the specified `shape`, an `ITensorData` `data` and an ITensorAllocator `allocator`.
    /// </summary>
    protected internal Tensor(TensorShape shape, ITensorData data = null, ITensorAllocator allocator = null)
    {
        m_Shape = shape;
        tensorOnDevice = data;
        m_TensorAllocator = allocator;
        ClearCache();
    }

    /// <summary>
    /// Dispose of the tensor and any associated memory.
    /// </summary>
    ~Tensor()
    {
        Dispose();
    }

    void PinToDevice(ITensorData onDevice, bool disposeUnpinned = true)
    {
        Logger.AssertIsTrue(onDevice?.maxCapacity >= shape.length || onDevice == null, "Tensor.PinToDevice: not enough capacity on device to pin tensor or device null");

        if (m_TensorAllocator != null)
            m_TensorAllocator.MoveToDevice(this, onDevice, m_TensorOnDevice, disposeUnpinned);
        else if (disposeUnpinned)
            m_TensorOnDevice?.Dispose();

        tensorOnDevice = onDevice;
    }

    /// <summary>
    /// Upload tensor values to the device, by associating the tensor with the uninitialized block of data on the device.
    ///
    /// You should allocate `destination` on the device. `UploadToDevice` overwrites the current contents of `destination`.
    ///
    /// By default Sentis discards the local cache after you call this method. Set `invalidateCacheAfterUpload` to false to keep the cache.
    /// </summary>
    public void UploadToDevice(ITensorData destination, bool invalidateCacheAfterUpload = true)
    {
        if (m_TensorOnDevice == destination && !m_CacheIsDirty)
            return;

        PrepareCacheForAccess();
        PinToDevice(destination, disposeUnpinned: true);

        m_CacheIsDirty = true;
        if (invalidateCacheAfterUpload)
            UploadAndInvalidateCache();
        else
            UploadIfDirty();
    }

    /// <summary>
    /// Upload tensor values to the device, by associating the tensor with the uninitialized block of data on the device.
    ///
    /// You should allocate `destination` on the device. `UploadToDevice` overwrites the current contents of `destination`.
    ///
    /// Sentis doesn't copy or initialize content from the tensor, regardless of the current cache or data on the device.
    /// </summary>
    public void AllocateOnDevice(ITensorData destination)
    {
        if (m_TensorOnDevice == destination)
            return;

        PinToDevice(destination, disposeUnpinned: true);
        ClearCache();
    }

    /// <summary>
    /// Associates a tensor with the block of data on a device. Sentis downloads from `source` on first access.
    ///
    /// Make sure `source` contains initialized and valid data that represents tensor values.
    ///
    /// Refer to `PrepareCacheForAccess()` if you need to schedule the download as soon as possible.
    /// </summary>
    public void AttachToDevice(ITensorData source)
    {
        if (m_TensorOnDevice == source && !m_CacheIsDirty)
            return;

        UploadIfDirty();
        PinToDevice(source, disposeUnpinned: true);
        if (!isCacheNull)
            PrepareCacheForAccess();
    }

    /// <summary>
    /// Synchronizes the tensor cache with the data on the device, then remove the tensor from the device.
    /// </summary>
    public ITensorData DetachFromDevice(bool disposeDeviceData = true)
    {
        PrepareCacheForAccess();

        ITensorData unpinned = (disposeDeviceData) ? null : m_TensorOnDevice;
        PinToDevice(null, disposeDeviceData);
        return unpinned;
    }

    protected abstract void UploadIfDirty();

    void InvalidateCache()
    {
        // Removes the tensor cache. If the tensor isn't pinned to the device, the cache holds the only copy, so Sentis can't remove it.
        if (m_TensorOnDevice == null)
            return;

        ClearCache();
    }

    void UploadAndInvalidateCache()
    {
        UploadIfDirty();
        InvalidateCache();
    }

    /// <summary>
    /// Read data from the device and write it to the cache.
    ///
    /// The default value of `blocking` is `true`, which means this method is a blocking read.
    ///
    /// When the value of `blocking` is false, the read is non-blocking. You can keep calling the method to get the status of the asynchronous download. You can access the tensor data when the method returns `true`.
    /// </summary>
    public abstract bool PrepareCacheForAccess(bool blocking = true);

    /// <summary>
    /// Upload the tensor cache to device memory and delete the tensor cache.
    /// </summary>
    public void FlushCache(bool uploadCache)
    {
        if(uploadCache)
            UploadAndInvalidateCache();
        else
            InvalidateCache();
    }

    /// <summary>
    /// Returns a shallow copy of the `Tensor` with a new shape. The copy shares data storage with the original tensor.
    ///
    /// `newShape.length` must be equal to `this.shape.length`.
    /// </summary>
    public abstract Tensor ShallowReshape(TensorShape newShape);

    /// <summary>
    /// Returns a shallow copy of the current `Tensor`. The copy shares data storage with original tensor.
    /// </summary>
    public Tensor ShallowCopy()
    {
        return ShallowReshape(shape);
    }

    /// <summary>
    /// Returns a deep copy of the current Tensor.
    /// </summary>
    public abstract Tensor DeepCopy();

    /// <summary>
    /// Removes system references to the tensor. The caller assumes ownership.
    /// </summary>
    public void TakeOwnership()
    {
        m_TensorAllocator?.WaiveOwnership(this);
        m_TensorAllocator = null;
    }

    /// Called from ITensorAllocator, puts Tensor in the ready for reuse state.
    internal ITensorData Invalidate()
    {
        ITensorData unpinned = m_TensorOnDevice;
        PinToDevice(null, false);
        Assert.AreEqual(m_TensorOnDevice, null, "Tensor.Invalidate: tensorOnDevice not null");
        ClearCache();
        tensorOnDevice = null;
        m_TensorAllocator = null;
        return unpinned;
    }

    internal void Init(TensorShape shape, ITensorData buffer, ITensorAllocator allocator)
    {
        this.shape = shape;
        tensorOnDevice = buffer;
        m_TensorAllocator = allocator;
        m_Disposed = false;
    }

    /// <summary>
    /// Disposes of the tensor and any associated memory.
    /// </summary>
    public virtual void Dispose()
    {
        if (m_TensorAllocator != null)
        {
            m_TensorAllocator.Release(this, true);
        }
        else if (m_TensorOnDevice != null)
        {
            m_TensorOnDevice.Dispose();
        }

        ClearCache();
        tensorOnDevice = null;
        m_TensorAllocator = null;
        m_Disposed = true;
    }

    protected abstract void ClearCache();

    /// <summary>
    /// Returns a string that represents the `Tensor`.
    /// </summary>
    public override string ToString()
    {
        return $"Tensor{dataType}{shape}";
    }
}

} // namespace Unity.Sentis
