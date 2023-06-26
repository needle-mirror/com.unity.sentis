using System;
using System.Threading;
using Unity.Jobs;
using UnityEngine.Assertions;

namespace Unity.Sentis {

/// <summary>
/// An interface that provides methods for converting custom tensor data to `BurstTensorData`.
/// </summary>
public interface IConvertibleToBurstTensorData
{
    /// <summary>
    /// Implement this method to convert to `BurstTensorData`.
    /// </summary>
    BurstTensorData ConvertToBurstTensorData(TensorShape shape);
}

/// <summary>
/// An interface that provides Job system dependency fences for the memory resource.
/// </summary>
public interface IDependableMemoryResource
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
public class BurstTensorData : ITensorData, IDependableMemoryResource, IConvertibleToComputeTensorData, IConvertibleToArrayTensorData
{
    JobHandle m_ReadFence;
    JobHandle m_WriteFence;
    bool m_SafeToDispose = true;
    NativeTensorArray m_Array;
    int m_Offset;
    int m_Count;
    TensorShape m_Shape;

    /// <summary>
    /// The shape of the tensor using this data as a `TensorShape`.
    /// </summary>
    public TensorShape shape => m_Shape;
    /// <inheritdoc/>
    public virtual DeviceType deviceType => DeviceType.CPU;
    /// <inheritdoc/>
    public int maxCapacity => m_Count;
    /// <summary>
    /// The `NativeTensorArray` managed array containing the `Tensor` data.
    /// </summary>
    public NativeTensorArray array => m_Array;
    /// <summary>
    /// The integer offset for the backing data.
    /// </summary>
    public int offset => m_Offset;
    /// <summary>
    /// The length of the tensor using this data.
    /// </summary>
    public int count => m_Count;

    /// <inheritdoc/>
    public JobHandle fence { get { return m_ReadFence; }  set { m_ReadFence = value; m_WriteFence = value; m_SafeToDispose = false; } }
    /// <inheritdoc/>
    public JobHandle reuse { get { return m_WriteFence; } set { m_WriteFence = JobHandle.CombineDependencies(value, m_WriteFence); m_SafeToDispose = false; } }

    /// <inheritdoc/>
    public unsafe void* rawPtr => m_Array.AddressAt<float>(m_Offset);

    /// <summary>
    /// Initializes and returns an instance of `BurstTensorData`, and allocates storage for a tensor with the shape of `shape`.
    /// </summary>
    /// <param name="shape">The shape of the tensor data to allocate.</param>
    /// <param name="clearOnInit">Whether to zero the data on allocation. The default value is `true`.</param>
    public BurstTensorData(TensorShape shape, bool clearOnInit = true)
    {
        m_Count = shape.length;
        m_Shape = shape;
        m_Array = new NativeTensorArray(m_Count, clearOnInit);
        m_Offset = 0;
    }

    /// <summary>
    /// Initializes and returns an instance of `BurstTensorData` from an `ArrayTensorData`.
    /// </summary>
    /// <param name="sharedArray">The `ArrayTensorData` to convert.</param>
    public BurstTensorData(ArrayTensorData sharedArray)
        : this(sharedArray.shape, sharedArray.array) { }

    /// <summary>
    /// Initializes and returns an instance of `BurstTensorData` from a `SharedArrayTensorData`.
    /// </summary>
    /// <param name="sharedArray">The `SharedArrayTensorData` to convert.</param>
    public BurstTensorData(SharedArrayTensorData sharedArray)
        : this(sharedArray.shape, sharedArray.array, sharedArray.offset) { }

    /// <summary>
    /// Initializes and returns an instance of `BurstTensorData` from a `TensorShape` and an `Array`.
    /// </summary>
    /// <param name="shape">The shape of the tensor data.</param>
    /// <param name="data">The values of the tensor data as an `Array`.</param>
    public BurstTensorData(TensorShape shape, Array data)
        : this(shape, new NativeTensorArrayFromManagedArray(data), 0) { }

    /// <summary>
    /// Initializes and returns an instance of `BurstTensorData` from a `NativeTensorArray` and an offset.
    /// </summary>
    /// <param name="shape">The shape of the tensor data.</param>
    /// <param name="data">The values of the tensor data as a `NativeTensorArray`.</param>
    /// <param name="offset">The integer offset for the backing data.</param>
    public BurstTensorData(TensorShape shape, NativeTensorArray data, int offset = 0)
    {
        m_Count = shape.length;
        m_Shape = shape;
        m_Array = data;
        m_Offset = offset;
        Logger.AssertIsTrue(m_Offset >= 0, "BurstTensorData.ValueError: negative offset {0} not supported", m_Offset);
        Logger.AssertIsTrue(m_Count >= 0, "BurstTensorData.ValueError: negative count {0} not supported", m_Count);
        Logger.AssertIsTrue(m_Offset + m_Count <= m_Array.Length, "BurstTensorData.ValueError: offset + count {0} is bigger than input buffer size {1}, copy will result in a out of bound memory access", m_Offset + m_Count, m_Array.Length);
    }

    /// <summary>
    /// Finalizes the `BurstTensorData`.
    /// </summary>
    ~BurstTensorData()
    {
        if (!m_SafeToDispose)
            D.LogWarning($"Found unreferenced, but undisposed BurstTensorData that potentially participates in an unfinished job and might lead to hazardous memory overwrites");
    }

    /// <summary>
    /// Disposes of the `BurstTensorData` and any associated memory.
    /// </summary>
    public void Dispose()
    {
        // It isn't safe to Complete jobs from a finalizer thread, so
        if (Thread.CurrentThread == CPUOps.MainThread)
            CompleteAllPendingOperations();
    }

    internal void CompleteAllPendingOperations()
    {
        fence.Complete();
        reuse.Complete();
        m_SafeToDispose = true;
    }

    /// <summary>
    /// Reserves storage for `count` elements.
    /// </summary>
    public void Reserve(int count)
    {
        if (count > maxCapacity)
        {
            // going to reallocate memory in base.Reserve()
            // thus need to finish current work
            CompleteAllPendingOperations();
        }
        if (count > maxCapacity)
        {
            m_Array = new NativeTensorArray(count);
            m_Offset = 0;
            m_Count = m_Array.Length;
        }
    }

    /// <summary>
    /// Uploads data to internal storage.
    /// </summary>
    public void Upload<T>(T[] data, int srcCount, int srcOffset = 0) where T : unmanaged
    {
        CompleteAllPendingOperations();

        var numItemToCopy = srcCount;
        var numItemAvailableInData = data.Length - srcOffset;
        Assert.IsTrue(srcOffset >= 0);
        Assert.IsTrue(numItemToCopy <= numItemAvailableInData);

        Reserve(numItemToCopy);
        NativeTensorArray.Copy(data, srcOffset, m_Array, m_Offset, numItemToCopy);
    }

    /// <summary>
    /// Returns data from internal storage.
    /// </summary>
    public T[] Download<T>(int dstCount, int srcOffset = 0) where T : unmanaged
    {
        // Download() as optimization gives direct access to the internal buffer
        // thus need to prepare internal buffer for potential writes
        CompleteAllPendingOperations();

        var downloadCount = dstCount;
        Logger.AssertIsTrue(m_Count >= downloadCount, "SharedArrayTensorData.Download.ValueError: cannot download {0} items from tensor of size {1}", downloadCount, m_Count);

        var dest = new T[downloadCount];
        NativeTensorArray.Copy(m_Array, srcOffset + m_Offset, dest, 0, downloadCount);
        return dest;
    }

    /// <summary>
    /// Returns the backing data and outputs the offset.
    /// </summary>
    /// <param name="offset">The integer offset in the backing data.</param>
    /// <returns>The internal `NativeTensorArray` backing data.</returns>
    public NativeTensorArray SharedAccess(out int offset)
    {
        // SharedAccess() by design gives direct access to the interna
        // thus need to prepare internal buffer for potential writes
        CompleteAllPendingOperations();

        offset = m_Offset;
        return m_Array;
    }

    /// <inheritdoc/>
    public ComputeTensorData ConvertToComputeTensorData(TensorShape shape)
    {
        CompleteAllPendingOperations();
        return new ComputeTensorData(shape, array, offset);
    }

    /// <inheritdoc/>
    public ArrayTensorData ConvertToArrayTensorData(TensorShape shape)
    {
        CompleteAllPendingOperations();
        return new ArrayTensorData(shape, array, offset);
    }

    /// <summary>
    /// Schedules asynchronous download of internal data.
    /// </summary>
    /// <returns>`true` if the download has finished.</returns>
    public bool ScheduleAsyncDownload()
    {
        return fence.IsCompleted;
    }

    /// <summary>
    /// Returns a string that represents the `BurstTensorData`.
    /// </summary>
    public override string ToString()
    {
        return string.Format("(CPU burst: [{0}], offset: {1} uploaded: {2})", m_Array?.Length, m_Offset, m_Count);
    }

    /// <summary>
    /// Moves a tensor into memory on the CPU backend device.
    /// </summary>
    /// <param name="X">The `Tensor` to move to the CPU.</param>
    /// <param name="uploadCache">Whether to also move the existing tensor data to the CPU. The default value is `true`.</param>
    public static BurstTensorData Pin(Tensor X, bool uploadCache = true)
    {
        X.FlushCache(uploadCache);

        var onDevice = X.tensorOnDevice;
        if (onDevice is BurstTensorData)
            return onDevice as BurstTensorData;

        if (onDevice is IConvertibleToBurstTensorData asConvertible)
        {
            X.AttachToDevice(asConvertible.ConvertToBurstTensorData(X.shape));
        }
        else
        {
            if (uploadCache)
                X.UploadToDevice(new BurstTensorData(X.shape)); // device is not compatible, create new array and upload
            else
                X.AllocateOnDevice(new BurstTensorData(X.shape, false)); // device is not compatible, create new array but do not upload nor 0-fill
        }

        return X.tensorOnDevice as BurstTensorData;
    }
}
} // namespace Unity.Sentis
