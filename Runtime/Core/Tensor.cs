using UnityEngine.Assertions;
using System;
using Unity.Collections;
using UnityEngine;

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
    ///   Data can be in a non readable type (GPU/NPU)
    ///      - Call CompleteAllPendingOperations to finish computing the tensor's data
    ///      - Call ReadbackAndClone or ReadBackAndCloneAsync to allow reading the tensor's data
    ///
    /// Data manipulation
    /// * ToReadOnlyArray returns a copy of the tensor's data
    /// * dataOnBackend can be manipulated directly to avoid a unnecessary copy
    ///   see ComputeTensorData/CPUTensorData for info
    /// </summary>
    public abstract class Tensor : IDisposable
    {
        private protected ITensorData m_DataOnBackend;
        private protected TensorShape m_Shape;
        private protected bool m_Disposed = false;
        private protected int m_Count;
        private protected DataType m_DataType;

        /// <summary>
        /// The data type of the elements of the tensor.
        /// </summary>
        public DataType dataType { get { return m_DataType; } }

        /// <summary>
        /// The length of the tensor (32 bit stride).
        /// </summary>
        public int count
        {
            get => m_Count;
            internal set => m_Count = value;
        }

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
            get {
                Logger.AssertIsTrue(m_DataOnBackend != null, "Tensor is empty and has no data on any backend");
                return m_DataOnBackend.backendType;
            }
        }

        internal bool disposed => m_Disposed;

        /// <summary>
        /// Changes the shape of a tensor without changing the backing data.
        ///
        /// The new shape must fit in the allocated backend tensor data, and the data cannot be on the GPUPixel backend.
        /// </summary>
        /// <param name="shape">The new shape for the tensor.</param>
        public abstract void Reshape(TensorShape shape);

        /// <summary>
        /// Associates a new tensor data to the tensor.
        /// </summary>
        /// <param name="tensorData">The new tensor data to associate to the tensor.</param>
        /// <param name="disposePrevious">Whether to dispose the previous tensor data.</param>
        public void AdoptTensorData(ITensorData tensorData, bool disposePrevious = true)
        {
            if (m_DataOnBackend == tensorData)
                return;

            Logger.AssertIsTrue(tensorData?.maxCapacity >= count || tensorData == null, "Tensor.AdoptTensorData: not enough capacity on device to pin tensor or device null");

            if (disposePrevious)
                m_DataOnBackend?.Dispose();

            m_DataOnBackend = tensorData;
        }

        /// <summary>
        /// Sets the tensor data to null and return the previous one.
        /// </summary>
        /// <returns>The tensor data.</returns>
        public ITensorData ReleaseTensorData()
        {
            var tensorData = m_DataOnBackend;
            m_DataOnBackend = null;
            return tensorData;
        }

        internal abstract Tensor CloneEmpty();

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
        public void ReadbackRequest()
        {
            m_DataOnBackend?.ReadbackRequest();
        }

        /// <summary>
        /// Blocking download task of the internal data.
        /// </summary>
        /// <returns>CPU copy of the tensor.</returns>
        public Tensor ReadbackAndClone()
        {
            var tensor = CloneEmpty();
            if (count == 0)
            {
                tensor.dataOnBackend = new CPUTensorData(0);
                return tensor;
            }

            var data = m_DataOnBackend.Download<int>(count);

            var cpuData = new CPUTensorData(count);
            NativeTensorArray.Copy(data, 0, cpuData.array, 0, count);

            tensor.dataOnBackend = cpuData;
            return tensor;
        }

        #if UNITY_2023_2_OR_NEWER
        /// <summary>
        /// Schedules asynchronous download task of the internal data.
        /// </summary>
        /// <returns>awaitable tensor on the cpu.</returns>
        public async Awaitable<Tensor> ReadbackAndCloneAsync()
        {
            var tensor = CloneEmpty();
            if (count == 0)
            {
                tensor.dataOnBackend = new CPUTensorData(0);
                return tensor;
            }

            var data = await m_DataOnBackend.DownloadAsync<int>(count);

            var cpuData = new CPUTensorData(count);
            NativeTensorArray.Copy(data, 0, cpuData.array, 0, count);

            tensor.dataOnBackend = cpuData;
            return tensor;
        }
        #endif

        /// <summary>
        /// Completes all scheduled tensor operations on device.
        /// </summary>
        public void CompleteAllPendingOperations()
        {
            m_DataOnBackend?.CompleteAllPendingOperations();
        }

        /// <summary>
        /// Disposes of the tensor and any associated memory.
        /// </summary>
        public void Dispose()
        {
            m_DataOnBackend?.Dispose();
            m_DataOnBackend = null;
            m_Disposed = true;
        }

        /// <summary>
        /// Returns a string that represents the `Tensor`.
        /// </summary>
        /// <returns>String representation of tensor.</returns>
        public override string ToString()
        {
            return $"{dataType}{shape}";
        }

        internal NativeArray<T>.ReadOnly AsReadOnlyNativeArray<T>() where T : unmanaged
        {
            if (count == 0)
                return new NativeArray<T>.ReadOnly();

            if (m_DataOnBackend is CPUTensorData rwData)
            {
                if (rwData.IsReadbackRequestDone())
                    return rwData.array.AsReadOnlyNativeArray<T>(shape.length);
                else
                    throw new InvalidOperationException("Tensor data is still pending, cannot read from tensor.");
            }
            else
                throw new InvalidOperationException("Tensor data cannot be read from, use .ReadbackAndClone() to allow reading from tensor.");
        }
        internal ReadOnlySpan<T> AsReadOnlySpan<T>() where T : unmanaged
        {
            if (count == 0)
                return ReadOnlySpan<T>.Empty;

            if (m_DataOnBackend is CPUTensorData rwData)
            {
                if (rwData.IsReadbackRequestDone())
                    return rwData.array.AsReadOnlySpan<T>(shape.length);
                else
                    throw new InvalidOperationException("Tensor data is still pending, cannot read from tensor.");
            }
            else
                throw new InvalidOperationException("Tensor data cannot be read from, use .ReadbackAndClone() to allow reading from tensor.");
        }
        internal T GetItem<T>(int d0) where T : unmanaged
        {
            if (m_DataOnBackend is CPUTensorData rwData)
            {
                if (rwData.IsReadbackRequestDone())
                    return rwData.array.Get<T>(d0);
                else
                    throw new InvalidOperationException("Tensor data is still pending, cannot read from tensor.");
            }
            else
                throw new InvalidOperationException("Tensor data cannot be read from, use .ReadbackAndClone() to allow reading from tensor.");
        }
        internal void SetItem<T>(int d0, T value) where T : unmanaged
        {
            if (m_DataOnBackend is CPUTensorData rwData)
            {
                if (rwData.IsReadbackRequestDone())
                    rwData.array.Set<T>(d0, value);
                else
                    throw new InvalidOperationException("Tensor data is still pending, cannot write to tensor.");
            }
            else
                throw new InvalidOperationException("Tensor data cannot be read from, use .ReadbackAndClone() to allow writting to the tensor.");
        }
    }
}
