using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

namespace Unity.Sentis
{
    /// <inheritdoc/>
    public class Tensor<T> : Tensor where T : unmanaged
    {
        /// <summary>
        /// Initializes and returns a tensor with specified `shape` and a T[] array of `srcData` data.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="srcData">The data elements of the tensor.</param>
        /// <param name="dataStartIndex">The index of the first tensor element in the srcData array.</param>
        public Tensor(TensorShape shape, T[] srcData, int dataStartIndex = 0)
        {
            m_Shape = shape;
            unsafe
            {
                m_Count = ((shape.length * sizeof(T) + sizeof(int) - 1) / sizeof(int));
            }
            Logger.AssertIsTrue((srcData.Length - dataStartIndex) >= count, "RangeError: array length {0} is too small compared to shape length {1}", count, shape);
            m_DataType = AllocatorUtils.ToDataType<T>();

            var CPUTensorData = new CPUTensorData(count);
            m_DataOnBackend = CPUTensorData;
            NativeTensorArray.Copy(srcData, dataStartIndex, CPUTensorData.array, 0, srcData.Length - dataStartIndex);
        }

        /// <summary>
        /// Initializes and returns a tensor with specified `shape` and a native T array of `srcData` data.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="srcData">The data elements of the tensor.</param>
        /// <param name="dataStartIndex">The index of the first tensor element in the srcData array.</param>
        public Tensor(TensorShape shape, NativeArray<T> srcData, int dataStartIndex = 0)
        {
            m_Shape = shape;
            unsafe
            {
                m_Count = ((shape.length * sizeof(T) + sizeof(int) - 1) / sizeof(int));
            }
            Logger.AssertIsTrue((srcData.Length - dataStartIndex) >= count, "RangeError: array length {0} is too small compared to shape length {1}", count, shape);
            m_DataType = AllocatorUtils.ToDataType<T>();

            var CPUTensorData = new CPUTensorData(count);
            m_DataOnBackend = CPUTensorData;
            NativeTensorArray.Copy(srcData, dataStartIndex, CPUTensorData.array, 0, srcData.Length - dataStartIndex);
        }

        /// <summary>
        /// Initializes and returns a Tensor with the specified `shape`, an `ITensorData` `data`.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="data">The optional tensor data.</param>
        public Tensor(TensorShape shape, ITensorData data)
        {
            m_Shape = shape;
            unsafe
            {
                m_Count = ((shape.length * sizeof(T) + sizeof(int) - 1) / sizeof(int));
            }
            m_DataOnBackend = data;
            m_DataType = AllocatorUtils.ToDataType<T>();

            if (m_DataOnBackend != null)
                Logger.AssertIsTrue(m_DataOnBackend.maxCapacity >= count, "RangeError: tensordata capacity {0} is too small compared to shape length {1}", m_DataOnBackend.maxCapacity, shape);
        }

        /// <summary>
        /// Initializes and returns a tensor with the specified `shape`.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="clearOnInit">Whether to clear the tensor data to zeros.</param>
        public Tensor(TensorShape shape, bool clearOnInit = true)
        {
            m_Shape = shape;
            unsafe
            {
                m_Count = ((shape.length * sizeof(T) + sizeof(int) - 1) / sizeof(int));
            }
            m_DataOnBackend = new CPUTensorData(count, clearOnInit);
            m_DataType = AllocatorUtils.ToDataType<T>();
        }

        /// <inheritdoc/>
        public override void Reshape(TensorShape shape)
        {
            if (shape == m_Shape)
                return;
            if (dataOnBackend != null)
            {
                Logger.AssertIsTrue(dataOnBackend is not TextureTensorData, "Tensor.Reshape: Sentis can only reshape when the dataOnBackend is not a TextureTensorData");
                Logger.AssertIsTrue(shape.length <= dataOnBackend.maxCapacity, "Tensor.Reshape: Sentis can only reshape when the new length fits in the number of elements allocated on the backend, got {0}, expected {1}", shape.length, dataOnBackend.maxCapacity);
            }
            m_Shape = shape;
            unsafe
            {
                m_Count = ((shape.length * sizeof(T) + sizeof(int) - 1) / sizeof(int));
            }
        }

        /// <summary>
        /// Uploads a contiguous block of tensor data to internal storage.
        /// </summary>
        /// <param name="srcData">The data to upload.</param>
        public void Upload(T[] srcData)
        {
            unsafe
            {
                fixed (T* src = &srcData[0])
                {
                    var nativeArray = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<T>((void*)src, srcData.Length, Allocator.None);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                    NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref nativeArray, AtomicSafetyHandle.Create());
#endif
                    m_DataOnBackend.Upload<T>(nativeArray, nativeArray.Length);
                }
            }
        }

        /// <summary>
        /// Uploads a contiguous block of tensor data to internal storage.
        /// </summary>
        /// <param name="srcData">The data to upload.</param>
        public void Upload(NativeArray<T> srcData)
        {
            m_DataOnBackend.Upload<T>(srcData, srcData.Length);
        }

        internal override Tensor CloneEmpty()
        {
            return new Tensor<T>(shape: shape, data: null);
        }

        /// <summary>
        /// Blocking download task of the internal data.
        /// </summary>
        /// <returns>returns cpu copy of the tensor.</returns>
        public new Tensor<T> ReadbackAndClone()
        {
            return base.ReadbackAndClone() as Tensor<T>;
        }

        #if UNITY_2023_2_OR_NEWER
        /// <summary>
        /// Schedules asynchronous download task of the internal data.
        /// </summary>
        /// <returns>awaitable tensor on the cpu.</returns>
        public new async Awaitable<Tensor<T>> ReadbackAndCloneAsync()
        {
            return (await base.ReadbackAndCloneAsync()) as Tensor<T>;
        }
        #endif

        /// <summary>
        /// Returns the tensor element at offset `(d7, d6, d5, d4, d3, d2, d1, d0)`, which is position `d7 * stride6 + d6 * stride5 + d5 * stride4 + d4 * stride3 + d3 * stride2 + d2 * stride1 + d1 * stride0 + d0`.
        /// </summary>
        /// <param name="d7">Axis 7.</param>
        /// <param name="d6">Axis 6.</param>
        /// <param name="d5">Axis 5.</param>
        /// <param name="d4">Axis 4.</param>
        /// <param name="d3">Axis 3.</param>
        /// <param name="d2">Axis 2.</param>
        /// <param name="d1">Axis 1.</param>
        /// <param name="d0">Axis 0.</param>
        public T this[int d7, int d6, int d5, int d4, int d3, int d2, int d1, int d0]
        {
            get { return this[shape.RavelIndex(d7, d6, d5, d4, d3, d2, d1, d0)]; }
            set { this[shape.RavelIndex(d7, d6, d5, d4, d3, d2, d1, d0)] = value;}
        }

        /// <summary>
        /// Returns the tensor element at offset `(d6, d5, d4, d3, d2, d1, d0)`, which is position `d6 * stride5 + d5 * stride4 + d4 * stride3 + d3 * stride2 + d2 * stride1 + d1 * stride0 + d0`.
        /// </summary>
        /// <param name="d6">Axis 6.</param>
        /// <param name="d5">Axis 5.</param>
        /// <param name="d4">Axis 4.</param>
        /// <param name="d3">Axis 3.</param>
        /// <param name="d2">Axis 2.</param>
        /// <param name="d1">Axis 1.</param>
        /// <param name="d0">Axis 0.</param>
        public T this[int d6, int d5, int d4, int d3, int d2, int d1, int d0]
        {
            get { return this[shape.RavelIndex(d6, d5, d4, d3, d2, d1, d0)]; }
            set { this[shape.RavelIndex(d6, d5, d4, d3, d2, d1, d0)] = value;}
        }
        /// <summary>
        /// Returns the tensor element at offset `(d5, d4, d3, d2, d1, d0)`, which is position `d5 * stride4 + d4 * stride3 + d3 * stride2 + d2 * stride1 + d1 * stride0 + d0`.
        /// </summary>
        /// <param name="d5">Axis 5.</param>
        /// <param name="d4">Axis 4.</param>
        /// <param name="d3">Axis 3.</param>
        /// <param name="d2">Axis 2.</param>
        /// <param name="d1">Axis 1.</param>
        /// <param name="d0">Axis 0.</param>
        public T this[int d5, int d4, int d3, int d2, int d1, int d0]
        {
            get { return this[shape.RavelIndex(d5, d4, d3, d2, d1, d0)]; }
            set { this[shape.RavelIndex(d5, d4, d3, d2, d1, d0)] = value;}
        }
        /// <summary>
        /// Returns the tensor element at offset `(d4, d3, d2, d1, d0)`, which is position `d4 * stride3 + d3 * stride2 + d2 * stride1 + d1 * stride0 + d0`.
        /// </summary>
        /// <param name="d4">Axis 4.</param>
        /// <param name="d3">Axis 3.</param>
        /// <param name="d2">Axis 2.</param>
        /// <param name="d1">Axis 1.</param>
        /// <param name="d0">Axis 0.</param>
        public T this[int d4, int d3, int d2, int d1, int d0]
        {
            get { return this[shape.RavelIndex(d4, d3, d2, d1, d0)]; }
            set { this[shape.RavelIndex(d4, d3, d2, d1, d0)] = value;}
        }
        /// <summary>
        /// Returns the tensor element at offset `(d3, d2, d1, d0)`, which is position `d3 * stride2 + d2 * stride1 + d1 * stride0 + d0` in this tensor.
        /// </summary>
        /// <param name="d3">Axis 3.</param>
        /// <param name="d2">Axis 2.</param>
        /// <param name="d1">Axis 1.</param>
        /// <param name="d0">Axis 0.</param>
        public T this[int d3, int d2, int d1, int d0]
        {
            get { return this[shape.RavelIndex(d3, d2, d1, d0)]; }
            set { this[shape.RavelIndex(d3, d2, d1, d0)] = value;}
        }
        /// <summary>
        /// Returns the tensor element at offset `(d2, d1, d0)`, which is position `d2 * stride1 + d1 * stride0 + d0`.
        /// </summary>
        /// <param name="d2">Axis 2.</param>
        /// <param name="d1">Axis 1.</param>
        /// <param name="d0">Axis 0.</param>
        public T this[int d2, int d1, int d0]
        {
            get { return this[shape.RavelIndex(d2, d1, d0)]; }
            set { this[shape.RavelIndex(d2, d1, d0)] = value;}
        }
        /// <summary>
        /// Returns the tensor element at offset `(d1, d0)`, which is position `d1 * stride0 + d0`.
        /// </summary>
        /// <param name="d1">Axis 1.</param>
        /// <param name="d0">Axis 0.</param>
        public T this[int d1, int d0]
        {
            get { return this[shape.RavelIndex(d1, d0)]; }
            set { this[shape.RavelIndex(d1, d0)] = value;}
        }

        /// <summary>
        /// Returns the tensor element at offset `d0`.
        /// </summary>
        /// <param name="d0">Axis 0.</param>
        public T this[int d0]
        {
            get { return base.GetItem<T>(d0); }
            set { base.SetItem<T>(d0, value); }
        }

        /// <summary>
        /// Blocking Call to return a copy of linear memory representation of the data in this tensor.
        ///
        /// the returned array is a deepcopy of the tensor, the caller of this methods is now responsible for it.
        /// If you modify the contents of the returned array, it will not modify the underlying tensor
        /// </summary>
        /// <returns>T array copy of tensor data.</returns>
        public T[] DownloadToArray()
        {
            if (count == 0)
                return Array.Empty<T>();
            return m_DataOnBackend.Download<T>(shape.length).ToArray();
        }

        /// <summary>
        /// Blocking Call to return a copy of linear memory representation of the data in this tensor.
        ///
        /// the returned native array is a deepcopy of the tensor, the caller of this methods is now responsible for it.
        /// If you modify the contents of the returned native array, it will not modify the underlying tensor
        /// </summary>
        /// <returns>T native array copy of tensor data.</returns>
        public NativeArray<T> DownloadToNativeArray()
        {
            return m_DataOnBackend.Download<T>(shape.length);
        }

        /// <summary>
        /// Exposes the linear memory data representation of this tensor as a readonly NativeArray.
        /// </summary>
        /// <returns>NativeArray of tensor data.</returns>
        public NativeArray<T>.ReadOnly AsReadOnlyNativeArray()
        {
            return base.AsReadOnlyNativeArray<T>();
        }

        /// <summary>
        /// Exposes the linear memory data representation of this tensor as a ReadOnlySpan.
        /// </summary>
        /// <returns>Span of tensor data.</returns>
        public ReadOnlySpan<T> AsReadOnlySpan()
        {
            return base.AsReadOnlySpan<T>();
        }
    }
}
