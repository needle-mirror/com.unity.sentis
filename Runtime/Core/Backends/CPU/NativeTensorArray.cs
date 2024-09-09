using System;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace Unity.Sentis
{
    ///see https://referencesource.microsoft.com/#mscorlib/system/runtime/interopservices/safehandle.cs
    class NativeMemorySafeHandle : SafeHandle
    {
        readonly Allocator m_AllocatorLabel;
        const int k_Alignment = sizeof(float);

        [ReliabilityContract(Consistency.WillNotCorruptState, Cer.MayFail)]
        public unsafe NativeMemorySafeHandle(long size, bool clearOnInit, Allocator allocator) : base(IntPtr.Zero, true)
        {
            m_AllocatorLabel = allocator;
            SetHandle((IntPtr)UnsafeUtility.Malloc(size, k_Alignment, allocator));
            if (clearOnInit)
                UnsafeUtility.MemClear((void*)handle, size);
        }

        public override bool IsInvalid {
            get { return handle == IntPtr.Zero; }
        }

        [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
        protected override unsafe bool ReleaseHandle()
        {
            UnsafeUtility.Free((void*)handle, m_AllocatorLabel);
            return true;
        }
    }

    class PinnedMemorySafeHandle : SafeHandle
    {
        private readonly GCHandle m_GCHandle;

        [ReliabilityContract(Consistency.WillNotCorruptState, Cer.MayFail)]
        public PinnedMemorySafeHandle(object managedObject) : base(IntPtr.Zero, true)
        {
            m_GCHandle = GCHandle.Alloc(managedObject, GCHandleType.Pinned);
            IntPtr pinnedPtr = m_GCHandle.AddrOfPinnedObject();
            SetHandle(pinnedPtr);
        }

        public override bool IsInvalid {
            get { return handle == IntPtr.Zero; }
        }

        [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
        protected override bool ReleaseHandle()
        {
            m_GCHandle.Free();
            return true;
        }
    }

    /// <summary>
    /// Options for the data type of a `Tensor`.
    /// </summary>
    public enum DataType
    {
        /// <summary>
        /// Use 32-bit floating point data.
        /// </summary>
        Float,
        /// <summary>
        /// Use 32-bit signed integer data.
        /// </summary>
        Int,
        /// <summary>
        /// Use 16-bit signed integer data - padded on a int32 buffer.
        /// </summary>
        Short,
        /// <summary>
        /// Use 8-bit signed integer data - padded on a int32 buffer.
        /// </summary>
        Byte,
        /// <summary>
        /// Use raw n-bit field data - padded on a int32 buffer.
        /// </summary>
        Custom
    }

    /// <summary>
    /// Represents an area of managed memory that's exposed as if it's native memory.
    /// </summary>
    public class NativeTensorArrayFromManagedArray : NativeTensorArray
    {
        readonly int m_PinnedMemoryByteOffset;

        /// <summary>
        /// Initializes and returns an instance of `NativeTensorArrayFromManagedArray` from an `Array` and a integer offset and count.
        /// </summary>
        /// <param name="srcData">The data for the `Tensor` as an `Array`.</param>
        /// <param name="srcOffset">The integer offset to use for the backing data.</param>
        /// <param name="numDestElement">The integer count to use for the backing data.</param>
        public NativeTensorArrayFromManagedArray(byte[] srcData, int srcOffset, int numDestElement)
            : this(srcData, srcOffset, sizeof(byte), numDestElement) { }

        /// <summary>
        /// Initializes and returns an instance of `NativeTensorArrayFromManagedArray` from an `Array` and a integer offset, size and count.
        /// </summary>
        /// <param name="srcData">The data for the `Tensor` as an `Array`.</param>
        /// <param name="srcElementOffset">The integer offset to use for the backing data.</param>
        /// <param name="srcElementSize">The integer size to use for the backing data in bytes.</param>
        /// <param name="numDestElement">The integer count to use for the backing data.</param>
        public NativeTensorArrayFromManagedArray(Array srcData, int srcElementOffset, int srcElementSize, int numDestElement)
            : base(new PinnedMemorySafeHandle(srcData), numDestElement)
        {
            m_PinnedMemoryByteOffset = srcElementSize * srcElementOffset;

            //Safety checks
            int srcLengthInByte = (srcData.Length - srcElementOffset) * srcElementSize;
            int dstLengthInByte = numDestElement * k_DataItemSize;
            if (srcElementOffset > srcData.Length)
                throw new ArgumentOutOfRangeException(nameof(srcElementOffset), "SrcElementOffset must be <= srcData.Length");
            if (dstLengthInByte > srcLengthInByte)
                throw new ArgumentOutOfRangeException(nameof(numDestElement), "NumDestElement too big for srcData and srcElementOffset");
            var neededSrcPaddedLengthInByte = numDestElement * k_DataItemSize;
            if (srcLengthInByte < neededSrcPaddedLengthInByte)
                throw new InvalidOperationException($"The NativeTensorArrayFromManagedArray source ptr (including offset) is to small to account for extra padding.");
        }

        /// <inheritdoc/>
        public override unsafe void* RawPtr => (byte*)base.RawPtr + m_PinnedMemoryByteOffset;

        /// <summary>
        /// Disposes of the array and any associated memory.
        /// </summary>
        public override void Dispose() {}
    }

    /// <summary>
    /// Represents an area of native memory that's exposed to managed code.
    /// </summary>
    public class NativeTensorArray : IDisposable
    {
        private protected readonly SafeHandle m_SafeHandle;
        readonly Allocator m_Allocator;
        readonly int m_Length;

        /// <summary>
        /// Size in bytes of an individual element.
        /// </summary>
        public const int k_DataItemSize = sizeof(float);

        /// <summary>
        /// Initializes and returns an instance of `NativeTensorArray` with a preexisting handle.
        /// </summary>
        /// <param name="safeHandle">The safe handle to the data.</param>
        /// <param name="dataLength">The integer number of elements.</param>
        protected NativeTensorArray(SafeHandle safeHandle, int dataLength)
        {
            m_Length = dataLength;
            m_SafeHandle = safeHandle;
            m_Allocator = Allocator.Persistent;
        }

        /// <summary>
        /// Initializes and returns an instance of `NativeTensorArray` with a given length.
        /// </summary>
        /// <param name="length">The integer number of elements to allocate.</param>
        /// <param name="clearOnInit">Whether to zero the data after allocating.</param>
        /// <param name="allocator">The allocation type to use as an `Allocator`.</param>
        public NativeTensorArray(int length, bool clearOnInit = false, Allocator allocator = Allocator.Persistent)
        {
            if (!UnsafeUtility.IsValidAllocator(allocator))
                throw new InvalidOperationException("The NativeTensorArray should use a valid allocator.");
            if (length <= 0)
                throw new ArgumentOutOfRangeException(nameof (length), "Length must be > 0");

            m_Length = length;
            m_SafeHandle = new NativeMemorySafeHandle(m_Length * k_DataItemSize, clearOnInit, allocator);
            m_Allocator = allocator;
        }

        /// <summary>
        /// Clears the allocated memory to zero.
        /// </summary>
        public unsafe void ZeroMemory()
        {
            var numByteToClear = m_Length * k_DataItemSize;
            UnsafeUtility.MemClear(RawPtr, numByteToClear);
        }

        /// <summary>
        /// Disposes of the array and any associated memory.
        /// </summary>
        public virtual void Dispose()
        {
            m_SafeHandle.Dispose();
        }

        /// <summary>
        /// The number of allocated elements.
        /// </summary>
        public int Length => m_Length;

        /// <summary>
        /// The raw pointer of the backing data.
        /// </summary>
        public virtual unsafe void* RawPtr
        {
            get
            {
                if (Disposed)
                    throw new InvalidOperationException("The NativeTensorArray was disposed.");
                return (void*)m_SafeHandle.DangerousGetHandle();
            }
        }

        /// <summary>
        /// Whether the backing data is disposed.
        /// </summary>
        public bool Disposed => m_SafeHandle.IsClosed;

        /// <summary>
        /// Returns the raw pointer of the backing data at a given index.
        /// </summary>
        /// <param name="index">The index of the element.</param>
        /// <typeparam name="T">The type of the element.</typeparam>
        /// <returns>The raw pointer to the element in the data.</returns>
        public unsafe T* AddressAt<T>(long index) where T : unmanaged
        {
            return ((T*)RawPtr) + index;
        }

        /// <summary>
        /// Returns the value of the backing data at a given index.
        /// </summary>
        /// <param name="index">The index of the element.</param>
        /// <typeparam name="T">The type of the element.</typeparam>
        /// <returns>The value of the element in the data.</returns>
        public unsafe T Get<T>(int index) where T : unmanaged
        {
            return UnsafeUtility.ReadArrayElement<T>(RawPtr, index);
        }

        /// <summary>
        /// Sets the value of the backing data at a given index.
        /// </summary>
        /// <param name="index">The index of the element.</param>
        /// <param name="value">The value to set at the index.</param>
        /// <typeparam name="T">The type of the element.</typeparam>
        public unsafe void Set<T>(int index, T value) where T : unmanaged
        {
            UnsafeUtility.WriteArrayElement<T>(RawPtr, index, value);
        }

        /// <summary>
        /// Returns the data converted to a `NativeArray`.
        /// </summary>
        /// <typeparam name="T">The type of the data.</typeparam>
        /// <returns>The converted native array from data.</returns>
        public NativeArray<T> GetNativeArrayHandle<T>() where T : unmanaged
        {
            unsafe
            {
                NativeArray<T> nativeArray = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<T>(RawPtr, m_Length, m_Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref nativeArray, AtomicSafetyHandle.Create());
#endif
                return nativeArray;
            }
        }

        /// <summary>
        /// Returns the data as a `NativeArray` constrained to read only operations.
        /// </summary>
        /// <param name="dstCount">The number of elements.</param>
        /// <param name="srcOffset">The index of the first element.</param>
        /// <typeparam name="T">The type of the data.</typeparam>
        /// <returns>The read only native array of the data.</returns>
        public NativeArray<T>.ReadOnly AsReadOnlyNativeArray<T>(int dstCount, int srcOffset = 0) where T : unmanaged
        {
            unsafe
            {
                NativeArray<T> nativeArray = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<T>((byte*)RawPtr + srcOffset * sizeof(T), dstCount, m_Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref nativeArray, AtomicSafetyHandle.Create());
#endif
                return nativeArray.AsReadOnly();
            }
        }

        /// <summary>
        /// Returns the data as a `ReadOnlySpan`.
        /// </summary>
        /// <param name="dstCount">The number of elements.</param>
        /// <param name="srcOffset">The index of the first element.</param>
        /// <typeparam name="T">The type of the data.</typeparam>
        /// <returns>The span of the data.</returns>
        public ReadOnlySpan<T> AsReadOnlySpan<T>(int dstCount, int srcOffset = 0) where T : unmanaged
        {
            unsafe
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                AtomicSafetyHandle.CheckReadAndThrow(AtomicSafetyHandle.Create());
#endif

                return new ReadOnlySpan<T>((byte*)RawPtr + srcOffset * sizeof(T), dstCount);
            }
        }

        /// <summary>
        /// Returns the data as an array.
        /// </summary>
        /// <param name="dstCount">The number of elements.</param>
        /// <param name="srcOffset">The index of the first element.</param>
        /// <typeparam name="T">The type of the data.</typeparam>
        /// <returns>The copied array of the data.</returns>
        public T[] ToArray<T>(int dstCount, int srcOffset = 0) where T : unmanaged
        {
            var array = new T[dstCount];
            Copy(this, srcOffset, array, 0, dstCount);
            return array;
        }

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        static void CheckCopyArguments(int srcLength, int srcIndex, int dstLength, int dstIndex, int length)
        {
            // all dims in byte
            if (length < 0)
                throw new ArgumentOutOfRangeException("length must be equal or greater than zero.");

            if (srcIndex < 0 || srcIndex > srcLength || (srcIndex == srcLength && srcLength > 0))
                throw new ArgumentOutOfRangeException("srcIndex is outside the range of valid indexes for the source buffer.");

            if (dstIndex < 0 || dstIndex > dstLength || (dstIndex == dstLength && dstLength > 0))
                throw new ArgumentOutOfRangeException("dstIndex is outside the range of valid indexes for the destination buffer.");

            if (srcIndex + length > srcLength)
                throw new ArgumentException("length is greater than the number of elements from srcIndex to the end of the source buffer.");

            if (srcIndex + length < 0)
                throw new ArgumentException("srcIndex + length causes an integer overflow");

            if (dstIndex + length > dstLength)
                throw new ArgumentException("length is greater than the number of elements from dstIndex to the end of the destination buffer.");

            if (dstIndex + length < 0)
                throw new ArgumentException("dstIndex + length causes an integer overflow");
        }
#endif

        /// <summary>
        /// Copies the data from a source `NativeTensorArray` to a destination `NativeTensorArray` up to a given length starting from given indexes.
        /// </summary>
        /// <param name="src">The array to copy from.</param>
        /// <param name="srcIndex">The index of the first element to copy from.</param>
        /// <param name="dst">The array to copy to.</param>
        /// <param name="dstIndex">The index of the first element to copy to.</param>
        /// <param name="length">The number of elements.</param>
        /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
        public static unsafe void Copy(NativeTensorArray src, int srcIndex, NativeTensorArray dst, int dstIndex, int length)
        {
            if (length == 0)
                return;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            CheckCopyArguments(src.Length, srcIndex, dst.Length, dstIndex, length);
#endif

            void* srcPtr = src.RawPtr;
            void* dstPtr = dst.RawPtr;
            UnsafeUtility.MemCpy((byte*)dstPtr + dstIndex * k_DataItemSize,
                                 (byte*)srcPtr + srcIndex * k_DataItemSize,
                                 length * k_DataItemSize);
        }

        /// <summary>
        /// Copies the data from a source `NativeTensorArray` to a destination array up to a given length starting from given indexes.
        /// </summary>
        /// <param name="src">The array to copy from.</param>
        /// <param name="srcIndex">The index of the first element to copy from.</param>
        /// <param name="dst">The array to copy to.</param>
        /// <param name="dstIndex">The index of the first element to copy to.</param>
        /// <param name="length">The number of elements.</param>
        /// <typeparam name="T">The data type of the elements.</typeparam>
        /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
        public static unsafe void Copy<T>(NativeTensorArray src, int srcIndex, T[] dst, int dstIndex, int length) where T : unmanaged
        {
            if (length == 0)
                return;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            CheckCopyArguments(src.Length * k_DataItemSize, srcIndex * k_DataItemSize, dst.Length * sizeof(T), dstIndex * sizeof(T), length * sizeof(T));
#endif

            fixed (void* dstPtr = &dst[0])
            {
                void* srcPtr = src.RawPtr;
                UnsafeUtility.MemCpy((byte*)dstPtr + dstIndex * sizeof(T),
                                    (byte*)srcPtr + srcIndex * k_DataItemSize,
                                    length * sizeof(T));
            }
        }

        /// <summary>
        /// Copies the data from a source `NativeTensorArray` to a destination array up to a given length starting from given indexes.
        /// </summary>
        /// <param name="src">The array to copy from.</param>
        /// <param name="srcIndex">The index of the first element to copy from.</param>
        /// <param name="dst">The array to copy to.</param>
        /// <param name="dstIndex">The index of the first element to copy to.</param>
        /// <param name="length">The number of elements.</param>
        /// <typeparam name="T">The data type of the elements.</typeparam>
        /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
        public static unsafe void Copy<T>(NativeTensorArray src, int srcIndex, NativeArray<T> dst, int dstIndex, int length) where T : unmanaged
        {
            if (length == 0)
                return;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            CheckCopyArguments(src.Length * k_DataItemSize, srcIndex * k_DataItemSize, dst.Length * sizeof(T), dstIndex * sizeof(T), length * sizeof(T));
#endif
            void* srcPtr = src.RawPtr;
            void* dstPtr = dst.GetUnsafeReadOnlyPtr<T>();
            UnsafeUtility.MemCpy((byte*)dstPtr + dstIndex * sizeof(T),
                                 (byte*)srcPtr + srcIndex * k_DataItemSize,
                                 length * sizeof(T));
        }

        /// <summary>
        /// Copies the data from a source array to a destination `NativeTensorArray` up to a given length starting from given indexes.
        /// </summary>
        /// <param name="src">The array to copy from.</param>
        /// <param name="srcIndex">The index of the first element to copy from.</param>
        /// <param name="dst">The array to copy to.</param>
        /// <param name="dstIndex">The index of the first element to copy to.</param>
        /// <param name="length">The number of elements.</param>
        /// <typeparam name="T">The data type of the elements.</typeparam>
        /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
        public static unsafe void Copy<T>(T[] src, int srcIndex, NativeTensorArray dst, int dstIndex, int length) where T : unmanaged
        {
            if (length == 0)
                return;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            CheckCopyArguments(src.Length * sizeof(T), srcIndex * sizeof(T), dst.Length * k_DataItemSize, dstIndex * k_DataItemSize, length * sizeof(T));
#endif
            fixed (void* srcPtr = &src[0])
            {
                void* dstPtr = dst.RawPtr;
                UnsafeUtility.MemCpy((byte*)dstPtr + dstIndex * k_DataItemSize,
                                     (byte*)srcPtr + srcIndex * sizeof(T),
                                     length * sizeof(T));
            }
        }

        /// <summary>
        /// Copies the data from a source `NativeArray` to a destination array up to a given length starting from given indexes.
        /// </summary>
        /// <param name="src">The array to copy from.</param>
        /// <param name="srcIndex">The index of the first element to copy from.</param>
        /// <param name="dst">The array to copy to.</param>
        /// <param name="dstIndex">The index of the first element to copy to.</param>
        /// <param name="length">The number of elements.</param>
        /// <typeparam name="T">The data type of the elements.</typeparam>
        /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
        public static unsafe void Copy<T>(NativeArray<T> src, int srcIndex, NativeTensorArray dst, int dstIndex, int length) where T : unmanaged
        {
            if (length == 0)
                return;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            CheckCopyArguments(src.Length * sizeof(T), srcIndex * sizeof(T), dst.Length * k_DataItemSize, dstIndex * k_DataItemSize, length * sizeof(T));
#endif
            void* srcPtr = src.GetUnsafeReadOnlyPtr<T>();
            void* dstPtr = dst.RawPtr;
            UnsafeUtility.MemCpy((byte*)dstPtr + dstIndex * k_DataItemSize,
                                 (byte*)srcPtr + srcIndex * sizeof(T),
                                 length * sizeof(T));
        }

        /// <summary>
        /// Copies the data from a source `NativeArray` to a destination array up to a given length starting from given indexes.
        /// </summary>
        /// <param name="src">The array to copy from.</param>
        /// <param name="srcIndex">The index of the first element to copy from.</param>
        /// <param name="dst">The array to copy to.</param>
        /// <param name="dstIndex">The index of the first element to copy to.</param>
        /// <param name="length">The number of elements.</param>
        /// <typeparam name="T">The data type of the elements.</typeparam>
        /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
        public static unsafe void Copy<T>(NativeArray<T>.ReadOnly src, int srcIndex, NativeTensorArray dst, int dstIndex, int length) where T : unmanaged
        {
            if (length == 0)
                return;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            CheckCopyArguments(src.Length * sizeof(T), srcIndex * sizeof(T), dst.Length * k_DataItemSize, dstIndex * k_DataItemSize, length * sizeof(T));
#endif
            void* srcPtr = src.GetUnsafeReadOnlyPtr<T>();
            void* dstPtr = dst.RawPtr;
            UnsafeUtility.MemCpy((byte*)dstPtr + dstIndex * k_DataItemSize,
                                 (byte*)srcPtr + srcIndex * sizeof(T),
                                 length * sizeof(T));
        }

        /// <summary>
        /// Copies the data from a source `NativeTensorArray` to a destination byte array up to a given length starting from given offsets.
        /// </summary>
        /// <param name="src">The array to copy from.</param>
        /// <param name="srcByteIndex">The index of the first element to copy from.</param>
        /// <param name="dst">The array to copy to.</param>
        /// <param name="dstByteIndex">The offset in bytes to copy to in the destination array.</param>
        /// <param name="lengthInBytes">The number of bytes to copy.</param>
        /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
        public static unsafe void BlockCopy(NativeTensorArray src, int srcByteIndex, byte[] dst, int dstByteIndex, int lengthInBytes)
        {
             if (lengthInBytes == 0)
                return;
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            CheckCopyArguments(src.Length * k_DataItemSize, srcByteIndex, dst.Length, dstByteIndex, lengthInBytes);
#endif

            fixed (void* dstPtr = &dst[0])
            {
                void* srcPtr = src.RawPtr;
                UnsafeUtility.MemCpy((byte*)dstPtr + dstByteIndex,
                                     (byte*)srcPtr + srcByteIndex,
                                     lengthInBytes);
            }
        }

        /// <summary>
        /// Copies the data from a source byte array to a destination `NativeTensorArray` up to a given length starting from given offsets.
        /// </summary>
        /// <param name="src">The array to copy from.</param>
        /// <param name="srcByteIndex">The offset in bytes to copy from.</param>
        /// <param name="dst">The array to copy to.</param>
        /// <param name="dstByteIndex">The offset in bytes to copy to in the destination array.</param>
        /// <param name="lengthInBytes">The number of bytes to copy.</param>
        /// <exception cref="ArgumentException">Thrown if the given indexes and length are out of bounds of the source or destination array.</exception>
        public static unsafe void BlockCopy(byte[] src, int srcByteIndex, NativeTensorArray dst, int dstByteIndex, int lengthInBytes)
        {
            if (lengthInBytes == 0)
                return;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
            CheckCopyArguments(src.Length, srcByteIndex, dst.Length * k_DataItemSize, dstByteIndex, lengthInBytes);
#endif

            fixed (void* srcPtr = &src[0])
            {
                void* dstPtr = dst.RawPtr;
                UnsafeUtility.MemCpy((byte*)dstPtr + dstByteIndex,
                                     (byte*)srcPtr + srcByteIndex,
                                     lengthInBytes);
            }
        }
    }
}
