using System;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;

namespace Unity.Sentis
{
    /// <summary>
    /// An interface that represents a device-dependent representation of the data in a tensor.
    /// </summary>
    public interface ITensorData : IDisposable
    {
        /// <summary>
        /// Uploads a contiguous block of tensor data to internal storage.
        /// </summary>
        /// <param name="data">The data to upload.</param>
        /// <param name="srcCount">The number of elements to upload.</param>
        /// <typeparam name="T">The type of data to upload.</typeparam>
        void Upload<T>(NativeArray<T> data, int srcCount) where T : unmanaged;

        /// <summary>
        /// Checks if asynchronous readback request is done.
        /// </summary>
        /// <returns>Whether async readback is successful.</returns>
        bool IsReadbackRequestDone();

        /// <summary>
        /// Schedules asynchronous readback of the internal data.
        /// </summary>
        void ReadbackRequest();

        /// <summary>
        /// Blocking call to make sure that internal data is correctly written to and available for CPU read back.
        /// </summary>
        void CompleteAllPendingOperations();

        /// <summary>
        /// Blocking call that returns a contiguous block of data from internal storage.
        /// </summary>
        /// <param name="dstCount">The number of elements to download.</param>
        /// <typeparam name="T">The data type of the elements.</typeparam>
        /// <returns>A native array of downloaded elements.</returns>
        NativeArray<T> Download<T>(int dstCount) where T : unmanaged;

        #if UNITY_2023_2_OR_NEWER
        /// <summary>
        /// Awaitable contiguous block of data from internal storage.
        /// </summary>
        /// <param name="dstCount">The number of elements to download.</param>
        /// <typeparam name="T">The data type of the elements.</typeparam>
        /// <returns>A awaitable native array of downloaded elements.</returns>
        Awaitable<NativeArray<T>> DownloadAsync<T>(int dstCount) where T : unmanaged;
        #endif

        /// <summary>
        /// The maximum count of the stored data elements.
        /// </summary>
        int maxCapacity { get; }

        /// <summary>
        /// On what backend are the data elements stored.
        /// </summary>
        BackendType backendType { get; }
    }
}
