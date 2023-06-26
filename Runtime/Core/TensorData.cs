using System;
using UnityEngine;

namespace Unity.Sentis {
/// <summary>
/// An interface that represents a device-dependent representation of the data in a tensor.
/// </summary>
public interface ITensorData : IDisposable
{
    /// <summary>
    /// Reserves memory for `count` elements.
    /// </summary>
    void Reserve(int count);

    /// <summary>
    /// Uploads the tensor data to internal storage.
    /// </summary>
    void Upload<T>(T[] data, int srcCount, int srcOffset = 0) where T : unmanaged;

    /// <summary>
    /// Schedules asynchronous download of the internal data.
    /// </summary>
    bool ScheduleAsyncDownload();

    /// <summary>
    /// Returns data from internal storage.
    /// </summary>
    T[] Download<T>(int dstCount, int srcOffset = 0) where T : unmanaged;

    /// <summary>
    /// The maximum count of the stored data elements.
    /// </summary>
    int maxCapacity { get; }

    /// <summary>
    /// On what device backend are the data elements stored
    /// </summary>
    DeviceType deviceType { get; }
}

} // namespace Unity.Sentis
