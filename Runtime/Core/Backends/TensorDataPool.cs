using System;
using System.Collections.Generic;
using Unity.Collections;

namespace Unity.Sentis
{
    /// <summary>
    /// Sorted list of tensordata by buffer size in byte
    ///
    /// Usage:
    /// * Algorithm goal:
    /// Given a pool of buffers and a size in byte
    /// Retrieve the smallest buffer which size is >= than provided size
    /// Ex: pool: [1, 4, 6, 9, 11] size = 3 => return 4
    /// Once found, remove tensor from pool
    /// We can append new buffers to the pool
    ///
    /// * Implementation: RB-Tree better, but works fine with a sorted list
    /// Keep sorted list of pool buffer size
    /// Search : Binary Search, positive index of the specified value if value is found. Otherwise bitwise complement of index of first greater value.
    ///     - found, remove buffer from sorted list
    ///     - not found, return null
    /// Insert : insert in list and sort
    /// Data structures
    /// list[int, size] for efficient search/insert/removal
    /// dict[size, buffer] for efficient removal/insert
    ///
    /// NB: can have multiple buffer of same size, so dict[buffer.size] won't work
    /// Solution: keep track of # of buffers with the same size and use a pairing function to get unique key from [buffer.size, size.count_in_pool]
    /// - on insert, increment size.count
    /// - on remove, return latest insert (buffer with size.count the highest) + decrement size.count
    /// </summary>
    class TensorDataPool<T> : IDisposable where T : ITensorData
    {
        NativeList<int> freeBufferSize = new NativeList<int>(0, Allocator.Persistent);
        Dictionary<int, int> bufferSizeCount = new Dictionary<int, int>();
        Dictionary<long, T> freeBuffers = new Dictionary<long, T>();

        // http://szudzik.com/ElegantPairing.pdf
        long SzudzikPairing(long a, long b)
        {
            return a >= b ? a * a + a + b : a + b * b;
        }

        public T AdoptFromPool(int size)
        {
            if (freeBufferSize.Length == 0)
                return default(T);

            ProfilerMarkers.TensorDataPoolAdopt.Begin();

            var found = freeBufferSize.BinarySearch(size);
            if (found < 0)
                found = ~found;
            if (found >= freeBufferSize.Length)
            {
                ProfilerMarkers.TensorDataPoolAdopt.End();
                return default(T);
            }

            var key = freeBufferSize[found];

            int index = bufferSizeCount[key];
            long keyIndex = SzudzikPairing(key, index);

            var buffer = freeBuffers[keyIndex];
            freeBuffers.Remove(keyIndex);
            freeBufferSize.RemoveAt(found);
            bufferSizeCount[key]--;

            ProfilerMarkers.TensorDataPoolAdopt.End();

            return buffer;
        }

        public void ReleaseToPool(T data)
        {
            ProfilerMarkers.TensorDataPoolRelease.Begin();

            int bufferSize = data.maxCapacity;
            if (!bufferSizeCount.ContainsKey(bufferSize))
                bufferSizeCount.Add(bufferSize, 0);

            bufferSizeCount[bufferSize] += 1;

            long index = SzudzikPairing(bufferSize, bufferSizeCount[bufferSize]);
            Logger.AssertIsFalse(freeBuffers.ContainsKey(index), "TensorDataPool.ReleaseToPool collision in hash or releasing already released buffer");
            freeBuffers[index] = data;

            int insertionIdx = freeBufferSize.BinarySearch(bufferSize);
            if (insertionIdx < 0)
                // BinarySearch returns a negative value if the search value was not found, but it's a negative of the index
                // where the value _would_ be if it were present
                insertionIdx = (~insertionIdx);
            if (insertionIdx == freeBufferSize.Length)
                freeBufferSize.Add(bufferSize);
            else
            {
                freeBufferSize.InsertRangeWithBeginEnd(insertionIdx, insertionIdx + 1);
                freeBufferSize[insertionIdx] = bufferSize;
            }

            ProfilerMarkers.TensorDataPoolRelease.End();
        }

        public void Dispose()
        {
            foreach (var tensor in freeBuffers.Values)
                tensor.Dispose();
            freeBuffers.Clear();
            freeBufferSize.Dispose();
            bufferSizeCount.Clear();
        }
    }
}
