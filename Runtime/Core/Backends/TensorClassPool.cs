using System;
using System.Collections.Generic;

namespace Unity.Sentis
{
    /// <summary>
    /// FIFO storage to handle tensor class re-use
    /// Usage:
    /// 1. ReleaseToPool
    ///     add a object to the re-use pool.
    /// 2. AdoptFromPool
    ///     try to adopt oldest object in pool.
    ///     if pool empty, return null
    /// 3. Dispose
    ///     disposes all object in the pool
    /// </summary>
    class TensorClassPool<T> where T : Tensor, IDisposable
    {
        Queue<T> freeTensors = new Queue<T>();

        public T AdoptFromPool()
        {
            if (freeTensors.Count > 0)
                return freeTensors.Dequeue();
            else
                return null;
        }

        public void ReleaseToPool(T tensor)
        {
            freeTensors.Enqueue(tensor);
        }

        public void Dispose()
        {
            foreach (var tensor in freeTensors)
                tensor.Dispose();
            freeTensors.Clear();
        }
    }
}
