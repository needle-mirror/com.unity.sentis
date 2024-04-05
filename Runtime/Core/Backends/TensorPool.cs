using System;
using System.Collections.Generic;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

namespace Unity.Sentis {

class TensorPool : IDisposable
{
    // tensor classes re-use pool
    TensorClassPool<TensorInt> m_TensorIntPool = new TensorClassPool<TensorInt>();
    TensorClassPool<TensorFloat> m_TensorFloatPool = new TensorClassPool<TensorFloat>();

    // tensor data re-use pool
    TensorDataPool<ComputeTensorData> m_computeMemoryPool = new TensorDataPool<ComputeTensorData>();
    TensorDataPool<BurstTensorData> m_cpuMemoryPool = new TensorDataPool<BurstTensorData>();

    public Tensor NewTensor(TensorShape shape, DataType dataType, BackendType backendType)
    {
        // adopt from pool or create
        Tensor tensor;
        switch (dataType)
        {
            case DataType.Float:
                tensor = m_TensorFloatPool.AdoptFromPool();
                if (tensor == null)
                    tensor = TensorFloat.AllocNoData(shape);
                break;
            case DataType.Int:
                tensor = m_TensorIntPool.AdoptFromPool();
                if (tensor == null)
                    tensor = TensorInt.AllocNoData(shape);
                break;
            default:
                throw new NotImplementedException();
        }
        tensor.shape = shape;
        ITensorData data; // alloc here or in ops?
        switch (backendType)
        {
            case BackendType.GPUCompute:
                data = m_computeMemoryPool.AdoptFromPool(shape.length);
                if (data == null)
                    data = new ComputeTensorData(shape.length);
                break;
            case BackendType.CPU:
                data = m_cpuMemoryPool.AdoptFromPool(shape.length);
                if (data == null)
                    data = new BurstTensorData(shape.length);
                break;
            default:
                data = null;
                break;
        }
        tensor.dataOnBackend = data;
        return tensor;
    }

    public void Dispose(Tensor tensor)
    {
        if (tensor.dataOnBackend != null) // 0-dim tensor have null tensor on device
        {
            switch (tensor.dataOnBackend.backendType)
            {
                case BackendType.GPUCompute:
                    m_computeMemoryPool.ReleaseToPool(tensor.dataOnBackend as ComputeTensorData);
                    break;
                case BackendType.CPU:
                    m_cpuMemoryPool.ReleaseToPool(tensor.dataOnBackend as BurstTensorData);
                    break;
                default:
                    tensor.dataOnBackend.Dispose();
                    break;
            }
            tensor.dataOnBackend = null;
        }
        switch (tensor.dataType)
        {
            case DataType.Float:
                m_TensorFloatPool.ReleaseToPool(tensor as TensorFloat);
                break;
            case DataType.Int:
                m_TensorIntPool.ReleaseToPool(tensor as TensorInt);
                break;
            default:
                throw new NotImplementedException();
        }
    }

    public void Dispose()
    {
        m_computeMemoryPool.Dispose();
        m_cpuMemoryPool.Dispose();
        m_TensorIntPool.Dispose();
        m_TensorFloatPool.Dispose();
    }
}

} // namespace Unity.Sentis
