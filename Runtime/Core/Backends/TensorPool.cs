using System;

namespace Unity.Sentis
{
    class TensorPool : IDisposable
    {
        // tensor classes re-use pool
        TensorClassPool<Tensor<int>> m_TensorIntPool = new TensorClassPool<Tensor<int>>();
        TensorClassPool<Tensor<float>> m_TensorFloatPool = new TensorClassPool<Tensor<float>>();

        // tensor data re-use pool
        TensorDataPool<ComputeTensorData> m_computeMemoryPool = new TensorDataPool<ComputeTensorData>();
        TensorDataPool<CPUTensorData> m_cpuMemoryPool = new TensorDataPool<CPUTensorData>();

        public Tensor NewTensor(TensorShape shape, DataType dataType, BackendType backendType)
        {
            // adopt from pool or create
            Tensor tensor;
            switch (dataType)
            {
                case DataType.Float:
                    tensor = m_TensorFloatPool.AdoptFromPool();
                    if (tensor == null)
                        tensor = new Tensor<float>(shape, data: null);
                    break;
                case DataType.Int:
                    tensor = m_TensorIntPool.AdoptFromPool();
                    if (tensor == null)
                        tensor = new Tensor<int>(shape, data: null);
                    break;
                default:
                    throw new NotImplementedException();
            }
            tensor.shape = shape;
            tensor.count = shape.length;

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
                        data = new CPUTensorData(shape.length);
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
                        m_cpuMemoryPool.ReleaseToPool(tensor.dataOnBackend as CPUTensorData);
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
                    m_TensorFloatPool.ReleaseToPool(tensor as Tensor<float>);
                    break;
                case DataType.Int:
                    m_TensorIntPool.ReleaseToPool(tensor as Tensor<int>);
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
}
