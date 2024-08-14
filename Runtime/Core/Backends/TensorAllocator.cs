using System;

namespace Unity.Sentis
{
    static class AllocatorUtils
    {
        internal static Tensor AllocTensor(DataType dataType, TensorShape shape, ITensorData buffer)
        {
            switch (dataType)
            {
                case DataType.Float:
                    return new Tensor<float>(shape, buffer);
                case DataType.Int:
                    return new Tensor<int>(shape, buffer);
                case DataType.Short:
                    return new Tensor<short>(shape, buffer);
                case DataType.Byte:
                    return new Tensor<byte>(shape, buffer);
                default:
                    throw new NotImplementedException();
            }
        }
        internal static DataType ToDataType<T>() where T : unmanaged
        {
            if (typeof(T) == typeof(float))
                return DataType.Float;
            else if (typeof(T) == typeof(int))
                return DataType.Int;
            else if (typeof(T) == typeof(short))
                return DataType.Short;
            else if (typeof(T) == typeof(byte))
                return DataType.Byte;
            else
                return DataType.Custom;
        }
    }
}
