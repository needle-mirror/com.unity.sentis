using System;
using Unity.Sentis;

static class AllocatorUtils
{
    internal static Tensor AllocTensor(DataType dataType, TensorShape shape, ITensorData buffer)
    {
        switch (dataType)
        {
            case DataType.Float:
                return new TensorFloat(shape, buffer);
            case DataType.Int:
                return new TensorInt(shape, buffer);
            case DataType.Short:
                return new TensorShort(shape, buffer);
            case DataType.Byte:
                return new TensorByte(shape, buffer);
            default:
                throw new NotImplementedException();
        }
    }
}
