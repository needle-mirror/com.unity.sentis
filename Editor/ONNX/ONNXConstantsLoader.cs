using Onnx;
using System;
using System.Linq;
using System.Runtime.CompilerServices;
using UnityEngine.Assertions;
using System.IO;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;

[assembly: InternalsVisibleToAttribute("Unity.Sentis.EditorTests")]

namespace Unity.Sentis.ONNX
{
    internal static class ONNXConstantsLoader
    {
        public static Constant LoadConstant(TensorProto tensorProto, FileStream weightStream)
        {
            var shape = GetShape(tensorProto);
            var dataType = GetDataType(tensorProto);

            var constant = new Constant(-1, shape, dataType, shape.length * NativeTensorArray.k_DataItemSize);
            if (shape.HasZeroDims())
            {
                return constant;
            }

            Assert.AreEqual(tensorProto.DataLocation, TensorProto.Types.DataLocation.External);

            var lengthField = tensorProto.ExternalData.SingleOrDefault(x => x.Key == "length");
            int length = lengthField != null ? int.Parse(lengthField.Value) : (int)weightStream.Length;

            byte[] byteArray = new byte[length];
            weightStream.Read(byteArray, 0, length);

            var tensorData = GetTensorData(byteArray, length, shape, (TensorProto.Types.DataType)tensorProto.DataType);
            constant.m_Weights = tensorData;
            return constant;
        }

        public static Constant LoadConstant(TensorProto tensorProto, string directoryPath = null)
        {
            if (tensorProto.ExternalData != null && tensorProto.ExternalData.Any(x => x.Key == "location"))
            {
                string name = tensorProto.ExternalData.SingleOrDefault(x => x.Key == "location").Value;
                using var weightStream = File.OpenRead(Path.Combine(directoryPath, name));
                return LoadConstant(tensorProto, weightStream);
            }

            var shape = GetShape(tensorProto);
            var dataType = GetDataType(tensorProto);

            var constant = new Constant(-1, shape, dataType, shape.length * NativeTensorArray.k_DataItemSize);

            if (shape.HasZeroDims())
                return constant;

            NativeTensorArrayFromManagedArray tensorData;

            if ((tensorProto.RawData != null) && (!tensorProto.RawData.IsEmpty))
            {
                tensorData = GetTensorData(tensorProto.RawData.ToByteArray(), tensorProto.RawData.Length, shape, (TensorProto.Types.DataType)tensorProto.DataType);
            }
            // Float32
            else if ((tensorProto.FloatData != null) && (tensorProto.FloatData.Count != 0))
            {
                Assert.IsTrue(shape.length == tensorProto.FloatData.Count);
                var arrayData = new float[shape.length];
                tensorProto.FloatData.CopyTo(arrayData, 0);
                tensorData = new NativeTensorArrayFromManagedArray(arrayData, 0, sizeof(float), shape.length);
            }
            // Int32
            else if ((tensorProto.Int32Data != null) && (tensorProto.Int32Data.Count != 0))
            {
                Assert.IsTrue(shape.length == tensorProto.Int32Data.Count);
                var arrayData = new int[shape.length];
                tensorProto.Int32Data.CopyTo(arrayData, 0);
                tensorData = new NativeTensorArrayFromManagedArray(arrayData, 0, sizeof(int), shape.length);
            }
            // Int64
            else if ((tensorProto.Int64Data != null) && (tensorProto.Int64Data.Count != 0))
            {
                Assert.IsTrue(shape.length == tensorProto.Int64Data.Count);
                var arrayData = new long[shape.length];
                tensorProto.Int64Data.CopyTo(arrayData, 0);
                var intArrayData = new int[shape.length];
                unsafe
                {
                    fixed (void* dataPtr = &arrayData[0],  dstPtr = &intArrayData[0])
                    {
                        var job = new BurstJobsCastTensor.LongBytesAsIntJob
                        {
                            src = (long*)dataPtr,
                            dst = (int*)dstPtr
                        };
                        var jobHandle = job.Schedule(shape.length, 32);
                        jobHandle.Complete();
                    }
                }
                tensorData = new NativeTensorArrayFromManagedArray(intArrayData, 0, sizeof(int), shape.length);
            }
            else
            {
                throw new OnnxLayerImportException("Could not read tensor data for constant tensor.");
            }

            constant.m_Weights = tensorData;
            return constant;
        }

        static TensorShape GetShape(TensorProto tensorProto)
        {
            var onnxShape = tensorProto.Dims.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
            return new TensorShape(onnxShape);
        }

        static DataType GetDataType(TensorProto tensorProto)
        {
            return ONNXNodeWrapper.DataTypeFromOnnxDataType((TensorProto.Types.DataType)tensorProto.DataType);
        }

        static NativeTensorArrayFromManagedArray GetTensorData(byte[] byteArray, int length, TensorShape shape, TensorProto.Types.DataType dataType)
        {
            if (shape.HasZeroDims())
                return null;

            NativeTensorArrayFromManagedArray data;

            // Double
            if (dataType == TensorProto.Types.DataType.Double)
            {
                Assert.IsTrue((sizeof(double) * shape.length) == length);
                float[] dstData = new float[shape.length];
                unsafe
                {
                    fixed (void* dataPtr = &byteArray[0], dstPtr = &dstData[0])
                    {
                        var job = new BurstJobsCastTensor.DoubleBytesAsFloatJob
                        {
                            src = (long*)dataPtr,
                            dst = (float*)dstPtr
                        };
                        var jobHandle = job.Schedule(shape.length, 32);
                        jobHandle.Complete();
                    }
                }
                data = new NativeTensorArrayFromManagedArray(dstData, 0, sizeof(float), shape.length);
            }
            // Float32
            else if (dataType == TensorProto.Types.DataType.Float)
            {
                Assert.IsTrue((sizeof(float) * shape.length) == length);
                data = new NativeTensorArrayFromManagedArray(byteArray, 0, sizeof(byte), shape.length);
            }
            // Float16
            else if (dataType == TensorProto.Types.DataType.Float16)
            {
                Assert.IsTrue((sizeof(ushort) * shape.length) == length);
                float[] dstData = new float[shape.length];
                unsafe
                {
                    fixed (void* dataPtr = &byteArray[0], dstPtr = &dstData[0])
                    {
                        var job = new BurstJobsCastTensor.Float16BytesAsFloatJob
                        {
                            src = (ushort*)dataPtr,
                            dst = (float*)dstPtr
                        };
                        var jobHandle = job.Schedule(shape.length, 32);
                        jobHandle.Complete();
                    }
                }
                data = new NativeTensorArrayFromManagedArray(dstData, 0, sizeof(float), shape.length);
            }
            // Int32
            else if (dataType == TensorProto.Types.DataType.Int32)
            {
                Assert.IsTrue((sizeof(int) * shape.length) == length);
                data = new NativeTensorArrayFromManagedArray(byteArray, 0, sizeof(byte), shape.length);
            }
            // Int64
            else if (dataType == TensorProto.Types.DataType.Int64)
            {
                Assert.IsTrue((sizeof(long) * shape.length) == length);
                int[] dstData = new int[shape.length];
                unsafe
                {
                    fixed (void* dataPtr = &byteArray[0], dstPtr = &dstData[0])
                    {
                        var job = new BurstJobsCastTensor.LongBytesAsIntJob
                        {
                            src = (long*)dataPtr,
                            dst = (int*)dstPtr
                        };
                        var jobHandle = job.Schedule(shape.length, 32);
                        jobHandle.Complete();
                    }
                }
                data = new NativeTensorArrayFromManagedArray(dstData, 0, sizeof(int), shape.length);
            }
            // Bool
            else if (dataType == TensorProto.Types.DataType.Bool)
            {
                Assert.IsTrue((sizeof(bool) * shape.length) == length);
                int[] dstData = new int[shape.length];
                unsafe
                {
                    fixed (void* dataPtr = &byteArray[0], dstPtr = &dstData[0])
                    {
                        var job = new BurstJobsCastTensor.BoolBytesAsIntJob
                        {
                            src = (bool*)dataPtr,
                            dst = (int*)dstPtr
                        };
                        var jobHandle = job.Schedule(shape.length, 32);
                        jobHandle.Complete();
                    }
                }
                data = new NativeTensorArrayFromManagedArray(dstData, 0, sizeof(int), shape.length);
            }
            // Uint8
            else if (dataType == TensorProto.Types.DataType.Uint8)
            {
                Assert.IsTrue((sizeof(byte) * shape.length) == length);
                int[] dstData = new int[shape.length];
                unsafe
                {
                    fixed (void* dataPtr = &byteArray[0], dstPtr = &dstData[0])
                    {
                        var job = new BurstJobsCastTensor.Uint8BytesAsIntJob
                        {
                            src = (byte*)dataPtr,
                            dst = (int*)dstPtr
                        };
                        var jobHandle = job.Schedule(shape.length, 32);
                        jobHandle.Complete();
                    }
                }
                data = new NativeTensorArrayFromManagedArray(dstData, 0, sizeof(int), shape.length);
            }
            // Int8
            else if (dataType == TensorProto.Types.DataType.Int8)
            {
                Assert.IsTrue((sizeof(sbyte) * shape.length) == length);
                int[] dstData = new int[shape.length];
                unsafe
                {
                    fixed (void* dataPtr = &byteArray[0], dstPtr = &dstData[0])
                    {
                        var job = new BurstJobsCastTensor.Int8BytesAsIntJob
                        {
                            src = (sbyte*)dataPtr,
                            dst = (int*)dstPtr
                        };
                        var jobHandle = job.Schedule(shape.length, 32);
                        jobHandle.Complete();
                    }
                }
                data = new NativeTensorArrayFromManagedArray(dstData, 0, sizeof(float), shape.length);
            }
            else
            {
                throw new OnnxLayerImportException($"Tensor data type {dataType} is not supported.");
            }

            return data;
        }
    }
}
