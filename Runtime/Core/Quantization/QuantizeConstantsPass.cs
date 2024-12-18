using System.Collections.Generic;
using Unity.Sentis.Layers;
using Unity.Jobs;
using UnityEngine;

namespace Unity.Sentis
{
    class QuantizeConstantsPass
    {
        QuantizationType m_QuantizationType;

        public QuantizeConstantsPass(QuantizationType quantizationType)
        {
            m_QuantizationType = quantizationType;
        }

        public void Run(ref Model model)
        {
            var constants = new Dictionary<int, (Constant, int)>();
            for (var i = 0; i < model.constants.Count; i++)
            {
                var c = model.constants[i];
                constants.Add(c.index, (c, i));
            }

            using var backend = new CPUBackend();
            using var min = new Tensor<float>(new TensorShape());
            using var max = new Tensor<float>(new TensorShape());

            var quantizeTensors = new HashSet<int>();
            for (var i = 0; i < model.layers.Count; i++)
            {
                var layer = model.layers[i];

                if (!(layer is Conv || layer is ConvTranspose || layer is Gather || layer is Dense || layer is MatMul || layer is MatMul2D))
                    continue;

                foreach (var input in layer.inputs)
                {
                    if ((input == -1) || quantizeTensors.Contains(input) || !constants.ContainsKey(input))
                        continue;

                    quantizeTensors.Add(input);

                    var constantIndex = constants[input];
                    var constant = constantIndex.Item1;
                    var index = constantIndex.Item2;
                    if (constant.dataType != DataType.Float)
                        continue;

                    if (m_QuantizationType == QuantizationType.Float16)
                    {
                        var quantizedTensor = new Tensor<short>(constant.shape, data: null);
                        var data = new int[quantizedTensor.count];
                        unsafe
                        {
                            fixed (void* dataPtr = &data[0])
                            {
                                var job = new BurstJobsQuantizeTensor.CastFloatToHalfJob
                                {
                                    src = (float*)constant.weights.RawPtr,
                                    dst = (ushort*)(dataPtr)
                                };
                                var jobHandle = job.Schedule(constant.shape.length, 32);
                                jobHandle.Complete();
                            }
                        }

                        var newIndex = model.GetUniqueIndex();
                        Constant quantizedConstant = new Constant(newIndex, constant.shape, DataType.Short, new NativeTensorArrayFromManagedArray(data, 0, sizeof(int), data.Length));
                        model.constants[index] = quantizedConstant;
                        model.layers.Insert(i, new Cast(constant.index, newIndex, DataType.Float));
                        i++;
                    }
                    else if (m_QuantizationType == QuantizationType.Uint8)
                    {
                        using var X = constant.WeightsToTensorWithSharedTensorData() as Tensor<float>;
                        backend.ReduceMin(X, min, null);
                        backend.ReduceMax(X, max, null);
                        min.CompleteAllPendingOperations();
                        max.CompleteAllPendingOperations();
                        var minValue = min.GetItem<float>(0);
                        var maxValue = max.GetItem<float>(0);
                        float scale = (Mathf.Max(0, maxValue) - Mathf.Min(0, minValue)) / 255f;
                        byte zeroPoint = (byte)Mathf.RoundToInt(Mathf.Clamp(-minValue / scale, 0, 255));

                        var quantizedTensor = new Tensor<byte>(constant.shape, null);
                        var data = new int[quantizedTensor.count];
                        unsafe
                        {
                            fixed (void* dataPtr = &data[0])
                            {
                                var job = new BurstJobsQuantizeTensor.QuantizeUint8Job
                                {
                                    src = (float*)constant.weights.RawPtr,
                                    dst = (byte*)(dataPtr),
                                    scale = scale,
                                    zeroPoint = zeroPoint
                                };
                                var jobHandle = job.Schedule(constant.shape.length, 32);
                                jobHandle.Complete();
                            }
                        }

                        var newIndex = model.GetUniqueIndex();
                        Constant quantizedConstant = new Constant(newIndex, constant.shape, DataType.Byte, new NativeTensorArrayFromManagedArray(data, 0, sizeof(int), data.Length));
                        model.constants[index] = quantizedConstant;
                        model.layers.Insert(i, new DequantizeUint8(constant.index, newIndex, scale, zeroPoint));
                        i++;
                    }
                }
            }
        }
    }
}
