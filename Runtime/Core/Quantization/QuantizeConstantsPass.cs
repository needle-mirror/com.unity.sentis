using System.Collections.Generic;
using Unity.Sentis.Layers;
using Unity.Jobs;
using UnityEngine;

namespace Unity.Sentis.Quantization
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
            var constants = new Dictionary<string, (Constant, int)>();
            for (var i = 0; i < model.constants.Count; i++)
            {
                var c = model.constants[i];
                constants.Add(c.index, (c, i));
            }

            using var backend = new CPUBackend();
            using var min = TensorFloat.AllocZeros(new TensorShape());
            using var max = TensorFloat.AllocZeros(new TensorShape());

            var quantizeTensors = new HashSet<string>();
            for (var i = 0; i < model.layers.Count; i++)
            {
                var layer = model.layers[i];

                if (!(layer is Conv || layer is ConvTranspose || layer is Gather || layer is Dense || layer is MatMul || layer is MatMul2D))
                    continue;

                foreach (var input in layer.inputs)
                {
                    if (string.IsNullOrEmpty(input) || quantizeTensors.Contains(input) || !constants.ContainsKey(input))
                        continue;

                    quantizeTensors.Add(input);

                    var constantIndex = constants[input];
                    var constant = constantIndex.Item1;
                    var index = constantIndex.Item2;
                    if (constant.dataType != DataType.Float)
                        continue;

                    if (m_QuantizationType == QuantizationType.Float16)
                    {
                        using var quantizedTensor = TensorShort.AllocZeros(constant.shape);
                        var data = BurstTensorData.Pin(quantizedTensor);
                        unsafe
                        {
                            var job = new BurstJobsQuantizeTensor.CastFloatToHalfJob
                            {
                                src = (float*)constant.weights.RawPtr,
                                dst = (ushort*)(data.rawPtr)
                            };
                            var jobHandle = job.Schedule(constant.shape.length, 1024);
                            jobHandle.Complete();
                        }

                        var newName = model.GetUniqueIndex(constant.index + "_fp16");
                        Constant quantizedConstant = new Constant(newName, constant.shape, DataType.Short, data.array);
                        model.constants[index] = quantizedConstant;
                        model.layers.Insert(i, new Cast(constant.index, newName, DataType.Float));
                        i++;
                    }
                    else if (m_QuantizationType == QuantizationType.Uint8)
                    {
                        using var X = constant.WeightsToTensorWithSharedTensorData() as TensorFloat;
                        backend.ReduceMin(X, min, null, false);
                        backend.ReduceMax(X, max, null, false);
                        var minValue = min.GetItem<float>(0);
                        var maxValue = max.GetItem<float>(0);
                        float scale = (Mathf.Max(0, maxValue) - Mathf.Min(0, minValue)) / 255f;
                        byte zeroPoint = (byte)Mathf.RoundToInt(Mathf.Clamp(-minValue / scale, 0, 255));

                        using var quantizedTensor = TensorByte.AllocZeros(constant.shape);
                        var data = BurstTensorData.Pin(quantizedTensor);
                        unsafe
                        {
                            var job = new BurstJobsQuantizeTensor.QuantizeUint8Job
                            {
                                src = (float*)constant.weights.RawPtr,
                                dst = (byte*)(data.rawPtr),
                                scale = scale,
                                zeroPoint = zeroPoint
                            };
                            var jobHandle = job.Schedule(constant.shape.length, 1024);
                            jobHandle.Complete();
                        }

                        var newName = model.GetUniqueIndex(constant.index + "_uint8");
                        Constant quantizedConstant = new Constant(newName, constant.shape, DataType.Byte, data.array);
                        model.constants[index] = quantizedConstant;
                        model.layers.Insert(i, new DequantizeUint8(constant.index, newName, scale, zeroPoint));
                        i++;
                    }
                }
            }
        }
    }
}
