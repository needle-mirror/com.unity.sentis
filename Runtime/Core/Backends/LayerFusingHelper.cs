using System;
using System.Collections.Generic;
using System.Linq; // ToArray(), ToDictionary()
using UnityEngine;

namespace Unity.Sentis
{
    class LinearLayerFusing : IDisposable
    {
        public static bool IsLayerLinear(Layers.Layer layer, Dictionary<string, Layers.Constant> constTensors)
        {
            var constInputs = layer.inputs.Count(x => constTensors.ContainsKey(x));
            bool allConstInputsButOne = (layer.inputs.Length - constInputs) == 1;

            return layer is Layers.Dense ||
                   layer is Layers.Conv ||
                   layer is Layers.ScaleBias ||
                   layer is Layers.ScalarMad ||
                   IsLayerLinearMathOp(layer) && allConstInputsButOne;
        }

        static bool IsLayerLinearMathOp(Layers.Layer layer)
        {
            return layer is Layers.Add ||
                   layer is Layers.Sub ||
                   layer is Layers.Mul;
        }

        public bool AreLayersFusable(Layers.Layer l0, Layers.Layer l1, Dictionary<string, Layers.Constant> constTensors)
        {
            bool conditions = true;
            // TODO: add fusing when bias is null
            if (l0.inputs.Any(string.IsNullOrEmpty) || l1.inputs.Any(string.IsNullOrEmpty))
                return false;

            if ((l0 is Layers.ScaleBias) && l1 is Layers.Conv)
                conditions = conditions && !(l1 as Layers.Conv).pads.Any(x => x != 0) && ((l1 as Layers.Conv).autoPad != Layers.AutoPad.NotSet); // padding breaks bias merging for non-zero bias
            else if (IsLayerLinearMathOp(l0) && (l1 is Layers.Conv))
            {
                if (!constTensors.ContainsKey(l0.inputs[0]) || !constTensors.ContainsKey(l0.inputs[1]))
                    return false;

                if (constTensors.ContainsKey(l0.inputs[0]))
                {
                    var constTensor = constTensors[l0.inputs[0]];
                    conditions = conditions && (constTensor.shape.rank == 1 && constTensor.shape[0] == constTensors[l1.inputs[2]].shape[0]);
                }
                else
                {
                    var constTensor = constTensors[l0.inputs[1]];
                    conditions = conditions && (constTensor.shape.rank == 1 && constTensor.shape[0] == constTensors[l1.inputs[2]].shape[0]);
                }
            }
            else if ((l0 is Layers.Conv) && IsLayerLinearMathOp(l1))
            {
                if (!constTensors.ContainsKey(l1.inputs[0]) || !constTensors.ContainsKey(l1.inputs[1]))
                    return false;

                if (constTensors.ContainsKey(l1.inputs[0]))
                {
                    var constTensor = constTensors[l1.inputs[0]];
                    conditions = conditions && (constTensor.shape.rank == 1 && constTensor.shape[0] == constTensors[l0.inputs[2]].shape[0]);
                }
                else
                {
                    var constTensor = constTensors[l1.inputs[1]];
                    conditions = conditions && (constTensor.shape.rank == 1 && constTensor.shape[0] == constTensors[l0.inputs[2]].shape[0]);
                }
            }
            else if (l0 is Layers.ScalarMad && l1 is Layers.Mul)
            {
                var lmad = l0 as Layers.ScalarMad;
                if (lmad.dataType != DataType.Float || lmad.bFloat != 0)
                    return false;
            }
            else if (l0 is Layers.Mul && l1 is Layers.ScalarMad)
            {
                var lmad = l1 as Layers.ScalarMad;
                if (lmad.dataType != DataType.Float || lmad.bFloat != 0)
                    return false;
            }
            else if (l0 is Layers.ScalarMad && l1 is Layers.Add)
            {
                var lmad = l0 as Layers.ScalarMad;
                if (lmad.dataType != DataType.Float || lmad.sFloat != 1)
                    return false;
            }
            else if (l0 is Layers.Add && l1 is Layers.ScalarMad)
            {
                var lmad = l1 as Layers.ScalarMad;
                if (lmad.dataType != DataType.Float || lmad.sFloat != 1)
                    return false;
            }
            else if (l0 is Layers.ScalarMad && l1 is Layers.Sub)
            {
                var lmad = l0 as Layers.ScalarMad;
                if (lmad.dataType != DataType.Float || lmad.sFloat != 1)
                    return false;
            }
            else if (l0 is Layers.Sub && l1 is Layers.ScalarMad)
            {
                var lmad = l1 as Layers.ScalarMad;
                if (lmad.dataType != DataType.Float || lmad.sFloat != 1)
                    return false;
            }

            return m_LayerFusers.ContainsKey((l0.GetType(), l1.GetType())) && conditions;
        }

        readonly CPUOps m_Ops = new CPUOps();

        readonly Dictionary<(Type, Type), Func<Layers.Layer, Layers.Layer, Dictionary<string, Layers.Constant>, Layers.Layer>> m_LayerFusers =
            new Dictionary<(Type, Type), Func<Layers.Layer, Layers.Layer, Dictionary<string, Layers.Constant>, Layers.Layer>>();

        public void Dispose()
        {
            m_Ops.Dispose();
        }

        void Add((Type, Type) layersType, Func<Layers.Layer, Layers.Layer, Dictionary<string, Layers.Constant>, Layers.Layer> opFuseAction)
        {
            m_LayerFusers.Add(layersType, opFuseAction);
        }

        public LinearLayerFusing()
        {
            Add((typeof(Layers.Add), typeof(Layers.Sub)), (l0, l1, constTensors) =>
            {
                using Tensor bias0 = constTensors.ContainsKey(l0.inputs[0]) ? constTensors[l0.inputs[0]].WeightsToTensor() : constTensors[l0.inputs[1]].WeightsToTensor();

                bool rightSub = constTensors.ContainsKey(l1.inputs[1]) ? true : false;

                using Tensor bias1 = rightSub ? constTensors[l1.inputs[1]].WeightsToTensor() : constTensors[l1.inputs[0]].WeightsToTensor() ;

                // rightsub  : (x+b0) - b1 = x + (b0-b1)
                // !rightsub : b1 - (x+b0) = (b1-b0) - x
                Tensor bias;
                if (bias0 is TensorInt)
                    bias = rightSub ? m_Ops.Sub(bias0 as TensorInt, bias1 as TensorInt) : m_Ops.Add(bias1 as TensorInt, bias0 as TensorInt);
                else
                    bias = rightSub ? m_Ops.Sub(bias0 as TensorFloat, bias1 as TensorFloat) : m_Ops.Add(bias1 as TensorFloat, bias0 as TensorFloat);

                Layers.Layer lmerged;
                if (rightSub)
                    lmerged = new Layers.Add(l0.index, l0.inputs[0], l0.inputs[1]);
                else
                    lmerged = new Layers.Sub(l0.index, l0.inputs[0], l0.inputs[1]);

                if (rightSub)
                    constTensors[lmerged.inputs[1]].TensorToDataSet(bias);
                else
                    constTensors[lmerged.inputs[0]].TensorToDataSet(bias);

                bias.Dispose();

                return lmerged;
            });
            Add((typeof(Layers.Sub), typeof(Layers.Add)), (l0, l1, constTensors) =>
            {
                using Tensor bias1 = constTensors.ContainsKey(l1.inputs[0]) ? constTensors[l1.inputs[0]].WeightsToTensor() : constTensors[l1.inputs[1]].WeightsToTensor();

                bool rightSub = constTensors.ContainsKey(l0.inputs[1]) ? true : false;
                using Tensor bias0 = rightSub ? constTensors[l0.inputs[1]].WeightsToTensor() : constTensors[l0.inputs[0]].WeightsToTensor();

                // rightsub  : (x-b0) + b1 = x + (b1-b0)
                // !rightsub : (b0-x) + b1 = (b0+b1) - x
                Tensor bias;
                if (bias0 is TensorInt)
                    bias = rightSub ? m_Ops.Sub(bias1 as TensorInt, bias0 as TensorInt) : m_Ops.Add(bias0 as TensorInt, bias1 as TensorInt);
                else
                    bias = rightSub ? m_Ops.Sub(bias1 as TensorFloat, bias0 as TensorFloat) : m_Ops.Add(bias0 as TensorFloat, bias1 as TensorFloat);

                Layers.Layer lmerged;
                if (rightSub)
                    lmerged = new Layers.Add(l0.index, l0.inputs[0], l0.inputs[1]);
                else
                    lmerged = new Layers.Sub(l0.index, l0.inputs[0], l0.inputs[1]);

                if (rightSub)
                    constTensors[lmerged.inputs[1]].TensorToDataSet(bias);
                else
                    constTensors[lmerged.inputs[0]].TensorToDataSet(bias);

                bias.Dispose();

                return lmerged;
            });
            Add((typeof(Layers.Add), typeof(Layers.Add)), (l0, l1, constTensors) =>
            {
                using Tensor bias0 = constTensors.ContainsKey(l0.inputs[0]) ? constTensors[l0.inputs[0]].WeightsToTensor() : constTensors[l0.inputs[1]].WeightsToTensor();
                using Tensor bias1 = constTensors.ContainsKey(l1.inputs[0]) ? constTensors[l1.inputs[0]].WeightsToTensor() : constTensors[l1.inputs[1]].WeightsToTensor();

                Tensor bias;
                if (bias0 is TensorInt)
                    bias = m_Ops.Add(bias0 as TensorInt, bias1 as TensorInt);
                else
                    bias = m_Ops.Add(bias0 as TensorFloat, bias1 as TensorFloat);

                Layers.Layer lmerged = new Layers.Add(l0.index, l0.inputs[0], l0.inputs[1]);

                if (constTensors.ContainsKey(lmerged.inputs[0]))
                    constTensors[lmerged.inputs[0]].TensorToDataSet(bias);
                else
                    constTensors[lmerged.inputs[1]].TensorToDataSet(bias);

                bias.Dispose();

                return lmerged;
            });
            Add((typeof(Layers.Mul), typeof(Layers.Mul)), (l0, l1, constTensors) =>
            {
                using Tensor scale0 = constTensors.ContainsKey(l0.inputs[0]) ? constTensors[l0.inputs[0]].WeightsToTensor() : constTensors[l0.inputs[1]].WeightsToTensor();
                using Tensor scale1 = constTensors.ContainsKey(l1.inputs[0]) ? constTensors[l1.inputs[0]].WeightsToTensor() : constTensors[l1.inputs[1]].WeightsToTensor();

                Tensor scale;
                if (scale0 is TensorInt)
                    scale = m_Ops.Mul(scale0 as TensorInt, scale1 as TensorInt);
                else
                    scale = m_Ops.Mul(scale0 as TensorFloat, scale1 as TensorFloat);

                Layers.Layer lmerged = new Layers.Mul(l0.index, l0.inputs[0], l0.inputs[1]);

                if(constTensors.ContainsKey(lmerged.inputs[0]))
                    constTensors[lmerged.inputs[0]].TensorToDataSet(scale);
                else
                    constTensors[lmerged.inputs[1]].TensorToDataSet(scale);

                scale.Dispose();

                return lmerged;
            });
            Add((typeof(Layers.ScaleBias), typeof(Layers.ScaleBias)), (l0, l1, constTensors) =>
            {
                using TensorFloat scale0 = constTensors[l0.inputs[1]].WeightsToTensor() as TensorFloat;
                using TensorFloat bias0 = constTensors[l0.inputs[2]].WeightsToTensor() as TensorFloat;

                using TensorFloat scale1 = constTensors[l1.inputs[1]].WeightsToTensor() as TensorFloat;
                using TensorFloat bias1 = constTensors[l1.inputs[2]].WeightsToTensor() as TensorFloat;

                Layers.Layer lmerged = new Layers.ScaleBias(l0.index, l0.inputs[0], l0.inputs[1], l0.inputs[2]);

                // s1*(s0*x + b0)+b1 = s1*s0*x + s1*b0+b1
                using TensorFloat scale = m_Ops.Mul(scale1, scale0);
                using TensorFloat mul = m_Ops.Mul(bias0, scale1);
                using TensorFloat bias = m_Ops.Add(mul, bias1);
                bias.shape = new TensorShape(bias.shape.length);

                constTensors[lmerged.inputs[1]].TensorToDataSet(scale);
                constTensors[lmerged.inputs[2]].TensorToDataSet(bias);

                return lmerged;
            });
            Add((typeof(Layers.ScaleBias), typeof(Layers.Dense)), (l0, l1, constTensors) =>
            {
                using TensorFloat scale0 = constTensors[l0.inputs[1]].WeightsToTensor() as TensorFloat;
                using TensorFloat bias0 = constTensors[l0.inputs[2]].WeightsToTensor() as TensorFloat;

                using TensorFloat weights1 = constTensors[l1.inputs[1]].WeightsToTensor() as TensorFloat;
                using TensorFloat bias1 = constTensors[l1.inputs[2]].WeightsToTensor() as TensorFloat;

                Layers.Layer lmerged = new Layers.Dense(l0.index, l0.inputs[0], l0.inputs[1], l0.inputs[2]);

                // b = W1 x b0 + b1``
                bias0.shape = new TensorShape(1, bias0.shape[0]);
                using TensorFloat bias = m_Ops.Dense(bias0, weights1, bias1);
                bias.shape = new TensorShape(bias.shape[1]);

                // W = W1 x s
                scale0.shape = new TensorShape(scale0.shape[0], 1);
                using TensorFloat weights = m_Ops.Mul(weights1, scale0);

                constTensors[lmerged.inputs[1]].TensorToDataSet(weights);
                constTensors[lmerged.inputs[2]].TensorToDataSet(bias);

                return lmerged;
            });
            Add((typeof(Layers.Mul), typeof(Layers.Conv)), (l0, l1, constTensors) =>
            {
                using TensorFloat scale0 = constTensors.ContainsKey(l0.inputs[0]) ? constTensors[l0.inputs[0]].WeightsToTensor() as TensorFloat : constTensors[l0.inputs[1]].WeightsToTensor() as TensorFloat;

                using TensorFloat kernel1 = constTensors[l1.inputs[1]].WeightsToTensor() as TensorFloat;
                using TensorFloat bias1 = l1.inputs.Length == 3 ? constTensors[l1.inputs[2]].WeightsToTensor() as TensorFloat : TensorFloat.AllocZeros(new TensorShape(kernel1.shape[0]));

                Layers.Layer lmerged = new Layers.Conv(l0.index, l0.inputs[0], l0.inputs[1], l0.inputs[2], (l1 as Layers.Conv).group, (l1 as Layers.Conv).strides, (l1 as Layers.Conv).pads, (l1 as Layers.Conv).dilations, (l1 as Layers.Conv).autoPad);
                // k = k * s
                using TensorFloat kernel = m_Ops.Mul(kernel1, scale0);

                constTensors[lmerged.inputs[1]].TensorToDataSet(kernel);
                constTensors[lmerged.inputs[2]].TensorToDataSet(bias1);

                return lmerged;
            });
            Add((typeof(Layers.Conv), typeof(Layers.Mul)), (l0, l1, constTensors) =>
            {
                using TensorFloat kernel0 = constTensors[l0.inputs[1]].WeightsToTensor() as TensorFloat;
                bool convHasBias = l0.inputs.Length == 3;
                using TensorFloat bias0 = convHasBias ? constTensors[l0.inputs[2]].WeightsToTensor() as TensorFloat : TensorFloat.AllocZeros(new TensorShape(kernel0.shape[0]));

                using TensorFloat scale1 = constTensors.ContainsKey(l1.inputs[0]) ? constTensors[l1.inputs[0]].WeightsToTensor() as TensorFloat : constTensors[l1.inputs[1]].WeightsToTensor() as TensorFloat;
                string biasName = convHasBias ? l0.inputs[2] : constTensors.ContainsKey(l1.inputs[0]) ? l1.inputs[0] : l1.inputs[1];

                Layers.Layer lmerged = new Layers.Conv(l0.index, l0.inputs[0], l0.inputs[1], biasName, (l0 as Layers.Conv).group, (l0 as Layers.Conv).strides, (l0 as Layers.Conv).pads, (l0 as Layers.Conv).dilations, (l0 as Layers.Conv).autoPad);

                // k = s1*k0
                using TensorFloat kernel = m_Ops.Mul(scale1, kernel0);
                // b = s1*b0
                using TensorFloat bias = m_Ops.Mul(scale1, bias0);

                constTensors[lmerged.inputs[1]].TensorToDataSet(kernel);
                constTensors[lmerged.inputs[2]].TensorToDataSet(bias);

                return lmerged;
            });
            Add((typeof(Layers.Add), typeof(Layers.Conv)), (l0, l1, constTensors) =>
            {
                using TensorFloat bias0 = constTensors.ContainsKey(l0.inputs[0]) ? constTensors[l0.inputs[0]].WeightsToTensor() as TensorFloat : constTensors[l0.inputs[1]].WeightsToTensor() as TensorFloat;

                using TensorFloat kernel1 = constTensors[l1.inputs[1]].WeightsToTensor() as TensorFloat;

                Layers.Layer lmerged = new Layers.Conv(l0.index, l0.inputs[0], l0.inputs[1], l0.inputs[2], (l1 as Layers.Conv).group, (l1 as Layers.Conv).strides, (l1 as Layers.Conv).pads, (l1 as Layers.Conv).dilations, (l1 as Layers.Conv).autoPad);

                // k = k
                // b = Sum_k[wk * beta] + b
                using TensorFloat bias = l1.inputs.Length == 3 ? constTensors[l1.inputs[2]].WeightsToTensor() as TensorFloat : TensorFloat.AllocZeros(new TensorShape(kernel1.shape[0]));

                var itK = new TensorNDIterator(kernel1.shape);
                itK = itK.RemoveDim(0);
                itK = itK.RemoveDim(0);

                for (int c = 0; c < kernel1.shape[1]; ++c)
                {
                    float beta = bias0[c % bias0.shape[-1]];

                    itK.Reset();
                    for (; itK.HasNext(); itK.MoveNext())
                    {
                        for (int k = 0; k < kernel1.shape[0]; ++k)
                        {
                            float w = kernel1[k * kernel1.shape[1] * itK.shape.length + c * itK.shape.length + itK.index];
                            bias[k] += w * beta;
                        }
                    }
                }

                constTensors[lmerged.inputs[1]].TensorToDataSet(kernel1);
                constTensors[lmerged.inputs[2]].TensorToDataSet(bias);

                return lmerged;
            });
            Add((typeof(Layers.Conv), typeof(Layers.Add)), (l0, l1, constTensors) =>
            {
                using TensorFloat kernel0 = constTensors[l0.inputs[1]].WeightsToTensor() as TensorFloat;

                bool convHasBias = l0.inputs.Length == 3;
                using TensorFloat bias0 = convHasBias ? constTensors[l0.inputs[2]].WeightsToTensor() as TensorFloat : TensorFloat.AllocZeros(new TensorShape(kernel0.shape[0]));

                using TensorFloat bias1 = constTensors.ContainsKey(l1.inputs[0]) ? constTensors[l1.inputs[0]].WeightsToTensor() as TensorFloat : constTensors[l1.inputs[1]].WeightsToTensor() as TensorFloat;
                string biasName = convHasBias ? l0.inputs[2] : constTensors.ContainsKey(l1.inputs[0]) ? l1.inputs[0] : l1.inputs[1];
                Layers.Layer lmerged = new Layers.Conv(l0.index, l0.inputs[0], l0.inputs[1], biasName, (l0 as Layers.Conv).group, (l0 as Layers.Conv).strides, (l0 as Layers.Conv).pads, (l0 as Layers.Conv).dilations, (l0 as Layers.Conv).autoPad);

                // b = b0+b1
                using TensorFloat bias = m_Ops.Add(bias0, bias1);

                constTensors[lmerged.inputs[1]].TensorToDataSet(kernel0);
                constTensors[lmerged.inputs[2]].TensorToDataSet(bias);

                return lmerged;
            });
            Add((typeof(Layers.Conv), typeof(Layers.ScaleBias)), (l0, l1, constTensors) =>
            {
                using TensorFloat kernel0 = constTensors[l0.inputs[1]].WeightsToTensor() as TensorFloat;
                bool convHasBias = l0.inputs.Length == 3;
                using TensorFloat bias0 = convHasBias ? constTensors[l0.inputs[2]].WeightsToTensor() as TensorFloat : TensorFloat.AllocZeros(new TensorShape(kernel0.shape[0]));

                using TensorFloat scale1 = constTensors[l1.inputs[1]].WeightsToTensor() as TensorFloat;
                using TensorFloat bias1 = constTensors[l1.inputs[2]].WeightsToTensor() as TensorFloat;

                // k = s1*k0
                using TensorFloat kernel = new TensorFloat(kernel0.shape, new float[kernel0.shape.length]);
                for(int i = 0; i < kernel0.shape.length; i++)
                {
                    kernel[i] = kernel0[i] * scale1[i / kernel0.shape.Length(1)];
                }
                // b = s1*b0+b1
                using TensorFloat mul = m_Ops.Mul(bias0, scale1);
                using TensorFloat bias = m_Ops.Add(mul, bias1);
                bias.shape = new TensorShape(bias.shape.length);

                string nameBias = convHasBias ? l0.inputs[2] : l1.inputs[2];
                Layers.Layer lmerged = new Layers.Conv(l0.index, l0.inputs[0], l0.inputs[1], nameBias, (l0 as Layers.Conv).group, (l0 as Layers.Conv).strides, (l0 as Layers.Conv).pads, (l0 as Layers.Conv).dilations, (l0 as Layers.Conv).autoPad);

                constTensors[lmerged.inputs[1]].TensorToDataSet(kernel);
                constTensors[lmerged.inputs[2]].TensorToDataSet(bias);

                return lmerged;
            });
            Add((typeof(Layers.ScaleBias), typeof(Layers.Conv)), (l0, l1, constTensors) =>
            {
                using TensorFloat scale0 = constTensors[l0.inputs[1]].WeightsToTensor() as TensorFloat;
                using TensorFloat bias0 = constTensors[l0.inputs[2]].WeightsToTensor() as TensorFloat;

                using TensorFloat kernel1 = constTensors[l1.inputs[1]].WeightsToTensor() as TensorFloat;

                Layers.Layer lmerged = new Layers.Conv(l0.index, l0.inputs[0], l0.inputs[1], l0.inputs[2], (l1 as Layers.Conv).group, (l1 as Layers.Conv).strides, (l1 as Layers.Conv).pads, (l1 as Layers.Conv).dilations, (l1 as Layers.Conv).autoPad);

                // k = k * s
                using TensorFloat kernel = new TensorFloat(kernel1.shape, new float[kernel1.shape.length]);
                // b = Sum_k[wk * beta] + b
                using TensorFloat bias = (l1.inputs.Length == 3) ? constTensors[l1.inputs[2]].WeightsToTensor() as TensorFloat : TensorFloat.AllocZeros(new TensorShape(kernel1.shape[0]));

                var itK = new TensorNDIterator(kernel1.shape);
                itK = itK.RemoveDim(0);
                itK = itK.RemoveDim(0);

                for (int c = 0; c < kernel1.shape[1]; ++c)
                {
                    float beta = bias0[c];
                    float gamma = scale0[c];

                    itK.Reset();
                    for (; itK.HasNext(); itK.MoveNext())
                    {
                        for (int k = 0; k < kernel1.shape[0]; ++k)
                        {
                            int indexk = k * kernel1.shape[1] * itK.shape.length + c * itK.shape.length + itK.index;
                            float w = kernel1[indexk];
                            kernel[indexk] = gamma * w;
                            bias[k] += w * beta;
                        }
                    }
                }

                constTensors[lmerged.inputs[1]].TensorToDataSet(kernel);
                constTensors[lmerged.inputs[2]].TensorToDataSet(bias);

                return lmerged;
            });
            Add((typeof(Layers.Dense), typeof(Layers.Dense)), (l0, l1, constTensors) =>
            {
                using TensorFloat weights0 = constTensors[l0.inputs[1]].WeightsToTensor() as TensorFloat;
                using TensorFloat bias0 = constTensors[l0.inputs[2]].WeightsToTensor() as TensorFloat;

                using TensorFloat weights1 = constTensors[l1.inputs[1]].WeightsToTensor() as TensorFloat;
                using TensorFloat bias1 = constTensors[l1.inputs[2]].WeightsToTensor() as TensorFloat;

                // W = W1 x W0
                using TensorFloat weights = m_Ops.MatMul2D(weights0, weights1, false, false);
                // b = W1 x b0 + b1
                bias0.shape = new TensorShape(1, bias0.shape[0]);
                using TensorFloat bias = m_Ops.Dense(bias0, weights1, bias1);
                bias.shape = new TensorShape(bias.shape[1]);

                Layers.Layer lmerged = new Layers.Dense(l0.index, l0.inputs[0], l0.inputs[1], l0.inputs[2]);

                constTensors[lmerged.inputs[1]].TensorToDataSet(weights);
                constTensors[lmerged.inputs[2]].TensorToDataSet(bias);

                return lmerged;
            });
            Add((typeof(Layers.ScalarMad), typeof(Layers.ScalarMad)), (l0, l1, constTensors) =>
            {
                var madLayer0 = l0 as Layers.ScalarMad;
                var madLayer1 = l1 as Layers.ScalarMad;

                if (madLayer0.dataType == DataType.Int)
                    return new Layers.ScalarMad(l0.index, l0.inputs[0], madLayer1.sInt * madLayer0.sInt, madLayer1.sInt * madLayer0.bInt + madLayer1.bInt);
                return new Layers.ScalarMad(l0.index, l0.inputs[0], madLayer1.sFloat * madLayer0.sFloat, madLayer1.sFloat * madLayer0.bFloat + madLayer1.bFloat);
            });
            Add((typeof(Layers.ScalarMad), typeof(Layers.Mul)), (l0, l1, constTensors) =>
            {
                var madLayer0 = l0 as Layers.ScalarMad;
                var scale0 = madLayer0.sFloat;
                using Tensor scale1 = constTensors.ContainsKey(l1.inputs[0]) ? constTensors[l1.inputs[0]].WeightsToTensor() : constTensors[l1.inputs[1]].WeightsToTensor();

                Tensor scale;
                scale = m_Ops.ScalarMad(scale1 as TensorFloat, scale0, 0);

                Layers.Layer lmerged = new Layers.Mul(l0.index, l0.inputs[0], constTensors.ContainsKey(l1.inputs[0]) ? l1.inputs[0] : l1.inputs[1]);

                constTensors[lmerged.inputs[1]].TensorToDataSet(scale);

                scale.Dispose();

                return lmerged;
            });
            Add((typeof(Layers.Mul), typeof(Layers.ScalarMad)), (l0, l1, constTensors) =>
            {
                using Tensor scale0 = constTensors.ContainsKey(l0.inputs[0]) ? constTensors[l0.inputs[0]].WeightsToTensor() : constTensors[l0.inputs[1]].WeightsToTensor();
                var madLayer1 = l1 as Layers.ScalarMad;
                var scale1 = madLayer1.sFloat;

                Tensor scale;
                scale = m_Ops.ScalarMad(scale0 as TensorFloat, scale1, 0);

                Layers.Layer lmerged = new Layers.Mul(l0.index, l0.inputs[0], l0.inputs[1]);

                if(constTensors.ContainsKey(lmerged.inputs[0]))
                    constTensors[lmerged.inputs[0]].TensorToDataSet(scale);
                else
                    constTensors[lmerged.inputs[1]].TensorToDataSet(scale);

                scale.Dispose();

                return lmerged;
            });
            Add((typeof(Layers.ScalarMad), typeof(Layers.Add)), (l0, l1, constTensors) =>
            {
                var madLayer0 = l0 as Layers.ScalarMad;
                var bias0 = madLayer0.bFloat;
                using Tensor bias1 = constTensors.ContainsKey(l1.inputs[0]) ? constTensors[l1.inputs[0]].WeightsToTensor() : constTensors[l1.inputs[1]].WeightsToTensor();

                Tensor bias;
                bias = m_Ops.ScalarMad(bias1 as TensorFloat, 1, bias0);

                Layers.Layer lmerged = new Layers.Add(l0.index, l0.inputs[0], constTensors.ContainsKey(l1.inputs[0]) ? l1.inputs[0] : l1.inputs[1]);

                constTensors[lmerged.inputs[1]].TensorToDataSet(bias);

                bias.Dispose();

                return lmerged;
            });
            Add((typeof(Layers.Add), typeof(Layers.ScalarMad)), (l0, l1, constTensors) =>
            {
                using Tensor bias0 = constTensors.ContainsKey(l0.inputs[0]) ? constTensors[l0.inputs[0]].WeightsToTensor() : constTensors[l0.inputs[1]].WeightsToTensor();
                var madLayer1 = l1 as Layers.ScalarMad;
                var bias1 = madLayer1.bFloat;

                Tensor bias;
                bias = m_Ops.ScalarMad(bias0 as TensorFloat, 1, bias1);

                Layers.Layer lmerged = new Layers.Add(l0.index, l0.inputs[0], l0.inputs[1]);

                if(constTensors.ContainsKey(lmerged.inputs[0]))
                    constTensors[lmerged.inputs[0]].TensorToDataSet(bias);
                else
                    constTensors[lmerged.inputs[1]].TensorToDataSet(bias);

                bias.Dispose();

                return lmerged;
            });
            Add((typeof(Layers.ScalarMad), typeof(Layers.Sub)), (l0, l1, constTensors) =>
            {
                var madLayer0 = l0 as Layers.ScalarMad;
                var bias0 = madLayer0.bFloat;
                bias0 = constTensors.ContainsKey(l1.inputs[0]) ? bias0 : -bias0;
                using Tensor bias1 = constTensors.ContainsKey(l1.inputs[0]) ? constTensors[l1.inputs[0]].WeightsToTensor() : constTensors[l1.inputs[1]].WeightsToTensor();

                Tensor bias;
                bias = m_Ops.ScalarMad(bias1 as TensorFloat, 1, constTensors.ContainsKey(l1.inputs[0]) ? -bias0 : bias0);

                Layers.Layer lmerged = constTensors.ContainsKey(l1.inputs[0]) ? new Layers.Sub(l0.index, l1.inputs[0], l0.inputs[0]) : new Layers.Sub(l0.index, l0.inputs[0], l1.inputs[1]);

                if(constTensors.ContainsKey(lmerged.inputs[0]))
                    constTensors[lmerged.inputs[0]].TensorToDataSet(bias);
                else
                    constTensors[lmerged.inputs[1]].TensorToDataSet(bias);

                bias.Dispose();

                return lmerged;
            });
            Add((typeof(Layers.Sub), typeof(Layers.ScalarMad)), (l0, l1, constTensors) =>
            {
                using Tensor bias0 = constTensors.ContainsKey(l0.inputs[0]) ? constTensors[l0.inputs[0]].WeightsToTensor() : constTensors[l0.inputs[1]].WeightsToTensor();
                var madLayer1 = l1 as Layers.ScalarMad;
                var bias1 = madLayer1.bFloat;
                bias1 = constTensors.ContainsKey(l0.inputs[0]) ? bias1 : -bias1;

                Tensor bias;
                bias = m_Ops.ScalarMad(bias0 as TensorFloat, 1, bias1);

                Layers.Layer lmerged = new Layers.Sub(l0.index, l0.inputs[0], l0.inputs[1]);

                if(constTensors.ContainsKey(lmerged.inputs[0]))
                    constTensors[lmerged.inputs[0]].TensorToDataSet(bias);
                else
                    constTensors[lmerged.inputs[1]].TensorToDataSet(bias);

                bias.Dispose();

                return lmerged;
            });
        }

        public Layers.Layer FuseLayers(Layers.Layer l0, Layers.Layer l1, Dictionary<string, Layers.Constant> constTensors)
        {
            var fnFuse = m_LayerFusers[(l0.GetType(), l1.GetType())];
            return fnFuse(l0, l1, constTensors);
        }
    }
} // namespace Unity.Sentis

