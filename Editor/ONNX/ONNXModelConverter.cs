using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Google.Protobuf;
using Onnx;

[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.ONNX")]
[assembly: InternalsVisibleTo("Unity.Sentis.Editor")]

namespace Unity.Sentis.ONNX
{
    /// <summary>
    /// Represents a converter from an ONNX model to Sentis format.
    /// </summary>
    public class ONNXModelConverter
    {
        // Configuration
        string m_DirectoryPath;
        string m_FilePath;

        /// <summary>
        /// Converts an ONNX model to a Sentis `Model` object.
        /// </summary>
        /// <returns>The converted Sentis model.</returns>
        public Model Convert()
        {
            using var readStream = new FileStream(m_FilePath, FileMode.Open, FileAccess.Read);
            using var inputStream = new CodedInputStream(readStream);

            var onnxModel = new ModelProto();
            onnxModel.MergeFrom(inputStream);

            Model model = null;
            model = ConvertOnnxModel(onnxModel);

#if UNITY_EDITOR && UNITY_2023_2_OR_NEWER && ENABLE_CLOUD_SERVICES_ANALYTICS
            var data = new SentisAnalytics.Data()
            {
                allOperators = model.layers.Select(l => l.profilerTag).Distinct().ToArray(),
                importWarningSeverity = Warnings.Select(w => (int)w.MessageSeverity).ToArray(),
                importWarningMessages = Warnings.Select(w => w.Message).ToArray(),
                modelLayerCount = model.layers.Count,
            };
            SentisAnalytics.SendEvent(data);
#endif

            return model;
        }

        /// <summary>
        /// Initializes and returns an instance of `ONNXModelConverter`.
        /// </summary>
        /// <param name="filePath">The path of the asset to convert.</param>
        public ONNXModelConverter(string filePath)
        {
            m_FilePath = filePath;
            m_DirectoryPath = Path.GetDirectoryName(m_FilePath);
        }

        void OnNode(Model model, long defaultOpsetVersion, ONNXNodeWrapper node)
        {
            var opType = node.OperatorType;
            if (opType == "Constant")
            {
                node.UnsupportedAttribute("sparse_value");
                var constant = ONNXConstantsLoader.LoadConstant(node.GetRequiredTensor("value"), m_DirectoryPath);
                constant.index = node.Name;
                model.AddConstant(constant);
            }
            // Layer.Activation
            else if (opType == "Celu")
            {
                var alpha = node.GetOptionalFloat("alpha", 1f);
                model.AddLayer(new Layers.Celu(node.Name, node.Input0, alpha));
            }
            else if (opType == "Elu")
            {
                var alpha = node.GetOptionalFloat("alpha", 1f);
                model.AddLayer(new Layers.Elu(node.Name, node.Input0, alpha));
            }
            else if (opType == "Erf")
            {
                model.AddLayer(new Layers.Erf(node.Name, node.Input0));
            }
            else if (opType == "Gelu")
            {
                model.AddLayer(new Layers.Gelu(node.Name, node.Input0));
            }
            else if (opType == "Hardmax")
            {
                var axis = node.GetOptionalInt("axis", -1);
                model.AddLayer(new Layers.Hardmax(node.Name, node.Input0, axis));
            }
            else if (opType == "HardSigmoid")
            {
                var alpha = node.GetOptionalFloat("alpha", 0.2f);
                var beta = node.GetOptionalFloat("beta", 0.5f);
                model.AddLayer(new Layers.HardSigmoid(node.Name, node.Input0, alpha, beta));
            }
            else if (opType == "HardSwish")
            {
                model.AddLayer(new Layers.HardSwish(node.Name, node.Input0));
            }
            else if (opType == "LeakyRelu")
            {
                var alpha = node.GetOptionalFloat("alpha", 0.01f);
                model.AddLayer(new Layers.LeakyRelu(node.Name, node.Input0, alpha));
            }
            else if (opType == "PRelu")
            {
                model.AddLayer(new Layers.PRelu(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Relu")
            {
                model.AddLayer(new Layers.Relu(node.Name, node.Input0));
            }
            else if (opType == "Selu")
            {
                var alpha = node.GetOptionalFloat("alpha", defaultOpsetVersion < 6 ? 1.6732f : 1.67326319f);
                var gamma = node.GetOptionalFloat("gamma", defaultOpsetVersion < 6 ? 1.0507f : 1.05070102f);
                model.AddLayer(new Layers.Selu(node.Name, node.Input0, alpha, gamma));
            }
            else if (opType == "Sigmoid")
            {
                model.AddLayer(new Layers.Sigmoid(node.Name, node.Input0));
            }
            else if (opType == "Softplus")
            {
                model.AddLayer(new Layers.Softplus(node.Name, node.Input0));
            }
            else if (opType == "Softsign")
            {
                model.AddLayer(new Layers.Softsign(node.Name, node.Input0));
            }
            else if (opType == "Tanh")
            {
                model.AddLayer(new Layers.Tanh(node.Name, node.Input0));
            }
            else if (opType == "ThresholdedRelu")
            {
                var alpha = node.GetOptionalFloat("alpha", 1f);
                model.AddLayer(new Layers.ThresholdedRelu(node.Name, node.Input0, alpha));
            }
            // Layer.ActivationNonLinear
            else if (opType == "LogSoftmax")
            {
                var axis = node.GetOptionalInt("axis", -1);
                model.AddLayer(new Layers.LogSoftmax(node.Name, node.Input0, axis));
            }
            else if (opType == "Softmax")
            {
                var axis = node.GetOptionalInt("axis", -1);
                model.AddLayer(new Layers.Softmax(node.Name, node.Input0, axis));
            }
            // Layer.Convolution
            else if (opType == "Conv")
            {
                // Conv-1, Conv-11

                var autoPad = node.AutoPadMode();
                var kernelShape = node.GetOptionalIntArray("kernel_shape", null);
                var dilations = node.GetOptionalIntArray("dilations", null);
                var group = node.GetOptionalInt("group", 1);
                var pads = node.GetOptionalIntArray("pads", null);
                var strides = node.GetOptionalIntArray("strides", null);

                model.AddLayer(new Layers.Conv(node.Name, node.Input0, node.Input1, node.OptionalInput(2), group, strides, pads, dilations, autoPad, kernelShape));
            }
            else if (opType == "ConvTranspose")
            {
                // ConvTranspose-1, ConvTranspose-11

                node.UnsupportedAttribute("output_shape", "null");

                var outputPadding = node.GetOptionalIntArray("output_padding", null);
                var autoPad = node.AutoPadMode();
                var kernelShape = node.GetOptionalIntArray("kernel_shape", null);
                node.UnsupportedAttribute("dilations", "null");
                node.UnsupportedAttribute("group", 1);
                var pads = node.GetOptionalIntArray("pads", null);
                var strides = node.GetOptionalIntArray("strides", null);

                model.AddLayer(new Layers.ConvTranspose(node.Name, node.Input0, node.Input1, node.OptionalInput(2), strides, pads, autoPad, outputPadding, kernelShape));
            }
            // Layer.Dimension
            else if (opType == "Shape")
            {
                // Shape-1, Shape-13, Shape-15
                var start = node.GetOptionalInt("start", 0);
                var end = node.GetOptionalInt("end", TensorShape.maxRank);
                model.AddLayer(new Layers.Shape(node.Name, node.Input0, start, end));
            }
            else if (opType == "Size")
            {
                // Size-1, Size-13
                model.AddLayer(new Layers.Size(node.Name, node.Input0));
            }
            // Layer.Generator
            else if (opType == "ConstantOfShape")
            {
                UnityEngine.Debug.Assert(node.InputCount > 0);

                if (!node.HasAttribute("value"))
                {
                    model.AddLayer(new Layers.ConstantOfShape(node.Name, node.Input0, 0.0f));
                    return;
                }

                var constant = ONNXConstantsLoader.LoadConstant(node.GetRequiredTensor("value"), m_DirectoryPath);
                if (constant.dataType == DataType.Int)
                {
                    var value = constant.weights.Get<int>(0);
                    model.AddLayer(new Layers.ConstantOfShape(node.Name, node.Input0, value));
                }
                else
                {
                    var value = constant.weights.Get<float>(0);
                    model.AddLayer(new Layers.ConstantOfShape(node.Name, node.Input0, value));
                }
                constant.weights.Dispose();
            }
            else if (opType == "Range")
            {
                model.AddLayer(new Layers.Range(node.Name, node.Input0, node.Input1, node.Input2));
            }
            else if (opType == "OneHot")
            {
                // OneHot-9, OneHot-11
                var axis = node.GetOptionalInt("axis", -1);
                model.AddLayer(new Layers.OneHot(node.Name, node.Input0, node.Input1, node.Input2, axis));
            }
            // Layer.Indexing
            else if (opType == "ArgMax")
            {
                var axis = node.GetOptionalInt("axis", 0);
                var keepdims = node.GetOptionalInt("keepdims", 1) == 1;
                var selectLastIndex = node.GetOptionalInt("select_last_index", 0) == 1;
                model.AddLayer(new Layers.ArgMax(node.Name, node.Input0, axis, keepdims, selectLastIndex));
            }
            else if (opType == "ArgMin")
            {
                var axis = node.GetOptionalInt("axis", 0);
                var keepdims = node.GetOptionalInt("keepdims", 1) == 1;
                var selectLastIndex = node.GetOptionalInt("select_last_index", 0) == 1;
                model.AddLayer(new Layers.ArgMin(node.Name, node.Input0, axis, keepdims, selectLastIndex));
            }
            else if (opType == "Gather")
            {
                var axis = node.GetOptionalInt("axis", 0);
                model.AddLayer(new Layers.Gather(node.Name, node.Input0, node.Input1, axis));
            }
            else if (opType == "GatherElements")
            {
                var axis = node.GetOptionalInt("axis", 0);
                model.AddLayer(new Layers.GatherElements(node.Name, node.Input0, node.Input1, axis));
            }
            else if (opType == "GatherND")
            {
                var batchDims = node.GetOptionalInt("batch_dims", 0);
                model.AddLayer(new Layers.GatherND(node.Name, node.Input0, node.Input1, batchDims));
            }
            else if (opType == "NonZero")
            {
                model.AddLayer(new Layers.NonZero(node.Name, node.Input0));
            }
            else if (opType == "Scatter")
            {
                // Scatter-9 maps to ScatterElements
                var axis = node.GetOptionalInt("axis", 0);
                model.AddLayer(new Layers.ScatterElements(node.Name, node.Input0, node.Input1, node.Input2, axis, Layers.ScatterReductionMode.None));
            }
            else if (opType == "ScatterElements")
            {
                var axis = node.GetOptionalInt("axis", 0);
                var reduction = node.ScatterReductionMode();
                model.AddLayer(new Layers.ScatterElements(node.Name, node.Input0, node.Input1, node.Input2, axis, reduction));
            }
            else if (opType == "ScatterND")
            {
                var reduction = node.ScatterReductionMode();
                model.AddLayer(new Layers.ScatterND(node.Name, node.Input0, node.Input1, node.Input2, reduction));
            }
            else if (opType == "TopK")
            {
                string[] outputs = { node.Outputs[0], node.Outputs[1] };
                var axis = node.GetOptionalInt("axis", -1);
                var largest = node.GetOptionalInt("largest", 1) == 1;
                var sorted = node.GetOptionalInt("sorted", 1) == 1;
                if (defaultOpsetVersion < 10)
                {
                    // TopK-1
                    var k = node.GetRequiredInt("k");
                    var kConstant = new Layers.Constant(model.GetUniqueIndex(node.Name + "_k"), new[] { k });
                    model.AddConstant(kConstant);
                    model.AddLayer(new Layers.TopK(node.Output0, node.Output1, node.Input0, kConstant.index, axis, largest, sorted));
                }
                else
                {
                    // TopK-10, TopK-11
                    model.AddLayer(new Layers.TopK(node.Output0, node.Output1, node.Input0, node.Input1, axis, largest, sorted));
                }
            }
            // Layer.Logical
            else if (opType == "And")
            {
                model.AddLayer(new Layers.And(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Compress")
            {
                int? axis = node.HasAttribute("axis") ? node.GetRequiredInt("axis") : null;
                model.AddLayer(new Layers.Compress(node.Name, node.Input0, node.Input1, axis));
            }
            else if (opType == "Equal")
            {
                model.AddLayer(new Layers.Equal(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Greater")
            {
                model.AddLayer(new Layers.Greater(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "GreaterOrEqual")
            {
                model.AddLayer(new Layers.GreaterOrEqual(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "IsInf")
            {
                var detectNegative = node.GetOptionalInt("detect_negative", 1) != 0;
                var detectPositive = node.GetOptionalInt("detect_positive", 1) != 0;
                model.AddLayer(new Layers.IsInf(node.Name, node.Input0, detectNegative, detectPositive));
            }
            else if (opType == "IsNaN")
            {
                model.AddLayer(new Layers.IsNaN(node.Name, node.Input0));
            }
            else if (opType == "Less")
            {
                model.AddLayer(new Layers.Less(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "LessOrEqual")
            {
                model.AddLayer(new Layers.LessOrEqual(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Not")
            {
                model.AddLayer(new Layers.Not(node.Name, node.Input0));
            }
            else if (opType == "Or")
            {
                model.AddLayer(new Layers.Or(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Xor")
            {
                model.AddLayer(new Layers.Xor(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Where")
            {
                model.AddLayer(new Layers.Where(node.Name, node.Input0, node.Input1, node.Input2));
            }
            // Layer.Math
            else if (opType == "Abs")
            {
                model.AddLayer(new Layers.Abs(node.Name, node.Input0));
            }
            else if (opType == "Add")
            {
                model.AddLayer(new Layers.Add(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Ceil")
            {
                model.AddLayer(new Layers.Ceil(node.Name, node.Input0));
            }
            else if (opType == "Clip")
            {
                if (defaultOpsetVersion < 11)
                {
                    // Clip-1, Clip-6
                    var min = node.GetOptionalFloat("min", float.MinValue);
                    var minConstant = new Layers.Constant(model.GetUniqueIndex(node.Name + "_min"), new[] { min });
                    model.AddConstant(minConstant);
                    var max = node.GetOptionalFloat("max", float.MaxValue);
                    var maxConstant = new Layers.Constant(model.GetUniqueIndex(node.Name + "_max"), new[] { max });
                    model.AddConstant(maxConstant);
                    model.AddLayer(new Layers.Clip(node.Name, node.Input0, minConstant.index, maxConstant.index));
                }
                else
                {
                    // Clip-11, Clip-12, Clip-13 or Clip-1, Clip-6 with no min or max
                    model.AddLayer(new Layers.Clip(node.Name, node.Input0, node.OptionalInput(1), node.OptionalInput(2)));
                }
            }
            else if (opType == "CumSum")
            {
                var reverse = node.GetOptionalInt("reverse", 0) == 1;
                var exclusive = node.GetOptionalInt("exclusive", 0) == 1;
                model.AddLayer(new Layers.CumSum(node.Name, node.Input0, node.Input1, reverse, exclusive));
            }
            else if (opType == "Div")
            {
                model.AddLayer(new Layers.Div(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Einsum")
            {
                model.AddLayer(new Layers.Einsum(node.Name, node.Inputs, node.GetRequiredString("equation")));
            }
            else if (opType == "Exp")
            {
                model.AddLayer(new Layers.Exp(node.Name, node.Input0));
            }
            else if (opType == "Floor")
            {
                model.AddLayer(new Layers.Floor(node.Name, node.Input0));
            }
            else if (opType == "Gemm")
            {
                var transposeA = node.GetOptionalInt("transA", 0) == 1;
                var transposeB = node.GetOptionalInt("transB", 0) == 1;

                var alpha = node.GetOptionalFloat("alpha", 1.0f);
                var scalarMadA = model.GetUniqueIndex(node.Input0 + "_ScalarMad");
                model.AddLayer(new Layers.ScalarMad(scalarMadA, node.Input0, alpha, 0));

                var name = node.Name;
                var matMulName = name;

                var hasC = node.InputCount == 3 && !string.IsNullOrEmpty(node.Inputs[2]);
                if (hasC)
                    matMulName = model.GetUniqueIndex(name + "_MatMul");

                model.AddLayer(new Layers.MatMul2D(matMulName, scalarMadA, transposeA, node.Input1, transposeB));

                if (hasC)
                {
                    var input2 = node.Input2;
                    var beta = node.GetOptionalFloat("beta", 1.0f);
                    var scalarMadC = model.GetUniqueIndex(node.Input2 + "_ScalarMad");
                    model.AddLayer(new Layers.ScalarMad(scalarMadC, input2, beta, 0));
                    model.AddLayer(new Layers.Add(name, matMulName, scalarMadC));
                }
            }
            else if (opType == "Log")
            {
                model.AddLayer(new Layers.Log(node.Name, node.Input0));
            }
            else if (opType == "MatMul")
            {
                model.AddLayer(new Layers.MatMul(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Max")
            {
                model.AddLayer(new Layers.Max(node.Name, node.Inputs));
            }
            else if (opType == "Mean")
            {
                model.AddLayer(new Layers.Mean(node.Name, node.Inputs));
            }
            else if (opType == "Min")
            {
                model.AddLayer(new Layers.Min(node.Name, node.Inputs));
            }
            else if (opType == "Mod")
            {
                model.AddLayer(new Layers.Mod(node.Name, node.Input0, node.Input1, node.GetOptionalInt("fmod", 0) != 0));
            }
            else if (opType == "Mul")
            {
                model.AddLayer(new Layers.Mul(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Neg")
            {
                model.AddLayer(new Layers.Neg(node.Name, node.Input0));
            }
            else if (opType == "Pow")
            {
                // Pow-1, Pow-7, Pow-12, Pow-13
                model.AddLayer(new Layers.Pow(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Reciprocal")
            {
                model.AddLayer(new Layers.Reciprocal(node.Name, node.Input0));
            }
            else if (opType == "Round")
            {
                model.AddLayer(new Layers.Round(node.Name, node.Input0));
            }
            else if (opType == "Shrink")
            {
                model.AddLayer(new Layers.Shrink(node.Name, node.Input0, node.GetOptionalFloat("bias", 0f), node.GetOptionalFloat("lambd", 0.5f)));
            }
            else if (opType == "Sign")
            {
                model.AddLayer(new Layers.Sign(node.Name, node.Input0));
            }
            else if (opType == "Sqrt")
            {
                model.AddLayer(new Layers.Sqrt(node.Name, node.Input0));
            }
            else if (opType == "Sub")
            {
                model.AddLayer(new Layers.Sub(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Sum")
            {
                model.AddLayer(new Layers.Sum(node.Name, node.Inputs));
            }
            // Layer.Normalization
            else if (opType == "BatchNormalization")
            {
                var epsilon = node.GetOptionalFloat("epsilon", 1e-5f);
                model.AddLayer(new Layers.BatchNormalization(node.Name, node.Input0, node.Input1, node.Input2, node.Input3, node.Input4, epsilon));
            }
            else if (opType == "InstanceNormalization")
            {
                var epsilon = node.GetOptionalFloat("epsilon", 1e-5f);
                model.AddLayer(new Layers.InstanceNormalization(node.Name, node.Input0, node.Input1, node.Input2, epsilon));
            }
            else if (opType == "LayerNormalization")
            {
                var epsilon = node.GetOptionalFloat("epsilon", 1e-5f);
                node.UnsupportedAttribute("axis", -1);
                model.AddLayer(new Layers.LayerNormalization(node.Name, node.Input0, node.Input1, node.OptionalInput(2), epsilon));
            }
            else if (opType == "LRN")
            {
                var alpha = node.GetOptionalFloat("alpha", 0.0001f);
                var beta = node.GetOptionalFloat("beta", 0.75f);
                var bias = node.GetOptionalFloat("bias", 1.0f);
                var size = node.GetRequiredInt("size");
                model.AddLayer(new Layers.LRN(node.Name, node.Input0, alpha, beta, bias, size));
            }
            // Layer.ObjectDetection
            else if (opType == "NonMaxSuppression")
            {
                var centerPointBox = (node.GetOptionalInt("center_point_box", 0) == 0) ? Layers.CenterPointBox.Corners : Layers.CenterPointBox.Center;
                model.AddLayer(new Layers.NonMaxSuppression(node.Name, node.Input0, node.Input1, node.OptionalInput(2), node.OptionalInput(3), node.OptionalInput(4), centerPointBox));
            }
            else if (opType == "RoiAlign")
            {
                node.UnsupportedAttribute("coordinate_transformation_mode", "half_pixel");
                var mode = node.GetOptionalString("mode", "avg") == "avg" ? Layers.RoiPoolingMode.Avg : Layers.RoiPoolingMode.Max;
                var outputHeight = node.GetOptionalInt("output_height", 1);
                var outputWidth = node.GetOptionalInt("output_width", 1);
                var samplingRatio = node.GetOptionalInt("sampling_ratio", 0);
                var spatialScale = node.GetOptionalFloat("spatial_scale", 1.0f);

                model.AddLayer(new Layers.RoiAlign(node.Name, node.Input0, node.Input1, node.Input2, mode, outputHeight, outputWidth, samplingRatio, spatialScale));
            }
            // Layer.Pooling
            else if (opType == "AveragePool")
            {
                node.UnsupportedAttribute("ceil_mode", 0);
                node.UnsupportedAttribute("dilations", new[] { 1, 1 });
                node.UnsupportedAttribute("storage_order", 0);
                node.UnsupportedAttribute("count_include_pad", 0);

                var autopad = node.AutoPadMode();

                var kernelShape = node.GetRequiredIntArray("kernel_shape");
                var pads = node.GetOptionalIntArray("pads", null);
                var strides = node.GetOptionalIntArray("strides", null);

                model.AddLayer(new Layers.AveragePool(node.Name, node.Input0, kernelShape, strides, pads, autopad));
            }
            else if (opType == "GlobalAveragePool")
            {
                model.AddLayer(new Layers.GlobalAveragePool(node.Name, node.Input0));
            }
            else if (opType == "GlobalMaxPool")
            {
                model.AddLayer(new Layers.GlobalMaxPool(node.Name, node.Input0));
            }
            else if (opType == "MaxPool")
            {
                node.UnsupportedAttribute("ceil_mode", 0);
                node.UnsupportedAttribute("dilations", new[] { 1, 1 });
                node.UnsupportedAttribute("storage_order", 0);

                var autopad = node.AutoPadMode();

                var kernelShape = node.GetRequiredIntArray("kernel_shape");
                var pads = node.GetOptionalIntArray("pads", null);
                var strides = node.GetOptionalIntArray("strides", null);

                model.AddLayer(new Layers.MaxPool(node.Name, node.Input0, kernelShape, strides, pads, autopad));
            }
            // Layer.Random
            else if (opType == "Bernoulli")
            {
                var dataType = node.GetDataType(defaultValue: DataType.Float);
                model.AddLayer(new Layers.Bernoulli(node.Name, node.Input0, dataType, node.Seed));
            }
            else if (opType == "Multinomial")
            {
                node.IgnoredAttribute("dtype", "dtype can only be int32 or int64 which both map to TensorInt");
                var samples = node.GetOptionalInt("sample_size", 1);
                model.AddLayer(new Layers.Multinomial(node.Name, node.Input0, samples, node.Seed));
            }
            else if (opType == "RandomNormal")
            {
                var mean = node.GetOptionalFloat("mean", 0.0f);
                var scale = node.GetOptionalFloat("scale", 1.0f);
                var shape = node.GetRequiredIntArray("shape");
                model.AddLayer(new Layers.RandomNormal(node.Name, shape, mean, scale, node.Seed));
            }
            else if (opType == "RandomNormalLike")
            {
                var mean = node.GetOptionalFloat("mean", 0.0f);
                var scale = node.GetOptionalFloat("scale", 1.0f);
                model.AddLayer(new Layers.RandomNormalLike(node.Name, node.Input0, mean, scale, node.Seed));
            }
            else if (opType == "RandomUniform")
            {
                var low = node.GetOptionalFloat("low", 0.0f);
                var high = node.GetOptionalFloat("high", 1.0f);
                var shape = node.GetRequiredIntArray("shape");
                model.AddLayer(new Layers.RandomUniform(node.Name, shape, low, high, node.Seed));
            }
            else if (opType == "RandomUniformLike")
            {
                var low = node.GetOptionalFloat("low", 0.0f);
                var high = node.GetOptionalFloat("high", 1.0f);
                model.AddLayer(new Layers.RandomUniformLike(node.Name, node.Input0, low, high, node.Seed));
            }
            // Layer.Recurrent
            else if (opType == "LSTM")
            {
                var hiddenSize = node.GetRequiredInt("hidden_size");
                var direction = node.Direction();
                var activations = node.Activations();
                var activationAlpha = node.GetOptionalFloatArray("activation_alpha", null);
                var activationBeta = node.GetOptionalFloatArray("activation_beta", null);
                var clip = node.GetOptionalFloat("clip", float.MaxValue);
                var inputForget = node.GetOptionalInt("input_forget", 0) != 0;
                var layout = node.Layout();

                model.AddLayer(new Layers.LSTM(node.Output0, node.Input0, node.Input1, node.Input2, hiddenSize, Y_h: node.OptionalOutput(1), Y_c: node.OptionalOutput(2), B: node.OptionalInput(3), sequenceLens: node.OptionalInput(4), initialH: node.OptionalInput(5), initialC: node.OptionalInput(6), P: node.OptionalInput(7), direction: direction, activations: activations, activationAlpha: activationAlpha, activationBeta: activationBeta, clip: clip, inputForget: inputForget, layout: layout));
            }
            // Layer.Reduction
            else if (opType == "ReduceL1")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = string.Empty;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = model.GetUniqueIndex(node.Name + "_axes");
                        var axesConstant = new Layers.Constant(axesIndex, axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = node.OptionalInput(1);
                }

                model.AddLayer(new Layers.ReduceL1(node.Name, node.Input0, axesIndex, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceL2")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = string.Empty;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = model.GetUniqueIndex(node.Name + "_axes");
                        var axesConstant = new Layers.Constant(axesIndex, axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = node.OptionalInput(1);
                }

                model.AddLayer(new Layers.ReduceL2(node.Name, node.Input0, axesIndex, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceLogSum")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = string.Empty;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = model.GetUniqueIndex(node.Name + "_axes");
                        var axesConstant = new Layers.Constant(axesIndex, axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = node.OptionalInput(1);
                }

                model.AddLayer(new Layers.ReduceLogSum(node.Name, node.Input0, axesIndex, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceLogSumExp")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = string.Empty;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = model.GetUniqueIndex(node.Name + "_axes");
                        var axesConstant = new Layers.Constant(axesIndex, axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = node.OptionalInput(1);
                }

                model.AddLayer(new Layers.ReduceLogSumExp(node.Name, node.Input0, axesIndex, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceMax")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = string.Empty;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = model.GetUniqueIndex(node.Name + "_axes");
                        var axesConstant = new Layers.Constant(axesIndex, axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = node.OptionalInput(1);
                }

                model.AddLayer(new Layers.ReduceMax(node.Name, node.Input0, axesIndex, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceMean")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = string.Empty;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = model.GetUniqueIndex(node.Name + "_axes");
                        var axesConstant = new Layers.Constant(axesIndex, axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = node.OptionalInput(1);
                }

                model.AddLayer(new Layers.ReduceMean(node.Name, node.Input0, axesIndex, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceMin")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = string.Empty;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = model.GetUniqueIndex(node.Name + "_axes");
                        var axesConstant = new Layers.Constant(axesIndex, axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = node.OptionalInput(1);
                }

                model.AddLayer(new Layers.ReduceMin(node.Name, node.Input0, axesIndex, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceProd")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = string.Empty;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = model.GetUniqueIndex(node.Name + "_axes");
                        var axesConstant = new Layers.Constant(axesIndex, axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = node.OptionalInput(1);
                }

                model.AddLayer(new Layers.ReduceProd(node.Name, node.Input0, axesIndex, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceSum")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = string.Empty;
                if (defaultOpsetVersion < 13)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = model.GetUniqueIndex(node.Name + "_axes");
                        var axesConstant = new Layers.Constant(axesIndex, axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = node.OptionalInput(1);
                }

                model.AddLayer(new Layers.ReduceSum(node.Name, node.Input0, axesIndex, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceSumSquare")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = string.Empty;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = model.GetUniqueIndex(node.Name + "_axes");
                        var axesConstant = new Layers.Constant(axesIndex, axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = node.OptionalInput(1);
                }

                model.AddLayer(new Layers.ReduceSumSquare(node.Name, node.Input0, axesIndex, keepDims, noopWithEmptyAxes));
            }
            // Layer.Transformation
            else if (opType == "Cast")
            {
                var toOnnxType = (TensorProto.Types.DataType)node.GetRequiredInt("to");
                var toDataType = ONNXNodeWrapper.DataTypeFromOnnxDataType(toOnnxType, OnUnsupported: () =>
                {
                    Warn(WarningType.Error, $"Unsupported tensor dataType: {toOnnxType}.");
                    Debug.LogError(Warnings.Last().Message);
                });
                model.AddLayer(new Layers.Cast(node.Name, node.Input0, toDataType));
            }
            else if (opType == "CastLike")
            {
                model.AddLayer(new Layers.CastLike(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Concat")
            {
                var axis = node.GetRequiredInt("axis");
                model.AddLayer(new Layers.Concat(node.Name, node.Inputs, axis));
            }
            else if (opType == "DepthToSpace")
            {
                var modeType = node.GetOptionalString("mode", "DCR");
                var mode = modeType == "DCR" ? Layers.DepthToSpaceMode.DepthColumnRow : Layers.DepthToSpaceMode.ColumnRowDepth;
                var blocksize = node.GetRequiredInt("blocksize");
                model.AddLayer(new Layers.DepthToSpace(node.Name, node.Input0, blocksize, mode));
            }
            else if (opType == "Expand")
            {
                // Expand-8, Expand-13
                model.AddLayer(new Layers.Expand(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Flatten")
            {
                var axis = node.GetOptionalInt("axis", 1);
                model.AddLayer(new Layers.Flatten(node.Name, node.Input0, axis));
            }
            else if (opType == "Dropout")
            {
                model.AddLayer(new Layers.Identity(node.Name, node.Input0));
            }
            else if (opType == "Identity")
            {
                model.AddLayer(new Layers.Identity(node.Name, node.Input0));
            }
            else if (opType == "Pad")
            {
                var mode = node.PadMode();
                if (defaultOpsetVersion < 11)
                {
                    // Pad-1 or Pad-2
                    var pads = node.GetRequiredIntArray(node.HasAttribute("pads") ? "pads" : "paddings");
                    var padsConstant = new Layers.Constant(model.GetUniqueIndex(node.Name + "_pads"), pads);
                    model.AddConstant(padsConstant);
                    var value = node.GetOptionalFloat("value", 0f);
                    var valueConstant = new Layers.Constant(model.GetUniqueIndex(node.Name + "_value"), new[] { value });
                    model.AddConstant(valueConstant);
                    model.AddLayer(new Layers.Pad(node.Name, node.Input0, padsConstant.index, valueConstant.index, mode: mode));
                }
                else
                {
                    // Pad-11, Pad-13, Pad-18
                    var constantValue = node.InputCount > 2 ? node.Inputs[2] : string.Empty;
                    var axes = node.InputCount > 3 ? node.Inputs[3] : string.Empty;
                    model.AddLayer(new Layers.Pad(node.Name, node.Input0, node.Input1, constantValue, axes, mode));
                }
            }
            else if (opType == "Reshape")
            {
                if (defaultOpsetVersion < 5)
                {
                    // Reshape-1
                    var shape = node.GetRequiredIntArray("shape");
                    var shapeConstant = new Layers.Constant(model.GetUniqueIndex(node.Name + "_shape"), shape);
                    model.AddConstant(shapeConstant);
                    model.AddLayer(new Layers.Reshape(node.Name, node.Input0, shapeConstant.index));
                }
                else
                {
                    // Reshape-5, Reshape-13, Reshape-14
                    var allowZero = node.GetOptionalInt("allowzero", 0) != 0;
                    model.AddLayer(new Layers.Reshape(node.Name, node.Input0, node.Input1, allowZero));
                }
            }
            else if (opType == "Resize")
            {
                var mode = node.InterpolationMode();
                var axes = node.GetOptionalIntArray("axes", null);
                if (defaultOpsetVersion < 11)
                {
                    // Resize-10
                    model.AddLayer(new Layers.Resize(node.Name, node.Input0, node.Input1, Layers.ScaleMode.Scales, mode, Layers.CoordTransformMode.Asymmetric, Layers.NearestMode.Floor, axes));
                }
                else
                {
                    node.UnsupportedAttribute("cubic_coeff_a", -0.75f);
                    node.UnsupportedAttribute("exclude_outside", 0);
                    node.UnsupportedAttribute("extrapolation_value", 0);
                    var coordinateTransformMode = node.CoordinateTransformMode();
                    var nearestMode = node.NearestMode();
                    if (node.InputCount == 3 || string.IsNullOrEmpty(node.Inputs[3]))
                    {
                        // Resize-11, Resize-13, Resize-18 with scales
                        model.AddLayer(new Layers.Resize(node.Name, node.Input0, node.Input2, Layers.ScaleMode.Scales, mode, coordinateTransformMode, nearestMode, axes));
                    }
                    else if (node.InputCount == 4)
                    {
                        // Resize-11, Resize-13, Resize-18 with sizes
                        model.AddLayer(new Layers.Resize(node.Name, node.Input0, node.Input3, Layers.ScaleMode.Sizes, mode, coordinateTransformMode, nearestMode, axes));
                    }
                }
            }
            else if (opType == "Slice")
            {
                if (defaultOpsetVersion < 10)
                {
                    // Slice-1
                    var starts = node.GetRequiredIntArray("starts");
                    var startsConstant = new Layers.Constant(model.GetUniqueIndex(node.Name + "_starts"), starts);
                    model.AddConstant(startsConstant);
                    var ends = node.GetRequiredIntArray("ends");
                    var endsConstant = new Layers.Constant(model.GetUniqueIndex(node.Name + "_ends"), ends);
                    model.AddConstant(endsConstant);
                    if (node.HasAttribute("axes"))
                    {
                        var axes = node.GetRequiredIntArray("axes");
                        var axesConstant = new Layers.Constant(model.GetUniqueIndex(node.Name + "_axes"), axes);
                        model.AddConstant(axesConstant);
                        model.AddLayer(new Layers.Slice(node.Name, node.Input0, startsConstant.index, endsConstant.index, axesConstant.index));
                    }
                    else
                    {
                        model.AddLayer(new Layers.Slice(node.Name, node.Input0, startsConstant.index, endsConstant.index));
                    }
                }
                else
                {
                    // Slice-10, Slice-11, Slice-13
                    model.AddLayer(new Layers.Slice(node.Name, node.Input0, node.Input1, node.Input2, node.OptionalInput(3), node.OptionalInput(4)));
                }
            }
            else if (opType == "SpaceToDepth")
            {
                var blocksize = node.GetRequiredInt("blocksize");
                model.AddLayer(new Layers.SpaceToDepth(node.Name, node.Input0, blocksize));
            }
            else if (opType == "Split")
            {
                var axis = node.GetOptionalInt("axis", 0);
                if (node.HasAttribute("split"))
                {
                    // Split-1, Split-2, Split-11 with "split" attribute
                    var split = node.GetRequiredIntArray("split");
                    var splitConstant = new Layers.Constant(model.GetUniqueIndex(node.Name + "_split"), split);
                    model.AddConstant(splitConstant);
                    model.AddLayer(new Layers.Split(node.Outputs, node.Input0, splitConstant.index, axis));
                }
                else if (!string.IsNullOrEmpty(node.OptionalInput(1)))
                {
                    // Split-1, Split-2, Split-11, Split-13, Split-18 with split tensor
                    model.AddLayer(new Layers.Split(node.Outputs, node.Input0, node.Input1, axis));
                }
                else
                {
                    // Split-1, Split-2, Split-11, Split-13, Split-18 with num_outputs
                    model.AddLayer(new Layers.Split(node.Outputs, node.Input0, axis: axis, numOutputs: node.GetOptionalInt("num_outputs", node.Outputs.Length)));
                }
            }
            else if (opType == "Squeeze")
            {
                if (defaultOpsetVersion < 13 && node.HasAttribute("axes"))
                {
                    // Squeeze-1, Squeeze-11 with given axes
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(model.GetUniqueIndex(node.Name + "_axes"), axes);
                    model.AddConstant(axesConstant);
                    model.AddLayer(new Layers.Squeeze(node.Name, node.Input0, axesConstant.index));
                }
                else
                {
                    // Squeeze-13 or Squeeze-1, Squeeze-11 without given axes
                    model.AddLayer(new Layers.Squeeze(node.Name, node.Input0, node.OptionalInput(1)));
                }
            }
            else if (opType == "Tile")
            {
                model.AddLayer(new Layers.Tile(node.Name, node.Input0, node.Input1));
            }
            else if (opType == "Transpose")
            {
                var permutations = node.GetOptionalIntArray("perm", null);
                model.AddLayer(new Layers.Transpose(node.Name, node.Input0, permutations));
            }
            else if (opType == "Trilu")
            {
                var upper = node.GetOptionalInt("upper", 1);
                model.AddLayer(new Layers.Trilu(node.Name, node.Input0, node.OptionalInput(1), (Layers.TriluMode)upper));
            }
            else if (opType == "Upsample")
            {
                var coordinateTransformMode = Layers.CoordTransformMode.Asymmetric;
                var mode = node.InterpolationMode();
                var nearestMode = Layers.NearestMode.Floor;
                if (defaultOpsetVersion < 9)
                {
                    // Upsample-7
                    var scales = node.GetRequiredFloatArray("scales");
                    var scalesConstant = new Layers.Constant(model.GetUniqueIndex(node.Name + "_scales"), scales);
                    model.AddConstant(scalesConstant);
                    model.AddLayer(new Layers.Resize(node.Name, node.Input0, scalesConstant.index, Layers.ScaleMode.Scales, mode, coordinateTransformMode, nearestMode, null));
                }
                else
                {
                    // Upsample-9
                    model.AddLayer(new Layers.Resize(node.Name, node.Input0, node.Input1, Layers.ScaleMode.Scales, mode, coordinateTransformMode, nearestMode, null));
                }
            }
            else if (opType == "Unsqueeze")
            {
                if (defaultOpsetVersion < 13)
                {
                    // Unsqueeze-1, Unsqueeze-11
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(model.GetUniqueIndex(node.Name + "_axes"), axes);
                    model.AddConstant(axesConstant);
                    model.AddLayer(new Layers.Unsqueeze(node.Name, node.Input0, axesConstant.index));
                }
                else
                {
                    // Unsqueeze-13
                    model.AddLayer(new Layers.Unsqueeze(node.Name, node.Input0, node.Input1));
                }
            }
            // Layer.Trigonometric
            else if (opType == "Acos")
            {
                model.AddLayer(new Layers.Acos(node.Name, node.Input0));
            }
            else if (opType == "Acosh")
            {
                model.AddLayer(new Layers.Acosh(node.Name, node.Input0));
            }
            else if (opType == "Asin")
            {
                model.AddLayer(new Layers.Asin(node.Name, node.Input0));
            }
            else if (opType == "Asinh")
            {
                model.AddLayer(new Layers.Asinh(node.Name, node.Input0));
            }
            else if (opType == "Atan")
            {
                model.AddLayer(new Layers.Atan(node.Name, node.Input0));
            }
            else if (opType == "Atanh")
            {
                model.AddLayer(new Layers.Atanh(node.Name, node.Input0));
            }
            else if (opType == "Cos")
            {
                model.AddLayer(new Layers.Cos(node.Name, node.Input0));
            }
            else if (opType == "Cosh")
            {
                model.AddLayer(new Layers.Cosh(node.Name, node.Input0));
            }
            else if (opType == "Sin")
            {
                model.AddLayer(new Layers.Sin(node.Name, node.Input0));
            }
            else if (opType == "Sinh")
            {
                model.AddLayer(new Layers.Sinh(node.Name, node.Input0));
            }
            else if (opType == "Tan")
            {
                model.AddLayer(new Layers.Tan(node.Name, node.Input0));
            }
            // Non standard ONNX
            else if (opType == "Swish")
            {
                model.AddLayer(new Layers.Swish(node.Name, node.Input0));
            }
            else if (opType == "ImageScaler")
            {
                var attrBias = node.GetRequiredFloatArray("bias");
                var maxElements = attrBias.Length;
                var attrScale = Enumerable.Repeat(node.GetOptionalFloat("scale", 1.0f), maxElements).ToArray();

                using var scale = new TensorFloat(new TensorShape(maxElements), attrScale);
                using var bias = new TensorFloat(new TensorShape(maxElements), attrBias);

                var scaleConstantName = model.GetUniqueIndex($"{node.Name}_Scale");
                model.AddConstant(new Layers.Constant(scaleConstantName, scale));
                var biasConstantName = model.GetUniqueIndex($"{node.Name}_Bias");
                model.AddConstant(new Layers.Constant(biasConstantName, bias));
                model.AddLayer(new Layers.ScaleBias(node.Name, node.Input0, scaleConstantName, biasConstantName));
            }
            else
            {
                Warn(WarningType.Error, $"{opType} not supported");
                Debug.LogError(Warnings.Last().Message);
            }
        }

        // NOTE: It's questionable whether we should be doing this since the ONNX specification requires the graph to be
        // topologically sorted, but at least one network encountered that was exported from keras2onnx v1.7.0 produced
        // an incorrectly sorted graph. related example: https://github.com/onnx/keras-onnx/issues/184
        static List<NodeProto> SortTopologically(ModelProto onnxModel)
        {
            GraphProto onnxGraph = onnxModel.Graph;
            HashSet<string> encounteredNodes = new HashSet<string>();
            foreach (var i in onnxGraph.Input)
                encounteredNodes.Add(i.Name);
            foreach (var i in onnxGraph.Initializer)
                encounteredNodes.Add(i.Name);

            var sortedGraph = new List<NodeProto>();
            bool graphInSortedOrder = true;
            foreach (NodeProto node in onnxGraph.Node)
            {
                foreach (var input in node.Input)
                    graphInSortedOrder &= encounteredNodes.Contains(input);

                if (!graphInSortedOrder)
                    break;

                foreach (var output in node.Output)
                    encounteredNodes.Add(output);
                sortedGraph.Add(node);
            }

            if (graphInSortedOrder)
                return sortedGraph;

            sortedGraph.Clear();
            var nodesToSort = new Queue<NodeProto>();
            foreach (NodeProto node in onnxGraph.Node)
            {
                nodesToSort.Enqueue(node);
            }

            var requeueNodes = new Queue<NodeProto>();
            while (nodesToSort.Count > 0)
            {
                NodeProto node = nodesToSort.Dequeue();

                var allInputsExist = true;
                foreach (string input in node.Input)
                {
                    if (string.IsNullOrEmpty(input))
                        continue;

                    if (!sortedGraph.Exists(n => n.Output.Any(o => o == input))
                        && !onnxGraph.Input.Any(i => i.Name == input)
                        && !onnxGraph.Initializer.Any(i => i.Name == input))
                    {
                        allInputsExist = false;
                        break;
                    }
                }

                if (!allInputsExist)
                {
                    if (nodesToSort.Count != 0)
                    {
                        // Mark for re-processing again when (potentially) all inputs have been processed
                        // We use a separate list, so we don't continually spin on nodes that are missing inputs
                        if (!requeueNodes.Contains(node))
                            requeueNodes.Enqueue(node);
                        continue;
                    }

                    // Something must've gone wrong
                    throw new OnnxImportException($"Missing inputs to node {node.Name}, but there are no nodes to process.");
                }

                if (!sortedGraph.Contains(node))
                    sortedGraph.Add(node);

                // Now that we have at least processed a single new node, let's requeue
                while (requeueNodes.Count > 0)
                    nodesToSort.Enqueue(requeueNodes.Dequeue());
            }

            return sortedGraph;
        }

        Model ConvertOnnxModel(ModelProto onnxModel)
        {
            var model = new Model();
            long defaultOpsetVersion = 15;

            // Parse producer meta data
            foreach (var opsetSetIdProto in onnxModel.OpsetImport)
            {
                if (string.IsNullOrEmpty(opsetSetIdProto.Domain))
                    defaultOpsetVersion = opsetSetIdProto.Version;
            }
            model.ProducerName = onnxModel.ProducerName;
            if (!string.IsNullOrEmpty(onnxModel.ProducerVersion))
                model.ProducerName += $" v{onnxModel.ProducerVersion}";

            // Convert graph inputs & outputs
            var initializersByName = onnxModel.Graph.Initializer.ToDictionary(i => i.Name, i => true);
            var namedDims = new List<string>();
            foreach (var input in onnxModel.Graph.Input)
            {
                // skip input tensors that have initializer data, they are constant tensors not global inputs
                // also skip nodes that should be trimmed
                if (initializersByName.ContainsKey(input.Name))
                    continue;

                var onnxShape = input.Type.TensorType.Shape;
                var inputShape = SymbolicTensorShape.UnknownOfRank(onnxShape.Dim.Count);

                for (var i = 0; i < inputShape.rank; i++)
                {
                    var dim = onnxShape.Dim[i];
                    switch (dim.ValueCase)
                    {
                        case TensorShapeProto.Types.Dimension.ValueOneofCase.None:
                            inputShape[i] = SymbolicTensorDim.Unknown;
                            break;
                        case TensorShapeProto.Types.Dimension.ValueOneofCase.DimParam:
                            var index = namedDims.IndexOf(dim.DimParam);
                            if (index < 0)
                            {
                                index = namedDims.Count;
                                namedDims.Add(dim.DimParam);
                            }
                            inputShape[i] = SymbolicTensorDim.Param((byte)index);
                            break;
                        case TensorShapeProto.Types.Dimension.ValueOneofCase.DimValue:
                            if (dim.DimValue < 0)
                                Warn(WarningType.Warning, "Tensor shape has negative index, treating as unknown dimension");
                            else
                                inputShape[i] = SymbolicTensorDim.Int(dim.DimValue > int.MaxValue ? int.MaxValue : (int)dim.DimValue);
                            break;
                        default:
                            throw new ArgumentOutOfRangeException();
                    }
                }

                var inputDataType = ONNXNodeWrapper.DataTypeFromOnnxDataType((TensorProto.Types.DataType)input.Type.TensorType.ElemType);

                model.AddInput(input.Name, inputDataType, inputShape);
            }

            foreach (ValueInfoProto o in onnxModel.Graph.Output)
                model.AddOutput(o.Name);

            var weightsStream = new Dictionary<string, FileStream>();
            // Read constants from initializer list
            foreach (TensorProto initializer in onnxModel.Graph.Initializer)
            {
                if (initializer.DataLocation == TensorProto.Types.DataLocation.External)
                {
                    string name = initializer.ExternalData.Single(x => x.Key == "location").Value;
                    if (!weightsStream.ContainsKey(name))
                    {
                        string filePath = Path.Combine(m_DirectoryPath, name);
                        if (File.Exists(filePath))
                            weightsStream.Add(name, File.OpenRead(Path.Combine(m_DirectoryPath, name)));
                        else
                        {
                            Warn(WarningType.Error, $"External Weights file not found! Expecting: {filePath}");
                            return null;
                        }
                    }
                    var stream = weightsStream[name];
                    var constant = ONNXConstantsLoader.LoadConstant(initializer, stream);
                    model.AddConstant(constant);
                }
                else
                {
                    var constant = ONNXConstantsLoader.LoadConstant(initializer);
                    model.AddConstant(constant);
                }
            }
            foreach (var stream in weightsStream.Values)
                stream.Dispose();

            // Nodes are supposed to be sorted, but this isn't always the case
            var sortedGraph = SortTopologically(onnxModel);

            // Convert graph nodes
            foreach (NodeProto onnxNode in sortedGraph)
            {
                var node = new ONNXNodeWrapper(onnxNode);

                try
                {
                    OnNode(model, defaultOpsetVersion, node);
                }
                catch (Exception e)
                {
                    Warn(WarningType.Error, e.Message);
                    throw new OnnxImportException(Warnings.Last().Message);
                }
            }

            // delete unused outputs
            HashSet<string> outputsToRemove = model.outputs.Select(o => o.index).ToHashSet();
            foreach (var layer in model.layers)
            {
                for (var i = 0; i < layer.outputs.Length; i++)
                    outputsToRemove.Remove(layer.outputs[i]);
            }
            model.outputs.RemoveAll(x => outputsToRemove.Contains(x.index));

            // strip :0 at the end of string name for TF import
            model = TrimTensorflowNames(model);

            // validate imported model
            if (!Warnings.Any(w => w.MessageSeverity == WarningType.Error))
            {
                model = ModelValidator.ValidateModel(model);
            }
            if (!Warnings.Any(w => w.MessageSeverity == WarningType.Error))
            {
                ModelOptimizer.OptimizeModel(ref model);
                ModelOptimizer.RunCPUFallbackPass(ref model);
            }

            return model;
        }

        static Model TrimTensorflowNames(Model model)
        {
            model.inputs   = model.inputs.Select(i   => {
                i.index = TrimTensorflowName(i.index);
                i.name = TrimTensorflowName(i.name);
                return i;
            }).ToList();

            model.outputs   = model.outputs.Select(o   => {
                o.index = TrimTensorflowName(o.index);
                o.name = TrimTensorflowName(o.name);
                return o;
            }).ToList();

            model.constants = model.constants.Select(c => {
                c.index = TrimTensorflowName(c.index);
                return c;
            }).ToList();

            model.layers   = model.layers.Select(l   => {
                for (int i = 0; i < l.inputs.Length; i++)
                    l.inputs[i] = TrimTensorflowName(l.inputs[i]);
                for (int i = 0; i < l.outputs.Length; i++)
                    l.outputs[i] = TrimTensorflowName(l.outputs[i]);
                return l;
            }).ToList();

            return model;
        }

        static string TrimTensorflowName(string name)
        {
            if (name.EndsWith(":0"))
                return name.Remove(name.Length-2);
            return name;
        }

        // Logging helpers
        void Warn(WarningType severity, string message)
        {
            Warnings.Add(new ImporterWarning(message, severity));
        }

        /// <summary>
        /// The warnings from the model importer.
        /// </summary>
        public List<ImporterWarning> Warnings { get; } = new List<ImporterWarning>();

        /// <summary>
        /// Represents types of warning from the model importer.
        /// </summary>
        public enum WarningType
        {
            /// <summary>
            /// No error.
            /// </summary>
            None = 0,

            /// <summary>
            /// Information. Execution should run without errors.
            /// </summary>
            Info = 1,

            /// <summary>
            /// Warning. Execution should run, but may have issues with precision or speed.
            /// </summary>
            Warning = 2,

            /// <summary>
            /// Error. Execution won't run.
            /// </summary>
            Error = 3
        }

        /// <summary>
        /// Represents the data structure for a warning from the model importer.
        /// </summary>
        public class ImporterWarning
        {
            /// <summary>
            /// A message.
            /// </summary>
            public string Message { get; }

            /// <summary>
            /// The severity of a warning.
            /// </summary>
            public WarningType MessageSeverity { get; }

            /// <summary>
            /// Initializes and returns an instance of `ImporterWarning`.
            /// </summary>
            /// <param name="severity">The severity of the warning as a `WarningType`</param>
            /// <param name="msg">The message text of the warning</param>
            public ImporterWarning(string msg, WarningType severity)
            {
                Message = msg;
                MessageSeverity = severity;
            }
        }
    }

    /// <summary>
    /// Represents an exception during the import of an ONNX model.
    /// </summary>
    public class OnnxImportException : Exception
    {
        /// <summary>
        /// Initializes and returns an instance of `OnnxImportException`.
        /// </summary>
        /// <param name="message">message</param>
        public OnnxImportException(string message) : base(message) { }
    }

    /// <summary>
    /// Represents an exception during the import of a ONNX layer.
    /// </summary>
    public class OnnxLayerImportException : Exception
    {
        /// <summary>
        /// Initializes and returns an instance of `ONNXLayerImportException`.
        /// </summary>
        /// <param name="message">message</param>
        public OnnxLayerImportException(string message) : base(message) { }
    }
}
