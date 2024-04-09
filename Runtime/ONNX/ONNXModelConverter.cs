using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Google.Protobuf;
using Onnx;

[assembly: InternalsVisibleTo("Unity.Sentis.Tests")]
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
        /// Calls the methods in its invocation list when the model is imported.
        /// </summary>
        public static event Action<object, Model> ModelImported;

        void Add(string opType, Action<Model, ONNXNodeWrapper> opImportAction)
        {
            m_NodeImporters.Add(opType, opImportAction);
        }

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
            SetupImporter();
            model = ConvertOnnxModel(onnxModel);
            ModelImported?.Invoke(onnxModel, model);

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

        internal void SetupImporter()
        {
            m_NodeImporters.Clear();

            Add("Constant", (net, node) =>
            {
                node.UnsupportedAttribute("sparse_value");
                var constant = ONNXConstantsLoader.LoadConstant(node.ValueAsTensor, m_DirectoryPath);
                constant.index = node.Name;
                net.AddConstant(constant);
            });

            // Layer.Activation
            Add("Celu", (net, node) => { net.AddLayer(new Layers.Celu(node.Name, node.Input0, node.GetOptionalFloat("alpha", 1f))); });
            Add("Elu", (net, node) => { net.AddLayer(new Layers.Elu(node.Name, node.Input0, node.AlphaOptional(1f))); });
            Add("Erf", (net, node) => { net.AddLayer(new Layers.Erf(node.Name, node.Input0)); });
            Add("Gelu", (net, node) =>{ net.AddLayer(new Layers.Gelu(node.Name, node.Input0)); });
            Add("Hardmax", (net, node) =>
            {
                var axis = node.AxisOptional(net.DefaultOpsetVersion > 11 ? -1 : 1);
                net.AddLayer(new Layers.Hardmax(node.Name, node.Input0, axis));
            });
            Add("HardSigmoid", (net, node) => { net.AddLayer(new Layers.HardSigmoid(node.Name, node.Input0, node.AlphaOptional(0.2f), node.BetaOptional(0.5f))); });
            Add("HardSwish", (net, node) => { net.AddLayer(new Layers.HardSwish(node.Name, node.Input0)); });
            Add("LeakyRelu", (net, node) => { net.AddLayer(new Layers.LeakyRelu(node.Name, node.Input0, node.AlphaOptional(0.01f))); });
            Add("PRelu", (net, node) => { net.AddLayer(new Layers.PRelu(node.Name, node.Input0, node.Input1)); });
            Add("Relu", (net, node) => { net.AddLayer(new Layers.Relu(node.Name, node.Input0)); });
            Add("Selu", (net, node) => { net.AddLayer(new Layers.Selu(node.Name, node.Input0, node.AlphaOptional(1.67326f), node.GammaOptional(1.0507f))); });
            Add("Sigmoid", (net, node) => { net.AddLayer(new Layers.Sigmoid(node.Name, node.Input0)); });
            Add("Softplus", (net, node) => { net.AddLayer(new Layers.Softplus(node.Name, node.Input0)); });
            Add("Softsign", (net, node) => { net.AddLayer(new Layers.Softsign(node.Name, node.Input0)); });
            Add("Tanh", (net, node) => { net.AddLayer(new Layers.Tanh(node.Name, node.Input0)); });
            Add("ThresholdedRelu", (net, node) => { net.AddLayer(new Layers.ThresholdedRelu(node.Name, node.Input0, node.GetOptionalFloat("alpha", 1f))); });

            // Layer.ActivationNonLinear
            Add("LogSoftmax", (net, node) =>
            {
                var axis = node.AxisOptional(net.DefaultOpsetVersion > 11 ? -1 : 1);
                net.AddLayer(new Layers.LogSoftmax(node.Name, node.Input0, axis));
            });
            Add("Softmax", (net, node) =>
            {
                var axis = node.AxisOptional(net.DefaultOpsetVersion > 11 ? -1 : 1);
                net.AddLayer(new Layers.Softmax(node.Name, node.Input0, axis));
            });

            // Layer.Convolution
            Add("Conv", (net, node) =>
            {
                // Conv-1, Conv-11

                var autoPad = node.AutoPadMode();
                var kernelShape = node.GetOptionalIntArray("kernel_shape", null);
                var dilations = node.GetOptionalIntArray("dilations", null);
                var group = node.GetOptionalInt("group", 1);
                var pads = node.GetOptionalIntArray("pads", null);
                var strides = node.GetOptionalIntArray("strides", null);

                if (node.InputCount == 2)
                    net.AddLayer(new Layers.Conv(node.Name, node.Input0, node.Input1, group, strides, pads, dilations, autoPad, kernelShape));
                else
                    net.AddLayer(new Layers.Conv(node.Name, node.Input0, node.Input1, node.Input2, group, strides, pads, dilations, autoPad, kernelShape));
            });
            Add("ConvTranspose", (net, node) =>
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

                if (node.InputCount == 2)
                    net.AddLayer(new Layers.ConvTranspose(node.Name, node.Input0, node.Input1, strides, pads, autoPad, outputPadding, kernelShape));
                else
                    net.AddLayer(new Layers.ConvTranspose(node.Name, node.Input0, node.Input1, node.Input2, strides, pads, autoPad, outputPadding, kernelShape));
            });

            // Layer.Dimension
            Add("Shape", (net, node) =>
            {
                // Shape-1, Shape-13, Shape-15
                var start = node.GetOptionalInt("start", 0);
                var end = node.GetOptionalInt("end", TensorShape.maxRank);
                net.AddLayer(new Layers.Shape(node.Name, node.Input0, start, end));
            });
            Add("Size", (net, node) =>
            {
                // Size-1, Size-13
                net.AddLayer(new Layers.Size(node.Name, node.Input0));
            });

            // Layer.Generator
            Add("ConstantOfShape", (net, node) =>
            {
                UnityEngine.Debug.Assert(node.InputCount > 0);

                if (!node.HasAttribute("value"))
                {
                    net.AddLayer(new Layers.ConstantOfShape(node.Name, node.Input0, 0.0f));
                    return;
                }

                var constant = ONNXConstantsLoader.LoadConstant(node.ValueAsTensor, m_DirectoryPath);
                if (constant.dataType == DataType.Int)
                {
                    var value = constant.weights.Get<int>(0);
                    net.AddLayer(new Layers.ConstantOfShape(node.Name, node.Input0, value));
                }
                else
                {
                    var value = constant.weights.Get<float>(0);
                    net.AddLayer(new Layers.ConstantOfShape(node.Name, node.Input0, value));
                }
                constant.weights.Dispose();
            });
            Add("Range", (net, node) =>
            {
                net.AddLayer(new Layers.Range(node.Name, node.Input0, node.Input1, node.Input2));
            });
            Add("OneHot", (net, node) =>
            {
                // OneHot-9, OneHot-11
                var axis = node.AxisOptional(-1);
                net.AddLayer(new Layers.OneHot(node.Name, node.Input0, node.Input1, node.Input2, axis));
            });

            // Layer.Indexing
            Add("ArgMax", (net, node) =>
            {
                var keepdims = node.GetOptionalInt("keepdims", 1) == 1;
                var selectLastIndex = node.GetOptionalInt("select_last_index", 0) == 1;
                var axis = node.AxisOptional(0);
                net.AddLayer(new Layers.ArgMax(node.Name, node.Input0, axis, keepdims, selectLastIndex));
            });
            Add("ArgMin", (net, node) =>
            {
                var keepdims = node.GetOptionalInt("keepdims", 1) == 1;
                var selectLastIndex = node.GetOptionalInt("select_last_index", 0) == 1;
                var axis = node.AxisOptional(0);
                net.AddLayer(new Layers.ArgMin(node.Name, node.Input0, axis, keepdims, selectLastIndex));
            });
            Add("Gather", (net, node) =>
            {
                var axis = node.AxisOptional(0);
                net.AddLayer(new Layers.Gather(node.Name, node.Input0, node.Input1, axis));
            });
            Add("GatherElements", (net, node) =>
            {
                var axis = node.AxisOptional(0);
                net.AddLayer(new Layers.GatherElements(node.Name, node.Input0, node.Input1, axis));
            });
            Add("GatherND", (net, node) =>
            {
                var batchDims = node.GetOptionalInt("batch_dims", 0);
                net.AddLayer(new Layers.GatherND(node.Name, node.Input0, node.Input1, batchDims));
            });
            Add("NonZero", (net, node) =>
            {
                net.AddLayer(new Layers.NonZero(node.Name, node.Input0));
            });
            Add("Scatter", (net, node) =>
            {
                // Scatter-9 maps to ScatterElements
                var axis = node.AxisOptional(0);
                net.AddLayer(new Layers.ScatterElements(node.Name, node.Input0, node.Input1, node.Input2, axis, Layers.ScatterReductionMode.None));
            });
            Add("ScatterElements", (net, node) =>
            {
                int axis = node.AxisOptional(0);
                Layers.ScatterReductionMode reduction = node.ScatterReductionMode();
                net.AddLayer(new Layers.ScatterElements(node.Name, node.Input0, node.Input1, node.Input2, axis, reduction));
            });
            Add("ScatterND", (net, node) =>
            {
                Layers.ScatterReductionMode reduction = node.ScatterReductionMode();
                net.AddLayer(new Layers.ScatterND(node.Name, node.Input0, node.Input1, node.Input2, reduction));
            });
            Add("TopK", (net, node) =>
            {
                string[] outputs = { node.Outputs[0], node.Outputs[1] };
                var axis = node.AxisOptional(-1);
                var largest = node.GetOptionalInt("largest", 1) == 1;
                var sorted = node.GetOptionalInt("sorted", 1) == 1;
                if (node.HasAttribute("k"))
                {
                    // TopK-1
                    var k = node.GetRequiredInt("k");
                    var kConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_k"), new[] { k });
                    net.AddConstant(kConstant);
                    net.AddLayer(new Layers.TopK(node.Name, node.Input0, kConstant.index, axis, largest, sorted, outputs));
                }
                else
                {
                    // TopK-10, TopK-11
                    net.AddLayer(new Layers.TopK(node.Name, node.Input0, node.Input1, axis, largest, sorted, outputs));
                }
            });

            // Layer.Logical
            Add("And", (net, node) => { net.AddLayer(new Layers.And(node.Name, node.Input0, node.Input1)); });
            Add("Compress", (net, node) =>
            {
                if (node.HasAttribute("axis"))
                    net.AddLayer(new Layers.Compress(node.Name, node.Input0, node.Input1, node.Axis));
                else
                    net.AddLayer(new Layers.Compress(node.Name, node.Input0, node.Input1));
            });
            Add("Equal", (net, node) => { net.AddLayer(new Layers.Equal(node.Name, node.Input0, node.Input1)); });
            Add("Greater", (net, node) => { net.AddLayer(new Layers.Greater(node.Name, node.Input0, node.Input1)); });
            Add("GreaterOrEqual", (net, node) => { net.AddLayer(new Layers.GreaterOrEqual(node.Name, node.Input0, node.Input1)); });
            Add("IsInf", (net, node) =>
            {
                var detectNegative = node.GetOptionalInt("detect_negative", 1) != 0;
                var detectPositive = node.GetOptionalInt("detect_positive", 1) != 0;
                net.AddLayer(new Layers.IsInf(node.Name, node.Input0, detectNegative, detectPositive));
            });
            Add("IsNaN", (net, node) => { net.AddLayer(new Layers.IsNaN(node.Name, node.Input0)); });
            Add("Less", (net, node) => { net.AddLayer(new Layers.Less(node.Name, node.Input0, node.Input1)); });
            Add("LessOrEqual", (net, node) => { net.AddLayer(new Layers.LessOrEqual(node.Name, node.Input0, node.Input1)); });
            Add("Not", (net, node) => { net.AddLayer(new Layers.Not(node.Name, node.Input0)); });
            Add("Or", (net, node) => { net.AddLayer(new Layers.Or(node.Name, node.Input0, node.Input1)); });
            Add("Xor", (net, node) => { net.AddLayer(new Layers.Xor(node.Name, node.Input0, node.Input1)); });
            Add("Where", (net, node) => { net.AddLayer(new Layers.Where(node.Name, node.Input0, node.Input1, node.Input2)); });

            // Layer.Math
            Add("Abs", (net, node) => { net.AddLayer(new Layers.Abs(node.Name, node.Input0)); });
            Add("Add", (net, node) => { net.AddLayer(new Layers.Add(node.Name, node.Input0, node.Input1)); });
            Add("Ceil", (net, node) => { net.AddLayer(new Layers.Ceil(node.Name, node.Input0)); });
            Add("Clip", (net, node) =>
            {
                if (node.HasAttribute("min") || node.HasAttribute("max"))
                {
                    // Clip-1, Clip-6 with at least one attribute from min/max
                    var min = node.GetOptionalFloat("min", float.MinValue);
                    var minConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_min"), new[] { min });
                    net.AddConstant(minConstant);
                    var max = node.GetOptionalFloat("max", float.MaxValue);
                    var maxConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_max"), new[] { max });
                    net.AddConstant(maxConstant);
                    net.AddLayer(new Layers.Clip(node.Name, node.Input0, minConstant.index, maxConstant.index));
                }
                else
                {
                    // Clip-11, Clip-12, Clip-13 or Clip-1, Clip-6 with no min or max
                    var minInput = node.InputCount >= 2 ? node.Inputs[1] : "";
                    var maxInput = node.InputCount >= 3 ? node.Inputs[2] : "";
                    if (string.IsNullOrEmpty(minInput))
                        net.AddLayer(new Layers.Clip(node.Name, node.Input0));
                    else if (string.IsNullOrEmpty(maxInput))
                        net.AddLayer(new Layers.Clip(node.Name, node.Input0, node.Input1));
                    else
                        net.AddLayer(new Layers.Clip(node.Name, node.Input0, node.Input1, node.Input2));
                }
            });
            Add("CumSum", (net, node) =>
            {
                var reverse = node.GetOptionalInt("reverse", 0) == 1;
                var exclusive = node.GetOptionalInt("exclusive", 0) == 1;
                net.AddLayer(new Layers.CumSum(node.Name, node.Input0, node.Input1, reverse, exclusive));
            });
            Add("Div", (net, node) => { net.AddLayer(new Layers.Div(node.Name, node.Input0, node.Input1)); });
            Add("Einsum", (net, node) =>
            {
                net.AddLayer(new Layers.Einsum(node.Name, node.Inputs, node.GetRequiredString("equation")));
            });
            Add("Exp", (net, node) => { net.AddLayer(new Layers.Exp(node.Name, node.Input0)); });
            Add("Floor", (net, node) => { net.AddLayer(new Layers.Floor(node.Name, node.Input0)); });
            Add("Gemm", (net, node) =>
            {
                node.UnsupportedAttribute("alpha", 1.0f);
                node.UnsupportedAttribute("beta", 1.0f);

                var transposeA = node.GetOptionalInt("transA", 0) == 1;
                var transposeB = node.GetOptionalInt("transB", 0) == 1;

                var name = node.Name;
                if (node.InputCount == 3)
                    name += "_Gemm";

                net.AddLayer(new Layers.MatMul2D(name, node.Input0, transposeA, node.Input1, transposeB));

                if (node.InputCount == 3)
                {
                    net.AddLayer(new Layers.Add(node.Name, name, node.Input2));
                }
            });
            Add("Log", (net, node) => { net.AddLayer(new Layers.Log(node.Name, node.Input0)); });
            Add("MatMul", (net, node) =>
            {
                net.AddLayer(new Layers.MatMul(node.Name, node.Input0, node.Input1));
            });
            Add("Max", (net, node) => { net.AddLayer(new Layers.Max(node.Name, node.Inputs)); });
            Add("Mean", (net, node) => { net.AddLayer(new Layers.Mean(node.Name, node.Inputs)); });
            Add("Min", (net, node) => { net.AddLayer(new Layers.Min(node.Name, node.Inputs)); });
            Add("Mod", (net, node) => { net.AddLayer(new Layers.Mod(node.Name, node.Input0, node.Input1, node.GetOptionalInt("fmod", 0) != 0)); });
            Add("Mul", (net, node) => { net.AddLayer(new Layers.Mul(node.Name, node.Input0, node.Input1)); });
            Add("Neg", (net, node) => { net.AddLayer(new Layers.Neg(node.Name, node.Input0)); });
            Add("Pow", (net, node) =>
            {
                // Pow-1, Pow-7, Pow-12, Pow-13
                net.AddLayer(new Layers.Pow(node.Name, node.Input0, node.Input1));
            });
            Add("Reciprocal", (net, node) => { net.AddLayer(new Layers.Reciprocal(node.Name, node.Input0)); });
            Add("Round", (net, node) => { net.AddLayer(new Layers.Round(node.Name, node.Input0)); });
            Add("Shrink", (net, node) => { net.AddLayer(new Layers.Shrink(node.Name, node.Input0, node.GetOptionalFloat("bias", 0f), node.GetOptionalFloat("lambd", 0.5f))); });
            Add("Sign", (net, node) => { net.AddLayer(new Layers.Sign(node.Name, node.Input0)); });
            Add("Sqrt", (net, node) => { net.AddLayer(new Layers.Sqrt(node.Name, node.Input0)); });
            Add("Sub", (net, node) => { net.AddLayer(new Layers.Sub(node.Name, node.Input0, node.Input1)); });
            Add("Sum", (net, node) => { net.AddLayer(new Layers.Sum(node.Name, node.Inputs)); });

            // Layer.Normalization
            Add("BatchNormalization", (net, node) =>
            {
                net.AddLayer(new Layers.BatchNormalization(node.Name, node.Input0, node.Input1, node.Input2, node.Input3, node.Input4, node.EpsilonOptional()));
            });
            Add("InstanceNormalization", (net, node) =>
            {
                net.AddLayer(new Layers.InstanceNormalization(node.Name, node.Input0, node.Input1, node.Input2, node.EpsilonOptional()));
            });
            Add("LayerNormalization", (net, node) =>
            {
                node.UnsupportedAttribute("axis", -1);
                net.AddLayer(new Layers.LayerNormalization(node.Name, node.Input0, node.Input1, node.Input2, node.EpsilonOptional()));
            });
            Add("LRN", (net, node) =>
            {
                var bias = node.GetOptionalFloat("bias", 1.0f);
                var size = node.GetRequiredInt("size");
                net.AddLayer(new Layers.LRN(node.Name, node.Input0, node.AlphaOptional(0.0001f), node.BetaOptional(0.75f), bias, size));
            });

            // Layer.ObjectDetection
            Add("NonMaxSuppression", (net, node) =>
            {
                var centerPointBox = (node.GetOptionalInt("center_point_box", 0) == 0) ? Layers.CenterPointBox.Corners : Layers.CenterPointBox.Center;
                var scoreThreshold = node.InputCount == 5 ? node.Input4 : null;
                var iouThreshold = node.InputCount >= 4 ? node.Input3 : null;
                var maxOutputBoxesPerClass = node.InputCount >= 3 ? node.Input2 : null;
                net.AddLayer(new Layers.NonMaxSuppression(node.Name, node.Input0, node.Input1, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox));
            });
            Add("RoiAlign", (net, node) =>
            {
                Layers.RoiPoolingMode mode = node.RoiPoolingMode();

                int output_height = node.GetOptionalInt("output_height", 1);
                int output_width = node.GetOptionalInt("output_width", 1);
                int sampling_ratio = node.GetOptionalInt("sampling_ratio", 0);
                float spatial_scale = node.GetOptionalFloat("spatial_scale", 1.0f);

                net.AddLayer(new Layers.RoiAlign(node.Name, node.Input0, node.Input1, node.Input2, mode, output_height, output_width, sampling_ratio, spatial_scale));
            });

            // Layer.Pooling
            Add("AveragePool", (net, node) =>
            {
                node.UnsupportedAttribute("ceil_mode", 0);
                node.UnsupportedAttribute("dilations", new[] { 1, 1 });
                node.UnsupportedAttribute("storage_order", 0);
                node.UnsupportedAttribute("count_include_pad", 0);

                var autopad = node.AutoPadMode();

                var kernelShape = node.GetRequiredIntArray("kernel_shape");
                var pads = node.GetOptionalIntArray("pads", null);
                var strides = node.GetOptionalIntArray("strides", null);

                net.AddLayer(new Layers.AveragePool(node.Name, node.Input0, kernelShape, strides, pads, autopad));
            });
            Add("GlobalAveragePool", (net, node) =>
            {
                net.AddLayer(new Layers.GlobalAveragePool(node.Name, node.Input0));
            });
            Add("GlobalMaxPool", (net, node) =>
            {
                net.AddLayer(new Layers.GlobalMaxPool(node.Name, node.Input0));
            });
            Add("MaxPool", (net, node) =>
            {
                node.UnsupportedAttribute("ceil_mode", 0);
                node.UnsupportedAttribute("dilations", new[] { 1, 1 });
                node.UnsupportedAttribute("storage_order", 0);

                var autopad = node.AutoPadMode();

                var kernelShape = node.GetRequiredIntArray("kernel_shape");
                var pads = node.GetOptionalIntArray("pads", null);
                var strides = node.GetOptionalIntArray("strides", null);

                net.AddLayer(new Layers.MaxPool(node.Name, node.Input0, kernelShape, strides, pads, autopad));
            });

            // Layer.Random
            Add("Bernoulli", (net, node) =>
            {
                var dataType = node.GetDataType(defaultValue: DataType.Float);
                net.AddLayer(new Layers.Bernoulli(node.Name, node.Input0, dataType, node.Seed));
            });
            Add("Multinomial", (net, node) =>
            {
                node.IgnoredAttribute("dtype", "dtype can only be int32 or int64 which both map to TensorInt");
                var samples = node.GetOptionalInt("sample_size", 1);
                net.AddLayer(new Layers.Multinomial(node.Name, node.Input0, samples, node.Seed));
            });
            Add("RandomNormal", (net, node) =>
            {
                var mean = node.GetOptionalFloat("mean", 0.0f);
                var scale = node.GetOptionalFloat("scale", 1.0f);
                net.AddLayer(new Layers.RandomNormal(node.Name, node.Shape, mean, scale, node.Seed));
            });
            Add("RandomNormalLike", (net, node) =>
            {
                var mean = node.GetOptionalFloat("mean", 0.0f);
                var scale = node.GetOptionalFloat("scale", 1.0f);
                net.AddLayer(new Layers.RandomNormalLike(node.Name, node.Input0, mean, scale, node.Seed));
            });
            Add("RandomUniform", (net, node) =>
            {
                var low = node.GetOptionalFloat("low", 0.0f);
                var high = node.GetOptionalFloat("high", 1.0f);
                net.AddLayer(new Layers.RandomUniform(node.Name, node.Shape, low, high, node.Seed));
            });
            Add("RandomUniformLike", (net, node) =>
            {
                var low = node.GetOptionalFloat("low", 0.0f);
                var high = node.GetOptionalFloat("high", 1.0f);
                net.AddLayer(new Layers.RandomUniformLike(node.Name, node.Input0, low, high, node.Seed));
            });

            // Layer.Recurrent
            Add("LSTM", (net, node) =>
            {
                var hiddenSize = node.GetRequiredInt("hidden_size");
                var direction = node.Direction();
                var activations = node.Activations();
                var activationAlpha = node.GetOptionalFloatArray("activation_alpha", null);
                var activationBeta = node.GetOptionalFloatArray("activation_beta", null);
                var clip = node.GetOptionalFloat("clip", float.MaxValue);
                var inputForget = node.GetOptionalInt("input_forget", 0) != 0;
                var layout = node.Layout();

                net.AddLayer(new Layers.LSTM(node.Name, node.Inputs, node.Outputs, hiddenSize, direction, activations, activationAlpha, activationBeta, clip, inputForget, layout));
            });

            // Layer.Reduction
            Add("ReduceL1", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.index };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceL1(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceL2", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.index };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceL2(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceLogSum", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.index };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceLogSum(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceLogSumExp", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.index };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceLogSumExp(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceMax", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.index };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceMax(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceMean", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.index };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceMean(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceMin", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.index };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceMin(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceProd", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.index };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceProd(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceSum", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.index };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceSum(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceSumSquare", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.index };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceSumSquare(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });

            // Layer.Transformation
            Add("Cast", (net, node) =>
            {
                var toOnnxType = (TensorProto.Types.DataType)node.GetRequiredInt("to");
                var toDataType = ONNXNodeWrapper.DataTypeFromOnnxDataType(toOnnxType, OnUnsupported: () =>
                {
                    Warn(WarningType.Error, $"Unsupported tensor dataType: {toOnnxType}.");
                    Debug.LogError(Warnings.Last().Message);
                });
                net.AddLayer(new Layers.Cast(node.Name, node.Input0, toDataType));
            });
            Add("CastLike", (net, node) =>
            {
                net.AddLayer(new Layers.CastLike(node.Name, node.Input0, node.Input1));
            });
            Add("Concat", (net, node) =>
            {
                net.AddLayer(new Layers.Concat(node.Name, node.Inputs, node.Axis));
            });
            Add("DepthToSpace", (net, node) =>
            {
                var modeType = node.ModeOptional("DCR");
                var mode = modeType == "DCR" ? Layers.DepthToSpaceMode.DepthColumnRow : Layers.DepthToSpaceMode.ColumnRowDepth;
                net.AddLayer(new Layers.DepthToSpace(node.Name, node.Input0, node.BlockSize, mode));
            });
            Add("Expand", (net, node) =>
            {
                // Expand-8, Expand-13
                net.AddLayer(new Layers.Expand(node.Name, node.Input0, node.Input1));
            });
            Add("Flatten", (net, node) =>
            {
                var axis = node.AxisOptional(1);
                net.AddLayer(new Layers.Flatten(node.Name, node.Input0, axis));
            });
            Add("Dropout", (net, node) => { net.AddLayer(new Layers.Identity(node.Name, node.Input0)); });
            Add("Identity", (net, node) => { net.AddLayer(new Layers.Identity(node.Name, node.Input0)); });
            Add("Pad", (net, node) =>
            {
                var mode = node.PadMode();
                if (node.InputCount == 1)
                {
                    // Pad-1 or Pad-2
                    var pads = node.GetRequiredIntArray(node.HasAttribute("pads") ? "pads" : "paddings");
                    var padsConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_pads"), pads);
                    net.AddConstant(padsConstant);
                    var value = node.GetOptionalFloat("value", 0f);
                    var valueConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_value"), new[] { value });
                    net.AddConstant(valueConstant);
                    net.AddLayer(new Layers.Pad(node.Name, node.Input0, padsConstant.index, valueConstant.index, mode));
                }
                else if (node.InputCount == 2 || (node.InputCount == 3 && string.IsNullOrEmpty(node.Inputs[2])))
                {
                    // Pad-11, Pad-13, Pad-18 no constant value
                    net.AddLayer(new Layers.Pad(node.Name, node.Input0, node.Input1, mode));
                }
                else if (node.InputCount == 3 || (node.InputCount == 4 && string.IsNullOrEmpty(node.Inputs[3])))
                {
                    // Pad-11, Pad-13, Pad-18 with constant value and no axes
                    net.AddLayer(new Layers.Pad(node.Name, node.Input0, node.Input1, node.Input2, mode));
                }
                else
                {
                    // Pad-18 with axes
                    net.AddLayer(new Layers.Pad(node.Name, node.Input0, node.Input1, node.Inputs[2], node.Input3, mode));
                }
            });
            Add("Reshape", (net, node) =>
            {
                if (node.HasAttribute("shape"))
                {
                    // Reshape-1, Reshape-5
                    var shape = node.GetRequiredIntArray("shape");
                    var shapeConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_shape"), shape);
                    net.AddConstant(shapeConstant);
                    net.AddLayer(new Layers.Reshape(node.Name, node.Input0, shapeConstant.index));
                }
                else
                {
                    // Reshape-13, Reshape-14
                    var allowZero = node.GetOptionalInt("allowzero", 0) != 0;
                    net.AddLayer(new Layers.Reshape(node.Name, node.Input0, node.Input1, allowZero));
                }
            });
            Add("Resize", (net, node) =>
            {
                node.UnsupportedAttribute("cubic_coeff_a", -0.75f);
                node.UnsupportedAttribute("exclude_outside", 0);
                node.UnsupportedAttribute("extrapolation_value", 0);
                var coordinateTransformMode = node.CoordinateTransformMode();
                var mode = node.InterpolationMode();
                var nearestMode = node.NearestMode();
                var axes = node.GetOptionalIntArray("axes", null);

                if (node.InputCount == 2)
                {
                    // Resize-10
                    net.AddLayer(new Layers.Resize(node.Name, node.Input0, node.Input1, Layers.ScaleMode.Scales, mode, Layers.CoordTransformMode.Asymmetric, Layers.NearestMode.Floor, axes));
                }
                else if (node.InputCount == 3)
                {
                    // Resize-11, Resize-13 with scales
                    net.AddLayer(new Layers.Resize(node.Name, node.Input0, node.Input2, Layers.ScaleMode.Scales, mode, coordinateTransformMode, nearestMode, axes));
                }
                else if (node.InputCount == 4)
                {
                    // Resize-11, Resize-13 with sizes
                    net.AddLayer(new Layers.Resize(node.Name, node.Input0, node.Input3, Layers.ScaleMode.Sizes, mode, coordinateTransformMode, nearestMode, axes));
                }
            });
            Add("Slice", (net, node) =>
            {
                if (node.HasAttribute("starts"))
                {
                    // Slice-1
                    var starts = node.GetRequiredIntArray("starts");
                    var startsConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_starts"), starts);
                    net.AddConstant(startsConstant);
                    var ends = node.GetRequiredIntArray("ends");
                    var endsConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_ends"), ends);
                    net.AddConstant(endsConstant);
                    if (node.HasAttribute("axes"))
                    {
                        var axes = node.GetRequiredIntArray("axes");
                        var axesConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_axes"), axes);
                        net.AddConstant(axesConstant);
                        net.AddLayer(new Layers.Slice(node.Name, node.Input0, startsConstant.index, endsConstant.index, axesConstant.index));
                    }
                    else
                    {
                        net.AddLayer(new Layers.Slice(node.Name, node.Input0, startsConstant.index, endsConstant.index));
                    }
                }
                else
                {
                    // Slice-10, Slice-11, Slice-13
                    if (node.InputCount == 3)
                        net.AddLayer(new Layers.Slice(node.Name, node.Input0, node.Input1, node.Input2));
                    else if (node.InputCount == 4)
                        net.AddLayer(new Layers.Slice(node.Name, node.Input0, node.Input1, node.Input2, node.Input3));
                    else if (node.InputCount == 5)
                        net.AddLayer(new Layers.Slice(node.Name, node.Input0, node.Input1, node.Input2, node.Input3, node.Input4));
                }
            });
            Add("SpaceToDepth", (net, node) =>
            {
                net.AddLayer(new Layers.SpaceToDepth(node.Name, node.Input0, node.BlockSize));
            });
            Add("Split", (net, node) =>
            {
                var axis = node.AxisOptional(0);
                if (node.HasAttribute("num_outputs"))
                {
                    // Split-18 with "num_outputs" attribute
                    net.AddLayer(new Layers.Split(node.Name, node.Input0, node.Outputs, axis, node.GetRequiredInt("num_outputs")));
                }
                else if (node.HasAttribute("split"))
                {
                    // Split-1, Split-2, Split-11 with "split" attribute
                    var split = node.GetRequiredIntArray("split");
                    var splitConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_split"), split);
                    net.AddConstant(splitConstant);
                    net.AddLayer(new Layers.Split(node.Name, node.Input0, splitConstant.index, node.Outputs, axis));
                }
                else if (node.InputCount == 2)
                {
                    // Split-1, Split-13, Split-18 with "split" input
                    net.AddLayer(new Layers.Split(node.Name, node.Input0, node.Input1, node.Outputs, axis));
                }
                else
                {
                    // Split-1, Split-2, Split-11, Split-13, Split-18 with no given "split" or "num_outputs"
                    net.AddLayer(new Layers.Split(node.Name, node.Input0, node.Outputs, axis, node.Outputs.Length));
                }
            });
            Add("Squeeze", (net, node) =>
            {
                if (node.HasAttribute("axes"))
                {
                    // Squeeze-1, Squeeze-11 with given axes
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    net.AddLayer(new Layers.Squeeze(node.Name, node.Input0, axesConstant.index));
                }
                else
                {
                    // Squeeze-13 or Squeeze-1, Squeeze-11 without given axes
                    if (node.InputCount == 2)
                        net.AddLayer(new Layers.Squeeze(node.Name, node.Input0, node.Input1));
                    else
                        net.AddLayer(new Layers.Squeeze(node.Name, node.Input0));
                }
            });
            Add("Tile", (net, node) =>
            {
                net.AddLayer(new Layers.Tile(node.Name, node.Input0, node.Input1));
            });
            Add("Transpose", (net, node) =>
            {
                var permutations = node.GetOptionalIntArray("perm", null);
                net.AddLayer(new Layers.Transpose(node.Name, node.Input0, permutations));
            });
            Add("Trilu", (net, node) =>
            {
                var upper = node.GetOptionalInt("upper", 1);

                if (node.InputCount == 1)
                    net.AddLayer(new Layers.Trilu(node.Name, node.Input0, (Layers.TriluMode)upper));
                else
                    net.AddLayer(new Layers.Trilu(node.Name, node.Input0, node.Input1, (Layers.TriluMode)upper));
            });
            Add("Upsample", (net, node) =>
            {
                var coordinateTransformMode = Layers.CoordTransformMode.Asymmetric;
                var mode = node.InterpolationMode();
                var nearestMode = Layers.NearestMode.Floor;
                if (node.HasAttribute("scales"))
                {
                    // Upsample-7
                    var scales = node.GetRequiredFloatArray("scales");
                    var scalesConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_scales"), scales);
                    net.AddConstant(scalesConstant);
                    net.AddLayer(new Layers.Resize(node.Name, node.Input0, scalesConstant.index, Layers.ScaleMode.Scales, mode, coordinateTransformMode, nearestMode, null));
                }
                else
                {
                    // Upsample-9
                    net.AddLayer(new Layers.Resize(node.Name, node.Input0, node.Input1, Layers.ScaleMode.Scales, mode, coordinateTransformMode, nearestMode, null));
                }
            });
            Add("Unsqueeze", (net, node) =>
            {
                if (node.HasAttribute("axes"))
                {
                    // Unsqueeze-1, Unsqueeze-11
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueIndex(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    net.AddLayer(new Layers.Unsqueeze(node.Name, node.Input0, axesConstant.index));
                }
                else
                {
                    // Unsqueeze-13
                    net.AddLayer(new Layers.Unsqueeze(node.Name, node.Input0, node.Input1));
                }
            });

            // Layer.Trigonometric
            Add("Acos", (net, node) => { net.AddLayer(new Layers.Acos(node.Name, node.Input0)); });
            Add("Acosh", (net, node) => { net.AddLayer(new Layers.Acosh(node.Name, node.Input0)); });
            Add("Asin", (net, node) => { net.AddLayer(new Layers.Asin(node.Name, node.Input0)); });
            Add("Asinh", (net, node) => { net.AddLayer(new Layers.Asinh(node.Name, node.Input0)); });
            Add("Atan", (net, node) => { net.AddLayer(new Layers.Atan(node.Name, node.Input0)); });
            Add("Atanh", (net, node) => { net.AddLayer(new Layers.Atanh(node.Name, node.Input0)); });
            Add("Cos", (net, node) => { net.AddLayer(new Layers.Cos(node.Name, node.Input0)); });
            Add("Cosh", (net, node) => { net.AddLayer(new Layers.Cosh(node.Name, node.Input0)); });
            Add("Sin", (net, node) => { net.AddLayer(new Layers.Sin(node.Name, node.Input0)); });
            Add("Sinh", (net, node) => { net.AddLayer(new Layers.Sinh(node.Name, node.Input0)); });
            Add("Tan", (net, node) => { net.AddLayer(new Layers.Tan(node.Name, node.Input0)); });

            // Non standard ONNX
            Add("Swish", (net, node) => { net.AddLayer(new Layers.Swish(node.Name, node.Input0)); });
            Add("ImageScaler", (net, node) =>
            {
                var attrBias = node.Bias;
                var maxElements = attrBias.Length;
                var attrScale = Enumerable.Repeat(node.GetOptionalFloat("scale", 1.0f), maxElements).ToArray();

                using var scale = new TensorFloat(new TensorShape(maxElements), attrScale);
                using var bias = new TensorFloat(new TensorShape(maxElements), attrBias);

                var scaleConstantName = net.GetUniqueIndex($"{node.Name}_Scale");
                net.AddConstant(new Layers.Constant(scaleConstantName, scale));
                var biasConstantName = net.GetUniqueIndex($"{node.Name}_Bias");
                net.AddConstant(new Layers.Constant(biasConstantName, bias));
                net.AddLayer(new Layers.ScaleBias(node.Name, node.Input0, scaleConstantName, biasConstantName));
            });
        }

        internal static readonly Dictionary<string, Action<Model, ONNXNodeWrapper>> m_NodeImporters =
            new Dictionary<string, Action<Model, ONNXNodeWrapper>>();

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

            // Parse producer meta data
            foreach (var opsetSetIdProto in onnxModel.OpsetImport)
            {
                if (string.IsNullOrEmpty(opsetSetIdProto.Domain))
                    model.DefaultOpsetVersion = opsetSetIdProto.Version;
            }
            model.ProducerName = onnxModel.ProducerName;
            if (!String.IsNullOrEmpty(onnxModel.ProducerVersion))
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
                            return model;
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
                var nodeId = node.Name;
                var opType = node.OperatorType;

                if (!m_NodeImporters.ContainsKey(opType))
                {
                    Warn(WarningType.Error, $"{opType} not supported");
                    Debug.LogError(Warnings.Last().Message);
                    continue;
                }

                try
                {
                    m_NodeImporters[opType](model, node);
                }
                catch (Exception e)
                {
                    Warn(WarningType.Error, e.Message);
                    Debug.LogError(Warnings.Last().Message);
                }
            }

            // delete unused outputs
            HashSet<string> outputsToRemove = model.outputs.Select(o => o.index).ToHashSet();
            foreach (var layer in model.layers)
            {
                outputsToRemove.Remove(layer.index);
                for (var i = 1; i < layer.outputs?.Length; i++)
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
                l.index = TrimTensorflowName(l.index);
                for(int i = 0; i < l.inputs.Length; i++)
                    l.inputs[i] = TrimTensorflowName(l.inputs[i]);
                if (l.outputs != null)
                {
                    for (int i = 0; i < l.outputs.Length; i++)
                        l.outputs[i] = TrimTensorflowName(l.outputs[i]);
                }
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
