using Onnx;
using UnityEngine;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Unity.Mathematics;
using static Unity.Sentis.ONNX.ONNXModelConverter;

[assembly: InternalsVisibleToAttribute("Unity.Sentis.EditorTests")]

namespace Unity.Sentis.ONNX
{
    internal class ONNXNodeWrapper
    {
        // Layer identification (name and op)
        public string GetName()
        {
            var name = m_ONNXNode.Output.Count > 0 ? m_ONNXNode.Output[0] : "";
            if (string.IsNullOrEmpty(name))
                return $"{m_ONNXNode.OpType}_{m_ONNXNode.ToString().GetHashCode()}";
            return name;
        }
        public string Name => GetName();
        public string OperatorType => m_ONNXNode.OpType;

        // Outputs
        public string[] Outputs
        {
            get
            {
                var outputs = new string[m_ONNXNode.Output.Count];
                for (var i = 0; i < m_ONNXNode.Output.Count; i++)
                    outputs[i] = OptionalOutput(i);
                return outputs;
            }
        }
        public int OutputCount => m_ONNXNode.Output.Count;
        public string Output0 => RequiredOutput(0);
        public string Output1 => RequiredOutput(1);
        public string Output2 => RequiredOutput(2);

        // Inputs
        public int InputCount => m_ONNXNode.Input.Count;
        public string[] Inputs => m_ONNXNode.Input.ToArray();
        public string Input0 => RequiredInput(0);
        public string Input1 => RequiredInput(1);
        public string Input2 => RequiredInput(2);
        public string Input3 => RequiredInput(3);
        public string Input4 => RequiredInput(4);
        public string Input5 => RequiredInput(5);
        public string Input6 => RequiredInput(6);
        public string Input7 => RequiredInput(7);
        public string Input8 => RequiredInput(8);

        public int? Seed
        {
            get
            {
                if (HasAttribute("seed"))
                    return math.asint(GetRequiredFloat("seed"));
                return null;
            }
        }

        // ---------------------------------------------------------------------------------
        // Implementation
        private NodeProto m_ONNXNode;
        private List<ImporterWarning> m_ImporterWarnings;

        public ONNXNodeWrapper(NodeProto ONNXNode)
        {
            m_ONNXNode = ONNXNode;
            m_ImporterWarnings = new List<ImporterWarning>();
        }

        // Logging helpers
        public void Warn(string message, WarningType severity)
        {
            m_ImporterWarnings.Add(new ImporterWarning(message, severity));
            Debug.LogWarning(message);
        }

        public bool HasAttribute(string name)
        {
            AttributeProto attr;
            return TryFindAttribute(name, out attr);
        }

        public void UnsupportedAttribute(string name)
        {
            AttributeProto attr;
            if (TryFindAttribute(name, out attr))
                Warn($"<b>{OperatorType}:</b> Unsupported attribute `<b>{name}</b>`. Value will be ignored.", WarningType.Warning);
        }
        public void UnsupportedAttribute(string name, int defaultValue)
        {
            if (GetOptionalInt(name, defaultValue) != defaultValue)
                Warn($"<b>{OperatorType}:</b> Unsupported attribute `<b>{name}</b>`. Value will be ignored and defaulted to [{string.Join(", ", defaultValue)}].", WarningType.Warning);
        }
        public void UnsupportedAttribute(string name, float defaultValue)
        {
            if (GetOptionalFloat(name, defaultValue) != defaultValue)
                Warn($"<b>{OperatorType}:</b> Unsupported attribute `<b>{name}</b>`. Value will be ignored and defaulted to [{string.Join(", ", defaultValue)}].", WarningType.Warning);
        }
        public void UnsupportedAttribute(string name, string defaultValue)
        {
            if (GetOptionalString(name, defaultValue) != defaultValue)
                Warn($"<b>{OperatorType}:</b> Unsupported attribute `<b>{name}</b>`. Value will be ignored and defaulted to [{string.Join(", ", defaultValue)}].", WarningType.Warning);
        }
        public void UnsupportedAttribute(string name, int[] defaultValue)
        {
            var valueArray = GetOptionalIntArray(name, defaultValue);
            if (!Enumerable.SequenceEqual(valueArray, defaultValue))
                Warn($"<b>{OperatorType}:</b> Unsupported attribute `<b>{name}</b>`. Value will be ignored and defaulted to [{string.Join(", ", defaultValue)}].", WarningType.Warning);
        }
        public void UnsupportedAttribute(string name, string[] defaultValue)
        {
            var stringArray = GetOptionalStringArray(name, defaultValue);
            if (!Enumerable.SequenceEqual(stringArray, defaultValue))
                Warn($"<b>{OperatorType}:</b> Unsupported attribute `<b>{name}</b>`. Value will be ignored and defaulted to [{string.Join(", ", defaultValue)}].", WarningType.Warning);
        }
        public void UnsupportedAttribute(string name, Func<int, bool> predicate, int[] defaultValue)
        {
            var valueArray = GetOptionalIntArray(name, defaultValue);
            if (!Enumerable.All(valueArray, predicate))
                Warn($"<b>{OperatorType}:</b> Unsupported attribute `<b>{name}</b>`. Value will be ignored and defaulted to [{string.Join(", ", defaultValue)}].", WarningType.Warning);
        }
        public void IgnoredAttribute(string name, string reasonToIgnore) { }

        // Input helpers
        internal string RequiredInput(int inputIndex)
        {
            if ((inputIndex >= m_ONNXNode.Input.Count) || (m_ONNXNode.Input[inputIndex] == ""))
                throw new OnnxLayerImportException($"required Input {inputIndex} was not found.");

            return m_ONNXNode.Input[inputIndex];
        }
        internal string OptionalInput(int inputIndex)
        {
            if (inputIndex >= m_ONNXNode.Input.Count || string.IsNullOrEmpty(m_ONNXNode.Input[inputIndex]))
                return string.Empty;

            return m_ONNXNode.Input[inputIndex];
        }

        // Output helpers
        internal string RequiredOutput(int outputIndex)
        {
            if (outputIndex == 0)
                return GetName();
            if (outputIndex >= m_ONNXNode.Output.Count || string.IsNullOrEmpty(m_ONNXNode.Output[outputIndex]))
                throw new OnnxLayerImportException($"required Output {outputIndex} was not found.");

            return m_ONNXNode.Output[outputIndex];
        }
        internal string OptionalOutput(int outputIndex)
        {
            if (outputIndex == 0)
                return GetName();
            if (outputIndex >= m_ONNXNode.Output.Count || string.IsNullOrEmpty(m_ONNXNode.Output[outputIndex]))
                return string.Empty;

            return m_ONNXNode.Output[outputIndex];
        }

        // Attribute helpers
        internal bool TryFindAttribute(string name, out AttributeProto attr)
        {
            return TryFindAttribute(name, AttributeProto.Types.AttributeType.Undefined, out attr);
        }
        internal bool TryFindAttribute(string name, AttributeProto.Types.AttributeType type, out AttributeProto attr)
        {
            const AttributeProto.Types.AttributeType undefined = AttributeProto.Types.AttributeType.Undefined;
            var attributes = m_ONNXNode.Attribute;
            for (var i = 0; i < attributes.Count; ++i)
            {
                attr = attributes[i];
                if (attr.Name == name && (attr.Type == type || attr.Type == undefined || type == undefined))
                    return true;
            }
            attr = null;
            return false;
        }
        internal AttributeProto FindAttribute(string name, AttributeProto.Types.AttributeType type = AttributeProto.Types.AttributeType.Undefined)
        {
            AttributeProto attr = null;
            if (TryFindAttribute(name, type, out attr))
                return attr;

            throw new OnnxLayerImportException($"Couldn't find attribute {name} of type {type}");
        }
        public float GetOptionalFloat(string name, float defaultValue)
        {
            try { return GetRequiredFloat(name); }
            catch (OnnxLayerImportException) { return defaultValue; }
        }
        public float GetRequiredFloat(string name)
        {
            return FindAttribute(name, AttributeProto.Types.AttributeType.Float).F;
        }
        public float[] GetOptionalFloatArray(string name, float[] defaultValue)
        {
            try { return GetRequiredFloatArray(name); }
            catch (OnnxLayerImportException) { return defaultValue; }
        }
        public float[] GetRequiredFloatArray(string name)
        {
            var attribute = FindAttribute(name, AttributeProto.Types.AttributeType.Floats);
            return attribute.Floats.ToArray();
        }
        public TensorProto GetRequiredTensor(string name)
        {
            return FindAttribute(name, AttributeProto.Types.AttributeType.Tensor).T;
        }
        public int GetOptionalInt(string name, int defaultValue)
        {
            try { return GetRequiredInt(name); }
            catch (OnnxLayerImportException) { return defaultValue; }
        }
        public int GetRequiredInt(string name)
        {
            long v = FindAttribute(name, AttributeProto.Types.AttributeType.Int).I;
            return v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v;
        }
        public int[] GetOptionalIntArray(string name, int[] defaultValue)
        {
            try { return GetRequiredIntArray(name); }
            catch (OnnxLayerImportException) { return defaultValue; }
        }
        public int[] GetRequiredIntArray(string name)
        {
            var attribute = FindAttribute(name,AttributeProto.Types.AttributeType.Ints);
            return attribute.Ints.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();
        }
        public string GetOptionalString(string name, string defaultValue)
        {
            try { return GetRequiredString(name); }
            catch (OnnxLayerImportException) { return defaultValue; }
        }
        public string GetRequiredString(string name)
        {
            var raw = FindAttribute(name, AttributeProto.Types.AttributeType.String).S;
            return raw.ToStringUtf8();
        }
        public string[] GetOptionalStringArray(string name, string[] defaultValue)
        {
            try { return GetRequiredStringArray(name); }
            catch (OnnxLayerImportException) { return defaultValue; }
        }
        public string[] GetRequiredStringArray(string name)
        {
            var attribute = FindAttribute(name,AttributeProto.Types.AttributeType.Strings);
            return attribute.Strings.Select(s => s.ToStringUtf8()).ToArray();
        }

        public DataType GetDataType(DataType defaultValue)
        {
            if (!HasAttribute("dtype"))
                return defaultValue;
            var dtype = (TensorProto.Types.DataType)GetRequiredInt("dtype");
            return DataTypeFromOnnxDataType(dtype, defaultValue, () => Warn($"DataType `{dtype}` is not supported.", WarningType.Warning));
        }

        public static DataType DataTypeFromOnnxDataType(TensorProto.Types.DataType dataType, DataType defaultValue = DataType.Float, Action OnUnsupported = null)
        {
            switch (dataType)
            {
                case TensorProto.Types.DataType.Undefined:
                    return defaultValue;
                case TensorProto.Types.DataType.Float:
                case TensorProto.Types.DataType.Float16:
                case TensorProto.Types.DataType.Double:
                case TensorProto.Types.DataType.Bfloat16:
                    return DataType.Float;
                case TensorProto.Types.DataType.Uint8:
                case TensorProto.Types.DataType.Int8:
                case TensorProto.Types.DataType.Uint16:
                case TensorProto.Types.DataType.Int16:
                case TensorProto.Types.DataType.Int32:
                case TensorProto.Types.DataType.Int64:
                case TensorProto.Types.DataType.Bool:
                case TensorProto.Types.DataType.Uint32:
                case TensorProto.Types.DataType.Uint64:
                    return DataType.Int;
                case TensorProto.Types.DataType.String:
                case TensorProto.Types.DataType.Complex64:
                case TensorProto.Types.DataType.Complex128:
                default:
                    OnUnsupported?.Invoke();
                    return defaultValue;
            }
        }

        public Layers.AutoPad AutoPadMode()
        {
            var autoPad = GetOptionalString("auto_pad", "NOTSET");
            Layers.AutoPad autoPadType = Layers.AutoPad.NotSet;
            if (autoPad == "VALID")
                autoPadType = Layers.AutoPad.Valid;
            else if (autoPad == "SAME_UPPER")
                autoPadType = Layers.AutoPad.SameUpper;
            else if (autoPad == "SAME_LOWER")
                autoPadType = Layers.AutoPad.SameLower;

            return autoPadType;
        }

        public Layers.PadMode PadMode()
        {
            var mode = GetOptionalString("mode", "constant");
            var modeType = Layers.PadMode.Constant;
            switch (mode)
            {
                case "constant":
                    modeType = Layers.PadMode.Constant;
                    break;
                case "reflect":
                    modeType = Layers.PadMode.Reflect;
                    break;
                case "edge":
                    modeType = Layers.PadMode.Edge;
                    break;
                case "wrap":
                    modeType = Layers.PadMode.Wrap;
                    break;
            }
            return modeType;
        }

        public Layers.RnnDirection Direction()
        {
            return GetOptionalString("direction", "forward") switch
            {
                "forward" => Layers.RnnDirection.Forward,
                "reverse" => Layers.RnnDirection.Reverse,
                "bidirectional" => Layers.RnnDirection.Bidirectional,
                _ => Layers.RnnDirection.Forward
            };
        }

        public Layers.RnnLayout Layout()
        {
            return (Layers.RnnLayout)GetOptionalInt("layout", 0);
        }

        public Layers.RnnActivation[] Activations()
        {
            if (!HasAttribute("activations"))
                return null;
            var activationStrings = GetRequiredStringArray("activations");
            var activations = new Layers.RnnActivation[activationStrings.Length];
            for (var i = 0; i < activations.Length; i++)
            {
                if (Enum.TryParse<Layers.RnnActivation>(activationStrings[i], out var activation))
                    activations[i] = activation;
                else
                    Warn($"RnnActivation `{activationStrings[i]}` is not supported for type {OperatorType}.", WarningType.Warning);
            }

            return activations;
        }

        public Layers.ScatterReductionMode ScatterReductionMode()
        {
            var reduction = GetOptionalString("reduction", "none");
            var reductionMode = Layers.ScatterReductionMode.None;
            if (reduction == "add")
                reductionMode = Layers.ScatterReductionMode.Add;
            else if (reduction == "mul")
                reductionMode = Layers.ScatterReductionMode.Mul;
            else if (reduction != "none")
                Warn($"Reduction `{reduction}` is not supported for type {OperatorType}.", WarningType.Warning);
            return reductionMode;
        }

        public Layers.InterpolationMode InterpolationMode()
        {
            string mode = GetOptionalString("mode", "nearest");

            Layers.InterpolationMode outputMode = Layers.InterpolationMode.Nearest;
            if (mode == "linear" || mode == "bilinear")
                outputMode = Layers.InterpolationMode.Linear;
            else if (mode != "nearest")
                Warn($"Mode `{mode}` is not supported for type {OperatorType}.", WarningType.Warning);

            return outputMode;
        }

        public Layers.CoordTransformMode CoordinateTransformMode()
        {
            string mode = GetOptionalString("coordinate_transformation_mode", "half_pixel");

            Layers.CoordTransformMode outputMode = Layers.CoordTransformMode.HalfPixel;
            if (mode == "half_pixel")
                outputMode = Layers.CoordTransformMode.HalfPixel;
            else if (mode == "pytorch_half_pixel")
                outputMode = Layers.CoordTransformMode.PytorchHalfPixel;
            else if (mode == "align_corners")
                outputMode = Layers.CoordTransformMode.AlignCorners;
            else if (mode == "asymmetric")
                outputMode = Layers.CoordTransformMode.Asymmetric;
            else
                Warn($"Mode `{mode}` is not supported for type {OperatorType}.", WarningType.Warning);

            return outputMode;
        }

        public Layers.NearestMode NearestMode()
        {
            string mode = GetOptionalString("nearest_mode", "round_prefer_floor");

            Layers.NearestMode outputMode = Layers.NearestMode.RoundPreferFloor;
            if (mode == "round_prefer_floor")
                outputMode = Layers.NearestMode.RoundPreferFloor;
            else if (mode == "round_prefer_ceil")
                outputMode = Layers.NearestMode.RoundPreferCeil;
            else if (mode == "floor")
                outputMode = Layers.NearestMode.Floor;
            else if (mode == "ceil")
                outputMode = Layers.NearestMode.Ceil;
            else
                Warn($"Mode `{mode}` is not supported for type {OperatorType}.", WarningType.Warning);

            return outputMode;
        }
    }
}
