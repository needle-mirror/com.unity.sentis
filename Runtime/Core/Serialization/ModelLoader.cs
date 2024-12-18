// #define DEBUG_TIMING
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using Unity.Sentis.Google.FlatBuffers;
using SentisFlatBuffer;
using Unity.Sentis.Layers;

[assembly: InternalsVisibleTo("Unity.Sentis.RuntimeTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis
{
    /// <summary>
    /// Provides methods for loading models.
    /// </summary>
    public static class ModelLoader
    {
        /// <summary>
        /// Converts a binary `ModelAsset` representation of a neural network to an object-oriented `Model` representation.
        /// </summary>
        /// <param name="modelAsset">The binary `ModelAsset` model</param>
        /// <returns>The loaded `Model`</returns>
        public static Model Load(ModelAsset modelAsset)
        {
            var modelDescriptionBytes = modelAsset.modelAssetData.value;
            var modelWeightsBytes = new byte[modelAsset.modelWeightsChunks.Length][];
            for (var i = 0; i < modelAsset.modelWeightsChunks.Length; i++)
                modelWeightsBytes[i] = modelAsset.modelWeightsChunks[i].value;
            return LoadModel(modelDescriptionBytes, modelWeightsBytes);
        }

        /// <summary>
        /// Loads a model that has been serialized to disk.
        /// </summary>
        /// <param name="path">The path of the binary serialized model</param>
        /// <returns>The loaded `Model`</returns>
        public static Model Load(string path)
        {
            using var fileStream = File.Open(path, FileMode.Open);
            return Load(fileStream);
        }

        /// <summary>
        /// Loads a model that has been serialized to a stream.
        /// </summary>
        /// <param name="stream">The stream to load the serialized model from.</param>
        /// <returns>The loaded `Model`.</returns>
        public static Model Load(Stream stream)
        {
            try
            {
                var model = new Model();

                var prefixSizeBytes = new byte[sizeof(int)];
                stream.Read(prefixSizeBytes);
                var modelDescriptionSize = BitConverter.ToInt32(prefixSizeBytes);
                var modelDescriptionBytes = new byte[modelDescriptionSize + sizeof(int)];
                System.Buffer.BlockCopy(prefixSizeBytes, 0, modelDescriptionBytes, 0, sizeof(int));
                stream.Read(modelDescriptionBytes, sizeof(int), modelDescriptionSize);
                var weightBuffersConstantsOffsets = LoadModelDescription(modelDescriptionBytes, ref model);

                for (var i = 0; i < weightBuffersConstantsOffsets.Length; i++)
                {
                    stream.Read(prefixSizeBytes);
                    var modelWeightsChunkSize = BitConverter.ToInt32(prefixSizeBytes);
                    var modelWeightsBufferBytes = new byte[modelWeightsChunkSize + sizeof(int)];
                    System.Buffer.BlockCopy(prefixSizeBytes, 0, modelWeightsBufferBytes, 0, sizeof(int));
                    stream.Read(modelWeightsBufferBytes, sizeof(int), modelWeightsChunkSize);
                    LoadModelWeights(modelWeightsBufferBytes, weightBuffersConstantsOffsets[i], ref model);
                }

                return model;
            }
            catch (Exception e)
            {
                D.LogError($"Failed to load serialized Sentis model, ensure model was exported with Sentis 1.4 or newer. ({e.Message})");
                return null;
            }
        }

        internal static int OptionalInput(this Chain chain, int index)
        {
            var input = index < chain.InputsLength ? chain.Inputs(index) : -1;
            return input;
        }

        internal static int RequiredInput(this Chain chain, int index)
        {
            Logger.AssertIsTrue(index < chain.InputsLength, "Input Required");
            var input = chain.Inputs(index);
            Logger.AssertIsTrue(input != -1, "Input Required");
            return input;
        }

        internal static int OptionalOutput(this Chain chain, int index)
        {
            var output = index < chain.OutputsLength ? chain.Outputs(index) : -1;
            return output;
        }

        internal static int RequiredOutput(this Chain chain, int index)
        {
            Logger.AssertIsTrue(index < chain.OutputsLength, "Output Required");
            var output = chain.Outputs(index);
            Logger.AssertIsTrue(output != -1, "Output Required");
            return output;
        }

        internal static Model LoadModelDescription(byte[] modelDescription)
        {
            var model = new Model();
            LoadModelDescription(modelDescription, ref model);
            return model;
        }

        internal static Model LoadModel(byte[] modelDescription, byte[][] modelWeights)
        {
            var model = new Model();
            var weightsConstantIndexesOffsets = LoadModelDescription(modelDescription, ref model);
            for (var i = 0; i < weightsConstantIndexesOffsets.Length; i++)
                LoadModelWeights(modelWeights[i], weightsConstantIndexesOffsets[i], ref model);
            return model;
        }

        static List<(int, int)>[] LoadModelDescription(byte[] modelDescription, ref Model model)
        {
            var bb = new ByteBuffer(modelDescription, sizeof(int));
            var program = Program.GetRootAsProgram(bb);
            var numWeightsBuffers = program.SegmentsLength;
            var weightBuffersConstantsOffsets = new List<(int, int)>[numWeightsBuffers];
            for (var i = 0; i < numWeightsBuffers; i++)
                weightBuffersConstantsOffsets[i] = new List<(int, int)>();

            try
            {
                if (program.Version < ModelWriter.version)
                    program = ModelUpgrader.Upgrade(program);
                if (program.Version > ModelWriter.version)
                    D.LogWarning("Serialized model was exported in a newer version of Sentis than the current installed version and may not work as expected. Update the Sentis package to ensure compatibility.");
                var executionPlan = program.ExecutionPlan.Value;
                int inputCount = executionPlan.InputsLength;
                for (int i = 0; i < inputCount; i++)
                {
                    int input = executionPlan.Inputs(i);
                    var inputDesc = executionPlan.Values(input).Value.ValAsTensor();
                    DynamicTensorShape shape;
                    if (inputDesc.ShapeDynamism == TensorShapeDynamism.STATIC)
                    {
                        shape = new DynamicTensorShape(new TensorShape(inputDesc.GetFixedSizesArray()));
                    }
                    else
                    {
                        shape = DynamicTensorShape.DynamicOfRank(inputDesc.DynamicSizesLength);
                        for (int k = 0; k < shape.rank; k++)
                        {
                            var ddim = inputDesc.DynamicSizes(k).Value;
                            shape[k] = ddim.ValType == SymbolicDim.Int ? DynamicTensorDim.Int(ddim.ValAsInt().IntVal) : DynamicTensorDim.Param(ddim.ValAsByte().ByteVal);
                        }
                    }

                    model.AddInput(executionPlan.InputsName(i), input, (DataType)inputDesc.ScalarType, shape);
                }
                model.outputs = new List<Model.Output>();
                for (int i = 0; i < executionPlan.OutputsLength; i++)
                {
                    model.AddOutput(executionPlan.OutputsName(i), executionPlan.Outputs(i));
                }

                HashSet<int> constants = new HashSet<int>();
                for (int i = 0; i < executionPlan.ChainsLength; i++)
                {
                    var chain = executionPlan.Chains(i).Value;
                    var index = chain.Outputs(0);

                    for (int k = 0; k < chain.InputsLength; k++)
                    {
                        var input = chain.Inputs(k);
                        if (input == -1)
                            continue;
                        if (constants.Contains(input))
                            continue;
                        var constantTensor = executionPlan.Values(input).Value.ValAsTensor();
                        if (constantTensor.ConstantBufferIdx == 0)
                            continue;
                        var shape = new TensorShape(constantTensor.GetFixedSizesArray());
                        int lengthByte = constantTensor.LengthByte;
                        model.AddConstant(new Constant(input, shape, (DataType)constantTensor.ScalarType, lengthByte));
                        var idx = (int)(constantTensor.ConstantBufferIdx - 1);
                        var offset = constantTensor.StorageOffset;
                        weightBuffersConstantsOffsets[idx].Add((model.constants.Count - 1, offset));
                        constants.Add(input);
                    }

                    if (chain.Instructions(0).Value.InstrArgsType == InstructionArguments.NONE)
                        continue;

                    var kernel = chain.Instructions(0).Value.InstrArgsAsKernelCall();
                    string kernelName = executionPlan.Operators(kernel.OpIndex).Value.Name;
                    if (kernelName == "Reshape")
                    {
                        var input = chain.RequiredInput(0);
                        var shape = chain.RequiredInput(1);
                        bool allowZero = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new Reshape(index, input, shape, allowZero));
                    }
                    else if (kernelName == "Conv")
                    {
                        var input = chain.RequiredInput(0);
                        var weights = chain.RequiredInput(1);
                        var bias = chain.OptionalInput(2);
                        var autoPad = (AutoPad)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var dilations = executionPlan.Values(kernel.Args(1)).Value.Val<IntList>()?.GetItemsArray();
                        var group = executionPlan.Values(kernel.Args(2)).Value.ValAsInt().IntVal;
                        var pads = executionPlan.Values(kernel.Args(3)).Value.Val<IntList>()?.GetItemsArray();
                        var strides = executionPlan.Values(kernel.Args(4)).Value.Val<IntList>()?.GetItemsArray();
                        var kernelShape = executionPlan.Values(kernel.Args(5)).Value.Val<IntList>()?.GetItemsArray();
                        var fusedactivation = (FusableActivation)executionPlan.Values(kernel.Args(6)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Conv(index, input, weights, bias, group, strides, pads, dilations, autoPad, kernelShape, fusedactivation));
                    }
                    else if (kernelName == "MaxPool")
                    {
                        var input = chain.RequiredInput(0);
                        var kernelShape = executionPlan.Values(kernel.Args(0)).Value.Val<IntList>()?.GetItemsArray();
                        var strides = executionPlan.Values(kernel.Args(1)).Value.Val<IntList>()?.GetItemsArray();
                        var pads = executionPlan.Values(kernel.Args(2)).Value.Val<IntList>()?.GetItemsArray();
                        var autoPad = (AutoPad)executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        model.AddLayer(new MaxPool(index, input, kernelShape, strides, pads, autoPad));
                    }
                    else if (kernelName == "Celu")
                    {
                        var input = chain.RequiredInput(0);
                        var alpha = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new Celu(index, input, alpha));
                    }
                    else if (kernelName == "Elu")
                    {
                        var input = chain.RequiredInput(0);
                        var alpha = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new Elu(index, input, alpha));
                    }
                    else if (kernelName == "Gelu")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Gelu(index, input));
                    }
                    else if (kernelName == "GeluFast")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new GeluFast(index, input));
                    }
                    else if (kernelName == "Erf")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Erf(index, input));
                    }
                    else if (kernelName == "Hardmax")
                    {
                        var input = chain.RequiredInput(0);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Hardmax(index, input, axis));
                    }
                    else if (kernelName == "HardSigmoid")
                    {
                        var input = chain.RequiredInput(0);
                        var alpha = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var beta = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new HardSigmoid(index, input, alpha, beta));
                    }
                    else if (kernelName == "HardSwish")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new HardSwish(index, input));
                    }
                    else if (kernelName == "LeakyRelu")
                    {
                        var input = chain.RequiredInput(0);
                        var alpha = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new LeakyRelu(index, input, alpha));
                    }
                    else if (kernelName == "PRelu")
                    {
                        var input = chain.RequiredInput(0);
                        var slope = chain.RequiredInput(1);
                        model.AddLayer(new PRelu(index, input, slope));
                    }
                    else if (kernelName == "Relu")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Relu(index, input));
                    }
                    else if (kernelName == "Relu6")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Relu6(index, input));
                    }
                    else if (kernelName == "Selu")
                    {
                        var input = chain.RequiredInput(0);
                        var alpha = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var gamma = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new Selu(index, input, alpha, gamma));
                    }
                    else if (kernelName == "Sigmoid")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Sigmoid(index, input));
                    }
                    else if (kernelName == "Softplus")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Softplus(index, input));
                    }
                    else if (kernelName == "Softsign")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Softsign(index, input));
                    }
                    else if (kernelName == "Swish")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Swish(index, input));
                    }
                    else if (kernelName == "Tanh")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Tanh(index, input));
                    }
                    else if (kernelName == "ThresholdedRelu")
                    {
                        var input = chain.RequiredInput(0);
                        var alpha = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new ThresholdedRelu(index, input, alpha));
                    }
                    else if (kernelName == "LogSoftmax")
                    {
                        var input = chain.RequiredInput(0);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new LogSoftmax(index, input, axis));
                    }
                    else if (kernelName == "Softmax")
                    {
                        var input = chain.RequiredInput(0);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Softmax(index, input, axis));
                    }
                    else if (kernelName == "ConvTranspose")
                    {
                        var input = chain.RequiredInput(0);
                        var weights = chain.RequiredInput(1);
                        var bias = chain.OptionalInput(2);
                        var autoPad = (AutoPad)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var outputPadding = executionPlan.Values(kernel.Args(1)).Value.Val<IntList>()?.GetItemsArray();
                        var pads = executionPlan.Values(kernel.Args(2)).Value.Val<IntList>()?.GetItemsArray();
                        var strides = executionPlan.Values(kernel.Args(3)).Value.Val<IntList>()?.GetItemsArray();
                        var kernelShape = executionPlan.Values(kernel.Args(4)).Value.Val<IntList>()?.GetItemsArray();
                        var fusedactivation = (FusableActivation)executionPlan.Values(kernel.Args(5)).Value.ValAsInt().IntVal;
                        model.AddLayer(new ConvTranspose(index, input, weights, bias, strides, pads, autoPad, outputPadding, kernelShape, fusedactivation));
                    }
                    else if (kernelName == "Shape")
                    {
                        var input = chain.RequiredInput(0);
                        var start = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var end = executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Shape(index, input, start, end));
                    }
                    else if (kernelName == "Size")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Size(index, input));
                    }
                    else if (kernelName == "ConstantOfShape")
                    {
                        var input = chain.RequiredInput(0);
                        var dataType = (DataType)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var cf = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        var ci = executionPlan.Values(kernel.Args(2)).Value.ValAsInt().IntVal;
                        if (dataType == DataType.Float)
                            model.AddLayer(new ConstantOfShape(index, input, cf));
                        else if (dataType == DataType.Int)
                            model.AddLayer(new ConstantOfShape(index, input, ci));
                    }
                    else if (kernelName == "OneHot")
                    {
                        var indices = chain.RequiredInput(0);
                        var depth = chain.RequiredInput(1);
                        var values = chain.RequiredInput(2);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new OneHot(index, indices, depth, values, axis));
                    }
                    else if (kernelName == "Range")
                    {
                        var start = chain.RequiredInput(0);
                        var limit = chain.RequiredInput(1);
                        var delta = chain.RequiredInput(2);
                        model.AddLayer(new Layers.Range(index, start, limit, delta));
                    }
                    else if (kernelName == "ArgMax")
                    {
                        var input = chain.RequiredInput(0);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var keepdim = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        var selectLastIndex = executionPlan.Values(kernel.Args(2)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ArgMax(index, input, axis, keepdim, selectLastIndex));
                    }
                    else if (kernelName == "ArgMin")
                    {
                        var input = chain.RequiredInput(0);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var keepdim = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        var selectLastIndex = executionPlan.Values(kernel.Args(2)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ArgMin(index, input, axis, keepdim, selectLastIndex));
                    }
                    else if (kernelName == "Gather")
                    {
                        var input = chain.RequiredInput(0);
                        var indices = chain.RequiredInput(1);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Gather(index, input, indices, axis));
                    }
                    else if (kernelName == "GatherElements")
                    {
                        var input = chain.RequiredInput(0);
                        var indices = chain.RequiredInput(1);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new GatherElements(index, input, indices, axis));
                    }
                    else if (kernelName == "GatherND")
                    {
                        var input = chain.RequiredInput(0);
                        var indices = chain.RequiredInput(1);
                        var batchDims = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new GatherND(index, input, indices, batchDims));
                    }
                    else if (kernelName == "NonZero")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new NonZero(index, input));
                    }
                    else if (kernelName == "ScatterElements")
                    {
                        var input = chain.RequiredInput(0);
                        var indices = chain.RequiredInput(1);
                        var updates = chain.RequiredInput(2);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var reduction = (ScatterReductionMode)executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        model.AddLayer(new ScatterElements(index, input, indices, updates, axis, reduction));
                    }
                    else if (kernelName == "ScatterND")
                    {
                        var input = chain.RequiredInput(0);
                        var indices = chain.RequiredInput(1);
                        var updates = chain.RequiredInput(2);
                        var reduction = (ScatterReductionMode)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new ScatterND(index, input, indices, updates, reduction));
                    }
                    else if (kernelName == "TopK")
                    {
                        var values = chain.RequiredOutput(0);
                        var indices = chain.RequiredOutput(1);
                        var input = chain.RequiredInput(0);
                        var k = chain.RequiredInput(1);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var largest = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        var sorted = executionPlan.Values(kernel.Args(2)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new TopK(values, indices, input, k, axis, largest, sorted));
                    }
                    else if (kernelName == "And")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new And(index, a, b));
                    }
                    else if (kernelName == "Compress")
                    {
                        var input = chain.RequiredInput(0);
                        var conditions = chain.RequiredInput(1);
                        var hasAxis = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var axis = executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Compress(index, input, conditions, hasAxis ? (int?)axis: null));
                    }
                    else if (kernelName == "Equal")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new Equal(index, a, b));
                    }
                    else if (kernelName == "Greater")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new Greater(index, a, b));
                    }
                    else if (kernelName == "GreaterOrEqual")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new GreaterOrEqual(index, a, b));
                    }
                    else if (kernelName == "IsInf")
                    {
                        var input = chain.RequiredInput(0);
                        bool detectNegative = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        bool detectPositive = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new IsInf(index, input, detectNegative, detectPositive));
                    }
                    else if (kernelName == "IsNaN")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new IsNaN(index, input));
                    }
                    else if (kernelName == "Less")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new Less(index, a, b));
                    }
                    else if (kernelName == "LessOrEqual")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new LessOrEqual(index, a, b));
                    }
                    else if (kernelName == "Not")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Not(index, input));
                    }
                    else if (kernelName == "Or")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new Or(index, a, b));
                    }
                    else if (kernelName == "Xor")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new Xor(index, a, b));
                    }
                    else if (kernelName == "Where")
                    {
                        var c = chain.RequiredInput(0);
                        var a = chain.RequiredInput(1);
                        var b = chain.RequiredInput(2);
                        model.AddLayer(new Where(index, c, a, b));
                    }
                    else if (kernelName == "Abs")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Abs(index, input));
                    }
                    else if (kernelName == "Add")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new Add(index, a, b));
                    }
                    else if (kernelName == "Ceil")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Ceil(index, input));
                    }
                    else if (kernelName == "Clip")
                    {
                        var input = chain.RequiredInput(0);
                        var min = chain.OptionalInput(1);
                        var max = chain.OptionalInput(2);
                        model.AddLayer(new Clip(index, input, min, max));
                    }
                    else if (kernelName == "CumSum")
                    {
                        var input = chain.RequiredInput(0);
                        var axis = chain.RequiredInput(1);
                        bool reverse = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        bool exclusive = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new CumSum(index, input, axis, reverse, exclusive));
                    }
                    else if (kernelName == "Dense")
                    {
                        var input = chain.RequiredInput(0);
                        var weights = chain.RequiredInput(1);
                        var bias = chain.RequiredInput(2);
                        var fusedactivation = (FusableActivation)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Dense(index, input, weights, bias, fusedactivation));
                    }
                    else if (kernelName == "DenseBatched")
                    {
                        var input = chain.RequiredInput(0);
                        var weights = chain.RequiredInput(1);
                        var bias = chain.RequiredInput(2);
                        var fusedactivation = (FusableActivation)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new DenseBatched(index, input, weights, bias, fusedactivation));
                    }
                    else if (kernelName == "Div")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new Div(index, a, b));
                    }
                    else if (kernelName == "Einsum")
                    {
                        var inputs = new int[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.RequiredInput(ii);
                        var equation = executionPlan.Values(kernel.Args(0)).Value.ValAsString().StringVal;
                        model.AddLayer(new Einsum(index, inputs, equation));
                    }
                    else if (kernelName == "Exp")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Exp(index, input));
                    }
                    else if (kernelName == "Floor")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Floor(index, input));
                    }
                    else if (kernelName == "Log")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Log(index, input));
                    }
                    else if (kernelName == "MatMul")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new MatMul(index, a, b));
                    }
                    else if (kernelName == "MatMul2D")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        bool transposeA = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        bool transposeB = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new MatMul2D(index, a, transposeA, b, transposeB));
                    }
                    else if (kernelName == "Max")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new Max(index, a, b));
                    }
                    else if (kernelName == "Min")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new Min(index, a, b));
                    }
                    else if (kernelName == "Mod")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        bool fmod = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new Mod(index, a, b, fmod));
                    }
                    else if (kernelName == "Mul")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new Mul(index, a, b));
                    }
                    else if (kernelName == "Neg")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Neg(index, input));
                    }
                    else if (kernelName == "Pow")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new Pow(index, a, b));
                    }
                    else if (kernelName == "Reciprocal")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Reciprocal(index, input));
                    }
                    else if (kernelName == "Round")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Round(index, input));
                    }
                    else if (kernelName == "ScalarMad")
                    {
                        var input = chain.RequiredInput(0);
                        var dataType = (DataType)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var sFloat = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        var bFloat = executionPlan.Values(kernel.Args(2)).Value.ValAsFloat().FloatVal;
                        var sInt = executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        var bInt = executionPlan.Values(kernel.Args(4)).Value.ValAsInt().IntVal;
                        if (dataType == DataType.Float)
                            model.AddLayer(new ScalarMad(index, input, sFloat, bFloat));
                        else if (dataType == DataType.Int)
                            model.AddLayer(new ScalarMad(index, input, sInt, bInt));
                    }
                    else if (kernelName == "Shrink")
                    {
                        var input = chain.RequiredInput(0);
                        var bias = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var lamda = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new Shrink(index, input, bias, lamda));
                    }
                    else if (kernelName == "Sign")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Sign(index, input));
                    }
                    else if (kernelName == "Sqrt")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Sqrt(index, input));
                    }
                    else if (kernelName == "Square")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Square(index, input));
                    }
                    else if (kernelName == "Sub")
                    {
                        var a = chain.RequiredInput(0);
                        var b = chain.RequiredInput(1);
                        model.AddLayer(new Sub(index, a, b));
                    }
                    else if (kernelName == "ScaleBias")
                    {
                        var input = chain.RequiredInput(0);
                        var scale = chain.RequiredInput(1);
                        var bias = chain.RequiredInput(2);
                        model.AddLayer(new ScaleBias(index, input, scale, bias));
                    }
                    else if (kernelName == "InstanceNormalization")
                    {
                        var input = chain.RequiredInput(0);
                        var scale = chain.RequiredInput(1);
                        var bias = chain.RequiredInput(2);
                        var epsilon = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new InstanceNormalization(index, input, scale, bias, epsilon));
                    }
                    else if (kernelName == "LayerNormalization")
                    {
                        var input = chain.RequiredInput(0);
                        var scale = chain.RequiredInput(1);
                        var bias = chain.OptionalInput(2);
                        var epsilon = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new LayerNormalization(index, input, scale, bias, epsilon));
                    }
                    else if (kernelName == "RMSNormalization")
                    {
                        var input = chain.RequiredInput(0);
                        var scale = chain.RequiredInput(1);
                        var epsilon = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new RMSNormalization(index, input, scale, epsilon));
                    }
                    else if (kernelName == "BatchNormalization")
                    {
                        var input = chain.RequiredInput(0);
                        var scale = chain.RequiredInput(1);
                        var bias = chain.RequiredInput(2);
                        var mean = chain.RequiredInput(3);
                        var variance = chain.RequiredInput(4);
                        var epsilon = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new BatchNormalization(index, input, scale, bias, mean, variance, epsilon));
                    }
                    else if (kernelName == "LRN")
                    {
                        var input = chain.RequiredInput(0);
                        var alpha = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var beta = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        var bias = executionPlan.Values(kernel.Args(2)).Value.ValAsFloat().FloatVal;
                        var count = executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        model.AddLayer(new LRN(index, input, alpha, beta, bias, count));
                    }
                    else if (kernelName == "NonMaxSuppression")
                    {
                        var input = chain.RequiredInput(0);
                        var scores = chain.RequiredInput(1);
                        var maxOutputBoxesPerClass = chain.OptionalInput(2);
                        var iouThreshold = chain.OptionalInput(3);
                        var scoreThreshold = chain.OptionalInput(4);
                        var centerPointBox = (CenterPointBox)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new NonMaxSuppression(index, input, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox));
                    }
                    else if (kernelName == "RoiAlign")
                    {
                        var input = chain.RequiredInput(0);
                        var rois = chain.RequiredInput(1);
                        var batchIndices = chain.RequiredInput(2);
                        var mode = (RoiPoolingMode)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var outputHeight = executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        var outputWidth = executionPlan.Values(kernel.Args(2)).Value.ValAsInt().IntVal;
                        var samplingRatio = executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        var spatialScale = executionPlan.Values(kernel.Args(4)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new RoiAlign(index, input, rois, batchIndices, mode, outputHeight, outputWidth, samplingRatio, spatialScale));
                    }
                    else if (kernelName == "AveragePool")
                    {
                        var input = chain.RequiredInput(0);
                        var kernelShape = executionPlan.Values(kernel.Args(0)).Value.Val<IntList>()?.GetItemsArray();
                        var strides = executionPlan.Values(kernel.Args(1)).Value.Val<IntList>()?.GetItemsArray();
                        var pads = executionPlan.Values(kernel.Args(2)).Value.Val<IntList>()?.GetItemsArray();
                        var autopad = (AutoPad)executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        model.AddLayer(new AveragePool(index, input, kernelShape, strides, pads, autopad));
                    }
                    else if (kernelName == "GlobalAveragePool")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new GlobalAveragePool(index, input));
                    }
                    else if (kernelName == "GlobalMaxPool")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new GlobalMaxPool(index, input));
                    }
                    else if (kernelName == "MaxPool")
                    {
                        var input = chain.RequiredInput(0);
                        var kernelShape = executionPlan.Values(kernel.Args(0)).Value.Val<IntList>()?.GetItemsArray();
                        var strides = executionPlan.Values(kernel.Args(1)).Value.Val<IntList>()?.GetItemsArray();
                        var pads = executionPlan.Values(kernel.Args(2)).Value.Val<IntList>()?.GetItemsArray();
                        var autopad = (AutoPad)executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        model.AddLayer(new MaxPool(index, input, kernelShape, strides, pads, autopad));
                    }
                    else if (kernelName == "RandomNormal")
                    {
                        var mean = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var scale = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        var shape = executionPlan.Values(kernel.Args(2)).Value.Val<IntList>()?.GetItemsArray();
                        var hasSeed = executionPlan.Values(kernel.Args(3)).Value.ValAsBool().BoolVal;
                        var seed = executionPlan.Values(kernel.Args(4)).Value.ValAsInt().IntVal;
                        model.AddLayer(new RandomNormal(index, shape, mean, scale, hasSeed ? (int?)seed : null));
                    }
                    else if (kernelName == "RandomNormalLike")
                    {
                        var input = chain.RequiredInput(0);
                        var mean = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var scale = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        var hasSeed = executionPlan.Values(kernel.Args(2)).Value.ValAsBool().BoolVal;
                        var seed = executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        model.AddLayer(new RandomNormalLike(index, input, mean, scale, hasSeed ? (int?)seed : null));
                    }
                    else if (kernelName == "RandomUniform")
                    {
                        var low = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var high = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        var shape = executionPlan.Values(kernel.Args(2)).Value.Val<IntList>()?.GetItemsArray();
                        var hasSeed = executionPlan.Values(kernel.Args(3)).Value.ValAsBool().BoolVal;
                        var seed = executionPlan.Values(kernel.Args(4)).Value.ValAsInt().IntVal;
                        model.AddLayer(new RandomUniform(index, shape, low, high, hasSeed ? (int?)seed : null));
                    }
                    else if (kernelName == "RandomUniformLike")
                    {
                        var input = chain.RequiredInput(0);
                        var low = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var high = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        var hasSeed = executionPlan.Values(kernel.Args(2)).Value.ValAsBool().BoolVal;
                        var seed = executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        model.AddLayer(new RandomUniformLike(index, input, low, high, hasSeed ? (int?)seed : null));
                    }
                    else if (kernelName == "Bernoulli")
                    {
                        var input = chain.RequiredInput(0);
                        var dataType = (DataType)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var hasSeed = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        var seed = executionPlan.Values(kernel.Args(2)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Bernoulli(index, input, dataType, hasSeed ? (int?)seed : null));
                    }
                    else if (kernelName == "Multinomial")
                    {
                        var input = chain.RequiredInput(0);
                        var count = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var hasSeed = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        var seed = executionPlan.Values(kernel.Args(2)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Multinomial(index, input, count, hasSeed ? (int?)seed : null));
                    }
                    else if (kernelName == "LSTM")
                    {
                        var Y = chain.RequiredOutput(0);
                        var Y_h = chain.OptionalOutput(1);
                        var Y_c = chain.OptionalOutput(2);
                        var X = chain.RequiredInput(0);
                        var W = chain.RequiredInput(1);
                        var R = chain.RequiredInput(2);
                        var B = chain.OptionalInput(3);
                        var sequenceLens = chain.OptionalInput(4);
                        var initialH = chain.OptionalInput(5);
                        var initialC = chain.OptionalInput(6);
                        var P = chain.OptionalInput(7);
                        var hiddenSize = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var direction = (RnnDirection)executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        var activations = executionPlan.Values(kernel.Args(2)).Value.Val<IntList>()?.GetItemsArray();
                        RnnActivation[] activationsEnum = null;
                        if (activations != null)
                        {
                            activationsEnum = new RnnActivation[activations.Length];
                            for (int k = 0; k < activations.Length; k++)
                                activationsEnum[k] = (RnnActivation)activations[k];
                        }
                        var activationAlpha = executionPlan.Values(kernel.Args(3)).Value.Val<FloatList>()?.GetItemsArray();
                        var activationBeta = executionPlan.Values(kernel.Args(4)).Value.Val<FloatList>()?.GetItemsArray();
                        var clip = executionPlan.Values(kernel.Args(5)).Value.ValAsFloat().FloatVal;
                        var inputForget = executionPlan.Values(kernel.Args(6)).Value.ValAsBool().BoolVal;
                        var layout = (RnnLayout)executionPlan.Values(kernel.Args(7)).Value.ValAsInt().IntVal;
                        model.AddLayer(new LSTM(Y, X, W, R, hiddenSize, Y_h: Y_h, Y_c: Y_c, B: B, sequenceLens: sequenceLens, initialH: initialH, initialC: initialC, P: P, direction: direction, activations: activationsEnum, activationAlpha: activationAlpha, activationBeta: activationBeta, clip: clip, inputForget: inputForget, layout: layout));
                    }
                    else if (kernelName == "ReduceL1")
                    {
                        var data = chain.RequiredInput(0);
                        var axes = chain.OptionalInput(1);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceL1(index, data, axes, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceL2")
                    {
                        var data = chain.RequiredInput(0);
                        var axes = chain.OptionalInput(1);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceL2(index, data, axes, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceLogSum")
                    {
                        var data = chain.RequiredInput(0);
                        var axes = chain.OptionalInput(1);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceLogSum(index, data, axes, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceLogSumExp")
                    {
                        var data = chain.RequiredInput(0);
                        var axes = chain.OptionalInput(1);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceLogSumExp(index, data, axes, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceMax")
                    {
                        var data = chain.RequiredInput(0);
                        var axes = chain.OptionalInput(1);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceMax(index, data, axes, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceMean")
                    {
                        var data = chain.RequiredInput(0);
                        var axes = chain.OptionalInput(1);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceMean(index, data, axes, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceMin")
                    {
                        var data = chain.RequiredInput(0);
                        var axes = chain.OptionalInput(1);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceMin(index, data, axes, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceProd")
                    {
                        var data = chain.RequiredInput(0);
                        var axes = chain.OptionalInput(1);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceProd(index, data, axes, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceSum")
                    {
                        var data = chain.RequiredInput(0);
                        var axes = chain.OptionalInput(1);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceSum(index, data, axes, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceSumSquare")
                    {
                        var data = chain.RequiredInput(0);
                        var axes = chain.OptionalInput(1);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceSumSquare(index, data, axes, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "Cast")
                    {
                        var input = chain.RequiredInput(0);
                        var dataType = (DataType)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Cast(index, input, dataType));
                    }
                    else if (kernelName == "CastLike")
                    {
                        var input = chain.RequiredInput(0);
                        var targetType = chain.RequiredInput(1);
                        model.AddLayer(new CastLike(index, input, targetType));
                    }
                    else if (kernelName == "Concat")
                    {
                        var inputs = new int[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.RequiredInput(ii);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Concat(index, inputs, axis));
                    }
                    else if (kernelName == "DepthToSpace")
                    {
                        var input = chain.RequiredInput(0);
                        var blocksize = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var mode = (DepthToSpaceMode)executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        model.AddLayer(new DepthToSpace(index, input, blocksize, mode));
                    }
                    else if (kernelName == "Expand")
                    {
                        var input = chain.RequiredInput(0);
                        var shape = chain.RequiredInput(1);
                        model.AddLayer(new Expand(index, input, shape));
                    }
                    else if (kernelName == "Flatten")
                    {
                        var input = chain.RequiredInput(0);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Flatten(index, input, axis));
                    }
                    else if (kernelName == "GridSample")
                    {
                        var input = chain.RequiredInput(0);
                        var grid = chain.RequiredInput(1);
                        var mode = (InterpolationMode)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var paddingMode = (PaddingMode)executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        var alignCorners = executionPlan.Values(kernel.Args(2)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new GridSample(index, input, grid, mode, paddingMode, alignCorners));
                    }
                    else if (kernelName == "Identity")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Identity(index, input));
                    }
                    else if (kernelName == "MoveDim")
                    {
                        var input = chain.RequiredInput(0);
                        var source = executionPlan.Values(kernel.Args(0)).Value.Val<IntList>()?.GetItemsArray();
                        var destination = executionPlan.Values(kernel.Args(1)).Value.Val<IntList>()?.GetItemsArray();
                        model.AddLayer(new MoveDim(index, input, source, destination));
                    }
                    else if (kernelName == "Narrow")
                    {
                        var input = chain.RequiredInput(0);
                        var dim = chain.RequiredInput(1);
                        var start = chain.RequiredInput(2);
                        var length = chain.RequiredInput(3);
                        model.AddLayer(new Narrow(index, input, dim, start, length));
                    }
                    else if (kernelName == "Pad")
                    {
                        var data = chain.RequiredInput(0);
                        var pads = chain.RequiredInput(1);
                        var constantValue = chain.OptionalInput(2);
                        var axes = chain.OptionalInput(3);
                        var mode = (PadMode)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Pad(index, data, pads, constantValue, axes, mode));
                    }
                    else if (kernelName == "Reshape")
                    {
                        var input = chain.RequiredInput(0);
                        var reshape = chain.RequiredInput(1);
                        var allowZero = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new Reshape(index, input, reshape, allowZero));
                    }
                    else if (kernelName == "Resize")
                    {
                        var input = chain.RequiredInput(0);
                        var scalesOrSizes = chain.RequiredInput(1);
                        var scaleMode = (Layers.ScaleMode)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var coordTransformMode = (Layers.CoordTransformMode)executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        var mode = (Layers.InterpolationMode)executionPlan.Values(kernel.Args(2)).Value.ValAsInt().IntVal;
                        var nearestMode = (Layers.NearestMode)executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        var axes = executionPlan.Values(kernel.Args(4)).Value.Val<IntList>()?.GetItemsArray();
                        model.AddLayer(new Resize(index, input, scalesOrSizes, scaleMode, mode, coordTransformMode, nearestMode, axes));
                    }
                    else if (kernelName == "Select")
                    {
                        var input = chain.RequiredInput(0);
                        var dim = chain.RequiredInput(1);
                        var selectIndex = chain.RequiredInput(2);
                        model.AddLayer(new Select(index, input, dim, selectIndex));
                    }
                    else if (kernelName == "Slice")
                    {
                        var input = chain.RequiredInput(0);
                        var starts = chain.RequiredInput(1);
                        var ends = chain.RequiredInput(2);
                        var axes = chain.OptionalInput(3);
                        var steps = chain.OptionalInput(4);
                        model.AddLayer(new Slice(index, input, starts, ends, axes, steps));
                    }
                    else if (kernelName == "SliceSet")
                    {
                        var input = chain.RequiredInput(0);
                        var values = chain.RequiredInput(1);
                        var starts = chain.RequiredInput(2);
                        var ends = chain.RequiredInput(3);
                        var axes = chain.OptionalInput(4);
                        var steps = chain.OptionalInput(5);
                        model.AddLayer(new SliceSet(index, input, values, starts, ends, axes, steps));
                    }
                    else if (kernelName == "SpaceToDepth")
                    {
                        var input = chain.RequiredInput(0);
                        var blocksize = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new SpaceToDepth(index, input, blocksize));
                    }
                    else if (kernelName == "Split")
                    {
                        var input = chain.RequiredInput(0);
                        var split = chain.OptionalInput(1);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var numOutputs = executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        var outputs = new int[chain.OutputsLength];
                        for (int ii = 0; ii < outputs.Length; ii++)
                            outputs[ii] = chain.RequiredOutput(ii);
                        model.AddLayer(new Split(outputs, input, split, axis, numOutputs));
                    }
                    else if (kernelName == "Squeeze")
                    {
                        var input = chain.RequiredInput(0);
                        var axes = chain.OptionalInput(1);
                        model.AddLayer(new Squeeze(index, input, axes));
                    }
                    else if (kernelName == "Tile")
                    {
                        var input = chain.RequiredInput(0);
                        var repeats = chain.RequiredInput(1);
                        model.AddLayer(new Tile(index, input, repeats));
                    }
                    else if (kernelName == "Transpose")
                    {
                        var input = chain.RequiredInput(0);
                        var permutations = executionPlan.Values(kernel.Args(0)).Value.Val<IntList>()?.GetItemsArray();
                        model.AddLayer(new Transpose(index, input, permutations));
                    }
                    else if (kernelName == "Trilu")
                    {
                        var input = chain.RequiredInput(0);
                        var k = chain.OptionalInput(1);
                        var mode = (TriluMode)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Trilu(index, input, k, mode));
                    }
                    else if (kernelName == "Unsqueeze")
                    {
                        var input = chain.RequiredInput(0);
                        var axes = chain.RequiredInput(1);
                        model.AddLayer(new Unsqueeze(index, input, axes));
                    }
                    else if (kernelName == "Acos")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Acos(index, input));
                    }
                    else if (kernelName == "Acosh")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Acosh(index, input));
                    }
                    else if (kernelName == "Asin")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Asin(index, input));
                    }
                    else if (kernelName == "Asinh")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Asinh(index, input));
                    }
                    else if (kernelName == "Atan")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Atan(index, input));
                    }
                    else if (kernelName == "Atanh")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Atanh(index, input));
                    }
                    else if (kernelName == "Cos")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Cos(index, input));
                    }
                    else if (kernelName == "Cosh")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Cosh(index, input));
                    }
                    else if (kernelName == "Sin")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Sin(index, input));
                    }
                    else if (kernelName == "Sinh")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Sinh(index, input));
                    }
                    else if (kernelName == "Tan")
                    {
                        var input = chain.RequiredInput(0);
                        model.AddLayer(new Tan(index, input));
                    }
                    else if (kernelName == "DequantizeUint8")
                    {
                        var input = chain.RequiredInput(0);
                        var scale = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var zeroPoint = (byte)executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        model.AddLayer(new DequantizeUint8(index, input, scale, zeroPoint));
                    }
                    else
                        throw new NotImplementedException(kernelName);
                }
            }
            catch (Exception e)
            {
                D.LogError($"Failed to load serialized model description. ({e.Message})");
                throw;
            }

            return weightBuffersConstantsOffsets;
        }

        static void LoadModelWeights(byte[] modelWeightsBufferBytes, List<(int, int)> constantIndexesOffsets, ref Model model)
        {
            try
            {
                var bb = new ByteBuffer(modelWeightsBufferBytes, sizeof(int));
                var weightBuffer = SentisFlatBuffer.Buffer.GetRootAsBuffer(bb);
                var data = weightBuffer.GetStorageArray();
                foreach (var (constantIdx, offset) in constantIndexesOffsets)
                {
                    var constant = model.constants[constantIdx];
                    if (constant.lengthBytes == 0)
                        continue;
                    var elementCount = constant.lengthBytes / NativeTensorArray.k_DataItemSize;
                    constant.weights = new NativeTensorArrayFromManagedArray(data, offset, elementCount);
                }
            }
            catch (InvalidOperationException)
            {
                D.LogError("Failed to load serialized model weights.");
                throw;
            }
        }
    }
}
