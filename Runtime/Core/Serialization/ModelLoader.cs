// #define DEBUG_TIMING
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using Unity.Sentis.Google.FlatBuffers;
using SentisFlatBuffer;
using Unity.Sentis.Layers;

[assembly: InternalsVisibleTo("Unity.Sentis.Tests")]

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
            for (int i = 0; i < modelAsset.modelWeightsChunks.Length; i++)
                modelWeightsBytes[i] = modelAsset.modelWeightsChunks[i].value;
            Model model = new Model();
            LoadModelDescription(modelDescriptionBytes, ref model);
            LoadModelWeights(modelWeightsBytes, ref model);
            return model;
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
                var numModelWeightsChunks = LoadModelDescription(modelDescriptionBytes, ref model);

                var modelWeightsBytes = new byte[numModelWeightsChunks][];
                for (int i = 0; i < numModelWeightsChunks; i++)
                {
                    stream.Read(prefixSizeBytes);
                    var modelWeightsChunkSize = BitConverter.ToInt32(prefixSizeBytes);
                    modelWeightsBytes[i] = new byte[modelWeightsChunkSize + sizeof(int)];
                    System.Buffer.BlockCopy(prefixSizeBytes, 0, modelWeightsBytes[i], 0, sizeof(int));
                    stream.Read(modelWeightsBytes[i], sizeof(int), modelWeightsChunkSize);
                }

                LoadModelWeights(modelWeightsBytes, ref model);
                return model;
            }
            catch (InvalidOperationException)
            {
                D.LogError("Failed to load serialized Sentis model, ensure model was exported with Sentis 1.4 or newer.");
                return null;
            }
        }

        internal static string InputAsString(this Chain chain, int index)
        {
            var input = chain.Inputs(index);
            return input == -1 ? string.Empty : input.ToString();
        }

        internal static int LoadModelDescription(byte[] modelDescription, ref Model model)
        {
            var bb = new ByteBuffer(modelDescription, sizeof(int));
            var program = Program.GetRootAsProgram(bb);

            try
            {
                var executionPlan = program.ExecutionPlan.Value;
                if (program.Version > ModelWriter.version)
                    D.LogWarning("Serialized model was exported in a newer version of Sentis than the current installed version and may not work as expected. Update the Sentis package to ensure compatibility.");
                int inputCount = executionPlan.InputsLength;
                for (int i = 0; i < inputCount; i++)
                {
                    int input = executionPlan.Inputs(i);
                    var inputDesc = executionPlan.Values(input).Value.ValAsTensor();
                    SymbolicTensorShape shape;
                    if (inputDesc.ShapeDynamism == TensorShapeDynamism.STATIC)
                    {
                        shape = new SymbolicTensorShape(new TensorShape(inputDesc.GetFixedSizesArray()));
                    }
                    else
                    {
                        shape = SymbolicTensorShape.UnknownOfRank(inputDesc.DynamicSizesLength);
                        for (int k = 0; k < shape.rank; k++)
                        {
                            var ddim = inputDesc.DynamicSizes(k).Value;
                            shape[k] = ddim.ValType == SymbolicDim.Int ? SymbolicTensorDim.Int(ddim.ValAsInt().IntVal) : SymbolicTensorDim.Param(ddim.ValAsByte().ByteVal);
                        }
                    }

                    model.AddInput(executionPlan.InputsName(i), input.ToString(), (DataType)inputDesc.ScalarType, shape);
                }
                model.outputs = new List<Model.Output>();
                for (int i = 0; i < executionPlan.OutputsLength; i++)
                {
                    model.AddOutput(executionPlan.OutputsName(i), executionPlan.Outputs(i).ToString());
                }
                var backendPartition = executionPlan.BackendPartitioning.Value;
                for (int i = 0; i < backendPartition.ChainsLength; i++)
                {
                    model.LayerCPUFallback.Add(backendPartition.Chains(i).ToString());
                }

                HashSet<int> constants = new HashSet<int>();
                for (int i = 0; i < executionPlan.ChainsLength; i++)
                {
                    var chain = executionPlan.Chains(i).Value;
                    var name = chain.Outputs(0).ToString();

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
                        model.AddConstant(new Constant(input.ToString(), shape, (DataType)constantTensor.ScalarType, lengthByte));
                        constants.Add(input);
                    }

                    var kernel = chain.Instructions(0).Value.InstrArgsAsKernelCall();
                    string kernelName = executionPlan.Operators(kernel.OpIndex).Value.Name;
                    if (kernelName == "Reshape")
                    {
                        var input = chain.InputAsString(0);
                        var shape = chain.InputAsString(1);
                        bool allowZero = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new Reshape(name, input, shape, allowZero));
                    }
                    else if (kernelName == "Conv")
                    {
                        var input = chain.InputAsString(0);
                        var weights = chain.InputAsString(1);
                        var autoPad = (AutoPad)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var dilations = executionPlan.Values(kernel.Args(1)).Value.Val<IntList>()?.GetItemsArray();
                        var group = executionPlan.Values(kernel.Args(2)).Value.ValAsInt().IntVal;
                        var pads = executionPlan.Values(kernel.Args(3)).Value.Val<IntList>()?.GetItemsArray();
                        var strides = executionPlan.Values(kernel.Args(4)).Value.Val<IntList>()?.GetItemsArray();
                        var kernelShape = executionPlan.Values(kernel.Args(5)).Value.Val<IntList>()?.GetItemsArray();
                        var fusedactivation = (FusableActivation)executionPlan.Values(kernel.Args(6)).Value.ValAsInt().IntVal;
                        if (chain.InputsLength == 2)
                        {
                            model.AddLayer(new Conv(name, input, weights, group, strides, pads, dilations, autoPad, kernelShape, fusedactivation));
                        }
                        else
                        {
                            var bias = chain.InputAsString(2);
                            model.AddLayer(new Conv(name, input, weights, bias, group, strides, pads, dilations, autoPad, kernelShape, fusedactivation));
                        }
                    }
                    else if (kernelName == "MaxPool")
                    {
                        var input = chain.InputAsString(0);
                        var kernelShape = executionPlan.Values(kernel.Args(0)).Value.Val<IntList>()?.GetItemsArray();
                        var strides = executionPlan.Values(kernel.Args(1)).Value.Val<IntList>()?.GetItemsArray();
                        var pads = executionPlan.Values(kernel.Args(2)).Value.Val<IntList>()?.GetItemsArray();
                        var autoPad = (AutoPad)executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        model.AddLayer(new MaxPool(name, input, kernelShape, strides, pads, autoPad));
                    }
                    else if (kernelName == "Celu")
                    {
                        var input = chain.InputAsString(0);
                        var alpha = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new Celu(name, input, alpha));
                    }
                    else if (kernelName == "Elu")
                    {
                        var input = chain.InputAsString(0);
                        var alpha = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new Elu(name, input, alpha));
                    }
                    else if (kernelName == "Gelu")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Gelu(name, input));
                    }
                    else if (kernelName == "GeluFast")
                    {
                        var input = chain.Inputs(0).ToString();
                        model.AddLayer(new GeluFast(name, input));
                    }
                    else if (kernelName == "Erf")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Erf(name, input));
                    }
                    else if (kernelName == "Hardmax")
                    {
                        var input = chain.InputAsString(0);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Hardmax(name, input, axis));
                    }
                    else if (kernelName == "HardSigmoid")
                    {
                        var input = chain.InputAsString(0);
                        var alpha = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var beta = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new HardSigmoid(name, input, alpha, beta));
                    }
                    else if (kernelName == "HardSwish")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new HardSwish(name, input));
                    }
                    else if (kernelName == "LeakyRelu")
                    {
                        var input = chain.InputAsString(0);
                        var alpha = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new LeakyRelu(name, input, alpha));
                    }
                    else if (kernelName == "PRelu")
                    {
                        var input = chain.InputAsString(0);
                        var slope = chain.InputAsString(1);
                        model.AddLayer(new PRelu(name, input, slope));
                    }
                    else if (kernelName == "Relu")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Relu(name, input));
                    }
                    else if (kernelName == "Relu6")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Relu6(name, input));
                    }
                    else if (kernelName == "Selu")
                    {
                        var input = chain.InputAsString(0);
                        var alpha = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var gamma = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new Selu(name, input, alpha, gamma));
                    }
                    else if (kernelName == "Sigmoid")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Sigmoid(name, input));
                    }
                    else if (kernelName == "Softplus")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Softplus(name, input));
                    }
                    else if (kernelName == "Softsign")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Softsign(name, input));
                    }
                    else if (kernelName == "Swish")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Swish(name, input));
                    }
                    else if (kernelName == "Tanh")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Tanh(name, input));
                    }
                    else if (kernelName == "ThresholdedRelu")
                    {
                        var input = chain.InputAsString(0);
                        var alpha = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new ThresholdedRelu(name, input, alpha));
                    }
                    else if (kernelName == "LogSoftmax")
                    {
                        var input = chain.InputAsString(0);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new LogSoftmax(name, input, axis));
                    }
                    else if (kernelName == "Softmax")
                    {
                        var input = chain.InputAsString(0);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Softmax(name, input, axis));
                    }
                    else if (kernelName == "ConvTranspose")
                    {
                        var input = chain.InputAsString(0);
                        var weights = chain.InputAsString(1);
                        var autoPad = (AutoPad)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var outputPadding = executionPlan.Values(kernel.Args(1)).Value.Val<IntList>()?.GetItemsArray();
                        var pads = executionPlan.Values(kernel.Args(2)).Value.Val<IntList>()?.GetItemsArray();
                        var strides = executionPlan.Values(kernel.Args(3)).Value.Val<IntList>()?.GetItemsArray();
                        var kernelShape = executionPlan.Values(kernel.Args(4)).Value.Val<IntList>()?.GetItemsArray();
                        var fusedactivation = (FusableActivation)executionPlan.Values(kernel.Args(5)).Value.ValAsInt().IntVal;
                        if (chain.InputsLength == 2)
                        {
                            model.AddLayer(new ConvTranspose(name, input, weights, strides, pads, autoPad, outputPadding, kernelShape, fusedactivation));
                        }
                        else
                        {
                            var bias = chain.InputAsString(2);
                            model.AddLayer(new ConvTranspose(name, input, weights, bias, strides, pads, autoPad, outputPadding, kernelShape, fusedactivation));
                        }
                    }
                    else if (kernelName == "Shape")
                    {
                        var input = chain.InputAsString(0);
                        var start = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var end = executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Shape(name, input, start, end));
                    }
                    else if (kernelName == "Size")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Size(name, input));
                    }
                    else if (kernelName == "ConstantOfShape")
                    {
                        var input = chain.InputAsString(0);
                        var dataType = (DataType)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var cf = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        var ci = executionPlan.Values(kernel.Args(2)).Value.ValAsInt().IntVal;
                        if (dataType == DataType.Float)
                            model.AddLayer(new ConstantOfShape(name, input, cf));
                        else
                            model.AddLayer(new ConstantOfShape(name, input, ci));
                    }
                    else if (kernelName == "OneHot")
                    {
                        var indices = chain.InputAsString(0);
                        var depth = chain.InputAsString(1);
                        var values = chain.InputAsString(2);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new OneHot(name, indices, depth, values, axis));
                    }
                    else if (kernelName == "Range")
                    {
                        var start = chain.InputAsString(0);
                        var limit = chain.InputAsString(1);
                        var delta = chain.InputAsString(2);
                        model.AddLayer(new Layers.Range(name, start, limit, delta));
                    }
                    else if (kernelName == "ArgMax")
                    {
                        var input = chain.InputAsString(0);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var keepdim = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        var selectLastIndex = executionPlan.Values(kernel.Args(2)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ArgMax(name, input, axis, keepdim, selectLastIndex));
                    }
                    else if (kernelName == "ArgMin")
                    {
                        var input = chain.InputAsString(0);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var keepdim = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        var selectLastIndex = executionPlan.Values(kernel.Args(2)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ArgMin(name, input, axis, keepdim, selectLastIndex));
                    }
                    else if (kernelName == "Gather")
                    {
                        var input = chain.InputAsString(0);
                        var indices = chain.InputAsString(1);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Gather(name, input, indices, axis));
                    }
                    else if (kernelName == "GatherElements")
                    {
                        var input = chain.InputAsString(0);
                        var indices = chain.InputAsString(1);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new GatherElements(name, input, indices, axis));
                    }
                    else if (kernelName == "GatherND")
                    {
                        var input = chain.InputAsString(0);
                        var indices = chain.InputAsString(1);
                        var batchDims = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new GatherND(name, input, indices, batchDims));
                    }
                    else if (kernelName == "NonZero")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new NonZero(name, input));
                    }
                    else if (kernelName == "ScatterElements")
                    {
                        var input = chain.InputAsString(0);
                        var indices = chain.InputAsString(1);
                        var updates = chain.InputAsString(2);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var reduction = (ScatterReductionMode)executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        model.AddLayer(new ScatterElements(name, input, indices, updates, axis, reduction));
                    }
                    else if (kernelName == "ScatterND")
                    {
                        var input = chain.InputAsString(0);
                        var indices = chain.InputAsString(1);
                        var updates = chain.InputAsString(2);
                        var reduction = (ScatterReductionMode)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new ScatterND(name, input, indices, updates, reduction));
                    }
                    else if (kernelName == "TopK")
                    {
                        var input = chain.InputAsString(0);
                        var outputs = new string[chain.OutputsLength];
                        for (int ii = 0; ii < outputs.Length; ii++)
                            outputs[ii] = chain.Outputs(ii).ToString();
                        var k = chain.InputAsString(1);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var largest = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        var sorted = executionPlan.Values(kernel.Args(2)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new TopK(name, input, k, axis, largest, sorted, outputs));
                    }
                    else if (kernelName == "And")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        model.AddLayer(new And(name, a, b));
                    }
                    else if (kernelName == "Compress")
                    {
                        var input = chain.InputAsString(0);
                        var conditions = chain.InputAsString(1);
                        var hasAxis = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var axis = executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        if (hasAxis)
                            model.AddLayer(new Compress(name, input, conditions, axis));
                        else
                            model.AddLayer(new Compress(name, input, conditions));
                    }
                    else if (kernelName == "Equal")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        model.AddLayer(new Equal(name, a, b));
                    }
                    else if (kernelName == "Greater")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        model.AddLayer(new Greater(name, a, b));
                    }
                    else if (kernelName == "GreaterOrEqual")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        model.AddLayer(new GreaterOrEqual(name, a, b));
                    }
                    else if (kernelName == "IsInf")
                    {
                        var input = chain.InputAsString(0);
                        bool detectNegative = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        bool detectPositive = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new IsInf(name, input, detectNegative, detectPositive));
                    }
                    else if (kernelName == "IsNaN")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new IsNaN(name, input));
                    }
                    else if (kernelName == "Less")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        model.AddLayer(new Less(name, a, b));
                    }
                    else if (kernelName == "LessOrEqual")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        model.AddLayer(new LessOrEqual(name, a, b));
                    }
                    else if (kernelName == "Not")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Not(name, input));
                    }
                    else if (kernelName == "Or")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        model.AddLayer(new Or(name, a, b));
                    }
                    else if (kernelName == "Xor")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        model.AddLayer(new Xor(name, a, b));
                    }
                    else if (kernelName == "Where")
                    {
                        var c = chain.InputAsString(0);
                        var a = chain.InputAsString(1);
                        var b = chain.InputAsString(2);
                        model.AddLayer(new Where(name, c, a, b));
                    }
                    else if (kernelName == "Abs")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Abs(name, input));
                    }
                    else if (kernelName == "Add")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        model.AddLayer(new Add(name, a, b));
                    }
                    else if (kernelName == "Ceil")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Ceil(name, input));
                    }
                    else if (kernelName == "Clip")
                    {
                        var input = chain.InputAsString(0);
                        if (chain.InputsLength == 1)
                        {
                            model.AddLayer(new Clip(name, input));
                        }
                        else if (chain.InputsLength == 2)
                        {
                            var min = chain.InputAsString(1);
                            model.AddLayer(new Clip(name, input, min));
                        }
                        else
                        {
                            var min = chain.InputAsString(1);
                            var max = chain.InputAsString(2);
                            model.AddLayer(new Clip(name, input, min, max));
                        }
                    }
                    else if (kernelName == "CumSum")
                    {
                        var input = chain.InputAsString(0);
                        var axis = chain.InputAsString(1);
                        bool reverse = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        bool exclusive = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new CumSum(name, input, axis, reverse, exclusive));
                    }
                    else if (kernelName == "Dense")
                    {
                        var input = chain.InputAsString(0);
                        var weights = chain.InputAsString(1);
                        var bias = chain.InputAsString(2);
                        var fusedactivation = (FusableActivation)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Dense(name, input, weights, bias, fusedactivation));
                    }
                    else if (kernelName == "Div")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        model.AddLayer(new Div(name, a, b));
                    }
                    else if (kernelName == "Einsum")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        var equation = executionPlan.Values(kernel.Args(0)).Value.ValAsString().StringVal;
                        model.AddLayer(new Einsum(name, inputs, equation));
                    }
                    else if (kernelName == "Exp")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Exp(name, input));
                    }
                    else if (kernelName == "Floor")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Floor(name, input));
                    }
                    else if (kernelName == "Log")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Log(name, input));
                    }
                    else if (kernelName == "MatMul")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        model.AddLayer(new MatMul(name, a, b));
                    }
                    else if (kernelName == "MatMul2D")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        bool transposeA = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        bool transposeB = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new MatMul2D(name, a, transposeA, b, transposeB));
                    }
                    else if (kernelName == "Max")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        model.AddLayer(new Max(name, inputs));
                    }
                    else if (kernelName == "Mean")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        model.AddLayer(new Mean(name, inputs));
                    }
                    else if (kernelName == "Min")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        model.AddLayer(new Min(name, inputs));
                    }
                    else if (kernelName == "Mod")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        bool fmod = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new Mod(name, a, b, fmod));
                    }
                    else if (kernelName == "Mul")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        model.AddLayer(new Mul(name, a, b));
                    }
                    else if (kernelName == "Neg")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Neg(name, input));
                    }
                    else if (kernelName == "Pow")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        model.AddLayer(new Pow(name, a, b));
                    }
                    else if (kernelName == "Reciprocal")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Reciprocal(name, input));
                    }
                    else if (kernelName == "Round")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Round(name, input));
                    }
                    else if (kernelName == "ScalarMad")
                    {
                        var input = chain.InputAsString(0);
                        var dataType = (DataType)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var sFloat = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        var bFloat = executionPlan.Values(kernel.Args(2)).Value.ValAsFloat().FloatVal;
                        var sInt = executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        var bInt = executionPlan.Values(kernel.Args(4)).Value.ValAsInt().IntVal;
                        if (dataType == DataType.Float)
                            model.AddLayer(new ScalarMad(name, input, sFloat, bFloat));
                        if (dataType == DataType.Int)
                            model.AddLayer(new ScalarMad(name, input, sInt, bInt));
                    }
                    else if (kernelName == "Shrink")
                    {
                        var input = chain.InputAsString(0);
                        var bias = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var lamda = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new Shrink(name, input, bias, lamda));
                    }
                    else if (kernelName == "Sign")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Sign(name, input));
                    }
                    else if (kernelName == "Sqrt")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Sqrt(name, input));
                    }
                    else if (kernelName == "Square")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Square(name, input));
                    }
                    else if (kernelName == "Sub")
                    {
                        var a = chain.InputAsString(0);
                        var b = chain.InputAsString(1);
                        model.AddLayer(new Sub(name, a, b));
                    }
                    else if (kernelName == "Sum")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        model.AddLayer(new Sum(name, inputs));
                    }
                    else if (kernelName == "ScaleBias")
                    {
                        var input = chain.InputAsString(0);
                        var scale = chain.InputAsString(1);
                        var bias = chain.InputAsString(2);
                        model.AddLayer(new ScaleBias(name, input, scale, bias));
                    }
                    else if (kernelName == "InstanceNormalization")
                    {
                        var input = chain.InputAsString(0);
                        var scale = chain.InputAsString(1);
                        var bias = chain.InputAsString(2);
                        var epsilon = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new InstanceNormalization(name, input, scale, bias, epsilon));
                    }
                    else if (kernelName == "LayerNormalization")
                    {
                        var input = chain.InputAsString(0);
                        var scale = chain.InputAsString(1);
                        var bias = chain.InputAsString(2);
                        var epsilon = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new LayerNormalization(name, input, scale, bias, epsilon));
                    }
                    else if (kernelName == "BatchNormalization")
                    {
                        var input = chain.InputAsString(0);
                        var scale = chain.InputAsString(1);
                        var bias = chain.InputAsString(2);
                        var mean = chain.InputAsString(3);
                        var variance = chain.InputAsString(4);
                        var epsilon = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new BatchNormalization(name, input, scale, bias, mean, variance, epsilon));
                    }
                    else if (kernelName == "LRN")
                    {
                        var input = chain.InputAsString(0);
                        var alpha = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var beta = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        var bias = executionPlan.Values(kernel.Args(2)).Value.ValAsFloat().FloatVal;
                        var count = executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        model.AddLayer(new LRN(name, input, alpha, beta, bias, count));
                    }
                    else if (kernelName == "NonMaxSuppression")
                    {
                        var input = chain.InputAsString(0);
                        var scores = chain.InputAsString(1);
                        var centerPointBox = (CenterPointBox)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        if (chain.InputsLength == 2)
                        {
                            model.AddLayer(new NonMaxSuppression(name, input, scores, null, null, null, centerPointBox));
                        }
                        else if (chain.InputsLength == 3)
                        {
                            var maxOutputBoxesPerClass = chain.InputAsString(2);
                            model.AddLayer(new NonMaxSuppression(name, input, scores, maxOutputBoxesPerClass, null, null, centerPointBox));
                        }
                        else if (chain.InputsLength == 4)
                        {
                            var maxOutputBoxesPerClass = chain.InputAsString(2);
                            var iouThreshold = chain.InputAsString(3);
                            model.AddLayer(new NonMaxSuppression(name, input, scores, maxOutputBoxesPerClass, iouThreshold, null, centerPointBox));
                        }
                        else
                        {
                            var maxOutputBoxesPerClass = chain.InputAsString(2);
                            var iouThreshold = chain.InputAsString(3);
                            var scoreThreshold = chain.InputAsString(4);
                            model.AddLayer(new NonMaxSuppression(name, input, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox));
                        }
                    }
                    else if (kernelName == "RoiAlign")
                    {
                        var input = chain.InputAsString(0);
                        var rois = chain.InputAsString(1);
                        var batchIndices = chain.InputAsString(2);
                        var mode = (RoiPoolingMode)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var outputHeight = executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        var outputWidth = executionPlan.Values(kernel.Args(2)).Value.ValAsInt().IntVal;
                        var samplingRatio = executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        var spatialScale = executionPlan.Values(kernel.Args(4)).Value.ValAsFloat().FloatVal;
                        model.AddLayer(new RoiAlign(name, input, rois, batchIndices, mode, outputHeight, outputWidth, samplingRatio, spatialScale));
                    }
                    else if (kernelName == "AveragePool")
                    {
                        var input = chain.InputAsString(0);
                        var kernelShape = executionPlan.Values(kernel.Args(0)).Value.Val<IntList>()?.GetItemsArray();
                        var strides = executionPlan.Values(kernel.Args(1)).Value.Val<IntList>()?.GetItemsArray();
                        var pads = executionPlan.Values(kernel.Args(2)).Value.Val<IntList>()?.GetItemsArray();
                        var autopad = (AutoPad)executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        model.AddLayer(new AveragePool(name, input, kernelShape, strides, pads, autopad));
                    }
                    else if (kernelName == "GlobalAveragePool")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new GlobalAveragePool(name, input));
                    }
                    else if (kernelName == "GlobalMaxPool")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new GlobalMaxPool(name, input));
                    }
                    else if (kernelName == "MaxPool")
                    {
                        var input = chain.InputAsString(0);
                        var kernelShape = executionPlan.Values(kernel.Args(0)).Value.Val<IntList>()?.GetItemsArray();
                        var strides = executionPlan.Values(kernel.Args(1)).Value.Val<IntList>()?.GetItemsArray();
                        var pads = executionPlan.Values(kernel.Args(2)).Value.Val<IntList>()?.GetItemsArray();
                        var autopad = (AutoPad)executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        model.AddLayer(new MaxPool(name, input, kernelShape, strides, pads, autopad));
                    }
                    else if (kernelName == "RandomNormal")
                    {
                        var mean = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var scale = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        var shape = executionPlan.Values(kernel.Args(2)).Value.Val<IntList>()?.GetItemsArray();
                        var hasSeed = executionPlan.Values(kernel.Args(3)).Value.ValAsBool().BoolVal;
                        var seed = executionPlan.Values(kernel.Args(4)).Value.ValAsInt().IntVal;
                        model.AddLayer(new RandomNormal(name, shape, mean, scale, hasSeed ? seed : null));
                    }
                    else if (kernelName == "RandomNormalLike")
                    {
                        var input = chain.InputAsString(0);
                        var mean = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var scale = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        var hasSeed = executionPlan.Values(kernel.Args(2)).Value.ValAsBool().BoolVal;
                        var seed = executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        model.AddLayer(new RandomNormalLike(name, input, mean, scale, hasSeed ? seed : null));
                    }
                    else if (kernelName == "RandomUniform")
                    {
                        var low = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var high = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        var shape = executionPlan.Values(kernel.Args(2)).Value.Val<IntList>()?.GetItemsArray();
                        var hasSeed = executionPlan.Values(kernel.Args(3)).Value.ValAsBool().BoolVal;
                        var seed = executionPlan.Values(kernel.Args(4)).Value.ValAsInt().IntVal;
                        model.AddLayer(new RandomUniform(name, shape, low, high, hasSeed ? seed : null));
                    }
                    else if (kernelName == "RandomUniformLike")
                    {
                        var input = chain.InputAsString(0);
                        var low = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var high = executionPlan.Values(kernel.Args(1)).Value.ValAsFloat().FloatVal;
                        var hasSeed = executionPlan.Values(kernel.Args(2)).Value.ValAsBool().BoolVal;
                        var seed = executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        model.AddLayer(new RandomUniformLike(name, input, low, high, hasSeed ? seed : null));
                    }
                    else if (kernelName == "Bernoulli")
                    {
                        var input = chain.InputAsString(0);
                        var dataType = (DataType)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var hasSeed = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        var seed = executionPlan.Values(kernel.Args(2)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Bernoulli(name, input, dataType, hasSeed ? seed : null));
                    }
                    else if (kernelName == "Multinomial")
                    {
                        var input = chain.InputAsString(0);
                        var count = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var hasSeed = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        var seed = executionPlan.Values(kernel.Args(2)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Multinomial(name, input, count, hasSeed ? seed : null));
                    }
                    else if (kernelName == "LSTM")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        var outputs = new string[chain.OutputsLength];
                        for (int ii = 0; ii < outputs.Length; ii++)
                            outputs[ii] = chain.Outputs(ii).ToString();
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
                        model.AddLayer(new LSTM(name, inputs, outputs, hiddenSize, direction, activationsEnum, activationAlpha, activationBeta, clip, inputForget, layout));
                    }
                    else if (kernelName == "ReduceL1")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceL1(name, inputs, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceL2")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceL2(name, inputs, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceLogSum")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceLogSum(name, inputs, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceLogSumExp")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceLogSumExp(name, inputs, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceMax")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceMax(name, inputs, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceMean")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceMean(name, inputs, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceMin")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceMin(name, inputs, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceProd")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceProd(name, inputs, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceSum")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceSum(name, inputs, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "ReduceSumSquare")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        var keepdims = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        var noopWithEmptyAxes = executionPlan.Values(kernel.Args(1)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new ReduceSumSquare(name, inputs, keepdims, noopWithEmptyAxes));
                    }
                    else if (kernelName == "Cast")
                    {
                        var input = chain.InputAsString(0);
                        var dataType = (DataType)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Cast(name, input, dataType));
                    }
                    else if (kernelName == "CastLike")
                    {
                        var input = chain.InputAsString(0);
                        var targetType = chain.InputAsString(1);
                        model.AddLayer(new CastLike(name, input, targetType));
                    }
                    else if (kernelName == "Concat")
                    {
                        var inputs = new string[chain.InputsLength];
                        for (int ii = 0; ii < inputs.Length; ii++)
                            inputs[ii] = chain.InputAsString(ii);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Concat(name, inputs, axis));
                    }
                    else if (kernelName == "DepthToSpace")
                    {
                        var input = chain.InputAsString(0);
                        var blocksize = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var mode = (DepthToSpaceMode)executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        model.AddLayer(new DepthToSpace(name, input, blocksize, mode));
                    }
                    else if (kernelName == "Expand")
                    {
                        var input = chain.InputAsString(0);
                        var shape = chain.InputAsString(1);
                        model.AddLayer(new Expand(name, input, shape));
                    }
                    else if (kernelName == "Flatten")
                    {
                        var input = chain.InputAsString(0);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new Flatten(name, input, axis));
                    }
                    else if (kernelName == "Identity")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Identity(name, input));
                    }
                    else if (kernelName == "MoveDim")
                    {
                        var input = chain.InputAsString(0);
                        var source = executionPlan.Values(kernel.Args(0)).Value.Val<IntList>()?.GetItemsArray();
                        var destination = executionPlan.Values(kernel.Args(1)).Value.Val<IntList>()?.GetItemsArray();
                        model.AddLayer(new MoveDim(name, input, source, destination));
                    }
                    else if (kernelName == "Narrow")
                    {
                        var input = chain.InputAsString(0);
                        var dim = chain.InputAsString(1);
                        var start = chain.InputAsString(2);
                        var length = chain.InputAsString(3);
                        model.AddLayer(new Narrow(name, input, dim, start, length));
                    }
                    else if (kernelName == "Pad")
                    {
                        var data = chain.InputAsString(0);
                        var pads = chain.InputAsString(1);
                        var mode = (PadMode)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        if (chain.InputsLength == 2)
                        {
                            model.AddLayer(new Pad(name, data, pads, mode));
                        }
                        else if (chain.InputsLength == 3)
                        {
                            var constantValue = chain.InputAsString(2);
                            model.AddLayer(new Pad(name, data, pads, constantValue, mode));
                        }
                        else
                        {
                            var constantValue = chain.InputAsString(2);
                            var axes = chain.InputAsString(3);
                            model.AddLayer(new Pad(name, data, pads, constantValue, axes, mode));
                        }
                    }
                    else if (kernelName == "Reshape")
                    {
                        var input = chain.InputAsString(0);
                        var reshape = chain.InputAsString(1);
                        var allowZero = executionPlan.Values(kernel.Args(0)).Value.ValAsBool().BoolVal;
                        model.AddLayer(new Reshape(name, input, reshape, allowZero));
                    }
                    else if (kernelName == "Resize")
                    {
                        var input = chain.InputAsString(0);
                        var scalesOrSizes = chain.InputAsString(1);
                        var scaleMode = (Layers.ScaleMode)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var coordTransformMode = (Layers.CoordTransformMode)executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        var mode = (Layers.InterpolationMode)executionPlan.Values(kernel.Args(2)).Value.ValAsInt().IntVal;
                        var nearestMode = (Layers.NearestMode)executionPlan.Values(kernel.Args(3)).Value.ValAsInt().IntVal;
                        var axes = executionPlan.Values(kernel.Args(4)).Value.Val<IntList>()?.GetItemsArray();
                        model.AddLayer(new Resize(name, input, scalesOrSizes, scaleMode, mode, coordTransformMode, nearestMode, axes));
                    }
                    else if (kernelName == "Select")
                    {
                        var input = chain.InputAsString(0);
                        var dim = chain.InputAsString(1);
                        var index = chain.InputAsString(2);
                        model.AddLayer(new Select(name, input, dim, index));
                    }
                    else if (kernelName == "Slice")
                    {
                        var input = chain.InputAsString(0);
                        var starts = chain.InputAsString(1);
                        var ends = chain.InputAsString(2);
                        if (chain.InputsLength == 3)
                        {
                            model.AddLayer(new Slice(name, input, starts, ends));
                        }
                        else if (chain.InputsLength == 4)
                        {
                            var axes = chain.InputAsString(3);
                            model.AddLayer(new Slice(name, input, starts, ends, axes));
                        }
                        else
                        {
                            var axes = chain.InputAsString(3);
                            var steps = chain.InputAsString(4);
                            model.AddLayer(new Slice(name, input, starts, ends, axes, steps));
                        }
                    }
                    else if (kernelName == "SliceSet")
                    {
                        var input = chain.InputAsString(0);
                        var values = chain.InputAsString(1);
                        var starts = chain.InputAsString(2);
                        var ends = chain.InputAsString(3);
                        if (chain.InputsLength == 4)
                        {
                            model.AddLayer(new SliceSet(name, input, values, starts, ends));
                        }
                        else if (chain.InputsLength == 5)
                        {
                            var axes = chain.InputAsString(4);
                            model.AddLayer(new SliceSet(name, input, values, starts, ends, axes));
                        }
                        else
                        {
                            var axes = chain.InputAsString(4);
                            var steps = chain.InputAsString(5);
                            model.AddLayer(new SliceSet(name, input, values, starts, ends, axes, steps));
                        }
                    }
                    else if (kernelName == "SpaceToDepth")
                    {
                        var input = chain.InputAsString(0);
                        var blocksize = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        model.AddLayer(new SpaceToDepth(name, input, blocksize));
                    }
                    else if (kernelName == "Split")
                    {
                        var input = chain.InputAsString(0);
                        var axis = executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        var numOutputs = executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        var outputs = new string[chain.OutputsLength];
                        for (int ii = 0; ii < outputs.Length; ii++)
                            outputs[ii] = chain.Outputs(ii).ToString();
                        if (chain.InputsLength == 1)
                        {
                            model.AddLayer(new Split(name, input, outputs, axis, numOutputs));
                        }
                        else
                        {
                            var split = chain.InputAsString(1);
                            model.AddLayer(new Split(name, input, split, outputs, axis));
                        }
                    }
                    else if (kernelName == "Squeeze")
                    {
                        var input = chain.InputAsString(0);
                        if (chain.InputsLength == 1)
                        {
                            model.AddLayer(new Squeeze(name, input));
                        }
                        else
                        {
                            var axes = chain.InputAsString(1);
                            model.AddLayer(new Squeeze(name, input, axes));
                        }
                    }
                    else if (kernelName == "Tile")
                    {
                        var input = chain.InputAsString(0);
                        var repeats = chain.InputAsString(1);
                        model.AddLayer(new Tile(name, input, repeats));
                    }
                    else if (kernelName == "Transpose")
                    {
                        var input = chain.InputAsString(0);
                        var permutations = executionPlan.Values(kernel.Args(0)).Value.Val<IntList>()?.GetItemsArray();
                        model.AddLayer(new Transpose(name, input, permutations));
                    }
                    else if (kernelName == "Trilu")
                    {
                        var input = chain.InputAsString(0);
                        var mode = (TriluMode)executionPlan.Values(kernel.Args(0)).Value.ValAsInt().IntVal;
                        if (chain.InputsLength == 1)
                        {
                            model.AddLayer(new Trilu(name, input, mode));
                        }
                        else
                        {
                            var k = chain.InputAsString(1);
                            model.AddLayer(new Trilu(name, input, k, mode));
                        }
                    }
                    else if (kernelName == "Unsqueeze")
                    {
                        var input = chain.InputAsString(0);
                        var axes = chain.InputAsString(1);
                        model.AddLayer(new Unsqueeze(name, input, axes));
                    }
                    else if (kernelName == "Acos")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Acos(name, input));
                    }
                    else if (kernelName == "Acosh")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Acosh(name, input));
                    }
                    else if (kernelName == "Asin")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Asin(name, input));
                    }
                    else if (kernelName == "Asinh")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Asinh(name, input));
                    }
                    else if (kernelName == "Atan")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Atan(name, input));
                    }
                    else if (kernelName == "Atanh")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Atanh(name, input));
                    }
                    else if (kernelName == "Cos")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Cos(name, input));
                    }
                    else if (kernelName == "Cosh")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Cosh(name, input));
                    }
                    else if (kernelName == "Sin")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Sin(name, input));
                    }
                    else if (kernelName == "Sinh")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Sinh(name, input));
                    }
                    else if (kernelName == "Tan")
                    {
                        var input = chain.InputAsString(0);
                        model.AddLayer(new Tan(name, input));
                    }
                    else if (kernelName == "DequantizeUint8")
                    {
                        var input = chain.InputAsString(0);
                        var scale = executionPlan.Values(kernel.Args(0)).Value.ValAsFloat().FloatVal;
                        var zeroPoint = (byte)executionPlan.Values(kernel.Args(1)).Value.ValAsInt().IntVal;
                        model.AddLayer(new DequantizeUint8(name, input, scale, zeroPoint));
                    }
                    else
                        throw new NotImplementedException(kernelName);
                }

                return program.SegmentsLength;
            }
            catch (InvalidOperationException)
            {
                D.LogError("Failed to load serialized model description.");
                throw;
            }
        }

        internal static void LoadModelWeights(byte[][] modelWeightsBytes, ref Model model)
        {
            try
            {
                var buffers = new byte[modelWeightsBytes.Length][];
                for (int i = 0; i < modelWeightsBytes.Length; i++)
                {
                    var bb = new ByteBuffer(modelWeightsBytes[i], sizeof(int));
                    var weightBuffer = SentisFlatBuffer.Buffer.GetRootAsBuffer(bb);
                    buffers[i] = weightBuffer.GetStorageArray();
                }

                long constantBufferOffset = 0;
                int segmentIndex = 0;
                for (int i = 0; i < model.constants.Count; i++)
                {
                    var constant = model.constants[i];
                    int constantByteSize = constant.lengthBytes;

                    if (constantBufferOffset + constantByteSize > buffers[segmentIndex].Length)
                    {
                        segmentIndex++;
                        constantBufferOffset = 0;
                    }
                    var elementCount = constantByteSize / NativeTensorArray.k_DataItemSize;
                    var array = new NativeTensorArrayFromManagedArray(buffers[segmentIndex], (int)constantBufferOffset, elementCount);
                    model.constants[i] = new Constant(constant.index, constant.shape, constant.dataType, array);
                    constantBufferOffset += constantByteSize;
                }
            }
            catch (InvalidOperationException)
            {
                D.LogError("Failed to load serialized model weights.");
                throw;
            }
        }
    }
} // namespace Unity.Sentis
