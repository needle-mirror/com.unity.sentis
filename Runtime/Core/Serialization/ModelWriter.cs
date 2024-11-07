using System;
using System.Collections.Generic;
using System.IO;
using Unity.Sentis.Google.FlatBuffers;
using SentisFlatBuffer;
using System.Reflection;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Unity.Sentis.Editor")]

namespace Unity.Sentis
{
    /// <summary>
    /// Provides methods for saving models.
    /// </summary>
    public static class ModelWriter
    {
        // The overhead in bytes on top of the byte array data for a FlatBuffer with a prefix length
        const int k_WeightsFlatBufferOverhead = 32;
        // The maximum size of a FlatBuffer is int.MaxValue bytes, subtract the overhead to get the maximum size for a single constant
        const long k_MaxConstantSize = int.MaxValue - k_WeightsFlatBufferOverhead;
        internal const int version = 3;

        /// <summary>
        /// Serializes and saves the model description and weights to a file.
        /// </summary>
        /// <param name="fileName">The path to save the model file to.</param>
        /// <param name="model">The model to save to file.</param>
        public static void Save(string fileName, Model model)
        {
            using var fileStream = File.Create(fileName);
            Save(fileStream, model);
        }

        /// <summary>
        /// Serializes and saves the model description and weights to a stream.
        /// </summary>
        /// <param name="stream">The stream to save the model to.</param>
        /// <param name="model">The model to save.</param>
        public static void Save(Stream stream, Model model)
        {
            var modelDescriptionBytes = SaveModelDescription(model);
            stream.Write(modelDescriptionBytes);
            var modelWeightsChunksBytes = SaveModelWeights(model);
            foreach (var chunkBytes in modelWeightsChunksBytes)
                stream.Write(chunkBytes);
        }

        /// <summary>
        /// Serializes and saves the model asset bytes to a stream.
        /// </summary>
        /// <param name="stream">The stream to save the model to.</param>
        /// <param name="modelAsset">The model asset to save.</param>
        public static void Save(Stream stream, ModelAsset modelAsset)
        {
            stream.Write(modelAsset.modelAssetData.value);
            for (var i = 0; i < (modelAsset.modelWeightsChunks?.Length ?? 0); i++)
                stream.Write(modelAsset.modelWeightsChunks[i].value);
        }

        /// <summary>
        /// Serializes and saves the model description and weights to a file.
        /// </summary>
        /// <param name="fileName">The path to save the model file to</param>
        /// <param name="modelAsset">The model asset to save to file</param>
        public static void Save(string fileName, ModelAsset modelAsset)
        {
            using var fileStream = File.Create(fileName);
            Save(fileStream, modelAsset);
        }

        internal static byte[] SaveModelDescription(Model model)
        {
            ProfilerMarkers.SaveModelDesc.Begin();

            var builder = new FlatBufferBuilder(1);

            // values: tensor desc + float/int layer values
            var values = new List<Offset<EValue>>();
            int valuesCount = 0;

            // mapping of indexes in model to indexes in serialized model
            var indexMapping = new Dictionary<int, int>();

            // Inputs
            var inputsIndices = new int[model.inputs.Count];
            var inputsNames = new StringOffset[model.inputs.Count];
            for (int i = 0; i < model.inputs.Count; i++)
            {
                var input = model.inputs[i];

                inputsIndices[i] = i;
                inputsNames[i] = builder.CreateString(input.name);

                Offset<SentisFlatBuffer.Tensor> val;
                if (input.shape.IsStatic())
                {
                    var size = SentisFlatBuffer.Tensor.CreateFixedSizesVector(builder, input.shape.ToTensorShape().ToArray());
                    int lengthByte = input.shape.ToTensorShape().length * sizeof(int); // TODO<quantization> quantized inputs
                    val = SentisFlatBuffer.Tensor.CreateTensor(builder, (ScalarType)input.dataType, lengthByte, size);
                }
                else
                {
                    var dims = new Offset<EDim>[input.shape.rank];
                    for (int k = 0; k < input.shape.rank; k++)
                    {
                        if (input.shape[k].isValue)
                        {
                            var vv = Int.CreateInt(builder, input.shape[k].value);
                            dims[k] = EDim.CreateEDim(builder, SymbolicDim.Int, vv.Value);
                        }
                        else
                        {
                            var vv = SentisFlatBuffer.Byte.CreateByte(builder, input.shape[k].param);
                            dims[k] = EDim.CreateEDim(builder, SymbolicDim.Byte, vv.Value);
                        }
                    }
                    var size = SentisFlatBuffer.Tensor.CreateDynamicSizesVector(builder, dims);
                    val = SentisFlatBuffer.Tensor.CreateTensor(builder, (ScalarType)input.dataType, shape_dynamism: TensorShapeDynamism.DYNAMIC_UNBOUND, dynamic_sizesOffset: size);
                }
                values.Add(EValue.CreateEValue(builder, KernelTypes.Tensor, val.Value));
                indexMapping[input.index] = valuesCount;
                valuesCount++;
            }
            var epModelInputs = ExecutionPlan.CreateInputsVector(builder, inputsIndices);
            var epModelInputsNames = ExecutionPlan.CreateInputsNameVector(builder, inputsNames);

            // Model Settings
            var epModelName = builder.CreateString(model.ProducerName);

            // constants
            var constants = new Dictionary<int, Constant>();
            for (int i = 0; i < model.constants.Count; i++)
            {
                constants.Add(model.constants[i].index, model.constants[i]);
            }

            Dictionary<string, int> operatorNames = new Dictionary<string, int>();
            var operators = new List<Offset<Operator>>();
            var chains = new List<Offset<Chain>>();
            var chainCPU = new List<int>();
            var segmentLength = new List<int>();
            long constantBufferOffset = 0;
            long constantBuffersSize = 0;
            for (int i = 0; i < model.layers.Count; i++)
            {
                var layer = model.layers[i];
                if (!operatorNames.ContainsKey(layer.opName))
                {
                    var operationName = builder.CreateString(layer.opName);
                    var operation = Operator.CreateOperator(builder, operationName);
                    operators.Add(operation);
                    operatorNames[layer.opName] = operatorNames.Count;
                }

                // layer inputs
                var layerInputs = new int[layer.inputs.Length];
                for (int k = 0; k < layer.inputs.Length; k++)
                {
                    var input = layer.inputs[k];

                    if (input != -1 && !indexMapping.ContainsKey(input))
                    {
                        // layer input must be a constant
                        var constant = constants[input];
                        var size = SentisFlatBuffer.Tensor.CreateFixedSizesVector(builder, constant.shape.ToArray());
                        Logger.AssertIsTrue(constant.lengthBytes <= k_MaxConstantSize, "Constant of size {0} is larger than maximum serializable constant size {1}", constant.lengthBytes, k_MaxConstantSize);
                        if (constantBufferOffset + constant.lengthBytes > k_MaxConstantSize)
                        {
                            segmentLength.Add((int)constantBufferOffset);
                            constantBufferOffset = 0;
                        }
                        var val = SentisFlatBuffer.Tensor.CreateTensor(builder, (ScalarType)constant.dataType, constant.lengthBytes, size, (uint)(1 + segmentLength.Count), (int)constantBufferOffset);
                        values.Add(EValue.CreateEValue(builder, KernelTypes.Tensor, val.Value));
                        constantBufferOffset += constant.lengthBytes;
                        constantBuffersSize += constant.lengthBytes;
                        indexMapping[input] = valuesCount;
                        valuesCount++;
                    }

                    layerInputs[k] = (input == -1) ? -1 : indexMapping[input];
                }

                var layerAttributesInputs = new List<int>();
                var fields = layer.GetType().GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
                foreach (var field in fields)
                {
                    var name = field.Name;
                    if (name == "name" || name == "inputs" || name == "outputs")
                        continue;

                    var value = field.GetValue(layer);
                    if (value is bool bv)
                    {
                        var val = Bool.CreateBool(builder, bv);
                        values.Add(EValue.CreateEValue(builder, KernelTypes.Bool, val.Value));
                    }
                    else if (value is float fv)
                    {
                        var val = Float.CreateFloat(builder, fv);
                        values.Add(EValue.CreateEValue(builder, KernelTypes.Float, val.Value));
                    }
                    else if (value is int iv)
                    {
                        var val = Int.CreateInt(builder, iv);
                        values.Add(EValue.CreateEValue(builder, KernelTypes.Int, val.Value));
                    }
                    else if (value is byte bytev)
                    {
                        var val = Int.CreateInt(builder, (int)bytev);
                        values.Add(EValue.CreateEValue(builder, KernelTypes.Int, val.Value));
                    }
                    else if (value is int[] ia)
                    {
                        var item = IntList.CreateItemsVector(builder, ia);
                        var val = IntList.CreateIntList(builder, item);
                        values.Add(EValue.CreateEValue(builder, KernelTypes.IntList, val.Value));
                    }
                    else if (value is float[] fa)
                    {
                        var item = FloatList.CreateItemsVector(builder, fa);
                        var val = FloatList.CreateFloatList(builder, item);
                        values.Add(EValue.CreateEValue(builder, KernelTypes.FloatList, val.Value));
                    }
                    else if (value is string s)
                    {
                        var item = builder.CreateString(s);
                        var val = SentisFlatBuffer.String.CreateString(builder, item);
                        values.Add(EValue.CreateEValue(builder, KernelTypes.String, val.Value));
                    }
                    else if (value is Enum)
                    {
                        int e = (int)Convert.ChangeType(value, Enum.GetUnderlyingType(value.GetType()));
                        var val = Int.CreateInt(builder, e);
                        values.Add(EValue.CreateEValue(builder, KernelTypes.Int, val.Value));
                    }
                    else if (value is null)
                    {
                        values.Add(EValue.CreateEValue(builder, KernelTypes.NONE));
                    }
                    else
                    {
                        continue;
                    }
                    layerAttributesInputs.Add(valuesCount);
                    valuesCount++;
                }

                var layerOutputs = new List<int>(); // TODO layer outputs

                foreach (var output in layer.outputs)
                {
                    layerOutputs.Add(valuesCount);
                    var valTo = SentisFlatBuffer.Tensor.CreateTensor(builder);
                    values.Add(EValue.CreateEValue(builder, KernelTypes.Tensor, valTo.Value));
                    indexMapping[output] = valuesCount;
                    valuesCount++;
                }

                var layerInputVector = ExecutionPlan.CreateInputsVector(builder, layerInputs);
                var layerOutputVector = ExecutionPlan.CreateOutputsVector(builder, layerOutputs.ToArray());
                // attributes
                var layerAttributesVector = ExecutionPlan.CreateInputsVector(builder, layerAttributesInputs.ToArray());
                var kernelCall = KernelCall.CreateKernelCall(builder, operatorNames[layer.opName], layerAttributesVector);
                Instruction.StartInstruction(builder);
                Instruction.AddInstrArgsType(builder, InstructionArguments.KernelCall);
                Instruction.AddInstrArgs(builder, kernelCall.Value);
                var linstruction = Instruction.EndInstruction(builder);

                var lInstructionVector = Chain.CreateInstructionsVector(builder, new[] { linstruction });
                Chain.StartChain(builder);
                Chain.AddInputs(builder, layerInputVector);
                Chain.AddOutputs(builder, layerOutputVector);
                Chain.AddInstructions(builder, lInstructionVector);
                var lChain = Chain.EndChain(builder);
                chains.Add(lChain);
            }

            for (int i = 0; i < model.outputs.Count; i++)
            {
                var output = model.outputs[i];
                if (!constants.ContainsKey(output.index))
                    continue;

                var constant = constants[output.index];

                var size = SentisFlatBuffer.Tensor.CreateFixedSizesVector(builder, constant.shape.ToArray());
                var val = SentisFlatBuffer.Tensor.CreateTensor(builder, (ScalarType)constant.dataType, constant.lengthBytes, size, (uint)(1 + segmentLength.Count), (int)constantBufferOffset);
                values.Add(EValue.CreateEValue(builder, KernelTypes.Tensor, val.Value));
                constantBufferOffset += constant.lengthBytes;
                constantBuffersSize += constant.lengthBytes;

                var constantVector = ExecutionPlan.CreateInputsVector(builder, new[] { valuesCount });

                var kernelCall = KernelCall.CreateKernelCall(builder);
                Instruction.StartInstruction(builder);
                Instruction.AddInstrArgsType(builder, InstructionArguments.NONE);
                var linstruction = Instruction.EndInstruction(builder);

                var lInstructionVector = Chain.CreateInstructionsVector(builder, new[] { linstruction });
                Chain.StartChain(builder);
                Chain.AddInputs(builder, constantVector);
                Chain.AddInstructions(builder, lInstructionVector);
                var lChain = Chain.EndChain(builder);
                chains.Add(lChain);

                indexMapping[constant.index] = valuesCount;
                valuesCount++;
            }

            var outputIndices = new List<int>();
            var outputsNames = new List<StringOffset>();
            foreach (var output in model.outputs)
            {
                outputIndices.Add(indexMapping[output.index]);
                outputsNames.Add(builder.CreateString(output.name));
            }

            var epModelOutputsNames = ExecutionPlan.CreateOutputsNameVector(builder, outputsNames.ToArray());
            var epModelOutputs = ExecutionPlan.CreateOutputsVector(builder, outputIndices.ToArray());
            var epModelValues = ExecutionPlan.CreateValuesVector(builder, values.ToArray());
            var epModelOperators = ExecutionPlan.CreateOperatorsVector(builder, operators.ToArray());
            var epModelChains = ExecutionPlan.CreateChainsVector(builder, chains.ToArray());
            var epCPUChains = BackendPartitioning.CreateChainsVector(builder, chainCPU.ToArray());
            var epBackendPartitioning = BackendPartitioning.CreateBackendPartitioning(builder, epCPUChains, SentisFlatBuffer.BackendType.CPU);

            ExecutionPlan.StartExecutionPlan(builder);
            ExecutionPlan.AddName(builder, epModelName);
            ExecutionPlan.AddInputs(builder, epModelInputs);
            ExecutionPlan.AddInputsName(builder, epModelInputsNames);
            ExecutionPlan.AddOutputs(builder, epModelOutputs);
            ExecutionPlan.AddOutputsName(builder, epModelOutputsNames);
            ExecutionPlan.AddValues(builder, epModelValues);
            ExecutionPlan.AddOperators(builder, epModelOperators);
            ExecutionPlan.AddChains(builder, epModelChains);
            ExecutionPlan.AddBackendPartitioning(builder, epBackendPartitioning);
            var programExecutionPlan = ExecutionPlan.EndExecutionPlan(builder);

            segmentLength.Add((int)constantBufferOffset);
            var dataSegments = new Offset<DataSegment>[segmentLength.Count];
            for (int i = 0; i < segmentLength.Count; i++)
            {
                dataSegments[i] = DataSegment.CreateDataSegment(builder, (ulong)constantBuffersSize, (ulong)segmentLength[i]);
            }
            var programDataSegments = Program.CreateSegmentsVector(builder, dataSegments);

            Program.StartProgram(builder);
            Program.AddVersion(builder, version);
            Program.AddExecutionPlan(builder, programExecutionPlan);
            Program.AddSegments(builder, programDataSegments);
            var program = Program.EndProgram(builder);
            builder.FinishSizePrefixed(program.Value);

            ProfilerMarkers.SaveModelDesc.End();
            return builder.DataBuffer.ToSizedArray();
        }

        internal static byte[][] SaveModelWeights(Model model)
        {
            ProfilerMarkers.SaveModelWeights.Begin();

            var constants = new Dictionary<int, Constant>();
            var foundConstants = new HashSet<int>();
            for (int i = 0; i < model.constants.Count; i++)
            {
                constants.Add(model.constants[i].index, model.constants[i]);
            }
            var constantsInOrder = new List<Constant>();
            for (int i = 0; i < model.layers.Count; i++)
            {
                var layer = model.layers[i];
                foreach (var input in layer.inputs)
                {
                    if (!constants.ContainsKey(input))
                        continue;
                    if (foundConstants.Contains(input))
                        continue;
                    foundConstants.Add(input);
                    constantsInOrder.Add(constants[input]);
                }
            }
            for (int i = 0; i < model.outputs.Count; i++)
            {
                var output = model.outputs[i];
                if (!constants.ContainsKey(output.index))
                    continue;
                if (foundConstants.Contains(output.index))
                    continue;
                foundConstants.Add(output.index);
                constantsInOrder.Add(constants[output.index]);
            }

            List<byte[]> segmentData = new List<byte[]>();
            long constantBufferByteSize = 0;
            for (int i = 0; i < constantsInOrder.Count; i++)
            {
                int constantByteSize = constantsInOrder[i].lengthBytes;
                if (constantBufferByteSize + constantByteSize > k_MaxConstantSize)
                {
                    segmentData.Add(new byte[constantBufferByteSize]);
                    constantBufferByteSize = 0;
                }
                constantBufferByteSize += constantByteSize; // lengthbyte
            }
            segmentData.Add(new byte[constantBufferByteSize]);
            var segmentBuffers = new byte[segmentData.Count][];
            foundConstants.Clear();

            FlatBufferBuilder builder; VectorOffset storage; Offset<SentisFlatBuffer.Buffer> cb;

            int segmentIndex = 0;
            constantBufferByteSize = 0;
            for (int i = 0; i < constantsInOrder.Count; i++)
            {
                var constant = constantsInOrder[i];
                int constantByteSize = constant.lengthBytes;
                Logger.AssertIsTrue(constant.lengthBytes <= k_MaxConstantSize, "Constant of size {0} is larger than maximum serializable constant size {1}", constant.lengthBytes, k_MaxConstantSize);
                if (constantBufferByteSize + constantByteSize > k_MaxConstantSize)
                {
                    // Preallocate exact size for FlatBuffer otherwise allocator can overshoot max FlatBuffer size for large byte arrays
                    builder = new FlatBufferBuilder(k_WeightsFlatBufferOverhead + segmentData[segmentIndex].Length);
                    storage = SentisFlatBuffer.Buffer.CreateStorageVectorBlock(builder, segmentData[segmentIndex]);
                    SentisFlatBuffer.Buffer.StartBuffer(builder);
                    SentisFlatBuffer.Buffer.AddStorage(builder, storage);
                    cb = SentisFlatBuffer.Buffer.EndBuffer(builder);
                    builder.FinishSizePrefixed(cb.Value);
                    segmentBuffers[segmentIndex] = builder.DataBuffer.ToSizedArray();
                    segmentData[segmentIndex] = null;

                    segmentIndex++;
                    constantBufferByteSize = 0;
                }
                if (constant.weights != null)
                    NativeTensorArray.BlockCopy(constant.weights, 0, segmentData[segmentIndex], (int)constantBufferByteSize, constantByteSize);
                constantBufferByteSize += constantByteSize;
            }

            // Preallocate exact size for FlatBuffer otherwise allocator can overshoot max FlatBuffer size for large byte arrays
            builder = new FlatBufferBuilder(k_WeightsFlatBufferOverhead + segmentData[segmentIndex].Length);
            storage = SentisFlatBuffer.Buffer.CreateStorageVectorBlock(builder, segmentData[segmentIndex]);
            SentisFlatBuffer.Buffer.StartBuffer(builder);
            SentisFlatBuffer.Buffer.AddStorage(builder, storage);
            cb = SentisFlatBuffer.Buffer.EndBuffer(builder);
            builder.FinishSizePrefixed(cb.Value);
            segmentBuffers[segmentIndex] = builder.DataBuffer.ToSizedArray();

            ProfilerMarkers.SaveModelWeights.End();
            return segmentBuffers;
        }
    }
}
