using System;
using System.Collections.Generic;
using System.Linq;
using SentisFlatBuffer;
using Unity.Sentis.Google.FlatBuffers;

namespace Unity.Sentis
{
    static class ModelUpgrader
    {
        static T[] GetArray<T>(int length, Func<int, T> indexer)
        {
            var array = new T[length];
            for (var i = 0; i < length; i++)
                array[i] = indexer(i);
            return array;
        }

        static int AddTensorValue(FlatBufferBuilder builder, List<Offset<EValue>> valuesOffsets)
        {
            var index = valuesOffsets.Count;
            valuesOffsets.Add(EValue.CreateEValue(builder, KernelTypes.Tensor, SentisFlatBuffer.Tensor.CreateTensor(builder).Value));
            return index;
        }

        static int GetOperatorIndex(List<string> operators, string op)
        {
            var index = operators.IndexOf(op);
            if (index < 0)
            {
                index = operators.Count;
                operators.Add(op);
            }

            return index;
        }

        static int AddIntValue(FlatBufferBuilder builder, List<Offset<EValue>> valuesOffsets, int v)
        {
            var val = Int.CreateInt(builder, v);
            valuesOffsets.Add(EValue.CreateEValue(builder, KernelTypes.Int, val.Value));
            return valuesOffsets.Count - 1;
        }

        static int AddFloatValue(FlatBufferBuilder builder, List<Offset<EValue>> valuesOffsets, float v)
        {
            var val = Float.CreateFloat(builder, v);
            valuesOffsets.Add(EValue.CreateEValue(builder, KernelTypes.Float, val.Value));
            return valuesOffsets.Count - 1;
        }

        public static Program UpgradeFlatbuffer(Program program, uint toVersion, Func<Chain, FlatBufferBuilder, List<string>, List<Offset<Chain>>, List<Offset<EValue>>, bool> OnChain = null, Func<EValue, FlatBufferBuilder, List<Offset<EValue>>, bool> OnValue = null)
        {
            var executionPlan = program.ExecutionPlan.Value;

            var builder = new FlatBufferBuilder(1);

            var inputs = executionPlan.GetInputsArray();
            var inputsName = GetArray(executionPlan.InputsNameLength, executionPlan.InputsName);
            var outputs = executionPlan.GetOutputsArray();
            var outputsName = GetArray(executionPlan.OutputsNameLength, executionPlan.OutputsName);
            var operators = GetArray(executionPlan.OperatorsLength, i => executionPlan.Operators(i).Value.Name).ToList();

            var valuesOffsets = new List<Offset<EValue>>();
            for (var i = 0; i < executionPlan.ValuesLength; i++)
            {
                var valOffset = 0;
                var value = executionPlan.Values(i).Value;

                // check if upgrade code handles case
                if (OnValue != null && OnValue(value, builder, valuesOffsets))
                    continue;

                switch (value.ValType)
                {
                    case KernelTypes.NONE:
                        valOffset = default;
                        break;
                    case KernelTypes.Null:
                        valOffset = default;
                        break;
                    case KernelTypes.Int:
                        valOffset = Int.CreateInt(builder, value.ValAsInt().IntVal).Value;
                        break;
                    case KernelTypes.Float:
                        valOffset = Float.CreateFloat(builder, value.ValAsFloat().FloatVal).Value;
                        break;
                    case KernelTypes.Bool:
                        valOffset = Bool.CreateBool(builder, value.ValAsBool().BoolVal).Value;
                        break;
                    case KernelTypes.Byte:
                        valOffset = SentisFlatBuffer.Byte.CreateByte(builder, value.ValAsByte().ByteVal).Value;
                        break;
                    case KernelTypes.Tensor:
                        var inputDesc = value.ValAsTensor();
                        var dynamicSizesOffset = default(VectorOffset);
                        var fixedSizesOffset = default(VectorOffset);
                        var sizes = inputDesc.GetFixedSizesArray();
                        if (sizes != null)
                            fixedSizesOffset = SentisFlatBuffer.Tensor.CreateFixedSizesVector(builder, inputDesc.GetFixedSizesArray());
                        if (inputDesc.ShapeDynamism == TensorShapeDynamism.DYNAMIC_UNBOUND)
                        {
                            var dimOffsets = new Offset<EDim>[inputDesc.DynamicSizesLength];
                            for (var j = 0; j < inputDesc.DynamicSizesLength; j++)
                            {
                                var dim = inputDesc.DynamicSizes(j).Value;
                                var dimValOffset = dim.ValType switch
                                {
                                    SymbolicDim.NONE => 0,
                                    SymbolicDim.Int => Int.CreateInt(builder, dim.ValAsInt().IntVal).Value,
                                    SymbolicDim.Byte => SentisFlatBuffer.Byte.CreateByte(builder, dim.ValAsByte().ByteVal).Value,
                                    _ => throw new ArgumentOutOfRangeException()
                                };
                                dimOffsets[j] = EDim.CreateEDim(builder, dim.ValType, dimValOffset);
                            }
                            dynamicSizesOffset = SentisFlatBuffer.Tensor.CreateDynamicSizesVector(builder, dimOffsets);
                        }
                        var val = SentisFlatBuffer.Tensor.CreateTensor(
                            builder,
                            scalar_type: inputDesc.ScalarType,
                            length_byte: inputDesc.LengthByte,
                            shape_dynamism: inputDesc.ShapeDynamism,
                            fixed_sizesOffset: fixedSizesOffset,
                            constant_buffer_idx: inputDesc.ConstantBufferIdx,
                            dynamic_sizesOffset: dynamicSizesOffset,
                            storage_offset: inputDesc.StorageOffset);
                        valOffset = val.Value;
                        break;
                    case KernelTypes.String:
                        valOffset = SentisFlatBuffer.String.CreateString(builder, builder.CreateString(value.ValAsString().StringVal)).Value;
                        break;
                    case KernelTypes.IntList:
                        valOffset = IntList.CreateIntList(builder, IntList.CreateItemsVector(builder, value.ValAsIntList().GetItemsArray())).Value;
                        break;
                    case KernelTypes.FloatList:
                        valOffset = FloatList.CreateFloatList(builder, FloatList.CreateItemsVector(builder, value.ValAsFloatList().GetItemsArray())).Value;
                        break;
                    case KernelTypes.BoolList:
                        valOffset = BoolList.CreateBoolList(builder, BoolList.CreateItemsVector(builder, value.ValAsBoolList().GetItemsArray())).Value;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }

                valuesOffsets.Add(EValue.CreateEValue(builder, value.ValType, valOffset));
            }

            var chainsOffsets = new List<Offset<Chain>>();
            for (var i = 0; i < executionPlan.ChainsLength; i++)
            {
                var chain = executionPlan.Chains(i).Value;

                // check if upgrade code handles case
                if (OnChain != null && OnChain(chain, builder, operators, chainsOffsets, valuesOffsets))
                    continue;

                var chainInputs = chain.GetInputsArray();
                var chainOutputs = chain.GetOutputsArray();
                var instructionsOffsets = new List<Offset<Instruction>>();
                for (var j = 0; j < chain.InstructionsLength; j++)
                {
                    var instruction = chain.Instructions(j).Value;
                    var instrArgsOffset = 0;
                    switch (instruction.InstrArgsType)
                    {
                        case InstructionArguments.NONE:
                            break;
                        case InstructionArguments.KernelCall:
                            var kernelCall = instruction.InstrArgsAsKernelCall();
                            var args = kernelCall.GetArgsArray();
                            instrArgsOffset = KernelCall.CreateKernelCall(builder, kernelCall.OpIndex, ExecutionPlan.CreateInputsVector(builder, args)).Value;
                            break;
                        default:
                            throw new ArgumentOutOfRangeException();
                    }
                    var instructionOffset = Instruction.CreateInstruction(builder, instruction.InstrArgsType, instrArgsOffset);
                    instructionsOffsets.Add(instructionOffset);
                }

                var chainOffset = Chain.CreateChain(
                    builder,
                    chainInputs == null ? default : Chain.CreateInputsVector(builder, chainInputs),
                    chainOutputs == null ? default : Chain.CreateOutputsVector(builder, chainOutputs),
                    Chain.CreateInstructionsVector(builder, instructionsOffsets.ToArray()));
                chainsOffsets.Add(chainOffset);
            }

            var programExecutionPlan = ExecutionPlan.CreateExecutionPlan(builder,
                nameOffset: builder.CreateString(executionPlan.Name),
                valuesOffset: ExecutionPlan.CreateValuesVector(builder, valuesOffsets.ToArray()),
                inputsOffset: ExecutionPlan.CreateInputsVector(builder, inputs),
                inputs_nameOffset: ExecutionPlan.CreateInputsNameVector(builder, inputsName.Select(i => builder.CreateString(i)).ToArray()),
                outputsOffset: ExecutionPlan.CreateOutputsVector(builder, outputs),
                outputs_nameOffset: ExecutionPlan.CreateOutputsNameVector(builder, outputsName.Select(i => builder.CreateString(i)).ToArray()),
                chainsOffset: ExecutionPlan.CreateChainsVector(builder, chainsOffsets.ToArray()),
                operatorsOffset: ExecutionPlan.CreateOperatorsVector(builder, operators.Select(i => Operator.CreateOperator(builder, builder.CreateString(i))).ToArray()),
                backend_partitioningOffset: BackendPartitioning.CreateBackendPartitioning(builder, BackendPartitioning.CreateChainsVector(builder, Array.Empty<int>()), SentisFlatBuffer.BackendType.CPU)
            );

            var dataSegments = new Offset<DataSegment>[program.SegmentsLength];
            for (var i = 0; i < program.SegmentsLength; i++)
            {
                var segment = program.Segments(i).Value;
                dataSegments[i] = DataSegment.CreateDataSegment(builder, segment.Offset, segment.Size);
            }

            var programDataSegments = Program.CreateSegmentsVector(builder, dataSegments);

            var programOffset = Program.CreateProgram(builder,
                version: toVersion,
                execution_planOffset: programExecutionPlan,
                segmentsOffset: programDataSegments);
            builder.FinishSizePrefixed(programOffset.Value);

            ProfilerMarkers.SaveModelDesc.End();
            var modelDescription = builder.DataBuffer.ToSizedArray();
            var bb = new ByteBuffer(modelDescription, sizeof(int));
            return Program.GetRootAsProgram(bb);
        }

        static bool UpgradeChainV1toV2(Chain chain, FlatBufferBuilder builder, List<string> operators, List<Offset<Chain>> chainsOffsets, List<Offset<EValue>> valuesOffsets)
        {
            var instruction = chain.Instructions(0).Value;
            if (instruction.InstrArgsType != InstructionArguments.KernelCall)
                return false;
            var kernelCall = instruction.InstrArgsAsKernelCall();
            var k = operators[kernelCall.OpIndex];
            switch (k)
            {
                case "Max":
                case "Min":
                {
                    var instructionOffset = Instruction.CreateInstruction(builder, instruction.InstrArgsType, KernelCall.CreateKernelCall(builder, kernelCall.OpIndex, ExecutionPlan.CreateInputsVector(builder, Array.Empty<int>())).Value);

                    var lhs = chain.Inputs(0);
                    for (var j = 1; j < chain.InputsLength; j++)
                    {
                        var rhs = chain.Inputs(j);
                        var ret = j == chain.InputsLength - 1 ? chain.Outputs(0) : AddTensorValue(builder, valuesOffsets);
                        chainsOffsets.Add(Chain.CreateChain(
                            builder,
                            Chain.CreateInputsVector(builder, new[] { lhs, rhs }),
                            Chain.CreateOutputsVector(builder, new[] { ret }),
                            Chain.CreateInstructionsVector(builder, new[] { instructionOffset }))
                        );
                        lhs = ret;
                    }
                    return true;
                }
                case "Sum":
                {
                    var opIndex = GetOperatorIndex(operators, "Add");
                    var instructionOffset = Instruction.CreateInstruction(builder, instruction.InstrArgsType, KernelCall.CreateKernelCall(builder, opIndex, ExecutionPlan.CreateInputsVector(builder, Array.Empty<int>())).Value);

                    var lhs = chain.Inputs(0);
                    for (var j = 1; j < chain.InputsLength; j++)
                    {
                        var rhs = chain.Inputs(j);
                        var ret = j == chain.InputsLength - 1 ? chain.Outputs(0) : AddTensorValue(builder, valuesOffsets);
                        chainsOffsets.Add(Chain.CreateChain(
                            builder,
                            Chain.CreateInputsVector(builder, new[] { lhs, rhs }),
                            Chain.CreateOutputsVector(builder, new[] { ret }),
                            Chain.CreateInstructionsVector(builder, new[] { instructionOffset }))
                        );
                        lhs = ret;
                    }
                    return true;
                }
                case "Mean":
                {
                    var opIndex = GetOperatorIndex(operators, "Add");
                    var instructionOffset = Instruction.CreateInstruction(builder, instruction.InstrArgsType, KernelCall.CreateKernelCall(builder, opIndex, ExecutionPlan.CreateInputsVector(builder, Array.Empty<int>())).Value);

                    var lhs = chain.Inputs(0);
                    for (var j = 1; j < chain.InputsLength; j++)
                    {
                        var rhs = chain.Inputs(j);
                        var ret = AddTensorValue(builder, valuesOffsets);
                        chainsOffsets.Add(Chain.CreateChain(
                            builder,
                            Chain.CreateInputsVector(builder, new[] { lhs, rhs }),
                            Chain.CreateOutputsVector(builder, new[] { ret }),
                            Chain.CreateInstructionsVector(builder, new[] { instructionOffset }))
                        );
                        lhs = ret;
                    }

                    var scalarMadOpIndex = GetOperatorIndex(operators, "ScalarMad");
                    var args = new[]
                    {
                        AddIntValue(builder, valuesOffsets, (int)DataType.Float),
                        AddFloatValue(builder, valuesOffsets, 1 / (float)chain.InputsLength),
                        AddFloatValue(builder, valuesOffsets, 0),
                        AddIntValue(builder, valuesOffsets, 0),
                        AddIntValue(builder, valuesOffsets, 0),
                    };
                    var scalarMadInstructionOffset = Instruction.CreateInstruction(builder, instruction.InstrArgsType, KernelCall.CreateKernelCall(builder, scalarMadOpIndex, ExecutionPlan.CreateInputsVector(builder, args)).Value);
                    chainsOffsets.Add(Chain.CreateChain(
                        builder,
                        Chain.CreateInputsVector(builder, new[] { lhs }),
                        Chain.CreateOutputsVector(builder, new[] { chain.Outputs(0) }),
                        Chain.CreateInstructionsVector(builder, new[] { scalarMadInstructionOffset }))
                    );

                    return true;
                }
                default:
                    return false;
            }
        }

        static bool UpgradeChainV2toV3(Chain chain, FlatBufferBuilder builder, List<string> operators, List<Offset<Chain>> chainsOffsets, List<Offset<EValue>> valuesOffsets)
        {
            return false;
        }

        static bool UpgradeChainV3toV4(Chain chain, FlatBufferBuilder builder, List<string> operators, List<Offset<Chain>> chainsOffsets, List<Offset<EValue>> valuesOffsets)
        {
            return false;
        }

        public static Program Upgrade(Program program)
        {
            if (program.Version == 1)
                program = UpgradeFlatbuffer(program, 2, UpgradeChainV1toV2);
            if (program.Version == 2)
                program = UpgradeFlatbuffer(program, 3, UpgradeChainV2toV3);
            if (program.Version == 3)
                program = UpgradeFlatbuffer(program, 4, UpgradeChainV3toV4);

            return program;
        }
    }
}
