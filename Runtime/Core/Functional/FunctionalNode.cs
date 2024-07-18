using System;

namespace Unity.Sentis
{
    abstract class FunctionalNode
    {
        protected FunctionalTensor[] m_Inputs;
        protected DataType[] m_OutputDataTypes;
        protected int[] m_OutputIndices;

        public FunctionalTensor[] Inputs => m_Inputs;
        public int[] OutputIndices => m_OutputIndices;

        protected FunctionalNode(FunctionalTensor[] inputs, DataType[] outputDataTypes)
        {
            m_Inputs = new FunctionalTensor[inputs.Length];
            for (var i = 0; i < m_Inputs.Length; i++)
                m_Inputs[i] = inputs[i] == null ? null : new FunctionalTensor(inputs[i].DataType, inputs[i].Source, inputs[i].OutputIndex);
            m_OutputDataTypes = new DataType[outputDataTypes.Length];
            for (var i = 0; i < m_OutputDataTypes.Length; i++)
                m_OutputDataTypes[i] = outputDataTypes[i];
            m_OutputIndices = new int[outputDataTypes.Length];
        }

        public FunctionalTensor[] CreateOutputs()
        {
            var outputs = new FunctionalTensor[m_OutputIndices.Length];
            for (var i = 0; i < outputs.Length; i++)
                outputs[i] = new FunctionalTensor(m_OutputDataTypes[i], this, i);
            return outputs;
        }

        public abstract void AddToModel(Model model, ref int index);
    }

    class FunctionalInput : FunctionalNode
    {
        Model.Input m_Input;

        public FunctionalInput(Model.Input input)
            : base(Array.Empty<FunctionalTensor>(), new[] { input.dataType })
        {
            m_Input = input;
        }

        public override void AddToModel(Model model, ref int index)
        {
            var inputName = "input_" + model.inputs.Count;
            m_OutputIndices[0] = index;
            model.AddInput(inputName, index, m_Input.dataType, m_Input.shape);
            index++;
        }
    }

    class FunctionalOutput : FunctionalNode
    {
        public FunctionalOutput(FunctionalTensor input)
            : base(new[] { input }, Array.Empty<DataType>()) { }

        public override void AddToModel(Model model, ref int index)
        {
            var outputName = "output_" + model.outputs.Count;
            model.AddOutput(outputName, m_Inputs[0].Index);
        }
    }

    class FunctionalConstant : FunctionalNode
    {
        Layers.Constant m_Constant;

        public FunctionalConstant(Layers.Constant constant)
            : base(Array.Empty<FunctionalTensor>(), new[] { constant.dataType })
        {
            m_Constant = constant;
        }

        public override void AddToModel(Model model, ref int index)
        {
            m_Constant.index = index;
            m_OutputIndices[0] = index;
            model.constants.Add(m_Constant);
            index++;
        }
    }

    class FunctionalLayer : FunctionalNode
    {
        Layers.Layer m_Layer;

        public FunctionalLayer(FunctionalTensor[] inputs, DataType[] outputDataTypes, Layers.Layer layer)
            : base(inputs, outputDataTypes)
        {
            m_Layer = layer;
        }

        public override void AddToModel(Model model, ref int index)
        {
            for (var i = 0; i < m_Inputs.Length; i++)
                m_Layer.inputs[i] = m_Inputs[i] is null ? -1 : m_Inputs[i].Index;

            for (var i = 0; i < m_OutputIndices.Length; i++)
            {
                m_OutputIndices[i] = index;
                m_Layer.outputs[i] = index;
                index++;
            }

            model.layers.Add(m_Layer);
        }
    }
}
