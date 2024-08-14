using System;

namespace Unity.Sentis
{
    abstract class Node
    {
        protected FunctionalTensor[] m_Inputs;
        protected DataType[] m_OutputDataTypes;
        protected int[] m_OutputIndices;

        public FunctionalTensor[] Inputs => m_Inputs;
        public int[] OutputIndices => m_OutputIndices;

        protected Node(FunctionalTensor[] inputs, DataType[] outputDataTypes)
        {
            m_Inputs = new FunctionalTensor[inputs.Length];
            for (var i = 0; i < m_Inputs.Length; i++)
                m_Inputs[i] = inputs[i] == null ? null : new FunctionalTensor(inputs[i].dataType, inputs[i].source, inputs[i].outputIndex);
            m_OutputDataTypes = new DataType[outputDataTypes.Length];
            for (var i = 0; i < m_OutputDataTypes.Length; i++)
                m_OutputDataTypes[i] = outputDataTypes[i];
            m_OutputIndices = new int[outputDataTypes.Length];
        }

        public abstract void AddToModel(Model model, ref int index);
    }

    class InputNode : Node
    {
        int m_Index;
        DataType m_DataType;
        DynamicTensorShape m_Shape;

        public InputNode(int index, DataType dataType, DynamicTensorShape shape)
            : base(Array.Empty<FunctionalTensor>(), new[] { dataType })
        {
            m_Index = index;
            m_DataType = dataType;
            m_Shape = shape;
        }

        public override void AddToModel(Model model, ref int index)
        {
            var inputName = "input_" + model.inputs.Count;
            m_OutputIndices[0] = index;
            while (model.inputs.Count <= m_Index)
                model.inputs.Add(default);
            model.inputs[m_Index] = new Model.Input { name = inputName, index = index, dataType = m_DataType, shape = m_Shape };
            index++;
        }
    }

    class OutputNode : Node
    {
        int m_Index;

        public OutputNode(int index, FunctionalTensor input)
            : base(new[] { input }, Array.Empty<DataType>())
        {
            m_Index = index;
        }

        public override void AddToModel(Model model, ref int index)
        {
            var outputName = "output_" + m_Index;
            while (model.outputs.Count <= m_Index)
                model.outputs.Add(default);
            model.outputs[m_Index] = new Model.Output { name = outputName, index = m_Inputs[0].index };
        }
    }

    class ConstantNode : Node
    {
        Constant m_Constant;

        public ConstantNode(Constant constant)
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

    class LayerNode : Node
    {
        Layer m_Layer;

        public LayerNode(FunctionalTensor[] inputs, DataType[] outputDataTypes, Layer layer)
            : base(inputs, outputDataTypes)
        {
            m_Layer = layer;
        }

        public override void AddToModel(Model model, ref int index)
        {
            for (var i = 0; i < m_Inputs.Length; i++)
                m_Layer.inputs[i] = m_Inputs[i] is null ? -1 : m_Inputs[i].index;

            for (var i = 0; i < m_OutputIndices.Length; i++)
            {
                m_OutputIndices[i] = index;
                m_Layer.outputs[i] = index;
                index++;
            }

            model.layers.Add(m_Layer);
        }

        public FunctionalTensor[] CreateOutputs()
        {
            var outputs = new FunctionalTensor[m_OutputIndices.Length];
            for (var i = 0; i < outputs.Length; i++)
                outputs[i] = new FunctionalTensor(m_OutputDataTypes[i], this, i);
            return outputs;
        }
    }
}
