using System;

namespace Unity.Sentis
{
    abstract class FunctionalNode
    {
        protected FunctionalTensor[] m_Inputs;
        protected DataType[] m_OutputDataTypes;
        protected string[] m_OutputNames;

        public FunctionalTensor[] Inputs => m_Inputs;
        public string[] OutputNames => m_OutputNames;

        protected FunctionalNode(FunctionalTensor[] inputs, DataType[] outputDataTypes)
        {
            m_Inputs = new FunctionalTensor[inputs.Length];
            for (var i = 0; i < m_Inputs.Length; i++)
                m_Inputs[i] = inputs[i] == null ? null : new FunctionalTensor(inputs[i].DataType, inputs[i].Source, inputs[i].OutputIndex);
            m_OutputDataTypes = new DataType[outputDataTypes.Length];
            for (var i = 0; i < m_OutputDataTypes.Length; i++)
                m_OutputDataTypes[i] = outputDataTypes[i];
            m_OutputNames = new string[outputDataTypes.Length];
        }

        public FunctionalTensor[] CreateOutputs()
        {
            var outputs = new FunctionalTensor[m_OutputNames.Length];
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
            var indexString = index.ToString();
            index++;
            var inputName = "input_" + model.inputs.Count;
            m_OutputNames[0] = indexString;
            model.AddInput(inputName, indexString, m_Input.dataType, m_Input.shape);
        }
    }

    class FunctionalOutput : FunctionalNode
    {
        public FunctionalOutput(FunctionalTensor input)
            : base(new[] { input }, Array.Empty<DataType>()) { }

        public override void AddToModel(Model model, ref int index)
        {
            var indexString = m_Inputs[0].Name;
            var outputName = "output_" + model.outputs.Count;
            model.AddOutput(outputName, indexString);
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
            var tensorIndex = index.ToString();
            index++;
            m_Constant.index = tensorIndex;
            m_OutputNames[0] = tensorIndex;
            model.constants.Add(m_Constant);
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
                m_Layer.inputs[i] = m_Inputs[i] is null ? string.Empty : m_Inputs[i].Name;

            for (var i = 0; i < m_OutputNames.Length; i++)
            {
                var indexString = index.ToString();
                index++;
                m_OutputNames[i] = indexString;
                if (i == 0)
                    m_Layer.index = indexString;
                if (m_Layer.outputs != null)
                    m_Layer.outputs[i] = indexString;
            }

            model.layers.Add(m_Layer);
        }
    }
}
