using System.Collections.Generic;
using UnityEngine.Assertions;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents a model graph using the functional API.
    ///
    /// Input functional tensors can be added to the graph, then manipulated using the functional API methods.
    ///
    /// The functional graph can be compiled to return an optimized Sentis runtime model.
    /// </summary>
    public class FunctionalGraph
    {
        List<InputNode> m_Inputs = new();

        /// <summary>
        /// Append an input to the graph with an input def.
        /// </summary>
        /// <param name="dataType">The data type of the input.</param>
        /// <param name="shape">The shape of the input.</param>
        /// <returns>The functional tensor input.</returns>
        public FunctionalTensor AddInput(DataType dataType, DynamicTensorShape shape)
        {
            var index = m_Inputs.Count;
            var input = new InputNode(index, dataType, shape);
            m_Inputs.Add(input);
            if (shape.IsStatic())
                return new FunctionalTensor(dataType, shape.ToTensorShape(), input, 0);
            else
                return new FunctionalTensor(dataType, input, 0);
        }

        /// <summary>
        /// Append an input to the graph with an input def.
        /// </summary>
        /// <param name="dataType">The data type of the input.</param>
        /// <param name="shape">The shape of the input.</param>
        /// <returns>The functional tensor input.</returns>
        public FunctionalTensor AddInput(DataType dataType, TensorShape shape)
        {
            var index = m_Inputs.Count;
            var input = new InputNode(index, dataType, new DynamicTensorShape(shape));
            m_Inputs.Add(input);
            return new FunctionalTensor(dataType, shape, input, 0);
        }

        /// <summary>
        /// Append an input to the graph with a type T and dynamic tensor shape.
        /// </summary>
        /// <param name="shape">The shape of the input.</param>
        /// <typeparam name="T">The data type of the input.</typeparam>
        /// <returns>The functional tensor input.</returns>
        public FunctionalTensor AddInput<T>(DynamicTensorShape shape) where T : unmanaged
        {
            return AddInput(AllocatorUtils.ToDataType<T>(), shape);
        }

        /// <summary>
        /// Append an input to the graph with a type T and static tensor shape.
        /// </summary>
        /// <param name="shape">The shape of the input.</param>
        /// <typeparam name="T">The data type of the input.</typeparam>
        /// <returns>The functional tensor input.</returns>
        public FunctionalTensor AddInput<T>(TensorShape shape) where T : unmanaged
        {
            return AddInput(AllocatorUtils.ToDataType<T>(), shape);
        }

        /// <summary>
        /// Append an input to the graph matching a model input.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="index">The input index of the input in the provided model.</param>
        /// <returns>The functional tensor input.</returns>
        public FunctionalTensor AddInput(Model model, int index)
        {
            var modelInput = model.inputs[index];
            return AddInput(modelInput.dataType, modelInput.shape);
        }

        /// <summary>
        /// Append inputs to the graph matching all of a model's input.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns>The functional tensor input array.</returns>
        public FunctionalTensor[] AddInputs(Model model)
        {
            var inputTensors = new FunctionalTensor[model.inputs.Count];
            for (var i = 0; i < inputTensors.Length; i++)
                inputTensors[i] = AddInput(model, i);
            return inputTensors;
        }

        /// <summary>
        /// Compile and return an optimized runtime model with given outputs.
        /// </summary>
        /// <param name="outputs">The outputs.</param>
        /// <returns>The compiled runtime model.</returns>
        public Model Compile(params FunctionalTensor[] outputs)
        {
            var model = Build(outputs);
            ModelOptimizer.OptimizeModel(ref model);
            return model;
        }

        /// <summary>
        /// Build an unoptimized model.
        /// </summary>
        internal Model Build(params FunctionalTensor[] outputs)
        {
            List<OutputNode> outputNodes = new();
            for (var i = 0; i < outputs.Length; i++)
            {
                var output = outputs[i];
                if (output.source is not LayerNode)
                    output = output.Clone();

                outputNodes.Add(new OutputNode(i, output));
            }

            return Build(outputNodes);
        }

        enum NodeProgress
        {
            NotVisited,
            InProgress,
            Done
        }

        Model Build(List<OutputNode> outputs)
        {
            // create empty model
            var model = new Model();

            // create for post order traversal algorithm
            var nodeStack = new Stack<Node>(); // stack of nodes to inspect and then process
            var nodeProgress = new Dictionary<Node, NodeProgress>(); // nodes which have been processed and added to the model

            var tensorIndex = 0;

            // iterate inputs to ensure they are in the right order on the model
            foreach (var input in m_Inputs)
            {
                input.AddToModel(model, ref tensorIndex);
                nodeProgress[input] = NodeProgress.Done;
            }

            // queue nodes for the output expressions in reverse order
            for (var i = outputs.Count - 1; i >= 0; i--)
                nodeStack.Push(outputs[i]);

            // push dependency nodes ahead of current node in stack
            // only process node once dependencies have been processed
            while (nodeStack.TryPeek(out var n))
            {
                var nProgress = nodeProgress.GetValueOrDefault(n, NodeProgress.NotVisited);
                if (nProgress == NodeProgress.InProgress)
                {
                    // add node to model
                    Logger.AssertIsTrue(n is not InputNode, "Input expression from incorrect source.");
                    n.AddToModel(model, ref tensorIndex);
                    nodeProgress[n] = NodeProgress.Done;
                    nodeStack.Pop();
                    continue;
                }

                if (nProgress == NodeProgress.Done)
                {
                    // node already added to model
                    nodeStack.Pop();
                    continue;
                }

                // node is not visited, iterate descendants
                nodeProgress[n] = NodeProgress.InProgress;

                for (var i = n.Inputs.Length - 1; i >= 0; i--)
                {
                    if (n.Inputs[i] is null)
                        continue;
                    var m = n.Inputs[i].source;
                    var mProgress = nodeProgress.GetValueOrDefault(m, NodeProgress.NotVisited);
                    if (mProgress == NodeProgress.NotVisited)
                        nodeStack.Push(m);
                    else
                        Assert.IsTrue(mProgress != NodeProgress.InProgress, "Model graph has cycle");
                }
            }

            return model;
        }
    }
}
