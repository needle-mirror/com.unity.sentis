using System.Linq;
using Unity.Sentis.Layers;
using System.Collections.Generic;
using System;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class ContractSubExpressionPass : IModelPass
    {
        Dictionary<int, Constant> modelConstants = new Dictionary<int, Constant>();
        Dictionary<int, Layer> indexToLayer = new Dictionary<int, Layer>();
        Dictionary<int, int> indexToLayerIndex = new Dictionary<int, int>();
        Dictionary<int, List<Layer>> downstreamLayers = new Dictionary<int, List<Layer>>();
        List<Layer> layersInPattern = new List<Layer>();
        List<int> inputLayers = new List<int>();
        List<Constant> inputConstants = new List<Constant>();

        // Automatic Chain rule CAS
        // we construct a graph of successive operations by chaining operators and function calls
        //  (x + 2) / 3 + x
        // that creates a graph of INode storing each operations
        // we can call .Validate on that object and it will recursively walk the graph and check validity if layer inputs match the subgraph
        // leaf node:
        //  * constant, match if constant exists and has the expected value
        //  * input, always match
        // tree node:
        //  * check layer type matches INode type
        //  * returns true if all inputs are valid
        private abstract class INode
        {
            public static INode operator *(INode a, INode b)
            {
                return new LayerNode<Mul>(a, b);
            }
            public static INode operator *(INode a, float b)
            {
                return new LayerNode<Mul>(a, b);
            }
            public static INode operator *(float a, INode b)
            {
                return new LayerNode<Mul>(a, b);
            }
            public static INode operator +(INode a, INode b)
            {
                return new LayerNode<Add>(a, b);
            }
            public static INode operator +(INode a, float b)
            {
                return new LayerNode<Add>(a, b);
            }
            public static INode operator -(INode a, INode b)
            {
                return new LayerNode<Sub>(a, b);
            }
            public static INode operator /(INode a, INode b)
            {
                return new LayerNode<Div>(a, b);
            }
            public static INode operator /(INode a, float b)
            {
                return new LayerNode<Div>(a, b);
            }
            public static INode operator /(float a, INode b)
            {
                return new LayerNode<Div>(a, b);
            }
            public static INode Erf(INode a)
            {
                return new LayerNode<Erf>(a);
            }
            public static INode Tanh(INode a)
            {
                return new LayerNode<Tanh>(a);
            }
            public static INode Sigmoid(INode a)
            {
                return new LayerNode<Sigmoid>(a);
            }
            public static INode Softmax(INode a, int b)
            {
                return new LayerNode<Softmax>(a, b);
            }
            public static INode Transpose(INode a, int[] b)
            {
                return new LayerNode<Transpose>(a, b);
            }
            public static INode Reshape(INode a, INode b)
            {
                return new LayerNode<Reshape>(a, b);
            }
            public static INode MatMul(INode a, INode b)
            {
                return new LayerNode<MatMul>(a, b);
            }
            public static INode MatMul2D(INode a, INode b)
            {
                return new LayerNode<MatMul2D>(a, b);
            }
            public static INode Pow(INode a, float b)
            {
                return new LayerNode<Pow>(a, b);
            }
            public static INode Sqrt(INode a)
            {
                return new LayerNode<Sqrt>(a);
            }
            public static INode ReduceMean(INode a, int b)
            {
                return new LayerNode<ReduceMean>(a, b);
            }
        }

        private class InputNode : INode
        {
        }

        abstract class IConstantNode : INode
        {
            public abstract bool Validate(Constant constant);
        }

        private class ConstantFloatTensor : IConstantNode
        {
            public override bool Validate(Constant constant)
            {
                if (constant.dataType != DataType.Int)
                    return false;

                return true;
            }
        }

        class ScalarInt : IConstantNode
        {
            float m_Value;
            public ScalarInt(int v)
            {
                m_Value = v;
            }

            public override bool Validate(Constant constant)
            {
                if (constant.dataType != DataType.Int)
                    return false;

                return constant.shape.length == 1 && constant.shape.rank <= 1 && constant.weights.Get<int>(0) == m_Value;
            }
        }

        class ScalarFloat : IConstantNode
        {
            float m_Value;
            public ScalarFloat(float v)
            {
                m_Value = v;
            }

            public override bool Validate(Constant constant)
            {
                if (constant.dataType != DataType.Float)
                    return false;

                return constant.shape.length == 1 && constant.shape.rank <= 1 && constant.weights.Get<float>(0) == m_Value;
            }
        }

        class VariableScalarFloat : IConstantNode
        {
            public override bool Validate(Constant constant)
            {
                if (constant.dataType != DataType.Float)
                    return false;

                return constant.shape.length == 1 && constant.shape.rank <= 1;
            }
        }

        class VariableScalarInt : IConstantNode
        {
            public override bool Validate(Constant constant)
            {
                if (constant.dataType != DataType.Int)
                    return false;

                return constant.shape.length == 1 && constant.shape.rank <= 1;
            }
        }

        class VectorInt : IConstantNode
        {
            int[] m_Value;
            public VectorInt(int[] v)
            {
                m_Value = v;
            }

            public override bool Validate(Constant constant)
            {
                if (constant.dataType != DataType.Int)
                    return false;

                if (constant.shape.length != m_Value.Length)
                    return false;

                for (int i = 0; i < m_Value.Length; i++)
                {
                    if (constant.weights.Get<int>(i) != m_Value[i])
                        return false;
                }

                return true;
            }
        }

        abstract class ILayerNode : INode
        {
            public INode[] inputs;
            public abstract bool Validate(Layer layer);
        }

        class LayerNode<T> : ILayerNode where T : Layer
        {
            public LayerNode(INode i0)
            {
                inputs = new[] { i0 };
            }
            public LayerNode(INode i0, INode i1)
            {
                inputs = new[] { i0, i1 };
            }
            public LayerNode(INode i0, float i1)
            {
                inputs = new[] { i0, new ScalarFloat(i1) };
            }
            public LayerNode(float i0, INode i1)
            {
                inputs = new[] { new ScalarFloat(i0), i1 };
            }
            public LayerNode(INode i0, int i1)
            {
                inputs = new[] { i0, new ScalarInt(i1) };
            }
            public LayerNode(INode i0, int[] i1)
            {
                inputs = new[] { i0, new VectorInt(i1) };
            }

            public override bool Validate(Layer layer)
            {
                return layer is T;
            }
        }

        // remapping rules:
        // key: expression to test against
        // value: layer to spawn, layersInPattern is all the layers that match the expression
        Dictionary<Func<InputNode, INode>, Func<Layer, List<int>, List<Constant>, Layer>> remappingRules = new Dictionary<Func<InputNode, INode>, Func<Layer, List<int>, List<Constant>, Layer>>()
        {
            { x => INode.Pow(x, -1.0f),                                      (y, iLayers, iConstants) => new Reciprocal(y.outputs[0], iLayers[0]) },
            { x => INode.Pow(x, 0.5f),                                       (y, iLayers, iConstants) => new Sqrt(y.outputs[0], iLayers[0]) },
            { x => INode.Pow(x, 1.0f),                                       (y, iLayers, iConstants) => new Identity(y.outputs[0], iLayers[0]) },
            { x => INode.Pow(x, 2.0f),                                       (y, iLayers, iConstants) => new Square(y.outputs[0], iLayers[0]) },
            { x => (x * INode.Sigmoid(x)),                                   (y, iLayers, iConstants) => new Swish(y.outputs[0], iLayers[0]) },
            { x => (x * (INode.Erf((x / Mathf.Sqrt(2.0f))) + 1.0f)) * 0.5f,  (y, iLayers, iConstants) => new Gelu(y.outputs[0], iLayers[0]) },
            { x => (x * 0.5f) * (INode.Tanh((x + (INode.Pow(x, 3.0f) * 0.044714998453855515f)) * 0.7978845834732056f) + 1),  (y, iLayers, iConstants) => new GeluFast(y.outputs[0], iLayers[0]) },
            { x => {
                var mean = INode.ReduceMean(x, -1);
                var y = x - mean;
                var variance = INode.ReduceMean(INode.Pow(y, 2.0f), -1);
                var epsilon = new VariableScalarFloat();
                var v = y / INode.Sqrt(variance + epsilon);
                var scale = new InputNode();
                var bias = new InputNode();
                return v * scale + bias; },
                    (y, iLayers, iConstants) => {
                    float epsilon = iConstants[0].weights.Get<float>(0);
                    return new LayerNormalization(y.outputs[0], iLayers[iLayers.Count - 1], iLayers[1], iLayers[0], epsilon);
                }
            },
            {
                x =>
                {
                    var pow = INode.Pow(x, 2.0f);
                    var reduceMean = INode.ReduceMean(pow, -1);
                    var epsilon = new VariableScalarFloat();
                    var add = reduceMean + epsilon;
                    var sqrt = INode.Sqrt(add);
                    var div = 1.0f / sqrt;
                    var mul = x * div;
                    var scale = new InputNode();
                    return scale * mul;
                },
                (y, iLayers, iConstants) =>
                {
                    float epsilon = iConstants[0].weights.Get<float>(0);
                    return new RMSNormalization(y.outputs[0], iLayers[1], iLayers[2], epsilon);
                }
            },
            { x => x + new VariableScalarFloat(), (y, iLayers, iConstants) => new ScalarMad(y.outputs[0], iLayers[0], 1.0f, iConstants[0].weights.Get<float>(0)) },
            { x => new VariableScalarFloat() + x, (y, iLayers, iConstants) => new ScalarMad(y.outputs[0], iLayers[0], 1.0f, iConstants[0].weights.Get<float>(0)) },
            { x => x - new VariableScalarFloat(), (y, iLayers, iConstants) => new ScalarMad(y.outputs[0], iLayers[0], 1.0f, -iConstants[0].weights.Get<float>(0)) },
            { x => new VariableScalarFloat() - x, (y, iLayers, iConstants) => new ScalarMad(y.outputs[0], iLayers[0], -1.0f, iConstants[0].weights.Get<float>(0)) },
            { x => x * new VariableScalarFloat(), (y, iLayers, iConstants) => new ScalarMad(y.outputs[0], iLayers[0], iConstants[0].weights.Get<float>(0), 0.0f) },
            { x => new VariableScalarFloat() * x, (y, iLayers, iConstants) => new ScalarMad(y.outputs[0], iLayers[0], iConstants[0].weights.Get<float>(0), 0.0f) },
            { x => x / new VariableScalarFloat(), (y, iLayers, iConstants) => new ScalarMad(y.outputs[0], iLayers[0], 1.0f / iConstants[0].weights.Get<float>(0), 0.0f) },
            { x => x + new VariableScalarInt(), (y, iLayers, iConstants) => new ScalarMad(y.outputs[0], iLayers[0], 1, iConstants[0].weights.Get<int>(0)) },
            { x => new VariableScalarInt() + x, (y, iLayers, iConstants) => new ScalarMad(y.outputs[0], iLayers[0], 1, iConstants[0].weights.Get<int>(0)) },
            { x => x - new VariableScalarInt(), (y, iLayers, iConstants) => new ScalarMad(y.outputs[0], iLayers[0], 1, -iConstants[0].weights.Get<int>(0)) },
            { x => new VariableScalarInt() - x, (y, iLayers, iConstants) => new ScalarMad(y.outputs[0], iLayers[0], -1, iConstants[0].weights.Get<int>(0)) },
            { x => x * new VariableScalarInt(), (y, iLayers, iConstants) => new ScalarMad(y.outputs[0], iLayers[0], iConstants[0].weights.Get<int>(0), 0) },
            { x => new VariableScalarInt() * x, (y, iLayers, iConstants) => new ScalarMad(y.outputs[0], iLayers[0], iConstants[0].weights.Get<int>(0), 0) },
        };

        bool Validate(INode root, Layer input)
        {
            Stack<int> layerStack = new Stack<int>();
            Stack<INode> nodeStack = new Stack<INode>();

            nodeStack.Push(root);
            layerStack.Push(input.outputs[0]);
            while (nodeStack.Count != 0)
            {
                INode node = nodeStack.Pop();
                int index = layerStack.Pop();

                if (node is IConstantNode cNode)
                {
                    if (!modelConstants.TryGetValue(index, out Constant constant))
                        return false;

                    if (!cNode.Validate(constant))
                        return false;

                    inputConstants.Add(constant);
                }
                else if (node is ILayerNode lNode)
                {
                    if (!indexToLayer.TryGetValue(index, out Layer layer))
                        return false;

                    if (!lNode.Validate(layer))
                        return false;

                    layersInPattern.Add(layer);
                    for (int i = 0; i < layer.inputs.Length; i++)
                    {
                        layerStack.Push(layer.inputs[i]);
                        nodeStack.Push(lNode.inputs[i]);
                    }
                }
                else if (node is InputNode)
                {
                    inputLayers.Add(index);
                }
            }

            return true;
        }

        // Pattern is said to be fully included if
        // foreach layer in subgraph
        // * all input are in the subGraph or inputs of root layer
        // * all downstream layers are in the subGraph or downstream of root layer
        bool CheckGraphFullInclusion(Layer root, List<Layer> subGraph, out int patternInput)
        {
            patternInput = -1;
            HashSet<int> layersInSubGraph = new HashSet<int>();
            foreach (var layer in subGraph)
            {
                layersInSubGraph.Add(layer.outputs[0]);
                foreach (var input in layer.inputs)
                {
                    layersInSubGraph.Add(input);
                }
                foreach (var downStream in downstreamLayers[layer.outputs[0]])
                {
                    if (downStream != null)
                        layersInSubGraph.Add(downStream.outputs[0]);
                }
            }

            foreach (var layer in layersInPattern)
            {
                layersInSubGraph.Remove(layer.outputs[0]);
                foreach (var input in layer.inputs)
                {
                    if (modelConstants.ContainsKey(input))
                        layersInSubGraph.Remove(input);
                }
            }

            foreach (var downStream in downstreamLayers[root.outputs[0]])
            {
                if (downStream == null)
                    continue;
                if (!layersInSubGraph.Contains(downStream.outputs[0]))
                    return false;
                layersInSubGraph.Remove(downStream.outputs[0]);
            }

            if (layersInSubGraph.Count != 1)
                return false;

            patternInput = layersInSubGraph.ElementAt(0);

            return true;
        }

        public void Run(ref Model model)
        {
            HashSet<int> layersToRemove = new HashSet<int>();

            // build static helpers:
            // - index -> constant
            // - index -> layer index int
            modelConstants = new Dictionary<int, Constant>();
            foreach (var c in model.constants)
                modelConstants.Add(c.index, c);

            indexToLayer = new Dictionary<int, Layer>();
            indexToLayerIndex = new Dictionary<int, int>();
            downstreamLayers = new Dictionary<int, List<Layer>>();
            var outputs = new HashSet<int>();
            foreach (var o in model.outputs)
                outputs.Add(o.index);
            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layer layer = model.layers[l];
                indexToLayer.Add(layer.outputs[0], layer);
                indexToLayerIndex.Add(layer.outputs[0], l);

                foreach (var input in layer.inputs)
                {
                    if (downstreamLayers.ContainsKey(input))
                        downstreamLayers[input].Add(layer);
                    else
                        downstreamLayers[input] = new List<Layer> { layer };
                }

                if (outputs.Contains(layer.outputs[0]))
                {
                    downstreamLayers[layer.outputs[0]] = new List<Layer> { null };
                }
            }

            var x = new InputNode();
            layersInPattern = new List<Layer>();
            inputLayers = new List<int>();
            inputConstants = new List<Constant>();

            // Algorithm:
            // foreach layers
            //  foreach pattern
            //      check if pattern is matched walking up model inputs
            //      if matched, check if subgraph is fully enclosed
            //      insert new merged layer
            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layer layer = model.layers[l];

                foreach (var item in remappingRules)
                {
                    layersInPattern.Clear();
                    inputLayers.Clear();
                    inputConstants.Clear();

                    var pattern = item.Key(x);
                    if (!Validate(pattern, layer))
                        continue;

                    if (!CheckGraphFullInclusion(layer, layersInPattern, out int input))
                        continue;

                    var remapping = item.Value;
                    var remapLayer = remapping(layer, inputLayers, inputConstants);

                    bool unconnectedOutputs = false;
                    foreach (var layerToDelete in layersInPattern)
                    {
                        unconnectedOutputs |= (remapLayer.outputs[0] != layerToDelete.outputs[0]) && outputs.Contains(layerToDelete.outputs[0]);
                    }
                    if (unconnectedOutputs)
                        break;


                    model.layers[indexToLayerIndex[remapLayer.outputs[0]]] = remapLayer;

                    foreach (var layerToDelete in layersInPattern)
                    {
                        if (remapLayer.outputs[0] != layerToDelete.outputs[0])
                            layersToRemove.Add(layerToDelete.outputs[0]);
                    }

                    break;
                }
            }

            model.layers.RemoveAll(l => layersToRemove.Contains(l.outputs[0]));
        }
    }
}
