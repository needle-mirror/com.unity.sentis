using System;
using System.Collections.Generic;
using Unity.Sentis.Compiler.Passes.Optimization;
using UnityEngine.Assertions;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents the static functional methods for model building and compilation.
    /// </summary>
    public static partial class Functional
    {
        enum NodeProgress
        {
            NotVisited,
            InProgress,
            Done
        }

        /// <summary>
        /// Compiles a forward method into a model.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDefs">The input definitions to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor[], FunctionalTensor[]> forward, InputDef[] inputDefs)
        {
            // setup input expressions from inputDefs, to feed to forward method
            var inputExpressions = new FunctionalTensor[inputDefs.Length];
            for (var i = 0; i < inputExpressions.Length; i++)
            {
                var modelInput = new Model.Input
                {
                    shape = inputDefs[i].Shape,
                    dataType = inputDefs[i].DataType
                };
                inputExpressions[i] = FunctionalTensor.FromInput(modelInput);
            }

            // create empty model
            var model = new Model();

            // create for post order traversal algorithm
            var nodeStack = new Stack<FunctionalNode>(); // stack of nodes to inspect and then process
            var nodeProgress = new Dictionary<FunctionalNode, NodeProgress>(); // nodes which have been processed and added to the model

            var tensorIndex = 0;

            // iterate inputs to ensure they are in the right order on the model
            foreach (var inputExpression in inputExpressions)
            {
                var node = inputExpression.Source;
                node.AddToModel(model, ref tensorIndex);
                nodeProgress[node] = NodeProgress.Done;
            }

            // calculate output expressions
            var outputExpressions = forward(inputExpressions);

            // queue nodes for the output expressions
            for (var i = outputExpressions.Length - 1; i >= 0; i--)
            {
                if (outputExpressions[i].Source is not FunctionalLayer)
                    outputExpressions[i] = outputExpressions[i].Clone();
                var node = new FunctionalOutput(outputExpressions[i]);
                nodeStack.Push(node);
            }

            // push dependency nodes ahead of current node in stack
            // only process node once dependencies have been processed
            while (nodeStack.TryPeek(out var n))
            {
                var nProgress = nodeProgress.GetValueOrDefault(n, NodeProgress.NotVisited);
                if (nProgress == NodeProgress.InProgress)
                {
                    // add node to model
                    Logger.AssertIsTrue(n is not FunctionalInput, "Input expression from incorrect source.");
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
                    var m = n.Inputs[i].Source;
                    var mProgress = nodeProgress.GetValueOrDefault(m, NodeProgress.NotVisited);
                    if (mProgress == NodeProgress.NotVisited)
                        nodeStack.Push(m);
                    else
                        Assert.IsTrue(mProgress != NodeProgress.InProgress, "Model graph has cycle");
                }
            }

            ModelOptimizer.OptimizeModel(ref model);
            ModelOptimizer.RunCPUFallbackPass(ref model);

            return model;
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with no inputs and a single output.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor> forward)
        {
            return Compile(_ =>
            {
                var res = forward();
                return new[] { res };
            }, Array.Empty<InputDef>());
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with a single input and a single output.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDef">The input definition to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, FunctionalTensor> forward, InputDef inputDef)
        {
            return Compile(x =>
            {
                var res = forward(x[0]);
                return new[] { res };
            }, new[] { inputDef });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with two inputs and a single output.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDefs">The input definitions to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, FunctionalTensor, FunctionalTensor> forward, (InputDef, InputDef) inputDefs)
        {
            return Compile(x =>
            {
                var res = forward(x[0], x[1]);
                return new[] { res };
            }, new[] { inputDefs.Item1, inputDefs.Item2 });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with three inputs and a single output.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDefs">The input definitions to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, FunctionalTensor, FunctionalTensor, FunctionalTensor> forward, (InputDef, InputDef, InputDef) inputDefs)
        {
            return Compile(x =>
            {
                var res = forward(x[0], x[1], x[2]);
                return new[] { res };
            }, new[] { inputDefs.Item1, inputDefs.Item2, inputDefs.Item3 });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with four inputs and a single output.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDefs">The input definitions to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, FunctionalTensor, FunctionalTensor, FunctionalTensor, FunctionalTensor> forward, (InputDef, InputDef, InputDef, InputDef) inputDefs)
        {
            return Compile(x =>
            {
                var res = forward(x[0], x[1], x[2], x[3]);
                return new[] { res };
            }, new[] { inputDefs.Item1, inputDefs.Item2, inputDefs.Item3, inputDefs.Item4 });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with no inputs and two outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<(FunctionalTensor, FunctionalTensor)> forward)
        {
            return Compile(_ =>
            {
                var res = forward();
                return new[] { res.Item1, res.Item2 };
            }, Array.Empty<InputDef>());
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with a single input and two outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDef">The input definition to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, (FunctionalTensor, FunctionalTensor)> forward, InputDef inputDef)
        {
            return Compile(x =>
            {
                var res = forward(x[0]);
                return new[] { res.Item1, res.Item2 };
            }, new[] { inputDef });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with two inputs and two outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDefs">The input definitions to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, FunctionalTensor, (FunctionalTensor, FunctionalTensor)> forward, (InputDef, InputDef) inputDefs)
        {
            return Compile(x =>
            {
                var res = forward(x[0], x[1]);
                return new[] { res.Item1, res.Item2 };
            }, new[] { inputDefs.Item1, inputDefs.Item2 });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with three inputs and two outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDefs">The input definitions to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, FunctionalTensor, FunctionalTensor, (FunctionalTensor, FunctionalTensor)> forward, (InputDef, InputDef, InputDef) inputDefs)
        {
            return Compile(x =>
            {
                var res = forward(x[0], x[1], x[2]);
                return new[] { res.Item1, res.Item2 };
            }, new[] { inputDefs.Item1, inputDefs.Item2, inputDefs.Item3 });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with four inputs and two outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDefs">The input definitions to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, FunctionalTensor, FunctionalTensor, FunctionalTensor, (FunctionalTensor, FunctionalTensor)> forward, (InputDef, InputDef, InputDef, InputDef) inputDefs)
        {
            return Compile(x =>
            {
                var res = forward(x[0], x[1], x[2], x[3]);
                return new[] { res.Item1, res.Item2 };
            }, new[] { inputDefs.Item1, inputDefs.Item2, inputDefs.Item3, inputDefs.Item4 });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with no inputs and three outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<(FunctionalTensor, FunctionalTensor, FunctionalTensor)> forward)
        {
            return Compile(_ =>
            {
                var res = forward();
                return new[] { res.Item1, res.Item2, res.Item3 };
            }, Array.Empty<InputDef>());
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with a single input and three outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDef">The input definition to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, (FunctionalTensor, FunctionalTensor, FunctionalTensor)> forward, InputDef inputDef)
        {
            return Compile(x =>
            {
                var res = forward(x[0]);
                return new[] { res.Item1, res.Item2, res.Item3 };
            }, new[] { inputDef });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with two inputs and three outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDefs">The input definitions to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, FunctionalTensor, (FunctionalTensor, FunctionalTensor, FunctionalTensor)> forward, (InputDef, InputDef) inputDefs)
        {
            return Compile(x =>
            {
                var res = forward(x[0], x[1]);
                return new[] { res.Item1, res.Item2, res.Item3 };
            }, new[] { inputDefs.Item1, inputDefs.Item2 });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with three inputs and three outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDefs">The input definitions to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, FunctionalTensor, FunctionalTensor, (FunctionalTensor, FunctionalTensor, FunctionalTensor)> forward, (InputDef, InputDef, InputDef) inputDefs)
        {
            return Compile(x =>
            {
                var res = forward(x[0], x[1], x[2]);
                return new[] { res.Item1, res.Item2, res.Item3 };
            }, new[] { inputDefs.Item1, inputDefs.Item2, inputDefs.Item3 });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with four inputs and three outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDefs">The input definitions to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, FunctionalTensor, FunctionalTensor, FunctionalTensor, (FunctionalTensor, FunctionalTensor, FunctionalTensor)> forward, (InputDef, InputDef, InputDef, InputDef) inputDefs)
        {
            return Compile(x =>
            {
                var res = forward(x[0], x[1], x[2], x[3]);
                return new[] { res.Item1, res.Item2, res.Item3 };
            }, new[] { inputDefs.Item1, inputDefs.Item2, inputDefs.Item3, inputDefs.Item4 });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with no inputs and four outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<(FunctionalTensor, FunctionalTensor, FunctionalTensor, FunctionalTensor)> forward)
        {
            return Compile(_ =>
            {
                var res = forward();
                return new[] { res.Item1, res.Item2, res.Item3, res.Item4 };
            }, Array.Empty<InputDef>());
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with a single input and four outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDef">The input definition to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, (FunctionalTensor, FunctionalTensor, FunctionalTensor, FunctionalTensor)> forward, InputDef inputDef)
        {
            return Compile(x =>
            {
                var res = forward(x[0]);
                return new[] { res.Item1, res.Item2, res.Item3, res.Item4 };
            }, new[] { inputDef });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with two inputs and four outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDefs">The input definitions to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, FunctionalTensor, (FunctionalTensor, FunctionalTensor, FunctionalTensor, FunctionalTensor)> forward, (InputDef, InputDef) inputDefs)
        {
            return Compile(x =>
            {
                var res = forward(x[0], x[1]);
                return new[] { res.Item1, res.Item2, res.Item3, res.Item4 };
            }, new[] { inputDefs.Item1, inputDefs.Item2 });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with three inputs and four outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDefs">The input definitions to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, FunctionalTensor, FunctionalTensor, (FunctionalTensor, FunctionalTensor, FunctionalTensor, FunctionalTensor)> forward, (InputDef, InputDef, InputDef) inputDefs)
        {
            return Compile(x =>
            {
                var res = forward(x[0], x[1], x[2]);
                return new[] { res.Item1, res.Item2, res.Item3, res.Item4 };
            }, new[] { inputDefs.Item1, inputDefs.Item2, inputDefs.Item3 });
        }

        /// <summary>
        /// Compiles and returns a model from a forward method with four inputs and four outputs.
        /// </summary>
        /// <param name="forward">The forward method to compile.</param>
        /// <param name="inputDefs">The input definitions to compile.</param>
        /// <returns>The compiled model.</returns>
        public static Model Compile(this Func<FunctionalTensor, FunctionalTensor, FunctionalTensor, FunctionalTensor, (FunctionalTensor, FunctionalTensor, FunctionalTensor, FunctionalTensor)> forward, (InputDef, InputDef, InputDef, InputDef) inputDefs)
        {
            return Compile(x =>
            {
                var res = forward(x[0], x[1], x[2], x[3]);
                return new[] { res.Item1, res.Item2, res.Item3, res.Item4 };
            }, new[] { inputDefs.Item1, inputDefs.Item2, inputDefs.Item3, inputDefs.Item4 });
        }
    }
}
