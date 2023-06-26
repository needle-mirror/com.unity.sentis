using System;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Profiling;

namespace Unity.Sentis.Compiler.Analyser
{
    struct ShapeInferenceAnalysis
    {
        /// <summary>
        /// Add the constant, input and inferred layer shapes to the given shape inference context
        /// </summary>
        public static void InferModelShapes(Model model, ShapeInferenceContext ctx)
        {
            Profiler.BeginSample("Sentis.Compiler.Analyser.ShapeInferenceAnalysis.InferModelShapes");

            InferModelConstantShapes(model, ctx);
            InferModelInputShapes(model, ctx);
            InferModelLayerShapes(model, ctx);

            Profiler.EndSample();
        }

        /// <summary>
        /// Get the model input symbolic tensor shapes and add to the given shape inference context
        /// </summary>
        public static void InferModelInputShapes(Model model, ShapeInferenceContext ctx)
        {
            Profiler.BeginSample("Sentis.Compiler.Analyser.ShapeInferenceAnalysis.InferModelInputShapes");

            foreach (var input in model.inputs)
            {
                ctx.AddShape(input.name, new SymbolicTensorShape(input.shape));
            }

            Profiler.EndSample();
        }

        /// <summary>
        /// Get the model constant symbolic tensor shapes and add to the given shape inference context
        /// </summary>
        public static void InferModelConstantShapes(Model model, ShapeInferenceContext ctx)
        {
            Profiler.BeginSample("Sentis.Compiler.Analyser.ShapeInferenceAnalysis.InferModelConstantShapes");

            foreach (var constant in model.constants)
            {
                ctx.AddShape(constant.name, new SymbolicTensorShape(constant.shape));
                ctx.AddKnownTensor(constant.name, constant.DataSetToTensor());
            }

            Profiler.EndSample();
        }

        /// <summary>
        /// Infer the model layer symbolic tensor shapes (inputs and constants should already be inferred)
        /// and add to the given shape inference context
        /// </summary>
        public static void InferModelLayerShapes(Model model, ShapeInferenceContext ctx)
        {
            Profiler.BeginSample("Sentis.Compiler.Analyser.ShapeInferenceAnalysis.InferModelLayerShapes");

            foreach (var layer in model.layers)
            {
                var layerInputShapes = new SymbolicTensorShape[layer.inputs.Length];
                for (var i = 0; i < layer.inputs.Length; i++)
                {
                    layerInputShapes[i] = ctx.GetSymbolicTensorShape(layer.inputs[i]);
                }

                var layerOutputShape = layer.InferOutputShape(layerInputShapes, ctx);
                ctx.AddShape(layer.name, layerOutputShape);
            }

            Profiler.EndSample();
        }
    }
}

