using System.Collections.Generic;
using Unity.Sentis.Compiler.Analyser;
using Unity.Sentis;

namespace Unity.Sentis.Compiler.Validation
{
    struct ValidateShapeInference : IValidationPass
    {
        public void Run(Model model)
        {
            // validate shape inference / largest tensorshape to catch invalid model at import time
            var ctx = new ShapeInferenceContext();
            ShapeInferenceAnalysis.InferModelShapes(model, ctx);

            var isInputsFullyKnown = true;
            var inputShapes = new Dictionary<string, TensorShape>();
            foreach (var input in model.inputs)
            {
                if (input.shape.IsFullyKnown())
                    inputShapes[input.name] = input.shape.ToTensorShape();
                else
                {
                    isInputsFullyKnown = false;
                    break;
                }
            }
            if (isInputsFullyKnown)
                MemoryFootprintAnalysis.FindLargestNecessaryTensorShape(model, inputShapes);

            MemoryFootprintAnalysis.FindLayersThatRequireStorage(model);
        }
    }
}
