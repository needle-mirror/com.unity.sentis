using System;
using Unity.Sentis;
using Unity.Sentis.Layers;
using Unity.Sentis.ONNX;
using UnityEngine;

// Implement an importer and layer for the Binarizer operator from ai.onnx.ml.
// https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.Binarizer

// Define onnx op importer, the name here must match the name of the operator, in this case 'Binarizer'.
[OpImport("Binarizer")]
public class BinarizerOp : IOpImporter
{
    public void Import(Model model, OperatorNode node)
    {
        // Setup default value of 0 if attribute not present.
        var threshold = 0f;
        // Check for attribute and retrieve it from OperatorNode.
        if (node.HasAttribute("threshold"))
            threshold = node.GetFloatAttribute("threshold");
        // Add custom layer to model with node name (output), node inputs and attributes.
        model.AddLayer(new Binarizer(node.Name, node.Inputs[0], threshold));

        // Note: we could alternatively implement this operator entirely with existing Layers as follows
        // model.AddConstant(new Constant("threshold", new TensorFloat(new TensorShape(), new[] { threshold })));
        // model.AddLayer(new Greater(node.Name, node.Inputs[0], "threshold"));
    }
}

// Define custom Binarizer layer, this must be serializable so that the model is serializable.
[Serializable]
public class Binarizer : CustomLayer
{
    // Save attributes as fields on the layer. These must be serializable types.
    public float threshold;

    // Constructor is called from the importer with the output and input names and attributes.
    public Binarizer(string name, string input, float threshold)
    {
        this.name = name;
        inputs = new[] { input };
        this.threshold = threshold;
    }

    // This is where the data type of the output tensors is inferred from the data type of the input tensors at import time.
    public override DataType[] InferOutputDataTypes(DataType[] inputDataTypes)
    {
        return new[] { DataType.Float };
    }

    // This is where the execution of the layer happens and an output tensor is created from an input tensor.
    public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
    {
        // This layer only takes one input, this is the first element of the inputs array.
        var X = inputTensors[0] as TensorFloat;

        // Create the output tensor of the correct data type and shape.
        var O = ctx.backend.NewOutputTensorFloat(X.shape);
        // If the output tensor has length 0 then we can return it without doing any more calculations.
        if (O.shape.HasZeroDims())
            return O;
        // Create a scalar tensor from the threshold value to use in the Greater op.
        // Make sure intermediate values are disposed to avoid memory leaks.
        using TensorFloat thresholdTensor = ctx.backend.NewTempTensorFloat(new TensorShape());
        ctx.backend.MemSet(thresholdTensor, threshold);
        // When possible use the provided backend in the context to avoid unnecessary conversions.
        // e.g. uploading and downloading the data from the GPU
        // See the samples on CSharpJobs and ComputeBuffers for writing optimized code to run on each backend.
        using TensorInt Oint = ctx.backend.NewTempTensorInt(X.shape);
        ctx.backend.Greater(X, thresholdTensor, Oint);
        // The Greater op outputs a TensorInt so we must cast this to a TensorFloat using the Cast method.
        ctx.backend.Cast(Oint, O);

        // Return the filled output tensor, if a layer has multiple outputs the others should be stored in
        // the ExecutionContext e.g. ctx.vars.Store(outputs[1], o1);
        return O;
    }
}
