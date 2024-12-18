using System.Runtime.CompilerServices;
using UnityEngine.Assertions;

[assembly: InternalsVisibleTo("Unity.Sentis.Tests")]

namespace Unity.Sentis
{
    /// <summary>
    /// Represents the static functional methods for model building and compilation.
    /// </summary>
    public static partial class Functional
    {
        internal static FunctionalTensor[] FromLayer(Layer layer, DataType[] dataTypes, params FunctionalTensor[] inputs)
        {
            Assert.AreEqual(layer.inputs.Length, inputs.Length);
            var layerNode = new LayerNode(inputs, dataTypes, layer);
            return layerNode.CreateOutputs();
        }

        internal static FunctionalTensor FromLayer(Layer layer, DataType dataType, params FunctionalTensor[] inputs)
        {
            return FromLayer(layer, new[] { dataType }, inputs)[0];
        }
    }
}
