namespace Unity.Sentis
{
    /// <summary>
    /// Represents extension functions for the Sentis functional API.
    /// </summary>
    public static class FunctionalExtensions
    {
        internal static Model DeepCopy(this Model model)
        {
            ModelWriter.SaveModel(model, out var modelDescriptionBytes, out var modelWeightsBytes);
            return ModelLoader.LoadModel(modelDescriptionBytes, modelWeightsBytes);
        }
    }
}
