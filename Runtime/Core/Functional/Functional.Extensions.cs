namespace Unity.Sentis
{
    /// <summary>
    /// Represents extension functions for the Sentis functional API.
    /// </summary>
    public static class FunctionalExtensions
    {
        internal static Model DeepCopy(this Model model)
        {
            var modelDescriptionBytes = ModelWriter.SaveModelDescription(model);
            var modelWeightsBytes = ModelWriter.SaveModelWeights(model);
            var ret = new Model();
            ModelLoader.LoadModelDescription(modelDescriptionBytes, ref ret);
            ModelLoader.LoadModelWeights(modelWeightsBytes, ref ret);
            return ret;
        }
    }
}
