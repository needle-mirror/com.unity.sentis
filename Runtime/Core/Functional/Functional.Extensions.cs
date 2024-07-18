namespace Unity.Sentis
{
    /// <summary>
    /// Represents extension functions for the Sentis functional API.
    /// </summary>
    public static class FunctionalExtensions
    {
        /// <summary>
        /// Returns the functional outputs of the forward pass of a model. Sentis will perform destructive edits of the source model.
        /// </summary>
        /// <param name="model">The source model.</param>
        /// <param name="inputs">The functional inputs to the model.</param>
        /// <returns>The functional representation of the model.</returns>
        public static FunctionalTensor[] Forward(this Model model, params FunctionalTensor[] inputs)
        {
            return FunctionalTensor.FromModel(model, inputs);
        }

        /// <summary>
        /// Returns the functional outputs of the forward pass of a model. Sentis will perform a deep copy of the course model.
        /// </summary>
        /// <param name="model">The source model.</param>
        /// <param name="inputs">The functional inputs to the model.</param>
        /// <returns>The functional representation of the model.</returns>
        public static FunctionalTensor[] ForwardWithCopy(this Model model, params FunctionalTensor[] inputs)
        {
            return FunctionalTensor.FromModel(model, inputs, true);
        }

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
