using System;

namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns the indexes of the boxes with the highest scores, which pass the intersect-over-union test to other output boxes.
        /// </summary>
        /// <param name="boxes">The boxes tensor [N, 4] with (x1, y1, x2, y2) corners format with 0 ≤ x1 &lt; x2 ≤ 1 and 0 ≤ y1 &lt; y2 ≤ 1.</param>
        /// <param name="scores">The scores tensor [N].(</param>
        /// <param name="iouThreshold">The threshold above which overlapping boxes are discarded.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor NMS(FunctionalTensor boxes, FunctionalTensor scores, float iouThreshold)
        {
            boxes = boxes.Float();
            scores = scores.Float();
            var layer = new Layers.NonMaxSuppression(null, null, null)
            {
                inputs = new string[4]
            };
            return FunctionalTensor.FromLayer(layer, DataType.Int, new[] { boxes.Unsqueeze(0), scores.Reshape(new[] { 1, 1, -1 }), Tensor(-1), Tensor(iouThreshold) }).Select(1, 2);
        }
    }
}
