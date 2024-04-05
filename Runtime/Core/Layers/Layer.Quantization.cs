using System;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a dequantize layer where four uint8 values are packed per int value.
    /// The final float values are calculated as y = (x - zeroPoint) * scale.
    /// </summary>
    [Serializable]
    class DequantizeUint8 : Layer
    {
        /// <summary>
        /// The scale value to use for dequantization.
        /// </summary>
        public float scale;
        /// <summary>
        /// The zero point value to use for dequantization.
        /// </summary>
        public byte zeroPoint;

        /// <summary>
        /// Initializes and returns an instance of `DequantizeUint8` layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="scale">The scale value to use for dequantization.</param>
        /// <param name="zeroPoint">The zero point value to use for dequantization.</param>
        public DequantizeUint8(string name, string input, float scale, byte zeroPoint)
        {
            this.index = name;
            this.inputs = new[] { input };
            this.scale = scale;
            this.zeroPoint = zeroPoint;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;
            ctx.AddPartialTensor(index, new PartialTensor(DataType.Float, shapeX));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]) as TensorByte;
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.DequantizeLinear(X, O, scale, zeroPoint);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, scale: {scale}, zeroPoint: {zeroPoint}";
        }

        internal override string profilerTag => "DequantizeUint8";
    }
}
