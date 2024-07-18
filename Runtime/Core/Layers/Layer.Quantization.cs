using System;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a dequantize layer where four uint8 values are packed per int value.
    /// The final float values are calculated as y = (x - zeroPoint) * scale.
    /// </summary>
    class DequantizeUint8 : Layer
    {
        public float scale;
        public byte zeroPoint;

        public DequantizeUint8(int output, int input, float scale, byte zeroPoint)
            : base(new[] { output }, new[] { input })
        {
            this.scale = scale;
            this.zeroPoint = zeroPoint;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, shapeX));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorByte;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.DequantizeLinear(X, O, scale, zeroPoint);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, scale: {scale}, zeroPoint: {zeroPoint}";
        }

        internal override string profilerTag => "DequantizeUint8";
    }
}
