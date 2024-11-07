using System;
using Unity.Profiling;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a dequantize layer where four uint8 values are packed per int value.
    /// The final float values are calculated as y = (x - zeroPoint) * scale.
    /// </summary>
    class DequantizeUint8 : Layer
    {
        static readonly string k_OpName = "DequantizeUint8";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
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

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<byte>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.DequantizeLinear(X, O, scale, zeroPoint);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, scale: {scale}, zeroPoint: {zeroPoint}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }
}
