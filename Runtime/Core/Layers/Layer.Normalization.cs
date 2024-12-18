using System;
using Unity.Collections;
using Unity.Profiling;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents an element-wise `ScaleBias` normalization layer: f(x, s, b) = x * s + b.
    /// </summary>
    class ScaleBias : Layer
    {
        static readonly string k_OpName = "ScaleBias";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public ScaleBias(int output, int input, int scale, int bias)
            : base(new[] { output }, new[] { input, scale, bias }) { }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var scale = ctx.GetPartialTensor(inputs[1]);
            var bias = ctx.GetPartialTensor(inputs[2]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeScale = scale.shape;
            var shapeBias = bias.shape;
            var c = DynamicTensorDim.Unknown;
            shapeScale.DeclareRank(1);
            c = DynamicTensorDim.MaxDefinedDim(c, shapeScale[0]);
            shapeBias.DeclareRank(1);
            c = DynamicTensorDim.MaxDefinedDim(c, shapeBias[0]);
            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType));
                return;
            }

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 2 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 2, shapeX.rank);

            var shapeOut = new DynamicTensorShape(shapeX);
            shapeOut[1] = DynamicTensorDim.MaxDefinedDim(shapeOut[1], c);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.ScaleBias(X as Tensor<float>, ctx.storage.GetTensor(inputs[1]) as Tensor<float>, ctx.storage.GetTensor(inputs[2]) as Tensor<float>, O);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an `InstanceNormalization` normalization layer. This computes the mean variance on the spatial dims of the input tensor and normalizes them according to `scale` and `bias` tensors.
    /// </summary>
    class InstanceNormalization : Layer
    {
        static readonly string k_OpName = "InstanceNormalization";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public float epsilon;

        public InstanceNormalization(int output, int input, int scale, int bias, float epsilon = 1e-5f)
            : base(new[] { output }, new[] { input, scale, bias })
        {
            if (epsilon == 0)
                epsilon = Mathf.Epsilon; // safety check to prevent division by zero
            this.epsilon = epsilon;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var scale = ctx.GetPartialTensor(inputs[1]);
            var bias = ctx.GetPartialTensor(inputs[2]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeScale = scale.shape;
            var shapeBias = bias.shape;
            var c = DynamicTensorDim.Unknown;
            shapeScale.DeclareRank(1);
            c = DynamicTensorDim.MaxDefinedDim(c, shapeScale[0]);
            shapeBias.DeclareRank(1);
            c = DynamicTensorDim.MaxDefinedDim(c, shapeBias[0]);
            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType));
                return;
            }

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 2 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 2, shapeX.rank);
            shapeScale.DeclareRank(1);

            var shapeOut = new DynamicTensorShape(shapeX);
            shapeOut[1] = DynamicTensorDim.MaxDefinedDim(shapeOut[1], c);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.InstanceNormalization(X as Tensor<float>, ctx.storage.GetTensor(inputs[1]) as Tensor<float>, ctx.storage.GetTensor(inputs[2]) as Tensor<float>, O, epsilon);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, epsilon: {epsilon}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an `LayerNormalization` normalization layer. This computes the mean variance on the last dimension of the input tensor and normalizes it according to `scale` and `bias` tensors.
    /// </summary>
    class LayerNormalization : Layer
    {
        static readonly string k_OpName = "LayerNormalization";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public float epsilon;

        public LayerNormalization(int output, int input, int scale, int bias = -1, float epsilon = 1e-5f)
            : base(new[] { output }, new[] { input, scale, bias })
        {
            if (epsilon == 0)
                epsilon = Mathf.Epsilon; // safety check to prevent division by zero
            this.epsilon = epsilon;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var scale = ctx.GetPartialTensor(inputs[1]);
            var bias = ctx.GetPartialTensor(inputs[2]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeScale = scale.shape;
            var shapeBias = bias.shape;

            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, DynamicTensorShape.DynamicRank));
                return;
            }

            Logger.AssertIsTrue(shapeX.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            shapeScale.DeclareRank(1);
            shapeBias.DeclareRank(1);

            var shape = new DynamicTensorShape(shapeX);
            shape[-1] = DynamicTensorDim.MaxDefinedDim(shape[-1], DynamicTensorDim.MaxDefinedDim(shapeScale[0], shapeBias[0]));
            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.LayerNormalization(X as Tensor<float>, ctx.storage.GetTensor(inputs[1]) as Tensor<float>, ctx.storage.GetTensor(inputs[2]) as Tensor<float>, O, epsilon);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, epsilon: {epsilon}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `RMSNormalization` normalization layer. This computes the mean square variance on the last dimension of the input tensor and normalizes it according to `scale` tensor.
    /// </summary>
    class RMSNormalization : Layer
    {
        static readonly string k_OpName = "RMSNormalization";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public float epsilon;

        public RMSNormalization(int output, int input, int scale, float epsilon = 1e-5f)
            : base(new[] { output }, new[] { input, scale })
        {
            if (epsilon == 0)
                epsilon = Mathf.Epsilon; // safety check to prevent division by zero
            this.epsilon = epsilon;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var scale = ctx.GetPartialTensor(inputs[1]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeScale = scale.shape;

            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, DynamicTensorShape.DynamicRank));
                return;
            }

            Logger.AssertIsTrue(shapeX.rank >= 3, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            shapeScale.DeclareRank(1);

            var shape = new DynamicTensorShape(shapeX);
            shape[-1] = DynamicTensorDim.MaxDefinedDim(shape[-1], shapeScale[0]);
            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.RMSNormalization(X as Tensor<float>, ctx.storage.GetTensor(inputs[1]) as Tensor<float>, O, epsilon);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, epsilon: {epsilon}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an `BatchNormalization` normalization layer. This computes the mean variance on the second dimension of the input tensor and normalizes it according to `scale` and `bias` tensors.
    /// </summary>
    class BatchNormalization : Layer
    {
        static readonly string k_OpName = "BatchNormalization";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public float epsilon;

        public BatchNormalization(int output, int input, int scale, int bias, int mean, int variance, float epsilon = 1e-5f)
            : base(new[] { output }, new[] { input, scale, bias, mean, variance })
        {
            this.epsilon = epsilon;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            var scale = ctx.GetPartialTensor(inputs[1]);
            var bias = ctx.GetPartialTensor(inputs[2]);
            var mean = ctx.GetPartialTensor(inputs[3]);
            var var = ctx.GetPartialTensor(inputs[4]);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeScale = scale.shape;
            var shapeBias = bias.shape;
            var shapeMean = mean.shape;
            var shapeVar = var.shape;

            if (!shapeX.hasRank)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, DynamicTensorShape.DynamicRank));
                return;
            }

            Logger.AssertIsTrue(shapeX.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            shapeScale.DeclareRank(1);
            shapeBias.DeclareRank(1);
            shapeMean.DeclareRank(1);
            shapeVar.DeclareRank(1);

            var shape = new DynamicTensorShape(shapeX);
            if (shapeX.rank > 1)
                shape[1] = DynamicTensorDim.MaxDefinedDim(shape[1], DynamicTensorDim.MaxDefinedDim(shapeScale[0], DynamicTensorDim.MaxDefinedDim(shapeBias[0], DynamicTensorDim.MaxDefinedDim(shapeMean[0], shapeVar[0]))));
            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.BatchNormalization(X as Tensor<float>, ctx.storage.GetTensor(inputs[1]) as Tensor<float>, ctx.storage.GetTensor(inputs[2]) as Tensor<float>, ctx.storage.GetTensor(inputs[3]) as Tensor<float>, ctx.storage.GetTensor(inputs[4]) as Tensor<float>, O, epsilon);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, epsilon: {epsilon}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an `LRN` local response normalization layer. This normalizes the input tensor over local input regions.
    /// </summary>
    class LRN : Layer
    {
        static readonly string k_OpName = "LRN";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public float alpha;
        public float beta;
        public float bias;
        public int count;

        public LRN(int output, int input, float alpha, float beta, float bias, int count)
            : base(new[] { output }, new[] { input })
        {
            this.alpha = alpha;
            this.beta = beta;
            this.bias = bias;
            this.count = count;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var X = ctx.GetPartialTensor(inputs[0]);
            if (X.shape.hasRank)
                Logger.AssertIsTrue(X.shape.rank >= 2, "RankError: incorrect rank, expecting at least {0}, got {1}", 2, X.shape.rank);

            ctx.AddPartialTensor(outputs[0], new PartialTensor(X.dataType, X.shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;

            // pixel we don't know which dim to pin
            var outputBackendType = ctx.backend.backendType;
            if (outputBackendType == BackendType.GPUPixel)
                outputBackendType = BackendType.CPU;

            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, outputBackendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;

            // https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
            // However divide the sum by size to follow onnx and pytorch implementation
            // ONNX https://github.com/onnx/onnx/blob/master/docs/Operators.md#LRN
            // PYTORCH https://github.com/pytorch/pytorch/blob/1465970a343e61f2f2b104859ca7f5d7e03f5d02/torch/nn/functional.py#L2069
            // Tensorflow don't and follow the paper to the letter https://github.com/tensorflow/tensorflow/blob/e6faa845c51bb69465146d93646947fd2ba53efa/tensorflow/python/kernel_tests/lrn_op_test.py#L53
            // However they bake the division to alpha when exporting to ONNX https://github.com/onnx/tensorflow-onnx/blob/7c37ccb97e0fd478ce093910c4a1411b18e44fd7/tf2onnx/onnx_opset/math.py


            // need to download, if gpucompute need to execute commandbuffer and flush.
            if (ctx.backend is GPUComputeBackend gpuBackend)
                gpuBackend.ExecuteCommandBufferAndClear();

            var arrayX = (X as Tensor<float>).DownloadToNativeArray();
            var arrayO = new NativeArray<float>(O.shape.length, Allocator.Temp);

            float sizef = count;

            var itRemap = new TensorNDIterator(O.shape);
            for (var it = new TensorNDIterator(O.shape); it.HasNext(); it.MoveNext())
            {
                int c = it[1];
                float regionCenter = (sizef - 1.0f) / 2.0f;
                int regionStart = Math.Max(0, c - (int)Mathf.Floor(regionCenter));
                int regionEnd = Math.Min(X.shape[1], c + (int)Mathf.Ceil(regionCenter) + 1);
                float sumOfSquared = 0.0f;
                for (int ci = regionStart; ci < regionEnd; ++ci)
                {
                    itRemap.CopyNDIndex(it);
                    itRemap[1] = ci;
                    float regionValue = arrayX[itRemap.index];
                    sumOfSquared += regionValue * regionValue;
                }

                arrayO[it.index] = arrayX[it.index] / Mathf.Pow(bias + alpha * sumOfSquared / sizef, beta);
            }
            O.dataOnBackend.Upload(arrayO, arrayO.Length);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, alpha: {alpha}, beta: {beta}, bias: {bias}, count: {count}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }
}
