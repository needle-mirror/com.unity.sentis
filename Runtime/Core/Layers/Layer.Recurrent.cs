using System;
using Unity.Profiling;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Options for the direction of a recurrent layer.
    /// </summary>
    enum RnnDirection
    {
        /// <summary>
        /// Use only forward direction in the calculation.
        /// </summary>
        Forward = 0,
        /// <summary>
        /// Use only reverse direction in the calculation.
        /// </summary>
        Reverse = 1,
        /// <summary>
        /// Use both forward and reverse directions in the calculation.
        /// </summary>
        Bidirectional = 2,
    }

    /// <summary>
    /// Options for activation functions to apply in a recurrent layer.
    /// </summary>
    enum RnnActivation
    {
        /// <summary>
        /// Use `Relu` activation: f(x) = max(0, x).
        /// </summary>
        Relu = 0,
        /// <summary>
        /// Use `Tanh` activation: f(x) = (1 - e^{-2x}) / (1 + e^{-2x}).
        /// </summary>
        Tanh = 1,
        /// <summary>
        /// Use `Sigmoid` activation: f(x) = 1 / (1 + e^{-x}).
        /// </summary>
        Sigmoid = 2,
        /// <summary>
        /// Use `Affine` activation: f(x) = alpha * x + beta.
        /// </summary>
        Affine = 3,
        /// <summary>
        /// Use `LeakyRelu` activation: f(x) = x if x >= 0, otherwise f(x) = alpha * x.
        /// </summary>
        LeakyRelu = 4,
        /// <summary>
        /// Use `ThresholdedRelu` activation: f(x) = x if x >= alpha, otherwise f(x) = 0.
        /// </summary>
        ThresholdedRelu = 5,
        /// <summary>
        /// Use `ScaledTanh` activation: f(x) = alpha * tanh(beta * x).
        /// </summary>
        ScaledTanh = 6,
        /// <summary>
        /// Use `HardSigmoid` activation: f(x) = clamp(alpha * x + beta, 0, 1).
        /// </summary>
        HardSigmoid = 7,
        /// <summary>
        /// Use `Elu` activation: f(x) = x if x >= 0, otherwise f(x) = alpha * (e^x - 1).
        /// </summary>
        Elu = 8,
        /// <summary>
        /// Use `Softsign` activation: f(x) = x / (1 + |x|).
        /// </summary>
        Softsign = 9,
        /// <summary>
        /// Use `Softplus` activation: f(x) = log(1 + e^x).
        /// </summary>
        Softplus = 10,
    }

    /// <summary>
    /// Options for the layout of the tensor in a recurrent layer.
    /// </summary>
    enum RnnLayout
    {
        /// <summary>
        /// Use layout with sequence as the first dimension of the tensors.
        /// </summary>
        SequenceFirst = 0,
        /// <summary>
        /// Use layout with batch as the first dimension of the tensors.
        /// </summary>
        BatchFirst = 1,
    }

    /// <summary>
    /// Represents an `LSTM` recurrent layer. This generates an output tensor by computing a one-layer LSTM (long short-term memory) on an input tensor.
    /// </summary>
    class LSTM : Layer
    {
        static readonly string k_OpName = "LSTM";
        static readonly ProfilerMarker k_ProfilerMarker = new("Sentis.Layer." + k_OpName);
        public int hiddenSize;
        public RnnDirection direction;
        public RnnActivation[] activations;
        public float[] activationAlpha;
        public float[] activationBeta;
        public float clip;
        public bool inputForget;
        public RnnLayout layout;
        public int NumDirections => direction == RnnDirection.Bidirectional ? 2 : 1;

        public LSTM(int Y, int X, int W, int R, int hiddenSize, int Y_h = -1, int Y_c = -1, int B = -1, int sequenceLens = -1, int initialH = -1, int initialC = -1, int P = -1, RnnDirection direction = RnnDirection.Forward, RnnActivation[] activations = null, float[] activationAlpha = null, float[] activationBeta = null, float clip = float.MaxValue, bool inputForget = false, RnnLayout layout = RnnLayout.SequenceFirst)
            : base(new[] { Y, Y_h, Y_c }, new[] { X, W, R, B, sequenceLens, initialH, initialC, P })
        {
            this.hiddenSize = hiddenSize;
            this.direction = direction;
            this.activations = new RnnActivation[3 * NumDirections];
            this.activationAlpha = new float[3 * NumDirections];
            this.activationBeta = new float[3 * NumDirections];
            for (var i = 0; i < 3 * NumDirections; i++)
            {
                this.activations[i] = i % 3 == 0 ? RnnActivation.Sigmoid : RnnActivation.Tanh;
                if (activations != null && i < activations.Length)
                    this.activations[i] = activations[i];
                switch (this.activations[i])
                {
                    case RnnActivation.Affine:
                        this.activationAlpha[i] = 1.0f;
                        break;
                    case RnnActivation.LeakyRelu:
                        this.activationAlpha[i] = 0.01f;
                        break;
                    case RnnActivation.ThresholdedRelu:
                        this.activationAlpha[i] = 1.0f;
                        break;
                    case RnnActivation.ScaledTanh:
                        this.activationAlpha[i] = 1.0f;
                        this.activationBeta[i] = 1.0f;
                        break;
                    case RnnActivation.HardSigmoid:
                        this.activationAlpha[i] = 0.2f;
                        this.activationBeta[i] = 0.5f;
                        break;
                    case RnnActivation.Elu:
                        this.activationAlpha[i] = 1.0f;
                        break;
                }
                if (activationAlpha != null && i < activationAlpha.Length)
                    this.activationAlpha[i] = activationAlpha[i];
                if (activationBeta != null && i < activationBeta.Length)
                    this.activationBeta[i] = activationBeta[i];
            }

            this.clip = clip;
            this.inputForget = inputForget;
            this.layout = layout;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var inputTensors = ctx.GetPartialTensors(inputs);
            var shapeX = inputTensors[0].shape;
            var shapeW = inputTensors[1].shape;
            var shapeR = inputTensors[2].shape;

            var seqLength = DynamicTensorDim.Unknown;
            var batchSize = DynamicTensorDim.Unknown;

            shapeX.DeclareRank(3);
            shapeW.DeclareRank(3);
            shapeR.DeclareRank(3);

            seqLength = DynamicTensorDim.MaxDefinedDim(seqLength, layout == RnnLayout.SequenceFirst ? shapeX[0] : shapeX[1]);
            batchSize = DynamicTensorDim.MaxDefinedDim(batchSize, layout == RnnLayout.SequenceFirst ? shapeX[1] : shapeX[0]);

            if (inputTensors[3] != null && inputTensors[3].shape is var shapeB)
                shapeB.DeclareRank(2);

            if (inputTensors[4] != null && inputTensors[4].shape is var shapeSequenceLens)
            {
                shapeSequenceLens.DeclareRank(1);
                batchSize = DynamicTensorDim.MaxDefinedDim(batchSize, shapeSequenceLens[0]);
            }

            if (inputTensors[5] != null && inputTensors[5].shape is var shapeInitialH)
            {
                shapeInitialH.DeclareRank(3);
                batchSize = DynamicTensorDim.MaxDefinedDim(batchSize, layout == RnnLayout.SequenceFirst ? shapeInitialH[1] : shapeInitialH[0]);
            }

            if (inputTensors[6] != null && inputTensors[6].shape is var shapeInitialC)
            {
                shapeInitialC.DeclareRank(3);
                batchSize = DynamicTensorDim.MaxDefinedDim(batchSize, layout == RnnLayout.SequenceFirst ? shapeInitialC[1] : shapeInitialC[0]);
            }

            if (inputTensors[7] != null && inputTensors[7].shape is var shapeP)
                shapeP.DeclareRank(2);

            var numDirectionsDim = DynamicTensorDim.Int(NumDirections);
            var hiddenSizeDim = DynamicTensorDim.Int(hiddenSize);

            if (layout == RnnLayout.SequenceFirst)
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, new DynamicTensorShape(seqLength, numDirectionsDim, batchSize, hiddenSizeDim)));
                ctx.AddPartialTensor(outputs[1], new PartialTensor(DataType.Float, new DynamicTensorShape(numDirectionsDim, batchSize, hiddenSizeDim)));
                ctx.AddPartialTensor(outputs[2], new PartialTensor(DataType.Float, new DynamicTensorShape(numDirectionsDim, batchSize, hiddenSizeDim)));
            }
            else
            {
                ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, new DynamicTensorShape(batchSize, seqLength, numDirectionsDim, hiddenSizeDim)));
                ctx.AddPartialTensor(outputs[1], new PartialTensor(DataType.Float, new DynamicTensorShape(batchSize, numDirectionsDim, hiddenSizeDim)));
                ctx.AddPartialTensor(outputs[2], new PartialTensor(DataType.Float, new DynamicTensorShape(batchSize, numDirectionsDim, hiddenSizeDim)));
            }
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var W = ctx.storage.GetTensor(inputs[1]) as Tensor<float>;
            var R = ctx.storage.GetTensor(inputs[2]) as Tensor<float>;
            var B = ctx.storage.GetTensor(inputs[3]) as Tensor<float>;
            var sequenceLens = ctx.storage.GetTensor(inputs[4]) as Tensor<int>;
            var initialH = ctx.storage.GetTensor(inputs[5]) as Tensor<float>;
            var initialC = ctx.storage.GetTensor(inputs[6]) as Tensor<float>;
            var P = ctx.storage.GetTensor(inputs[7]) as Tensor<float>;

            ShapeInference.LSTM(X.shape, W.shape, R.shape, layout, out var shapeY, out var shapeY_h, out var shapeY_c);
            var Y = ctx.storage.AllocateTensorAndStore(outputs[0], shapeY, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            var Y_h = ctx.storage.AllocateTensorAndStore(outputs[1], shapeY_h, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            var Y_c = ctx.storage.AllocateTensorAndStore(outputs[2], shapeY_c, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (Y.shape.HasZeroDims())
                return;

            ctx.backend.LSTM(X, W, R, B, sequenceLens, initialH, initialC, P, Y, Y_h, Y_c, direction, activations, activationAlpha, activationBeta, inputForget, clip, layout);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, outputs: [{string.Join(", ", outputs)}], hiddenSize: {hiddenSize}, direction: {direction}, activations: [{string.Join(", ", activations)}], activationAlpha: [{string.Join(", ", activationAlpha)}], activationBeta: [{string.Join(", ", activationBeta)}], clip: {clip}, inputForget: {inputForget}, layout: {layout}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }
}
