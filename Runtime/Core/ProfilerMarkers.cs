using System.Collections.Generic;
using Unity.Profiling;

namespace Unity.Sentis
{
    static class ProfilerMarkers
    {
        public static readonly ProfilerMarker Schedule = new ProfilerMarker("Sentis.Schedule");
        public static readonly ProfilerMarker LoadComputeShader = new ProfilerMarker("Sentis.ComputeShader.Load");
        public static readonly ProfilerMarker LoadPixelShader = new ProfilerMarker("Sentis.PixelShader.Load");
        public static readonly ProfilerMarker ComputeTensorDataDownload = new ProfilerMarker("Sentis.ComputeTensorData.DownloadDataFromGPU");
        public static readonly ProfilerMarker TextureTensorDataDownload = new ProfilerMarker("Sentis.TextureTensorData.DownloadDataFromGPU");
        public static readonly ProfilerMarker TensorDataPoolAdopt = new ProfilerMarker("Sentis.TensorDataPool.Adopt");
        public static readonly ProfilerMarker TensorDataPoolRelease = new ProfilerMarker("Sentis.TensorDataPool.Release");
        public static readonly ProfilerMarker InferModelPartialTensors = new ProfilerMarker("Sentis.Compiler.Analyser.ShapeInferenceAnalysis.InferModelPartialTensors");
        public static readonly ProfilerMarker LoadModelDesc = new ProfilerMarker("Sentis.ModelLoader.LoadModelDesc");
        public static readonly ProfilerMarker LoadModelWeights = new ProfilerMarker("Sentis.ModelLoader.LoadModelWeights");
        public static readonly ProfilerMarker SaveModelDesc = new ProfilerMarker("Sentis.ModelLoader.SaveModelDesc");
        public static readonly ProfilerMarker SaveModelWeights = new ProfilerMarker("Sentis.ModelLoader.SaveModelWeights");
        public static readonly ProfilerMarker ComputeTensorDataNewEmpty = new ProfilerMarker("Sentis.ComputeTensorData.NewEmpty");
        public static readonly ProfilerMarker ComputeTensorDataNewArray = new ProfilerMarker("Sentis.ComputeTensorData.NewArray");

        public static readonly Dictionary<string, ProfilerMarker> LayerTypeMarkers = new Dictionary<string, ProfilerMarker>()
        {
            {"Celu", new ProfilerMarker("Sentis.Layer.Celu") },
            {"Elu", new ProfilerMarker("Sentis.Layer.Elu") },
            {"Gelu", new ProfilerMarker("Sentis.Layer.Gelu") },
            {"Erf", new ProfilerMarker("Sentis.Layer.Erf") },
            {"Hardmax", new ProfilerMarker("Sentis.Layer.Hardmax") },
            {"HardSigmoid", new ProfilerMarker("Sentis.Layer.HardSigmoid") },
            {"HardSwish", new ProfilerMarker("Sentis.Layer.HardSwish") },
            {"LeakyRelu", new ProfilerMarker("Sentis.Layer.LeakyRelu") },
            {"PRelu", new ProfilerMarker("Sentis.Layer.PRelu") },
            {"Relu", new ProfilerMarker("Sentis.Layer.Relu") },
            {"Relu6", new ProfilerMarker("Sentis.Layer.Relu6") },
            {"Selu", new ProfilerMarker("Sentis.Layer.Selu") },
            {"Sigmoid", new ProfilerMarker("Sentis.Layer.Sigmoid") },
            {"Softplus", new ProfilerMarker("Sentis.Layer.Softplus") },
            {"Softsign", new ProfilerMarker("Sentis.Layer.Softsign") },
            {"Swish", new ProfilerMarker("Sentis.Layer.Swish") },
            {"Tanh", new ProfilerMarker("Sentis.Layer.Tanh") },
            {"ThresholdedRelu", new ProfilerMarker("Sentis.Layer.ThresholdedRelu") },
            {"LogSoftmax", new ProfilerMarker("Sentis.Layer.LogSoftmax") },
            {"Softmax", new ProfilerMarker("Sentis.Layer.Softmax") },
            {"SingleHeadAttention", new ProfilerMarker("Sentis.Layer.SingleHeadAttention") },
            {"Conv", new ProfilerMarker("Sentis.Layer.Conv") },
            {"ConvTranspose", new ProfilerMarker("Sentis.Layer.ConvTranspose") },
            {"Shape", new ProfilerMarker("Sentis.Layer.Shape") },
            {"Size", new ProfilerMarker("Sentis.Layer.Size") },
            {"ConstantOfShape", new ProfilerMarker("Sentis.Layer.ConstantOfShape") },
            {"OneHot", new ProfilerMarker("Sentis.Layer.OneHot") },
            {"Range", new ProfilerMarker("Sentis.Layer.Range") },
            {"ArgMax", new ProfilerMarker("Sentis.Layer.ArgMax") },
            {"ArgMin", new ProfilerMarker("Sentis.Layer.ArgMin") },
            {"Gather", new ProfilerMarker("Sentis.Layer.Gather") },
            {"GatherElements", new ProfilerMarker("Sentis.Layer.GatherElements") },
            {"GatherND", new ProfilerMarker("Sentis.Layer.GatherND") },
            {"NonZero", new ProfilerMarker("Sentis.Layer.NonZero") },
            {"ScatterElements", new ProfilerMarker("Sentis.Layer.ScatterElements") },
            {"ScatterND", new ProfilerMarker("Sentis.Layer.ScatterND") },
            {"TopK", new ProfilerMarker("Sentis.Layer.TopK") },
            {"And", new ProfilerMarker("Sentis.Layer.And") },
            {"Compress", new ProfilerMarker("Sentis.Layer.Compress") },
            {"Equal", new ProfilerMarker("Sentis.Layer.Equal") },
            {"Greater", new ProfilerMarker("Sentis.Layer.Greater") },
            {"GreaterOrEqual", new ProfilerMarker("Sentis.Layer.GreaterOrEqual") },
            {"IsInf", new ProfilerMarker("Sentis.Layer.IsInf") },
            {"IsNaN", new ProfilerMarker("Sentis.Layer.IsNaN") },
            {"Less", new ProfilerMarker("Sentis.Layer.Less") },
            {"LessOrEqual", new ProfilerMarker("Sentis.Layer.LessOrEqual") },
            {"Not", new ProfilerMarker("Sentis.Layer.Not") },
            {"Or", new ProfilerMarker("Sentis.Layer.Or") },
            {"Xor", new ProfilerMarker("Sentis.Layer.Xor") },
            {"Where", new ProfilerMarker("Sentis.Layer.Where") },
            {"Abs", new ProfilerMarker("Sentis.Layer.Abs") },
            {"Add", new ProfilerMarker("Sentis.Layer.Add") },
            {"Ceil", new ProfilerMarker("Sentis.Layer.Ceil") },
            {"Clip", new ProfilerMarker("Sentis.Layer.Clip") },
            {"CumSum", new ProfilerMarker("Sentis.Layer.CumSum") },
            {"Dense", new ProfilerMarker("Sentis.Layer.Dense") },
            {"DenseBatched", new ProfilerMarker("Sentis.Layer.DenseBatched") },
            {"Div", new ProfilerMarker("Sentis.Layer.Div") },
            {"Einsum", new ProfilerMarker("Sentis.Layer.Einsum") },
            {"Exp", new ProfilerMarker("Sentis.Layer.Exp") },
            {"Floor", new ProfilerMarker("Sentis.Layer.Floor") },
            {"Log", new ProfilerMarker("Sentis.Layer.Log") },
            {"MatMul", new ProfilerMarker("Sentis.Layer.MatMul") },
            {"MatMul2D", new ProfilerMarker("Sentis.Layer.MatMul2D") },
            {"Max", new ProfilerMarker("Sentis.Layer.Max") },
            {"Mean", new ProfilerMarker("Sentis.Layer.Mean") },
            {"Min", new ProfilerMarker("Sentis.Layer.Min") },
            {"Mod", new ProfilerMarker("Sentis.Layer.Mod") },
            {"Mul", new ProfilerMarker("Sentis.Layer.Mul") },
            {"Neg", new ProfilerMarker("Sentis.Layer.Neg") },
            {"Pow", new ProfilerMarker("Sentis.Layer.Pow") },
            {"Reciprocal", new ProfilerMarker("Sentis.Layer.Reciprocal") },
            {"Round", new ProfilerMarker("Sentis.Layer.Round") },
            {"ScalarMad", new ProfilerMarker("Sentis.Layer.ScalarMad") },
            {"Shrink", new ProfilerMarker("Sentis.Layer.Shrink") },
            {"Sign", new ProfilerMarker("Sentis.Layer.Sign") },
            {"Sqrt", new ProfilerMarker("Sentis.Layer.Sqrt") },
            {"Square", new ProfilerMarker("Sentis.Layer.Square") },
            {"Sub", new ProfilerMarker("Sentis.Layer.Sub") },
            {"Sum", new ProfilerMarker("Sentis.Layer.Sum") },
            {"ScaleBias", new ProfilerMarker("Sentis.Layer.ScaleBias") },
            {"InstanceNormalization", new ProfilerMarker("Sentis.Layer.InstanceNormalization") },
            {"LayerNormalization", new ProfilerMarker("Sentis.Layer.LayerNormalization") },
            {"BatchNormalization", new ProfilerMarker("Sentis.Layer.BatchNormalization") },
            {"LRN", new ProfilerMarker("Sentis.Layer.LRN") },
            {"NonMaxSuppression", new ProfilerMarker("Sentis.Layer.NonMaxSuppression") },
            {"RoiAlign", new ProfilerMarker("Sentis.Layer.RoiAlign") },
            {"AveragePool", new ProfilerMarker("Sentis.Layer.AveragePool") },
            {"GlobalAveragePool", new ProfilerMarker("Sentis.Layer.GlobalAveragePool") },
            {"GlobalMaxPool", new ProfilerMarker("Sentis.Layer.GlobalMaxPool") },
            {"MaxPool", new ProfilerMarker("Sentis.Layer.MaxPool") },
            {"RandomNormal", new ProfilerMarker("Sentis.Layer.RandomNormal") },
            {"RandomNormalLike", new ProfilerMarker("Sentis.Layer.RandomNormalLike") },
            {"RandomUniform", new ProfilerMarker("Sentis.Layer.RandomUniform") },
            {"RandomUniformLike", new ProfilerMarker("Sentis.Layer.RandomUniformLike") },
            {"Bernoulli", new ProfilerMarker("Sentis.Layer.Bernoulli") },
            {"Multinomial", new ProfilerMarker("Sentis.Layer.Multinomial") },
            {"ReduceL1", new ProfilerMarker("Sentis.Layer.ReduceL1") },
            {"ReduceL2", new ProfilerMarker("Sentis.Layer.ReduceL2") },
            {"ReduceLogSum", new ProfilerMarker("Sentis.Layer.ReduceLogSum") },
            {"ReduceLogSumExp", new ProfilerMarker("Sentis.Layer.ReduceLogSumExp") },
            {"ReduceMax", new ProfilerMarker("Sentis.Layer.ReduceMax") },
            {"ReduceMean", new ProfilerMarker("Sentis.Layer.ReduceMean") },
            {"ReduceMin", new ProfilerMarker("Sentis.Layer.ReduceMin") },
            {"ReduceProd", new ProfilerMarker("Sentis.Layer.ReduceProd") },
            {"ReduceSum", new ProfilerMarker("Sentis.Layer.ReduceSum") },
            {"ReduceSumSquare", new ProfilerMarker("Sentis.Layer.ReduceSumSquare") },
            {"Cast", new ProfilerMarker("Sentis.Layer.Cast") },
            {"CastLike", new ProfilerMarker("Sentis.Layer.CastLike") },
            {"Concat", new ProfilerMarker("Sentis.Layer.Concat") },
            {"DepthToSpace", new ProfilerMarker("Sentis.Layer.DepthToSpace") },
            {"Expand", new ProfilerMarker("Sentis.Layer.Expand") },
            {"Flatten", new ProfilerMarker("Sentis.Layer.Flatten") },
            {"GridSample", new ProfilerMarker("Sentis.Layer.GridSample") },
            {"Identity", new ProfilerMarker("Sentis.Layer.Identity") },
            {"Pad", new ProfilerMarker("Sentis.Layer.Pad") },
            {"Reshape", new ProfilerMarker("Sentis.Layer.Reshape") },
            {"Resize", new ProfilerMarker("Sentis.Layer.Resize") },
            {"Slice", new ProfilerMarker("Sentis.Layer.Slice") },
            {"SpaceToDepth", new ProfilerMarker("Sentis.Layer.SpaceToDepth") },
            {"Split", new ProfilerMarker("Sentis.Layer.Split") },
            {"Squeeze", new ProfilerMarker("Sentis.Layer.Squeeze") },
            {"Tile", new ProfilerMarker("Sentis.Layer.Tile") },
            {"Transpose", new ProfilerMarker("Sentis.Layer.Transpose") },
            {"Trilu", new ProfilerMarker("Sentis.Layer.Trilu") },
            {"Unsqueeze", new ProfilerMarker("Sentis.Layer.Unsqueeze") },
            {"Acos", new ProfilerMarker("Sentis.Layer.Acos") },
            {"Acosh", new ProfilerMarker("Sentis.Layer.Acosh") },
            {"Asin", new ProfilerMarker("Sentis.Layer.Asin") },
            {"Asinh", new ProfilerMarker("Sentis.Layer.Asinh") },
            {"Atan", new ProfilerMarker("Sentis.Layer.Atan") },
            {"Atanh", new ProfilerMarker("Sentis.Layer.Atanh") },
            {"Cos", new ProfilerMarker("Sentis.Layer.Cos") },
            {"Cosh", new ProfilerMarker("Sentis.Layer.Cosh") },
            {"Sin", new ProfilerMarker("Sentis.Layer.Sin") },
            {"Sinh", new ProfilerMarker("Sentis.Layer.Sinh") },
            {"Tan", new ProfilerMarker("Sentis.Layer.Tan") },
        };
        public static readonly ProfilerMarker CustomLayerMarker = new ProfilerMarker("Sentis.Layer.Custom");

        public static ProfilerMarker LayerTypeProfilerMarker(string opName)
        {
            var marker = CustomLayerMarker;
            LayerTypeMarkers.TryGetValue(opName, out marker);
            return marker;
        }
    }
}
