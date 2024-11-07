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
    }
}
