using System;
using UnityEngine;

namespace Unity.Sentis {

static class BackendFactory
{
    public static IBackend CreateBackend(BackendType backendType)
    {
        switch (backendType)
        {
            case BackendType.GPUCompute:
                return new GPUComputeBackend();
            case BackendType.GPUCommandBuffer:
                return new GPUCommandBufferBackend();
            case BackendType.GPUPixel:
                return new GPUPixelBackend();
            default:
                return new CPUBackend();
        }
    }

    public static IWorker CreateWorker(BackendType backendType, Model model, WorkerFactory.WorkerConfiguration workerConfiguration)
    {
        if (WorkerFactory.IsType(backendType, DeviceType.GPU) && !SystemInfo.supportsComputeShaders && !Application.isEditor)
        {
            backendType = BackendType.GPUPixel;
        }

        IModelStorage vars = new ModelStorage();

        var backend = CreateBackend(backendType);

        return new GenericWorker(model, backend, vars, workerConfiguration.takeoverWeights);
    }
}
} // namespace Unity.Sentis
