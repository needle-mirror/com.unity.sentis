using System;
using UnityEngine;

namespace Unity.Sentis
{
    static class BackendFactory
    {
        public static IBackend CreateBackend(BackendType backendType)
        {
            switch (backendType)
            {
                case BackendType.GPUCompute:
#if UNITY_6000_1_OR_NEWER
                    if (SystemInfo.supportsMachineLearning)
                        return new GfxDeviceBackend();
#endif
                    return new GPUComputeBackend();
                case BackendType.GPUPixel:
                    return new GPUPixelBackend();
                default:
                    return new CPUBackend();
            }
        }
    }
}
