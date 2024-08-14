using System;
using UnityEngine;

namespace Unity.Sentis
{
    /// <summary>
    /// Types of devices that Sentis uses to execute a neural network.
    /// </summary>
    public enum DeviceType
    {
        /// <summary>
        /// Executes using the GPU.
        /// </summary>
        GPU = 1 << 8,

        /// <summary>
        /// Executes using the CPU.
        /// </summary>
        CPU = 1 << 9,
    }

    /// <summary>
    /// Types of backend that Sentis uses to execute a neural network.
    /// </summary>
    public enum BackendType
    {
        /// <summary>
        /// Executes using compute shaders on the GPU.
        /// </summary>
        GPUCompute = 0 | DeviceType.GPU,

        /// <summary>
        /// Executes using pixel shaders on the GPU.
        /// </summary>
        GPUPixel = 1 | DeviceType.GPU,

        /// <summary>
        /// Executes using Burst on the CPU.
        /// </summary>
        CPU = 0 | DeviceType.CPU,
    }
}
