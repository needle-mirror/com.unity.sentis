using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine; // CustomYieldInstruction
using UnityEngine.Assertions;
using UnityEngine.Rendering;

namespace Unity.Sentis {

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
    /// CommandBuffer implementation
    /// </summary>
    GPUCommandBuffer = 1 | DeviceType.GPU,

    /// <summary>
    /// Executes using pixel shaders on the GPU.
    /// </summary>
    GPUPixel = 2 | DeviceType.GPU,

    /// <summary>
    /// Executes using Burst on the CPU.
    /// </summary>
    CPU = 0 | DeviceType.CPU,
}

/// <summary>
/// An interface that allows you to execute neural networks (models).
///
/// `IWorker` abstracts implementation details on different hardware devices such as the CPU and the GPU. `IWorker` lets you do the following:
///
/// - Specify inputs.
/// - Schedule the work.
/// - Get outputs.
///
/// Internally, `IWorker` translates the neural network from a <see cref="Model"/> into a set of operations, then sends the operations to the hardware device for asynchronous execution.
///
/// Use `WorkerFactory.CreateWorker` or `Model.CreateWorker` to create a new instance of a worker.
/// </summary>
public interface IWorker : IDisposable
{
    /// <summary>
    /// Sets a tensor as a named input of the model.
    /// </summary>
    /// <param name="name">The name of the input to set.</param>
    /// <param name="inputTensor">The tensor to set as the input.</param>
    void SetInput(string name, Tensor inputTensor);

    /// <summary>
    /// Schedules the execution of the model on the worker. This is non-blocking.
    /// </summary>
    /// <returns>The `IWorker`.</returns>
    IWorker Execute();

    /// <summary>
    /// Sets a tensor as the default input of the model and schedules the execution of the model on the worker. This is non-blocking. For models with more than one input this sets the first input.
    /// </summary>
    /// <param name="inputTensor">The tensor to set to the default input of the model.</param>
    /// <returns>The `IWorker`.</returns>
    IWorker Execute(Tensor inputTensor);

    /// <summary>
    /// Sets multiple tensors as the inputs of the model and schedules execution of the model. This is non-blocking.
    /// </summary>
    /// <param name="inputTensors">The tensors to use as the inputs of the model as a dictionary mapping input names to tensors.</param>
    /// <returns>The `IWorker`.</returns>
    IWorker Execute(IDictionary<string, Tensor> inputTensors);

    /// <summary>
    /// Schedules the execution of the model one layer at a time. This is non-blocking.
    ///
    /// To schedule the execution of the next layer of the model, call `MoveNext` on the `IEnumerator` object this method returns.
    /// </summary>
    /// <returns>The `IEnumerator` for scheduling manual execution.</returns>
    IEnumerator ExecuteLayerByLayer();

    /// <summary>
    /// Sets a tensor as the default input of the model and schedules execution of the model one layer at a time. This is non-blocking. For models with more than one input this sets the first input.
    ///
    /// To schedule execution of the next layer of the model, call `MoveNext` on the `IEnumerator` object this method returns.
    /// </summary>
    /// <param name="inputTensor">The tensor to set to the default input of the model.</param>
    /// <returns>The `IEnumerator` for scheduling manual execution.</returns>
    IEnumerator ExecuteLayerByLayer(Tensor inputTensor);

    /// <summary>
    /// Sets multiple tensors as the inputs of the model and schedules execution of the model one layer at a time. This is non-blocking.
    ///
    /// To schedule execution of the next layer of the model, call `MoveNext` on the `IEnumerator` object this method returns.
    /// </summary>
    /// <param name="inputTensors">The tensors to use as the inputs of the model as a dictionary mapping input names to tensors.</param>
    /// <returns>The `IEnumerator` for scheduling manual execution.</returns>
    IEnumerator ExecuteLayerByLayer(IDictionary<string, Tensor> inputTensors);

    /// <summary>
    /// Returns the proportion of the model scheduled for execution since the last call to `ExecuteLayerByLayer`.
    ///
    /// Returns 0.0 after you call `ExecuteLayerByLayer`. Returns 1.0 when the model is fully scheduled.
    ///
    /// The value increases each time you iterate on the `IEnumerator` that `ExecuteLayerByLayer` returns.
    /// </summary>
    float scheduleProgress { get; }

    /// <summary>
    /// Returns a reference to the default output tensor. This is non-blocking.
    ///
    /// For models with more than one output this returns a reference to the first output tensor.
    ///
    /// The reference is valid only until you call `Execute()` or `Dispose()` on the worker.
    ///
    /// </summary>
    /// <returns>The output tensor reference.</returns>
    Tensor PeekOutput();

    /// <summary>
    /// Returns a reference to the default output tensor. This is non-blocking.
    ///
    /// For models with more than one output this returns a reference to the first output tensor.
    ///
    /// The reference is valid only until you call `Execute()` or `Dispose()` on the worker.
    /// </summary>
    /// <param name="name">The name of the output tensor to peek.</param>
    /// <returns>The output tensor reference.</returns>
    Tensor PeekOutput(string name);

    /// <summary>
    /// Takes ownership of the default output tensor. This is non-blocking.
    ///
    /// For models with more than one output this returns a reference to the first output tensor.
    /// </summary>
    /// <returns>The output tensor.</returns>
    Tensor TakeOutputOwnership();

    /// <summary>
    /// Takes ownership of an output tensor with a given `name`. This is non-blocking.
    /// </summary>
    /// <param name="name">The name of the output tensor to take ownership of.</param>
    /// <returns>The output tensor.</returns>
    Tensor TakeOutputOwnership(string name);

    /// <summary>
    /// Returns the backend used for execution.
    /// </summary>
    /// <returns>The `IBackend` used.</returns>
    IBackend GetBackend();
}

/// <summary>
/// Provides extension methods for the `IWorker` interface.
/// </summary>
public static class WorkerExtensions
{
    /// <summary>
    /// Execute model and returns a CPU copy of all outputs. This is a non blocking call.
    /// </summary>
    /// <param name="worker">The worker to execute.</param>
    /// <param name="inputs">The input tensors as a dictionary.</param>
    /// <param name="outputs">The output names as a list.</param>
    /// <returns>The async task.</returns>
    public static async Task<bool[]> ExecuteAndDownloadOutputsAsync(this IWorker worker, Dictionary<string, Tensor> inputs, List<string> outputs)
    {
        worker.Execute(inputs);
        Task<bool>[] tasks = new Task<bool>[outputs.Count];
        for (int i = 0; i < outputs.Count; i++)
        {
            tasks[i] = worker.PeekOutput(outputs[i]).CompleteOperationsAndDownloadAsync();
        }
        return await Task.WhenAll(tasks);
    }

    /// <summary>
    /// Non-blocking API that schedules network execution on CommandBuffer in one go.
    /// </summary>
    /// <param name="cb">The command buffer to schedule execution on.</param>
    /// <param name="worker">The worker to use for execution.</param>
    /// <param name="inputs">A dictionary of input tensors.</param>
    public static void ExecuteWorker(this CommandBuffer cb, IWorker worker, Dictionary<string, Tensor> inputs)
    {
        var backend = worker.GetBackend();
        Assert.IsTrue(backend is GPUCommandBufferBackend);
        (backend as GPUCommandBufferBackend).cb = cb;
        worker.Execute(inputs);
    }

    /// <summary>
    /// Non-blocking API that schedules network execution on CommandBuffer in one go.
    /// </summary>
    /// <param name="cb">The command buffer to schedule execution on.</param>
    /// <param name="worker">The worker to use for execution.</param>
    /// <param name="input">The input tensor.</param>
    public static void ExecuteWorker(this CommandBuffer cb, IWorker worker, Tensor input)
    {
        var backend = worker.GetBackend();
        Assert.IsTrue(backend is GPUCommandBufferBackend);
        (backend as GPUCommandBufferBackend).cb = cb;
        worker.Execute(input);
    }
}

/// <summary>
/// Provides methods for instantiating workers and ops on given back ends.
/// </summary>
public class WorkerFactory
{
    /// <summary>
    /// Represents the configuration for a `WorkerFactory`.
    /// </summary>
    public struct WorkerConfiguration
    {
        /// <summary>
        /// If true the worker is allowed to take ownership of the weights memory from the model
        /// this is useful so worker to limit memory pressure when the worker need to copy those
        /// weight to a different device.
        /// </summary>
        public bool takeoverWeights;

        /// <summary>
        /// Initializes and returns an instance of `WorkerConfiguration`.
        /// </summary>
        /// <param name="takeoverWeights">Whether to allow the worker to take ownership of the model weights memory.</param>
        public WorkerConfiguration(bool takeoverWeights = false)
        {
            this.takeoverWeights = takeoverWeights;
        }
    }

    /// <summary>
    /// Initializes and returns an instance of `IBackend` for a backend type..
    /// </summary>
    /// <param name="backendType">The type of backend to use.</param>
    /// <returns>The created `IBackend` instance.</returns>
    public static IBackend CreateBackend(BackendType backendType)
    {
        return BackendFactory.CreateBackend(backendType);
    }

    /// <summary>
    /// Initializes and returns an instance of `IWorker` on a given back end with a `model` to execute and `workerConfiguration`.
    /// </summary>
    /// <param name="backendType">The type of backend to use.</param>
    /// <param name="model">The model to execute with this `IWorker`.</param>
    /// <param name="workerConfiguration">The worker configuration to use when executing.</param>
    /// <returns>The created `IWorker` instance.</returns>
    public static IWorker CreateWorker(BackendType backendType, Model model, WorkerConfiguration workerConfiguration)
    {
        return BackendFactory.CreateWorker(backendType, model, workerConfiguration);
    }

    /// <summary>
    /// Initializes and returns an instance of `IWorker` on a given back end with a `model` to execute.
    /// </summary>
    /// <param name="backendType">The type of backend to use.</param>
    /// <param name="model">The model to execute with this `IWorker`.</param>
    /// <returns>The created `IWorker` instance.</returns>
    public static IWorker CreateWorker(BackendType backendType, Model model)
    {
        var workerConfiguration = new WorkerConfiguration();
        return CreateWorker(backendType, model, workerConfiguration);
    }

    /// <summary>
    /// Initializes and returns an instance of `IWorker` on a given device with a `model` to execute. Sentis selects the best backend type available for `deviceType`.
    /// </summary>
    /// <param name="deviceType">The type of device to use. Sentis selects the best backend type available for `deviceType`.</param>
    /// <param name="model">The model to execute with this `IWorker`.</param>
    /// <returns>The created `IWorker` instance.</returns>
    public static IWorker CreateWorker(Model model, DeviceType deviceType)
    {
        var type = GetBestTypeForDevice(deviceType);
        var workerConfiguration = new WorkerConfiguration();
        return CreateWorker(type, model, workerConfiguration);
    }

    /// <summary>
    /// Checks if a backend type matches a device type. For example, `IsType(BackendType.GPUCompute, DeviceType.GPU)` returns `true`.
    /// </summary>
    /// <param name="backendType">The backend type to check.</param>
    /// <param name="deviceType">The device type to check.</param>
    /// <returns>Whether the backend type matches the device type.</returns>
    public static bool IsType(BackendType backendType, DeviceType deviceType)
    {
        return ((int)backendType & (int)deviceType) == (int)deviceType;
    }

    /// <summary>
    /// Returns the best backend type for the given `deviceType`.
    /// </summary>
    /// <param name="deviceType">The device type.</param>
    /// <returns>The selected backend type for the device type.</returns>
    public static BackendType GetBestTypeForDevice(DeviceType deviceType)
    {
        switch (deviceType)
        {
            case DeviceType.GPU:
                if (SystemInfo.supportsComputeShaders && ComputeInfo.supportsCompute)
                {
                    return BackendType.GPUCompute;
                }
                else
                {
                    return BackendType.GPUPixel;
                }
            default:
                return BackendType.CPU;
        }
    }
}

} // namespace Unity.Sentis
