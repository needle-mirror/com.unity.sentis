using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine.Assertions;
using UnityEngine.Device;
using UnityEngine.Rendering;

[assembly: InternalsVisibleTo("Unity.Sentis.PerformanceTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.RuntimeTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis
{
    /// <summary>
    /// Represents a worker that allows you to execute neural networks (models).
    ///
    /// `Worker` abstracts implementation details on different hardware devices such as the CPU and the GPU. `Worker` lets you do the following:
    ///
    /// - Specify inputs.
    /// - Schedule the work.
    /// - Get outputs.
    ///
    /// Internally, `Worker` translates the neural network from a <see cref="Model"/> into a set of operations, then sends the operations to the hardware device for asynchronous execution.
    /// </summary>
    public class Worker : IDisposable
    {
        Model m_Model;
        Dictionary<string, TensorShape> m_InputShapes = new Dictionary<string, TensorShape>();
        Dictionary<string, int> m_InputIndexes = new Dictionary<string, int>();
        Dictionary<string, int> m_OutputIndexes = new Dictionary<string, int>();

        IBackend m_Backend;
        IModelStorage m_Storage;
        CPUBackend m_FallbackBackend;
        HashSet<int> m_LayerCPUFallback;

        /// <summary>
        /// Returns the backend type of the worker.
        /// </summary>
        public BackendType backendType => m_Backend.backendType;

        /// <summary>
        /// Initializes and returns an instance of `Worker` for the specified `model` and `ops`.
        /// </summary>
        /// <param name="model">The model to execute.</param>
        /// <param name="backend">The backend to use for execution.</param>
        /// <param name="storage">The stored tensor variables to use for execution.</param>
        /// <param name="takeoverWeights">Whether to allow the worker to take ownership of the model weights during execution.</param>
        internal Worker(Model model, IBackend backend, IModelStorage storage, bool takeoverWeights = false)
        {
            m_Model = model;

            foreach (var input in model.inputs)
                m_InputIndexes[input.name] = input.index;
            foreach (var output in model.outputs)
                m_OutputIndexes[output.name] = output.index;
            m_Storage = storage;
            m_Backend = backend;
            if (m_Backend is CPUBackend cpuBackend)
                m_FallbackBackend = cpuBackend;
            else
                m_FallbackBackend = new CPUBackend();

            m_LayerCPUFallback = CPUFallbackCalculator.Calculate(model, backend.backendType);

            m_Storage.PrepareStorage(m_Model, takeoverWeights);
        }

        /// <summary>
        /// Returns the best backend type for the given `deviceType`.
        /// </summary>
        /// <param name="deviceType">The device type.</param>
        /// <returns>The selected backend type for the device type.</returns>
        static BackendType GetBestTypeForDevice(DeviceType deviceType)
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

        /// <summary>
        /// Creates and initializes a Worker with a model on a backend type.
        /// </summary>
        /// <param name="model">The model for the worker.</param>
        /// <param name="backendType">The backend type for the worker.</param>
        public Worker(Model model, BackendType backendType)
            : this(model, BackendFactory.CreateBackend(backendType), new ModelStorage()) { }

        /// <summary>
        /// Creates and initializes a Worker with a model on a backend type.
        ///
        /// The fastest backend type for the device type will be chosen.
        /// </summary>
        /// <param name="model">The model for the worker.</param>
        /// <param name="deviceType">The device type for the worker.</param>
        public Worker(Model model, DeviceType deviceType)
            : this(model, GetBestTypeForDevice(deviceType)) { }

        /// <summary>
        /// Finalizes the `Worker`.
        /// </summary>
        ~Worker()
        {
            Dispose();
        }

        /// <summary>
        /// Gets the backend used by the worker for execution.
        /// </summary>
        /// <returns>The backend used for execution.</returns>
        internal IBackend GetBackend() { return m_Backend; }

        /// <summary>
        /// Disposes of the worker and any associated memory.
        /// </summary>
        public void Dispose()
        {
            m_Storage?.Dispose();
            m_Storage = null;
            m_Backend?.Dispose();
            m_Backend = null;
            m_FallbackBackend?.Dispose();
            m_FallbackBackend = null;
            m_InputShapes?.Clear();
            m_InputShapes = null;
        }

        /// <summary>
        /// Sets an input tensor with an index.
        /// </summary>
        /// <param name="index">The index of the input to set.</param>
        /// <param name="input">The input tensor.</param>
        public void SetInput(int index, Tensor input)
        {
            Logger.AssertIsTrue(index < m_Model.inputs.Count, "Cannot set input tensor at index {0} as model only contains {1} inputs", index, m_Model.inputs.Count);
            var modelInput = m_Model.inputs[index];
            Logger.AssertIsTrue(modelInput.dataType == input.dataType, "Cannot set input tensor {0} as data types do not match, expected {1} received {2}", index, modelInput.dataType, input.dataType);
            Logger.AssertIsTrue(DynamicTensorShape.IsCompatible(modelInput.shape, input.shape), "Cannot set input tensor {0} as shapes are not compatible, expected {1} received {2}", index, modelInput.shape, input.shape);
            m_Storage.SetInput(index, input);
        }

        /// <summary>
        /// Sets an input tensor with a name.
        /// </summary>
        /// <param name="name">The name of the input to set.</param>
        /// <param name="input">The input tensor.</param>
        public void SetInput(string name, Tensor input)
        {
            var success = m_InputIndexes.TryGetValue(name, out var index);
            Logger.AssertIsTrue(success, "Cannot set input tensor {0} as model does not contain an input with that name", name);
            SetInput(index, input);
        }

        /// <summary>
        /// Sets input tensors in order.
        /// </summary>
        /// <param name="input">The input tensors.</param>
        void SetInputs(Tensor input)
        {
            SetInput(0, input);
        }

        /// <summary>
        /// Sets input tensors in order.
        /// </summary>
        /// <param name="inputs">The input tensors.</param>
        void SetInputs(params Tensor[] inputs)
        {
            for (var i = 0; i < inputs.Length; i++)
                SetInput(i, inputs[i]);
        }

        /// <summary>
        /// Sets a command buffer to the worker.
        /// </summary>
        /// <param name="cb">The command buffer.</param>
        internal void SetCommandBuffer(CommandBuffer cb)
        {
            Logger.AssertIsTrue(m_Backend is GPUComputeBackend, "Cannot set command buffer on backend type {0}", backendType);
            (m_Backend as GPUComputeBackend).SetCommandBuffer(cb);
        }

        /// <summary>
        /// Schedules the execution of the model on the worker. This is non-blocking.
        /// </summary>
        public void Schedule()
        {
            ProfilerMarkers.Schedule.Begin();

            m_Storage.DisposeOnExecute();

            ExecutionContext ctx = new ExecutionContext
            {
                storage = m_Storage,
                cpuBackend = m_FallbackBackend,
            };

            foreach (var l in m_Model.layers)
            {
                ctx.backend = m_Backend;
                if (m_LayerCPUFallback.Contains(l.outputs[0]))
                    ctx.backend = m_FallbackBackend;

                var markerType = l.profilerMarker;
                markerType.Begin();
#if SENTIS_DEBUG
                Profiler.BeginSample(l.index);
#endif
                l.Execute(ctx);
#if SENTIS_DEBUG
                Profiler.EndSample();
#endif
                markerType.End();

                m_Storage.DisposeAfterLayer(l);
            }

            // gpucompute: need to execute commandbuffer and flush.
            if (ctx.backend is GPUComputeBackend gpuBackend && gpuBackend.InternalCommandBuffer())
                gpuBackend.ExecuteCommandBufferAndClear();

            ProfilerMarkers.Schedule.End();
        }

        /// <summary>
        /// Schedules the execution of the model on the worker. This is non-blocking.
        /// </summary>
        /// <param name="input">The tensor to set to the default input of the model.</param>
        public void Schedule(Tensor input)
        {
            SetInputs(input);
            Schedule();
        }

        /// <summary>
        /// Schedules the execution of the model on the worker. This is non-blocking.
        /// </summary>
        /// <param name="inputs">The tensors to set.</param>
        public void Schedule(params Tensor[] inputs)
        {
            SetInputs(inputs);
            Schedule();
        }

        /// <summary>
        /// Schedules the execution of the model in parts. This is non-blocking.
        ///
        /// To schedule the execution of the next layer of the model, call `MoveNext` on the `IEnumerator` object this method returns.
        /// </summary>
        /// <returns>The `IEnumerator` for iterating the scheduling.</returns>
        public IEnumerator ScheduleIterable()
        {
            ProfilerMarkers.Schedule.Begin();

            m_Storage.DisposeOnExecute();

            ExecutionContext ctx = new ExecutionContext
            {
                storage = m_Storage,
                cpuBackend = m_FallbackBackend
            };

            foreach (var l in m_Model.layers)
            {
                ctx.backend = m_Backend;
                bool cpuLayer = m_LayerCPUFallback.Contains(l.outputs[0]);
                if (cpuLayer)
                    ctx.backend = m_FallbackBackend;

                var markerType = l.profilerMarker;
                markerType.Begin();
#if SENTIS_DEBUG
                Profiler.BeginSample(l.index);
#endif
                l.Execute(ctx);
#if SENTIS_DEBUG
                Profiler.EndSample();
#endif
                markerType.End();

                m_Storage.DisposeAfterLayer(l);

                if (!cpuLayer)
                {
                    // gpucompute: need to execute commandbuffer and flush.
                    if (ctx.backend is GPUComputeBackend gpuBackend && gpuBackend.InternalCommandBuffer())
                        gpuBackend.ExecuteCommandBufferAndClear();

                    ProfilerMarkers.Schedule.End();
                    yield return null;
                    ProfilerMarkers.Schedule.Begin();
                }
            }

            ProfilerMarkers.Schedule.End();
        }

        /// <summary>
        /// Schedules the execution of the model in parts. This is non-blocking.
        ///
        /// To schedule the execution of the next layer of the model, call `MoveNext` on the `IEnumerator` object this method returns.
        /// </summary>
        /// <param name="input">The tensor to set to the default input of the model.</param>
        /// <returns>The `IEnumerator` for iterating the scheduling.</returns>
        public IEnumerator ScheduleIterable(Tensor input)
        {
            SetInputs(input);
            return ScheduleIterable();
        }

        /// <summary>
        /// Schedules the execution of the model in parts. This is non-blocking.
        ///
        /// To schedule the execution of the next layer of the model, call `MoveNext` on the `IEnumerator` object this method returns.
        /// </summary>
        /// <param name="inputs">The tensors to set.</param>
        /// <returns>The `IEnumerator` for iterating the scheduling.</returns>
        public IEnumerator ScheduleIterable(params Tensor[] inputs)
        {
            SetInputs(inputs);
            return ScheduleIterable();
        }

        /// <summary>
        /// Returns a reference to the default output tensor. This is non-blocking.
        ///
        /// For models with more than one output this returns a reference to the first output tensor.
        ///
        /// The reference is valid only until you call `Execute()` or `Dispose()` on the worker.
        /// </summary>
        /// <returns>The output tensor reference.</returns>
        public Tensor PeekOutput()
        {
            return m_Storage.PeekTensor(m_Model.outputs[0].index);
        }

        /// <summary>
        /// Returns a reference an output tensor. This is non-blocking.
        ///
        /// The reference is valid only until you call `Execute()` or `Dispose()` on the worker.
        /// </summary>
        /// <param name="index">The index of the output tensor to peek.</param>
        /// <returns>The output tensor reference.</returns>
        public Tensor PeekOutput(int index)
        {
            Logger.AssertIsTrue(index < m_Model.outputs.Count, "Cannot peek output tensor at index {0} as model only contains {1} outputs", index, m_Model.outputs.Count);
            return m_Storage.PeekTensor(m_Model.outputs[index].index);
        }

        /// <summary>
        /// Returns a reference to an output tensor. This is non-blocking.
        ///
        /// The reference is valid only until you call `Execute()` or `Dispose()` on the worker.
        /// </summary>
        /// <param name="name">The name of the output tensor to peek.</param>
        /// <returns>The output tensor reference.</returns>
        public Tensor PeekOutput(string name)
        {
            var success = m_OutputIndexes.TryGetValue(name, out var index);
            Logger.AssertIsTrue(success, "Cannot peek output tensor {0} as model does not contain an output with that name", name);
            return m_Storage.PeekTensor(index);
        }

        /// <summary>
        /// Schedule a copy of the output tensor at an index into a tensor.
        ///
        /// If, the input tensor is null, Sentis will allocate a new one
        /// If not, the input tensor dataType must match the output and have large enough capacity for the output shape.
        ///
        /// Sentis reshapes the provided tensor if the shape does not match.
        /// </summary>
        /// <param name="index">The index of the output.</param>
        /// <param name="tensor">The tensor to copy the output into.</param>
        public void CopyOutput(int index, ref Tensor tensor)
        {
            var src = PeekOutput(index);
            if (tensor == null)
                tensor = src.CloneEmpty();
            else
            {
                Logger.AssertAreEqual(tensor.dataType, src.dataType, "Cannot copy to tensor as data types do not match, expected {0} received {1}", src.dataType, tensor.dataType);
                tensor.Reshape(src.shape);
            }
            m_Backend.MemCopy(src, tensor);
            if (m_Backend is GPUComputeBackend gpuBackend && gpuBackend.InternalCommandBuffer())
                gpuBackend.ExecuteCommandBufferAndClear();
        }

        /// <summary>
        /// Schedule a copy of the output tensor with a name into a tensor.
        ///
        /// If, the input tensor is null, Sentis will allocate a new one
        /// If not, the input tensor dataType must match the output and have large enough capacity for the output shape.
        ///
        /// Sentis reshapes the provided tensor if the shape does not match.
        /// </summary>
        /// <param name="name">The name of the output.</param>
        /// <param name="tensor">The tensor to copy the output into.</param>
        public void CopyOutput(string name, ref Tensor tensor)
        {
            var success = m_OutputIndexes.TryGetValue(name, out var index);
            Logger.AssertIsTrue(success, "Cannot copy output {0} as model does not contain an output with that name", name);
            var src = m_Storage.PeekTensor(index);
            if (tensor == null)
                tensor = src.CloneEmpty();
            else
            {
                Logger.AssertAreEqual(tensor.dataType, src.dataType, "Cannot copy to tensor as data types do not match, expected {0} received {1}", src.dataType, tensor.dataType);
                tensor.Reshape(src.shape);
            }
            m_Backend.MemCopy(src, tensor);
            if (m_Backend is GPUComputeBackend gpuBackend && gpuBackend.InternalCommandBuffer())
                gpuBackend.ExecuteCommandBufferAndClear();
        }
    }

    /// <summary>
    /// Provides extension methods for scheduling the worker from a CommandBuffer.
    /// </summary>
    public static class CommandBufferWorkerExtensions
    {
        /// <summary>
        /// Schedules the execution of the model on a worker using a command buffer. This is non-blocking.
        ///
        /// Call Graphics.ExecuteCommandBuffer to execute the command buffer after scheduling.
        /// </summary>
        /// <param name="cb">The command buffer.</param>
        /// <param name="worker">The worker.</param>
        public static void ScheduleWorker(this CommandBuffer cb, Worker worker)
        {
            worker.SetCommandBuffer(cb);
            worker.Schedule();
        }

        /// <summary>
        /// Schedules the execution of the model on a worker using a command buffer. This is non-blocking.
        ///
        /// Call Graphics.ExecuteCommandBuffer to execute the command buffer after scheduling.
        /// </summary>
        /// <param name="cb">The command buffer.</param>
        /// <param name="worker">The worker.</param>
        /// <param name="input">The tensor to set to the default input of the model.</param>
        public static void ScheduleWorker(this CommandBuffer cb, Worker worker, Tensor input)
        {
            worker.SetCommandBuffer(cb);
            worker.Schedule(input);
        }

        /// <summary>
        /// Schedules the execution of the model on a worker using a command buffer. This is non-blocking.
        ///
        /// Call Graphics.ExecuteCommandBuffer to execute the command buffer after scheduling.
        /// </summary>
        /// <param name="cb">The command buffer.</param>
        /// <param name="worker">The worker.</param>
        /// <param name="inputs">The tensors to set to the inputs of the model.</param>
        public static void ScheduleWorker(this CommandBuffer cb, Worker worker, params Tensor[] inputs)
        {
            worker.SetCommandBuffer(cb);
            worker.Schedule(inputs);
        }

        /// <summary>
        /// Schedules the execution of the model on a worker using a command buffer in parts. This is non-blocking.
        ///
        /// Call Graphics.ExecuteCommandBuffer to execute the command buffer after scheduling.
        ///
        /// To schedule the execution of the next layer of the model, call `MoveNext` on the `IEnumerator` object this method returns.
        /// </summary>
        /// <param name="cb">The command buffer.</param>
        /// <param name="worker">The worker.</param>
        /// <returns>The `IEnumerator` for iterating the scheduling.</returns>
        public static IEnumerator ScheduleWorkerIterable(this CommandBuffer cb, Worker worker)
        {
            worker.SetCommandBuffer(cb);
            return worker.ScheduleIterable();
        }

        /// <summary>
        /// Schedules the execution of the model on a worker using a command buffer in parts. This is non-blocking.
        ///
        /// Call Graphics.ExecuteCommandBuffer to execute the command buffer after scheduling.
        ///
        /// To schedule the execution of the next layer of the model, call `MoveNext` on the `IEnumerator` object this method returns.
        /// </summary>
        /// <param name="cb">The command buffer.</param>
        /// <param name="worker">The worker.</param>
        /// <param name="input">The tensor to set to the default input of the model.</param>
        /// <returns>The `IEnumerator` for iterating the scheduling.</returns>
        public static IEnumerator ScheduleWorkerIterable(this CommandBuffer cb, Worker worker, Tensor input)
        {
            worker.SetCommandBuffer(cb);
            return worker.ScheduleIterable(input);
        }

        /// <summary>
        /// Schedules the execution of the model on a worker using a command buffer in parts. This is non-blocking.
        ///
        /// Call Graphics.ExecuteCommandBuffer to execute the command buffer after scheduling.
        ///
        /// To schedule the execution of the next layer of the model, call `MoveNext` on the `IEnumerator` object this method returns.
        /// </summary>
        /// <param name="cb">The command buffer.</param>
        /// <param name="worker">The worker.</param>
        /// <param name="inputs">The tensors to set to the inputs of the model.</param>
        /// <returns>The `IEnumerator` for iterating the scheduling.</returns>
        public static IEnumerator ScheduleWorkerIterable(this CommandBuffer cb, Worker worker, params Tensor[] inputs)
        {
            worker.SetCommandBuffer(cb);
            return worker.ScheduleIterable(inputs);
        }
    }
}
