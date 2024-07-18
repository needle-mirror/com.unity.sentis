using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Unity.Sentis.PerformanceTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.RuntimeTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis
{
    /// <summary>
    /// Represents a generic implementation of an <see cref="IWorker"/>.
    /// </summary>
    public class GenericWorker : IWorker
    {
        Model m_Model;
        Dictionary<string, TensorShape> m_InputShapes = new Dictionary<string, TensorShape>();
        Dictionary<string, int> m_InputIndexes = new Dictionary<string, int>();
        Dictionary<string, int> m_OutputIndexes = new Dictionary<string, int>();

        IBackend m_Backend;
        IModelStorage m_Storage;
        CPUBackend m_FallbackBackend;
        HashSet<int> m_LayerCPUFallback;

        float m_Progress = 0f;

        /// <summary>
        /// Initializes and returns an instance of `GenericWorker` for the specified `model` and `ops`.
        /// </summary>
        /// <param name="model">The model to execute.</param>
        /// <param name="backend">The backend to use for execution.</param>
        /// <param name="storage">The stored tensor variables to use for execution.</param>
        /// <param name="takeoverWeights">Whether to allow the worker to take ownership of the model weights during execution.</param>
        public GenericWorker(Model model, IBackend backend, IModelStorage storage, bool takeoverWeights = false)
        {
            m_Model = model;

            foreach (var input in model.inputs)
                m_InputIndexes[input.name] = input.index;
            foreach (var output in model.outputs)
                m_OutputIndexes[output.name] = output.index;
            m_Storage = storage;
            m_Backend = backend;
            if (m_Backend is CPUBackend)
                m_FallbackBackend = (m_Backend as CPUBackend);
            else
                m_FallbackBackend = new CPUBackend();

            m_LayerCPUFallback = CPUFallbackCalculator.Calculate(model, backend.backendType);

            m_Storage.PrepareStorage(m_Model, takeoverWeights);
        }

        /// <summary>
        /// Finalizes the `GenericWorker`.
        /// </summary>
        ~GenericWorker()
        {
            Dispose();
        }

        /// <summary>
        /// Gets the backend used by the worker for execution.
        /// </summary>
        /// <returns>The backend used for execution.</returns>
        public IBackend GetBackend() { return m_Backend; }

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

        /// <inheritdoc/>
        public void SetInput(string name, Tensor x)
        {
            // TODO<execute> bring back shape assert
            m_Storage.SetInput(m_InputIndexes[name], x);
        }

        /// <inheritdoc/>
        public IWorker Execute(IDictionary<string, Tensor> inputs)
        {
            foreach (var i in m_Model.inputs)
            {
                m_Storage.SetInput(i.index, inputs[i.name]);
            }
            return Execute();
        }

        /// <inheritdoc/>
        public IWorker Execute(Tensor input)
        {
            m_Storage.SetInput(m_Model.inputs[0].index, input);
            return Execute();
        }

        /// <inheritdoc/>
        public IWorker Execute()
        {
            ProfilerMarkers.Execute.Begin();

            m_Storage.DisposeOnExecute();

            ExecutionContext ctx = new ExecutionContext
            {
                storage = m_Storage,
                cpuBackend = m_FallbackBackend,
            };

            int idx = 0;
            foreach (var l in m_Model.layers)
            {
                idx++;
                m_Progress = idx / (float)m_Model.layers.Count;

                ctx.backend = m_Backend;
                if (m_LayerCPUFallback.Contains(l.outputs[0]))
                    ctx.backend = m_FallbackBackend;

                var markerType = ProfilerMarkers.LayerTypeProfilerMarker(l.profilerTag);
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

            ProfilerMarkers.Execute.End();
            return this;
        }

        /// <inheritdoc/>
        public IEnumerator ExecuteLayerByLayer(IDictionary<string, Tensor> inputs)
        {
            foreach (var entry in inputs)
                SetInput(entry.Key, entry.Value);
            return ExecuteLayerByLayer();
        }

        /// <inheritdoc/>
        public IEnumerator ExecuteLayerByLayer(Tensor input)
        {
            m_Storage.SetInput(m_Model.inputs[0].index, input);
            return ExecuteLayerByLayer();
        }

        /// <inheritdoc/>
        public float scheduleProgress => m_Progress;

        /// <inheritdoc/>
        public IEnumerator ExecuteLayerByLayer()
        {
            ProfilerMarkers.Execute.Begin();

            m_Storage.DisposeOnExecute();

            ExecutionContext ctx = new ExecutionContext
            {
                storage = m_Storage,
                cpuBackend = m_FallbackBackend
            };

            int idx = 0;
            foreach (var l in m_Model.layers)
            {
                idx++;

                m_Progress = idx / (float)m_Model.layers.Count;

                ctx.backend = m_Backend;
                bool cpuLayer = m_LayerCPUFallback.Contains(l.outputs[0]);
                if (cpuLayer)
                    ctx.backend = m_FallbackBackend;

                var markerType = ProfilerMarkers.LayerTypeProfilerMarker(l.profilerTag);
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
                    ProfilerMarkers.Execute.End();
                    yield return null;
                    ProfilerMarkers.Execute.Begin();
                }
            }

            ProfilerMarkers.Execute.End();
        }

        /// <inheritdoc/>
        public Tensor PeekOutput()
        {
            return m_Storage.PeekTensor(m_Model.outputs[0].index);
        }

        /// <inheritdoc/>
        public Tensor PeekOutput(string name)
        {
            return m_Storage.PeekTensor(m_OutputIndexes[name]);
        }

        /// <inheritdoc/>
        public Tensor TakeOutputOwnership()
        {
            return m_Storage.TakeTensorOwnership(m_Model.outputs[0].index);
        }

        /// <inheritdoc/>
        public Tensor TakeOutputOwnership(string name)
        {
            return m_Storage.TakeTensorOwnership(m_OutputIndexes[name]);
        }
    }
}
