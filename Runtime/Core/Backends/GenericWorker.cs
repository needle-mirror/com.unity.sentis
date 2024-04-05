using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Unity.Sentis.PerformanceTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.Tests")]

namespace Unity.Sentis
{

/// <summary>
/// Represents a generic implementation of an <see cref="IWorker"/>.
/// </summary>
public class GenericWorker : IWorker
{
    Model m_Model;
    Dictionary<string, TensorShape> m_InputShapes = new Dictionary<string, TensorShape>();
    Dictionary<string, string> m_InputIndexes = new Dictionary<string, string>();
    Dictionary<string, string> m_OutputIndexes = new Dictionary<string, string>();

    IBackend m_Backend;
    IModelStorage m_Vars;
    CPUBackend m_FallbackBackend;
    HashSet<string> m_LayerCPUFallback;

    float m_Progress = 0f;

    /// <summary>
    /// Initializes and returns an instance of `GenericWorker` for the specified `model` and `ops`.
    /// </summary>
    /// <param name="model">The model to execute.</param>
    /// <param name="backend">The backend to use for execution.</param>
    /// <param name="vars">The stored tensor variables to use for execution.</param>
    /// <param name="takeoverWeights">Whether to allow the worker to take ownership of the model weights during execution.</param>
    public GenericWorker(Model model, IBackend backend, IModelStorage vars, bool takeoverWeights = false)
    {
        m_Model = model;
        foreach (var input in model.inputs)
            m_InputIndexes[input.name] = input.index;
        foreach (var output in model.outputs)
            m_OutputIndexes[output.name] = output.index;
        m_Vars = vars;
        m_Backend = backend;
        if (m_Backend is CPUBackend)
            m_FallbackBackend = (m_Backend as CPUBackend);
        else
            m_FallbackBackend = new CPUBackend();

        m_LayerCPUFallback = model.LayerCPUFallback;

        m_Vars.PrepareStorage(m_Model, takeoverWeights);
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
        m_Vars?.Dispose();
        m_Vars = null;
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
        m_Vars.SetInput(m_InputIndexes[name], x);
    }

    /// <inheritdoc/>
    public IWorker Execute(IDictionary<string, Tensor> inputs)
    {
        foreach (var input in inputs)
        {
            SetInput(input.Key, input.Value);
        }
        return Execute();
    }

    /// <inheritdoc/>
    public IWorker Execute(Tensor input)
    {
        m_Vars.SetInput(m_Model.inputs[0].index, input);
        return Execute();
    }

    /// <inheritdoc/>
    public IWorker Execute()
    {
        ProfilerMarkers.Execute.Begin();

        m_Vars.DisposeOnExecute();

        ExecutionContext ctx = new ExecutionContext();
        ctx.vars = m_Vars;

        int idx = 0;
        foreach (var l in m_Model.layers)
        {
            idx++;
            m_Progress = idx / (float)m_Model.layers.Count;

            ctx.backend = m_Backend;
            if (m_LayerCPUFallback.Contains(l.index))
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

            m_Vars.DisposeAfterLayer(l);
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
        m_Vars.SetInput(m_Model.inputs[0].index, input);
        return ExecuteLayerByLayer();
    }

    /// <inheritdoc/>
    public float scheduleProgress => m_Progress;

    /// <inheritdoc/>
    public IEnumerator ExecuteLayerByLayer()
    {
        ProfilerMarkers.Execute.Begin();

        m_Vars.DisposeOnExecute();

        ExecutionContext ctx = new ExecutionContext();
        ctx.vars = m_Vars;

        int idx = 0;
        foreach (var l in m_Model.layers)
        {
            idx++;

            m_Progress = idx / (float)m_Model.layers.Count;

            ctx.backend = m_Backend;
            bool cpuLayer = m_LayerCPUFallback.Contains(l.index);
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

            m_Vars.DisposeAfterLayer(l);

            if (!cpuLayer)
            {
                ProfilerMarkers.Execute.End();
                yield return null;
            }
        }

        ProfilerMarkers.Execute.End();
    }

    /// <inheritdoc/>
    public Tensor PeekOutput()
    {
        return m_Vars.PeekTensor(m_Model.outputs[0].index);
    }

    /// <inheritdoc/>
    public Tensor PeekOutput(string name)
    {
        return m_Vars.PeekTensor(m_OutputIndexes[name]);
    }

    /// <inheritdoc/>
    public Tensor TakeOutputOwnership()
    {
        return m_Vars.TakeTensorOwnership(m_Model.outputs[0].index);
    }

    /// <inheritdoc/>
    public Tensor TakeOutputOwnership(string name)
    {
        return m_Vars.TakeTensorOwnership(m_OutputIndexes[name]);
    }

    }
} // namespace Unity.Sentis
