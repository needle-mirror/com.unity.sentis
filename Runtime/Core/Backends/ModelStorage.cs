using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

using System.Runtime.CompilerServices;
using Unity.Sentis.Compiler.Analyser;
using Unity.Profiling;

[assembly: InternalsVisibleTo("Unity.Sentis.PerformanceTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.Tests")]

namespace Unity.Sentis
{
/// <summary>
/// Handles allocation/dispose scheme for a given model
///
/// Usage:
/// 1. Setup allocation scheme: PrepareStorage
/// 2. Pass user inputs: SetInput
/// 3. Prepare before Execute: DisposeOnExecute
/// 3. execute worker:
///       foreach layer:
///         GetTensor(layer.input[0]) -> fetches ressources computed earlier
///         AllocateTensor -> allocs ressources (
///         Store -> gives ownership back to var
///         DisposeAfterLayer(layer) -> releases to the re-use pool any upstream ressources that only depends on layer
///
/// * PrepareStorage:
///  Computes optimal allocation scheme for the model
///    - which layer needs to be disposed before Execute
///      model outputs, un-connected layer outputs needs to be flushed on every successive Execute
///      call DisposeOnExecute to release said tensor to pool
///    - when a layer can be disposed
///      each layer is necessarily used as input by N layers
///      onces the deepest of these downstream layers have been reached, we know that this layer will not be used so we can safely dispose of it
///      algorithm consist in finding foreach layer the deepest downstream layer so we know when we can dispose of it
///
/// * TensorsByName
/// ModelStorageWithReuse stored all intermediate tensors in a in-use dict
/// Once no longer needed, those tensors are removed from the in-use dict and given back to the reuse pool
///
/// * GetTensor / StoreTensor
/// gets/append a tensor from the in-use pool
///
/// * AllocateTensor/DisposeTensor
/// Try to re-use a tensor in the re-use pool, if not allocate a new one
/// Gives a tensor to the re-use pool and removes it from the in-use pool
///
/// * DisposeAfterLayer
/// Dumps all tensors who's deepest dependency where the given layer
/// Sets all tensor given back to the pool to null in m_InUseTensorsPool
/// </summary>
class ModelStorage : IModelStorage
{
    Dictionary<string, Tensor> m_InUseTensorsPool = new Dictionary<string, Tensor>(); // in-use layer.index -> tensor pool
    HashSet<string> m_TensorsNotOwned = new HashSet<string>(); // model.inputs + outputs which user has taken ownership
    Dictionary<string, List<string>> m_TensorsToDisposeWhenLayerDone = new Dictionary<string, List<string>>();
    HashSet<string> m_UnconectedTensors = new HashSet<string>(); // model.outputs + unconnected layer.outputs

    TensorPool m_TensorPool = new TensorPool(); // re-use tensor pool

    public ModelStorage() { }

    /// <inheritdoc/>
    public Tensor GetTensor(string index)
    {
        return m_InUseTensorsPool[index];
    }

    /// <inheritdoc/>
    public Tensor AllocateTensor(TensorShape shape, DataType dataType, BackendType backendType)
    {
        return m_TensorPool.NewTensor(shape, dataType, backendType);
    }

    /// <inheritdoc/>
    public void Store(string index, Tensor result)
    {
        m_InUseTensorsPool[index] = result;
    }

    /// <inheritdoc/>
    public Tensor AllocateTensorAndStore(string index, TensorShape shape, DataType dataType, BackendType backendType)
    {
        var tensor = AllocateTensor(shape, dataType, backendType);
        Store(index, tensor);
        return tensor;
    }

    /// <summary>
    /// Disposes of the worker and any associated memory.
    /// </summary>
    public void Dispose()
    {
        m_TensorPool.Dispose();
        foreach (var input in m_TensorsNotOwned)
            m_InUseTensorsPool.Remove(input);
        foreach (var tensor in m_InUseTensorsPool.Values)
        {
            if (tensor == null)
                continue;
            tensor.Dispose();
        }
        m_InUseTensorsPool.Clear();

        m_UnconectedTensors.Clear();
        m_TensorsNotOwned.Clear();
        m_TensorsToDisposeWhenLayerDone.Clear();
    }

    protected bool ValidateInputShapes(Model model, IDictionary<string, TensorShape> inputShapes)
    {
        var valid = true;
        foreach (var input in model.inputs)
        {
            if (m_InUseTensorsPool.TryGetValue(input.index, out var tensor))
            {
                if (tensor.dataType != input.dataType)
                    D.LogWarning($"Given input data type: {tensor.dataType} does not match that of model input tensor: {input.dataType} for input: {input.index}");
                model.ValidateInputTensorShape(input, tensor.shape);
                continue;
            }

            if (inputShapes != null && inputShapes.TryGetValue(input.index, out var tensorShape))
            {
                model.ValidateInputTensorShape(input, tensorShape);
                continue;
            }

            D.LogWarning("Global input is missing: " + input.index);
            valid = false;
        }
        return valid;
    }

    /// <inheritdoc/>
    public void SetInput(string index, Tensor X)
    {
        m_InUseTensorsPool[index] = X;
    }

    /// <inheritdoc/>
    public void PrepareStorage(Model model, bool takeoverWeights)
    {
        m_TensorsToDisposeWhenLayerDone.Clear();
        m_InUseTensorsPool.Clear();
        m_TensorsNotOwned.Clear();
        m_UnconectedTensors.Clear();
        m_TensorsNotOwned = new HashSet<string>(model.inputs.Select(i => i.index));

        // TODO<Allocator dynamic load constants from disk
        HashSet<string> constants = new HashSet<string>();
        for(var i = 0; i < model.constants.Count; i++)
        {
            // TODO<Allocator> consider moving constants in memory pool to get disposed
            var constant = model.constants[i];
            Tensor tensor = AllocatorUtils.AllocTensor(constant.dataType, constant.shape, constant.shape.length == 0 ? null : new BurstTensorData(constant.weights));

            m_InUseTensorsPool[constant.index] = tensor;
            constants.Add(constant.index);
            if (takeoverWeights)
                constant.weights = null;
        }

        m_UnconectedTensors = new HashSet<string>(model.inputs.Select(i => i.index));

        // For each layer we find the latest downstream layer that has said layer as input
        // ex:
        // 0 -> 1 -> 4 -> 5 -> 8
        //   -> 2 -> 3  /     |
        //   -> 7 ------------/
        // latestDownstreamLayer:
        //  0 -> 7, 1 -> 4, 2 -> 3, 4 -> 5, 5 -> 8, 7 -> 8
        Dictionary<string, string> latestDownstreamLayer = new Dictionary<string, string>();
        foreach (var layer in model.layers)
        {
            foreach (var input in layer.inputs)
            {
                if (string.IsNullOrEmpty(input) || constants.Contains(input)) // constants are kept around until dispose
                    continue;
                m_UnconectedTensors.Remove(input);
                latestDownstreamLayer[input] = layer.index;
            }
            m_UnconectedTensors.Add(layer.index);
            if (layer.outputs == null || layer.outputs.Length == 0)
                continue;
            foreach (var output in layer.outputs)
                m_UnconectedTensors.Add(output);
        }
        foreach (var output in model.outputs)
        {
            if (constants.Contains(output.index)) // constants are kept around until dispose
                continue;
            m_UnconectedTensors.Add(output.index);
        }

        // now that we have the latestDownstreamLayer, we inverse the map
        // and compute when we reach a layer, what layers can I delete
        // in this case
        // 3 -> [2], 4 -> [1], 5 -> [4,3] , 7 -> [0], 8 -> [5,7]
        foreach (var entry in latestDownstreamLayer)
        {
            string layerIndex = entry.Key;
            string downstreamLayer = entry.Value;
            if (m_UnconectedTensors.Contains(layerIndex) || m_TensorsNotOwned.Contains(layerIndex))
                continue;

            if (m_TensorsToDisposeWhenLayerDone.ContainsKey(downstreamLayer))
                m_TensorsToDisposeWhenLayerDone[downstreamLayer].Add(layerIndex);
            else
                m_TensorsToDisposeWhenLayerDone[downstreamLayer] = new List<string>() { layerIndex };
        }
    }

    /// <inheritdoc/>
    public void DisposeOnExecute()
    {
        foreach (var layer in m_UnconectedTensors)
        {
            // N.B: outputs are disposed here in case of successive execute. On first execute call tensorbyname will be empty
            if (!m_InUseTensorsPool.ContainsKey(layer))
                continue;

            var tensor = m_InUseTensorsPool[layer];
            m_TensorPool.Dispose(tensor);
            m_InUseTensorsPool[layer] = null;
        }
    }

    /// <inheritdoc/>
    public void DisposeAfterLayer(Layers.Layer layer)
    {
        if (!m_TensorsToDisposeWhenLayerDone.ContainsKey(layer.index))
            return;

        var toDispose = m_TensorsToDisposeWhenLayerDone[layer.index];
        foreach (var upstreamTensor in toDispose)
        {
            var tensor = m_InUseTensorsPool[upstreamTensor];
            m_TensorPool.Dispose(tensor);
            m_InUseTensorsPool[upstreamTensor] = null;
        }
    }

    /// <inheritdoc/>
    public void Dispose(Tensor tensor)
    {
        m_TensorPool.Dispose(tensor);
    }

    /// <inheritdoc/>
    public Tensor PeekTensor(string index)
    {
        Tensor tensor;
        if (!m_InUseTensorsPool.TryGetValue(index, out tensor))
            D.LogWarning("ModelStorage missing variable: " + index);

        return tensor;
    }

    /// <inheritdoc/>
    public Tensor TakeTensorOwnership(string index)
    {
        Tensor tensor;
        if (!m_InUseTensorsPool.Remove(index, out tensor))
            D.LogWarning("ModelStorage missing variable: " + index);

        return tensor;
    }
}

} // namespace Unity.Sentis
