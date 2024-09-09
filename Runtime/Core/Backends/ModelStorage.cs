using System;
using System.Collections.Generic;
using System.Linq;

using UnityEngine;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Unity.Sentis.PerformanceTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.RuntimeTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

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
        Dictionary<int, Tensor> m_InUseTensorsPool = new Dictionary<int, Tensor>(); // in-use layer.index -> tensor pool
        HashSet<int> m_TensorsNotOwned = new HashSet<int>(); // model.inputs + outputs which user has taken ownership
        Dictionary<int, List<int>> m_TensorsToDisposeWhenLayerDone = new Dictionary<int, List<int>>();
        HashSet<int> m_UnconnectedTensors = new HashSet<int>(); // model.outputs + unconnected layer.outputs

        TensorPool m_TensorPool = new TensorPool(); // re-use tensor pool

        public ModelStorage() { }

        /// <inheritdoc/>
        public Tensor GetTensor(int index)
        {
            if (index == -1)
                return null;
            return m_InUseTensorsPool[index];
        }

        public TensorShape GetTensorShape(int tensorIndex)
        {
            return m_InUseTensorsPool[tensorIndex].shape;
        }

        public DataType GetDataType(int tensorIndex)
        {
            return m_InUseTensorsPool[tensorIndex].dataType;
        }

        /// <inheritdoc/>
        public int GetInt(int tensorIndex, int defaultValue)
        {
            var tensor = GetTensor(tensorIndex) as Tensor<int>;
            if (tensor is null)
                return defaultValue;

            if (tensor.dataOnBackend is CPUTensorData readableTensorData && readableTensorData.IsReadbackRequestDone())
                return readableTensorData.array.Get<int>(0);

            //D.LogWarning($"Tensor {tensorIndex} needs to be read on the CPU however is on the {tensor.backendType} backend, Sentis will download the tensor data which may be slow.");
            return tensor.dataOnBackend.Download<int>(1)[0];
        }

        /// <inheritdoc/>
        public float GetFloat(int tensorIndex, float defaultValue)
        {
            var tensor = GetTensor(tensorIndex) as Tensor<float>;
            if (tensor is null)
                return defaultValue;

            if (tensor.dataOnBackend is CPUTensorData readableTensorData && readableTensorData.IsReadbackRequestDone())
                return readableTensorData.array.Get<float>(0);

            //D.LogWarning($"Tensor {tensorIndex} needs to be read on the CPU however is on the {tensor.backendType} backend, Sentis will download the tensor data which may be slow.");
            return tensor.dataOnBackend.Download<float>(1)[0];
        }

        /// <inheritdoc/>
        public ReadOnlySpan<int> GetInts(int tensorIndex, ReadOnlySpan<int> defaultValue)
        {
            var tensor = GetTensor(tensorIndex) as Tensor<int>;
            if (tensor is null)
                return defaultValue;

            if (tensor.dataOnBackend.maxCapacity == 0)
                return ReadOnlySpan<int>.Empty;

            if (tensor.dataOnBackend is CPUTensorData readableTensorData && readableTensorData.IsReadbackRequestDone())
                return readableTensorData.array.AsReadOnlySpan<int>(tensor.shape.length);

            //D.LogWarning($"Tensor {tensorIndex} needs to be read on the CPU however is on the {tensor.backendType} backend, Sentis will download the tensor data which may be slow.");
            return tensor.dataOnBackend.Download<int>(tensor.shape.length).AsReadOnlySpan();
        }

        /// <inheritdoc/>
        public ReadOnlySpan<float> GetFloats(int tensorIndex, ReadOnlySpan<float> defaultValue)
        {
            var tensor = GetTensor(tensorIndex) as Tensor<float>;
            if (tensor is null)
                return defaultValue;

            if (tensor.dataOnBackend.maxCapacity == 0)
                return ReadOnlySpan<float>.Empty;

            if (tensor.dataOnBackend is CPUTensorData readableTensorData && readableTensorData.IsReadbackRequestDone())
                return readableTensorData.array.AsReadOnlySpan<float>(tensor.shape.length);

            //D.LogWarning($"Tensor {tensorIndex} needs to be read on the CPU however is on the {tensor.backendType} backend, Sentis will download the tensor data which may be slow.");
            return tensor.dataOnBackend.Download<float>(tensor.shape.length).AsReadOnlySpan();
        }

        /// <inheritdoc/>
        public Tensor AllocateTensor(TensorShape shape, DataType dataType, BackendType backendType)
        {
            return m_TensorPool.NewTensor(shape, dataType, backendType);
        }

        /// <inheritdoc/>
        public void Store(int index, Tensor result)
        {
            if (index == -1)
                return;
            m_InUseTensorsPool[index] = result;
        }

        /// <inheritdoc/>
        public Tensor AllocateTensorAndStore(int index, TensorShape shape, DataType dataType, BackendType backendType)
        {
            if (index == -1)
                return null;
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

            m_UnconnectedTensors.Clear();
            m_TensorsNotOwned.Clear();
            m_TensorsToDisposeWhenLayerDone.Clear();
        }

        protected bool ValidateInputShapes(Model model, IDictionary<int, TensorShape> inputShapes)
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
        public void SetInput(int index, Tensor X)
        {
            m_InUseTensorsPool[index] = X;
        }

        /// <inheritdoc/>
        public void PrepareStorage(Model model, bool takeoverWeights)
        {
            m_TensorsToDisposeWhenLayerDone.Clear();
            m_InUseTensorsPool.Clear();
            m_TensorsNotOwned.Clear();
            m_UnconnectedTensors.Clear();
            m_TensorsNotOwned = new HashSet<int>(model.inputs.Select(i => i.index));

            // TODO<Allocator dynamic load constants from disk
            HashSet<int> constants = new HashSet<int>();
            for(var i = 0; i < model.constants.Count; i++)
            {
                // TODO<Allocator> consider moving constants in memory pool to get disposed
                var constant = model.constants[i];
                Tensor tensor = AllocatorUtils.AllocTensor(constant.dataType, constant.shape, new CPUTensorData(constant.weights));

                m_InUseTensorsPool[constant.index] = tensor;
                constants.Add(constant.index);
                if (takeoverWeights)
                    constant.m_Weights = null;
            }

            m_UnconnectedTensors = new HashSet<int>(model.inputs.Select(i => i.index));

            // For each layer we find the latest downstream layer that has said layer as input
            // ex:
            // 0 -> 1 -> 4 -> 5 -> 8
            //   -> 2 -> 3  /     |
            //   -> 7 ------------/
            // latestDownstreamLayer:
            //  0 -> 7, 1 -> 4, 2 -> 3, 4 -> 5, 5 -> 8, 7 -> 8
            Dictionary<int, int> latestDownstreamLayer = new Dictionary<int, int>();
            foreach (var layer in model.layers)
            {
                foreach (var input in layer.inputs)
                {
                    if ((input == -1) || constants.Contains(input)) // constants are kept around until dispose
                        continue;
                    m_UnconnectedTensors.Remove(input);
                    latestDownstreamLayer[input] = layer.outputs[0];
                }
                foreach (var output in layer.outputs)
                    m_UnconnectedTensors.Add(output);
            }
            foreach (var output in model.outputs)
            {
                if (constants.Contains(output.index)) // constants are kept around until dispose
                    continue;
                m_UnconnectedTensors.Add(output.index);
            }

            // now that we have the latestDownstreamLayer, we inverse the map
            // and compute when we reach a layer, what layers can I delete
            // in this case
            // 3 -> [2], 4 -> [1], 5 -> [4,3] , 7 -> [0], 8 -> [5,7]
            foreach (var entry in latestDownstreamLayer)
            {
                int layerIndex = entry.Key;
                int downstreamLayer = entry.Value;
                if (m_UnconnectedTensors.Contains(layerIndex) || m_TensorsNotOwned.Contains(layerIndex))
                    continue;

                if (m_TensorsToDisposeWhenLayerDone.ContainsKey(downstreamLayer))
                    m_TensorsToDisposeWhenLayerDone[downstreamLayer].Add(layerIndex);
                else
                    m_TensorsToDisposeWhenLayerDone[downstreamLayer] = new List<int>() { layerIndex };
            }

            // Remove inputs from m_UnconnectedTensors so they don't get disposed
            foreach (var input in model.inputs)
            {
                m_UnconnectedTensors.Remove(input.index);
            }
        }

        /// <inheritdoc/>
        public void DisposeOnExecute()
        {
            foreach (var layer in m_UnconnectedTensors)
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
        public void DisposeAfterLayer(Layer layer)
        {
            if (!m_TensorsToDisposeWhenLayerDone.ContainsKey(layer.outputs[0]))
                return;

            var toDispose = m_TensorsToDisposeWhenLayerDone[layer.outputs[0]];
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
        public Tensor PeekTensor(int index)
        {
            Tensor tensor;
            if (!m_InUseTensorsPool.TryGetValue(index, out tensor))
                D.LogWarning("ModelStorage missing variable: " + index);

            return tensor;
        }

        /// <inheritdoc/>
        public Tensor TakeTensorOwnership(int index)
        {
            Tensor tensor;
            if (!m_InUseTensorsPool.Remove(index, out tensor))
                D.LogWarning("ModelStorage missing variable: " + index);

            return tensor;
        }
    }
}
