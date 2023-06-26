using System;
using System.Collections.Generic;

namespace Unity.Sentis
{
    class ShapeInferenceContext : IDisposable
    {
        Dictionary<string, SymbolicTensorShape> m_SymbolicTensorShapes;
        Dictionary<string, PartialTensor> m_PartialTensors;
        Dictionary<string, Tensor> m_KnownTensors;

        public ShapeInferenceContext(ITensorAllocator allocator = null)
        {
            m_SymbolicTensorShapes = new Dictionary<string, SymbolicTensorShape>();
            m_PartialTensors = new Dictionary<string, PartialTensor>();
            m_KnownTensors = new Dictionary<string, Tensor>();
        }

        public Dictionary<string, SymbolicTensorShape> SymbolicTensorShapes => m_SymbolicTensorShapes;
        public Dictionary<string, PartialTensor> PartialTensors => m_PartialTensors;
        public Dictionary<string, Tensor> KnownTensors => m_KnownTensors;

        public void AddShape(string name, SymbolicTensorShape shape)
        {
            if (m_SymbolicTensorShapes.TryGetValue(name, out var prevShape))
                shape = MaxDefinedTensorShape(shape, prevShape);
            m_SymbolicTensorShapes[name] = shape;
        }

        public void AddShape(string name, TensorShape t)
        {
            AddShape(name, new SymbolicTensorShape(t));
        }

        /// <summary>
        /// Returns a symbolic shape with the most known rank and dims from two
        /// given shapes that are known to be equal. Asserts if the shapes cannot be equal
        /// </summary>
        static SymbolicTensorShape MaxDefinedTensorShape(SymbolicTensorShape a, SymbolicTensorShape b)
        {
            if (!a.hasRank)
                return b;
            if (!b.hasRank)
                return a;
            Logger.AssertIsTrue(a.rank == b.rank, "InputError: incompatible tensor shapes");
            var shapeOut = SymbolicTensorShape.UnknownOfRank(a.rank);
            for (var i = 0; i < shapeOut.rank; i++)
            {
                shapeOut[i] = SymbolicTensorDim.MaxDefinedDim(a[i], b[i]);
            }

            return shapeOut;
        }

        public SymbolicTensorShape GetSymbolicTensorShape(string name)
        {
            if (string.IsNullOrEmpty(name))
                return SymbolicTensorShape.UnknownShape;
            return m_SymbolicTensorShapes.TryGetValue(name, out var symbolicTensorShape) ? symbolicTensorShape : SymbolicTensorShape.UnknownShape;
        }

        public SymbolicTensorShape[] GetShapes(string[] names)
        {
            var shapes = new SymbolicTensorShape[names.Length];

            for (var i = 0; i < shapes.Length; i++)
            {
                shapes[i] = GetSymbolicTensorShape(names[i]);
            }

            return shapes;
        }

        public void AddPartialTensor(string name, PartialTensor partialTensor, bool isTryAddFullTensor = true)
        {
            if (m_PartialTensors.TryGetValue(name, out var prevPartialTensor))
                partialTensor = MaxDefinedPartialTensor(partialTensor, prevPartialTensor);
            m_PartialTensors[name] = partialTensor;
            if (isTryAddFullTensor && partialTensor.IsFullyKnown())
                m_KnownTensors[name] = partialTensor.ToTensorInt();
        }

        public PartialTensor[] GetPartialTensors(string[] names)
        {
            var partialTensors = new PartialTensor[names.Length];

            for (var i = 0; i < partialTensors.Length; i++)
            {
                partialTensors[i] = GetPartialTensor(names[i]);
            }

            return partialTensors;
        }

        /// <summary>
        /// Returns a partial tensor with the most known elements from two
        /// given tensors that are known to be equal. Asserts if the tensors cannot be equal
        /// </summary>
        static PartialTensor MaxDefinedPartialTensor(PartialTensor a, PartialTensor b)
        {
            if (!a.isPartiallyKnown)
                return b;
            if (!b.isPartiallyKnown)
                return a;
            Logger.AssertIsTrue(a.shape == b.shape, "InputError: incompatible tensors");
            var tensorOut = new PartialTensor(a.shape);
            for (var i = 0; i < tensorOut.shape.length; i++)
            {
                tensorOut[i] = PartialTensorElement.MaxDefinedElement(a[i], b[i]);
            }

            return tensorOut;
        }

        public PartialTensor GetPartialTensor(string name)
        {
            if (string.IsNullOrEmpty(name))
                return PartialTensor.Unknown;
            return m_PartialTensors.TryGetValue(name, out var partialTensor) ? partialTensor : PartialTensor.Unknown;
        }

        public Tensor GetKnownTensor(string name)
        {
            if (string.IsNullOrEmpty(name))
                return null;
            return m_KnownTensors.TryGetValue(name, out var knownTensor) ? knownTensor : null;
        }

        public void AddKnownTensor(string name, Tensor tensor)
        {
            m_KnownTensors[name] = tensor;
            AddShape(name, tensor.shape);
            var partialTensor = PartialTensor.FromTensor(tensor);
            AddPartialTensor(name, partialTensor, false);
        }

        public bool TryGetKnownTensors(string[] names, out Tensor[] tensors)
        {
            tensors = new Tensor[names.Length];

            for (var i = 0; i < names.Length; i++)
            {
                if (string.IsNullOrEmpty(names[i]))
                    continue;
                if (m_KnownTensors.TryGetValue(names[i], out var tensor))
                    tensors[i] = tensor;
                else
                    return false;
            }

            return true;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            foreach (var tensor in m_KnownTensors.Values)
            {
                tensor.Dispose();
            }
        }
    }
}
