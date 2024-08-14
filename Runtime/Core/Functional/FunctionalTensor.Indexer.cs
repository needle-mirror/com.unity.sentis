using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Unity.Sentis
{
    public partial class FunctionalTensor
    {
        void IndexerSet(FunctionalTensor src, IEnumerable<IndexOrRange> indexOrRanges)
        {
            var starts = new List<int>();
            var ends = new List<int>();
            var axes = new List<int>();
            var axis = 0;
            foreach (var indexOrRange in indexOrRanges)
            {
                if (!indexOrRange.IsRangeAll())
                {
                    starts.Add(indexOrRange.Start());
                    ends.Add(indexOrRange.End());
                    axes.Add(axis);
                }

                if (indexOrRange.IsIndex)
                    src = src.Unsqueeze(axis);

                axis++;
            }
            var c = Functional.FromLayer(new Layers.SliceSet(-1, -1, -1, -1, -1, -1, -1), Functional.CommonType(this, src), new[] { this, src, Functional.Constant(starts.ToArray()), Functional.Constant(ends.ToArray()), Functional.Constant(axes.ToArray()), null });
            m_DataType = c.dataType;
            m_Source = c.source;
            m_OutputIndex = c.outputIndex;
        }

        FunctionalTensor IndexerGet(IEnumerable<IndexOrRange> indexOrRanges)
        {
            var starts = new List<int>();
            var ends = new List<int>();
            var axes = new List<int>();
            var squeezeAxes = new List<int>();
            var axis = 0;
            foreach (var indexOrRange in indexOrRanges)
            {
                if (!indexOrRange.IsRangeAll())
                {
                    starts.Add(indexOrRange.Start());
                    ends.Add(indexOrRange.End());
                    axes.Add(axis);
                    if (indexOrRange.IsIndex)
                        squeezeAxes.Add(axis);
                }

                axis++;
            }

            if (starts.Count == 0)
                return this;
            var startsArray = starts.ToArray();
            var endsArray = ends.ToArray();
            var axesArray = axes.ToArray();
            var slice = Functional.FromLayer(new Layers.Slice(-1, -1, -1, -1, -1, -1), dataType, new[] { this, Functional.Constant(startsArray), Functional.Constant(endsArray), Functional.Constant(axesArray), null });
            if (isShapeKnown)
            {
                var numAxes = startsArray.Length;
                Span<int> startsSpan = stackalloc int[numAxes];
                Span<int> endsSpan = stackalloc int[numAxes];
                Span<int> axesSpan = stackalloc int[numAxes];
                Span<int> stepsSpan = stackalloc int[numAxes];
                ShapeInference.Slice(shape, startsArray, endsArray, axesArray, null, ref startsSpan, ref endsSpan, ref axesSpan, ref stepsSpan);
                var oShape = shape.Slice(startsSpan, endsSpan, axesSpan, stepsSpan);
                slice.SetShape(oShape);
            }
            if (squeezeAxes.Count > 0)
                slice = slice.Squeeze(squeezeAxes.ToArray());
            return slice;
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="indices">The indexes to index with.</param>
        public FunctionalTensor this[params Index[] indices]
        {
            get => IndexerGet(indices.Select(i => new IndexOrRange(i)));
            set => IndexerSet(value, indices.Select(i => new IndexOrRange(i)));
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="ranges">The ranges to index with.</param>
        public FunctionalTensor this[params Range[] ranges]
        {
            get => IndexerGet(ranges.Select(r => new IndexOrRange(r)));
            set => IndexerSet(value, ranges.Select(r => new IndexOrRange(r)));
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        public FunctionalTensor this[Index i0, Range i1]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        public FunctionalTensor this[Range i0, Index i1]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        public FunctionalTensor this[Index i0, Index i1, Range i2]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        public FunctionalTensor this[Index i0, Range i1, Index i2]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        public FunctionalTensor this[Index i0, Range i1, Range i2]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        public FunctionalTensor this[Range i0, Index i1, Index i2]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        public FunctionalTensor this[Range i0, Index i1, Range i2]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        public FunctionalTensor this[Range i0, Range i1, Index i2]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        /// <param name="i3">The fourth index.</param>
        public FunctionalTensor this[Index i0, Index i1, Index i2, Range i3]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        /// <param name="i3">The fourth index.</param>
        public FunctionalTensor this[Index i0, Index i1, Range i2, Index i3]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        /// <param name="i3">The fourth index.</param>
        public FunctionalTensor this[Index i0, Index i1, Range i2, Range i3]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        /// <param name="i3">The fourth index.</param>
        public FunctionalTensor this[Index i0, Range i1, Index i2, Index i3]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        /// <param name="i3">The fourth index.</param>
        public FunctionalTensor this[Index i0, Range i1, Index i2, Range i3]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        /// <param name="i3">The fourth index.</param>
        public FunctionalTensor this[Index i0, Range i1, Range i2, Index i3]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        /// <param name="i3">The fourth index.</param>
        public FunctionalTensor this[Index i0, Range i1, Range i2, Range i3]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        /// <param name="i3">The fourth index.</param>
        public FunctionalTensor this[Range i0, Index i1, Index i2, Index i3]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        /// <param name="i3">The fourth index.</param>
        public FunctionalTensor this[Range i0, Index i1, Index i2, Range i3]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        /// <param name="i3">The fourth index.</param>
        public FunctionalTensor this[Range i0, Index i1, Range i2, Index i3]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        /// <param name="i3">The fourth index.</param>
        public FunctionalTensor this[Range i0, Index i1, Range i2, Range i3]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        /// <param name="i3">The fourth index.</param>
        public FunctionalTensor this[Range i0, Range i1, Index i2, Index i3]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        /// <param name="i3">The fourth index.</param>
        public FunctionalTensor this[Range i0, Range i1, Index i2, Range i3]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
        }

        /// <summary>
        /// Indexes the functional tensor.
        /// </summary>
        /// <param name="i0">The first index.</param>
        /// <param name="i1">The second index.</param>
        /// <param name="i2">The third index.</param>
        /// <param name="i3">The fourth index.</param>
        public FunctionalTensor this[Range i0, Range i1, Range i2, Index i3]
        {
            get => IndexerGet(new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
            set => IndexerSet(value, new[] { new IndexOrRange(i0), new IndexOrRange(i1), new IndexOrRange(i2), new IndexOrRange(i3) });
        }
    }

    struct IndexOrRange
    {
        enum IndexOrRangeType
        {
            Index,
            Range
        }

        IndexOrRangeType m_Type;
        Index m_Index;
        Range m_Range;

        public IndexOrRange(Index index)
        {
            m_Type = IndexOrRangeType.Index;
            m_Index = index;
            m_Range = default;
        }

        public IndexOrRange(Range range)
        {
            m_Type = IndexOrRangeType.Range;
            m_Index = default;
            m_Range = range;
        }

        public bool IsIndex => m_Type == IndexOrRangeType.Index;

        public int Start()
        {
            return m_Type switch
            {
                IndexOrRangeType.Index => m_Index.IsFromEnd ? m_Index.Value == 0 ? int.MaxValue : -m_Index.Value : m_Index.Value,
                IndexOrRangeType.Range => m_Range.Start.IsFromEnd ? m_Range.Start.Value == 0 ? int.MaxValue : -m_Range.Start.Value : m_Range.Start.Value,
                _ => throw new ArgumentOutOfRangeException()
            };
        }

        public int End()
        {
            return m_Type switch
            {
                IndexOrRangeType.Index => m_Index.IsFromEnd ? m_Index.Value == 1 ? int.MaxValue : -m_Index.Value + 1 : m_Index.Value + 1,
                IndexOrRangeType.Range => m_Range.End.IsFromEnd ? m_Range.End.Value == 0 ? int.MaxValue : -m_Range.End.Value : m_Range.End.Value,
                _ => throw new ArgumentOutOfRangeException()
            };
        }

        public bool IsRangeAll()
        {
            return m_Type == IndexOrRangeType.Range && m_Range.Start is { IsFromEnd: false, Value: 0 } && m_Range.End is { IsFromEnd: true, Value: 0 };
        }
    }
}
