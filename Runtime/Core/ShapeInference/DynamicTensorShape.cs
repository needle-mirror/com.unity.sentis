using System;
using System.Text;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents the shape of an input tensor, or the predicted shape of a tensor before Sentis executes.
    /// </summary>
    [Serializable]
    public unsafe struct DynamicTensorShape
    {
        DynamicTensorDim m_D7;
        DynamicTensorDim m_D6;
        DynamicTensorDim m_D5;
        DynamicTensorDim m_D4;
        DynamicTensorDim m_D3;
        DynamicTensorDim m_D2;
        DynamicTensorDim m_D1;
        DynamicTensorDim m_D0;
        bool m_IsRankDynamic;
        int m_Rank;

        internal bool hasRank => !m_IsRankDynamic;

        /// <summary>
        /// Whether the shape has a dynamic rank.
        /// </summary>
        public bool isRankDynamic => m_IsRankDynamic;

        /// <summary>
        /// The rank of a `DynamicTensorShape`, For example, a tensor of shape (5) has a rank of 1. A tensor of shape (7, 3, 5) has a rank of 3.
        ///
        /// This cannot be called if the shape has a dynamic rank. Call `isRankDynamic` first.
        /// </summary>
        public int rank
        {
            get
            {
                Logger.AssertIsTrue(!m_IsRankDynamic, "Cannot get the rank of a shape with a dynamic rank.");
                return m_Rank;
            }
        }

        /// <summary>
        /// Gets the tensor shape at a given axis.
        /// Ex:
        /// shape  (3, 4, 5, 6)
        /// index   0, 1, 2, 3
        ///        -4,-3,-2,-1
        /// shape  (7, 3, 2)
        /// index   0, 1, 2
        ///        -3,-2,-1
        /// </summary>
        /// <param name="axis">The axis to get or set.</param>
        internal DynamicTensorDim this[int axis]
        {
            get
            {
                if (!hasRank)
                {
                    Logger.AssertIsTrue(axis >= -TensorShape.maxRank && axis < TensorShape.maxRank, "IndexError: axis {0} is out of bounds shape of max rank, {1}", axis, TensorShape.maxRank);
                    return DynamicTensorDim.Unknown;
                }

                axis = Axis(axis);

                fixed (DynamicTensorDim* shape = &m_D7)
                {
                    return shape[(TensorShape.maxRank - rank) + axis];
                }
            }

            set
            {
                if (!hasRank)
                {
                    Logger.AssertIsTrue(axis >= -TensorShape.maxRank && axis < TensorShape.maxRank, "IndexError: axis {0} is out of bounds shape of max rank, {1}", axis, TensorShape.maxRank);
                    return;
                }

                axis = Axis(axis);

                fixed (DynamicTensorDim* shape = &m_D7)
                {
                    shape[(TensorShape.maxRank - rank) + axis] = value;
                }
            }
        }

        /// <summary>
        /// Checks if the `DynamicTensorShape` is static and can be converted to a `TensorShape`.
        /// </summary>
        /// <returns>Whether the `DynamicTensorShape` has static rank all static dimensions.</returns>
        public bool IsStatic()
        {
            if (!hasRank)
                return false;

            for (var i = 0; i < rank; i++)
            {
                if (!this[i].isValue)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// The length of the tensor shape as a dynamic tensor dimension.
        /// </summary>
        internal DynamicTensorDim Length()
        {
            if (!hasRank)
                return DynamicTensorDim.Unknown;

            var length = DynamicTensorDim.One;

            for (var i = 0; i < rank && !(length == 0); i++)
                length *= this[i];

            return length;
        }

        /// <summary>
        /// Converts the `DynamicTensorShape` to a `TensorShape`. You should call `IsStatic` before you call this method.
        /// </summary>
        /// <returns>The converted `TensorShape`.</returns>
        public TensorShape ToTensorShape()
        {
            Assert.IsTrue(hasRank, "ValueError: Cannot convert tensor of dynamic rank to TensorShape");

            var shapeOut = TensorShape.Ones(rank);
            for (var i = 0; i < rank; i++)
            {
                shapeOut[i] = this[i].value;
            }

            return shapeOut;
        }

        internal DynamicTensorShape(DynamicTensorDim d0, DynamicTensorDim d1, DynamicTensorDim d2, DynamicTensorDim d3, DynamicTensorDim d4, DynamicTensorDim d5, DynamicTensorDim d6, DynamicTensorDim d7)
        {
            m_D7 = d0;
            m_D6 = d1;
            m_D5 = d2;
            m_D4 = d3;
            m_D3 = d4;
            m_D2 = d5;
            m_D1 = d6;
            m_D0 = d7;

            m_IsRankDynamic = false;
            m_Rank = 8;
        }

        internal DynamicTensorShape(DynamicTensorDim d0, DynamicTensorDim d1, DynamicTensorDim d2, DynamicTensorDim d3, DynamicTensorDim d4, DynamicTensorDim d5, DynamicTensorDim d6)
        {
            m_D7 = default;
            m_D6 = d0;
            m_D5 = d1;
            m_D4 = d2;
            m_D3 = d3;
            m_D2 = d4;
            m_D1 = d5;
            m_D0 = d6;

            m_IsRankDynamic = false;
            m_Rank = 7;
        }

        internal DynamicTensorShape(DynamicTensorDim d0, DynamicTensorDim d1, DynamicTensorDim d2, DynamicTensorDim d3, DynamicTensorDim d4, DynamicTensorDim d5)
        {
            m_D7 = default;
            m_D6 = default;
            m_D5 = d0;
            m_D4 = d1;
            m_D3 = d2;
            m_D2 = d3;
            m_D1 = d4;
            m_D0 = d5;

            m_IsRankDynamic = false;
            m_Rank = 6;
        }

        internal DynamicTensorShape(DynamicTensorDim d0, DynamicTensorDim d1, DynamicTensorDim d2, DynamicTensorDim d3, DynamicTensorDim d4)
        {
            m_D7 = default;
            m_D6 = default;
            m_D5 = default;
            m_D4 = d0;
            m_D3 = d1;
            m_D2 = d2;
            m_D1 = d3;
            m_D0 = d4;

            m_IsRankDynamic = false;
            m_Rank = 5;
        }

        internal DynamicTensorShape(DynamicTensorDim d0, DynamicTensorDim d1, DynamicTensorDim d2, DynamicTensorDim d3)
        {
            m_D7 = default;
            m_D6 = default;
            m_D5 = default;
            m_D4 = default;
            m_D3 = d0;
            m_D2 = d1;
            m_D1 = d2;
            m_D0 = d3;

            m_IsRankDynamic = false;
            m_Rank = 4;
        }

        internal DynamicTensorShape(DynamicTensorDim d0, DynamicTensorDim d1, DynamicTensorDim d2)
        {
            m_D7 = default;
            m_D6 = default;
            m_D5 = default;
            m_D4 = default;
            m_D3 = default;
            m_D2 = d0;
            m_D1 = d1;
            m_D0 = d2;

            m_IsRankDynamic = false;
            m_Rank = 3;
        }

        internal DynamicTensorShape(DynamicTensorDim d0, DynamicTensorDim d1)
        {
            m_D7 = default;
            m_D6 = default;
            m_D5 = default;
            m_D4 = default;
            m_D3 = default;
            m_D2 = default;
            m_D1 = d0;
            m_D0 = d1;

            m_IsRankDynamic = false;
            m_Rank = 2;
        }

        internal DynamicTensorShape(DynamicTensorDim d0)
        {
            m_D7 = default;
            m_D6 = default;
            m_D5 = default;
            m_D4 = default;
            m_D3 = default;
            m_D2 = default;
            m_D1 = default;
            m_D0 = d0;

            m_IsRankDynamic = false;
            m_Rank = 1;
        }

        /// <summary>
        /// Initializes and returns an instance of `DynamicTensorShape` with rank 1.
        ///
        /// Dimensions with non-negative values are static and values of -1 are dynamic.
        /// </summary>
        /// <param name="d0">The dimension of axis 0.</param>
        public DynamicTensorShape(int d0)
            : this(DynamicTensorDim.FromInt(d0)) { }

        /// <summary>
        /// Initializes and returns an instance of `DynamicTensorShape` with rank 2.
        ///
        /// Dimensions with non-negative values are static and values of -1 are dynamic.
        /// </summary>
        /// <param name="d0">The dimension of axis 0.</param>
        /// <param name="d1">The dimension of axis 1.</param>
        public DynamicTensorShape(int d0, int d1)
            : this(DynamicTensorDim.FromInt(d0), DynamicTensorDim.FromInt(d1)) { }

        /// <summary>
        /// Initializes and returns an instance of `DynamicTensorShape` with rank 3.
        ///
        /// Dimensions with non-negative values are static and values of -1 are dynamic.
        /// </summary>
        /// <param name="d0">The dimension of axis 0.</param>
        /// <param name="d1">The dimension of axis 1.</param>
        /// <param name="d2">The dimension of axis 2.</param>
        public DynamicTensorShape(int d0, int d1, int d2)
            : this(DynamicTensorDim.FromInt(d0), DynamicTensorDim.FromInt(d1), DynamicTensorDim.FromInt(d2)) { }

        /// <summary>
        /// Initializes and returns an instance of `DynamicTensorShape` with rank 4.
        ///
        /// Dimensions with non-negative values are static and values of -1 are dynamic.
        /// </summary>
        /// <param name="d0">The dimension of axis 0.</param>
        /// <param name="d1">The dimension of axis 1.</param>
        /// <param name="d2">The dimension of axis 2.</param>
        /// <param name="d3">The dimension of axis 3.</param>
        public DynamicTensorShape(int d0, int d1, int d2, int d3)
            : this(DynamicTensorDim.FromInt(d0), DynamicTensorDim.FromInt(d1), DynamicTensorDim.FromInt(d2), DynamicTensorDim.FromInt(d3)) { }

        /// <summary>
        /// Initializes and returns an instance of `DynamicTensorShape` with rank 5.
        ///
        /// Dimensions with non-negative values are static and values of -1 are dynamic.
        /// </summary>
        /// <param name="d0">The dimension of axis 0.</param>
        /// <param name="d1">The dimension of axis 1.</param>
        /// <param name="d2">The dimension of axis 2.</param>
        /// <param name="d3">The dimension of axis 3.</param>
        /// <param name="d4">The dimension of axis 4.</param>
        public DynamicTensorShape(int d0, int d1, int d2, int d3, int d4)
            : this(DynamicTensorDim.FromInt(d0), DynamicTensorDim.FromInt(d1), DynamicTensorDim.FromInt(d2), DynamicTensorDim.FromInt(d3), DynamicTensorDim.FromInt(d4)) { }

        /// <summary>
        /// Initializes and returns an instance of `DynamicTensorShape` with rank 6.
        ///
        /// Dimensions with non-negative values are static and values of -1 are dynamic.
        /// </summary>
        /// <param name="d0">The dimension of axis 0.</param>
        /// <param name="d1">The dimension of axis 1.</param>
        /// <param name="d2">The dimension of axis 2.</param>
        /// <param name="d3">The dimension of axis 3.</param>
        /// <param name="d4">The dimension of axis 4.</param>
        /// <param name="d5">The dimension of axis 5.</param>
        public DynamicTensorShape(int d0, int d1, int d2, int d3, int d4, int d5)
            : this(DynamicTensorDim.FromInt(d0), DynamicTensorDim.FromInt(d1), DynamicTensorDim.FromInt(d2), DynamicTensorDim.FromInt(d3), DynamicTensorDim.FromInt(d4), DynamicTensorDim.FromInt(d5)) { }

        /// <summary>
        /// Initializes and returns an instance of `DynamicTensorShape` with rank 7.
        ///
        /// Dimensions with non-negative values are static and values of -1 are dynamic.
        /// </summary>
        /// <param name="d0">The dimension of axis 0.</param>
        /// <param name="d1">The dimension of axis 1.</param>
        /// <param name="d2">The dimension of axis 2.</param>
        /// <param name="d3">The dimension of axis 3.</param>
        /// <param name="d4">The dimension of axis 4.</param>
        /// <param name="d5">The dimension of axis 5.</param>
        /// <param name="d6">The dimension of axis 6.</param>
        public DynamicTensorShape(int d0, int d1, int d2, int d3, int d4, int d5, int d6)
            : this(DynamicTensorDim.FromInt(d0), DynamicTensorDim.FromInt(d1), DynamicTensorDim.FromInt(d2), DynamicTensorDim.FromInt(d3), DynamicTensorDim.FromInt(d4), DynamicTensorDim.FromInt(d5), DynamicTensorDim.FromInt(d6)) { }

        /// <summary>
        /// Initializes and returns an instance of `DynamicTensorShape` with rank 8.
        ///
        /// Dimensions with non-negative values are static and values of -1 are dynamic.
        /// </summary>
        /// <param name="d0">The dimension of axis 0.</param>
        /// <param name="d1">The dimension of axis 1.</param>
        /// <param name="d2">The dimension of axis 2.</param>
        /// <param name="d3">The dimension of axis 3.</param>
        /// <param name="d4">The dimension of axis 4.</param>
        /// <param name="d5">The dimension of axis 5.</param>
        /// <param name="d6">The dimension of axis 6.</param>
        /// <param name="d7">The dimension of axis 7.</param>
        public DynamicTensorShape(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7)
            : this(DynamicTensorDim.FromInt(d0), DynamicTensorDim.FromInt(d1), DynamicTensorDim.FromInt(d2), DynamicTensorDim.FromInt(d3), DynamicTensorDim.FromInt(d4), DynamicTensorDim.FromInt(d5), DynamicTensorDim.FromInt(d6), DynamicTensorDim.FromInt(d7)) { }

        /// <summary>
        /// Initializes and returns an instance of `DynamicTensorShape` with given dims.
        ///
        /// Values of -1 are
        /// </summary>
        /// <param name="shape">The shape as a span.</param>
        public DynamicTensorShape(ReadOnlySpan<int> shape)
        {
            Logger.AssertIsTrue(shape.Length <= TensorShape.maxRank, "ValueError: DynamicTensorShape are capped to rank=8, cannot create DynamicTensorShape of rank {0}", shape.Length);
            m_IsRankDynamic = false;
            m_Rank = shape.Length;
            m_D0 = m_Rank > 0 ? DynamicTensorDim.FromInt(shape[m_Rank - 1 - 0]) : default;
            m_D1 = m_Rank > 1 ? DynamicTensorDim.FromInt(shape[m_Rank - 1 - 1]) : default;
            m_D2 = m_Rank > 2 ? DynamicTensorDim.FromInt(shape[m_Rank - 1 - 2]) : default;
            m_D3 = m_Rank > 3 ? DynamicTensorDim.FromInt(shape[m_Rank - 1 - 3]) : default;
            m_D4 = m_Rank > 4 ? DynamicTensorDim.FromInt(shape[m_Rank - 1 - 4]) : default;
            m_D5 = m_Rank > 5 ? DynamicTensorDim.FromInt(shape[m_Rank - 1 - 5]) : default;
            m_D6 = m_Rank > 6 ? DynamicTensorDim.FromInt(shape[m_Rank - 1 - 6]) : default;
            m_D7 = m_Rank > 7 ? DynamicTensorDim.FromInt(shape[m_Rank - 1 - 7]) : default;
        }

        /// <summary>
        /// Initializes and returns an instance of `DynamicTensorShape` of dynamic rank.
        /// </summary>
        internal static DynamicTensorShape DynamicRank
        {
            get
            {
                var shape = new DynamicTensorShape();
                shape.m_IsRankDynamic = true;
                return shape;
            }
        }

        /// <summary>
        /// Initializes and returns an instance of `DynamicTensorShape` with a given `TensorShape`, and dynamic dimensions. For example: `DynamicTensorShape(new TensorShape(3, 4, 5, 6))` returns a dynamic tensor shape of (3, 4, 5, 6).
        /// </summary>
        /// <param name="other">The `TensorShape` to copy.</param>
        public DynamicTensorShape(TensorShape other)
        {
            m_Rank = other.rank;
            m_IsRankDynamic = false;

            m_D7 = m_Rank > 7 ? DynamicTensorDim.Int(other[m_Rank - 8]) : default;
            m_D6 = m_Rank > 6 ? DynamicTensorDim.Int(other[m_Rank - 7]) : default;
            m_D5 = m_Rank > 5 ? DynamicTensorDim.Int(other[m_Rank - 6]) : default;
            m_D4 = m_Rank > 4 ? DynamicTensorDim.Int(other[m_Rank - 5]) : default;
            m_D3 = m_Rank > 3 ? DynamicTensorDim.Int(other[m_Rank - 4]) : default;
            m_D2 = m_Rank > 2 ? DynamicTensorDim.Int(other[m_Rank - 3]) : default;
            m_D1 = m_Rank > 1 ? DynamicTensorDim.Int(other[m_Rank - 2]) : default;
            m_D0 = m_Rank > 0 ? DynamicTensorDim.Int(other[m_Rank - 1]) : default;
        }

        /// <summary>
        /// Returns a copy of another `DynamicTensorShape`.
        /// </summary>
        /// <param name="other">The `DynamicTensorShape` to copy.</param>
        public DynamicTensorShape(DynamicTensorShape other)
        {
            m_Rank = other.m_Rank;
            m_IsRankDynamic = other.m_IsRankDynamic;

            m_D7 = other.m_D7;
            m_D6 = other.m_D6;
            m_D5 = other.m_D5;
            m_D4 = other.m_D4;
            m_D3 = other.m_D3;
            m_D2 = other.m_D2;
            m_D1 = other.m_D1;
            m_D0 = other.m_D0;
        }

        /// <summary>
        /// Creates and returns a `DynamicTensorShape` with given rank and all dimensions dynamic.
        /// </summary>
        /// <param name="rank">The rank of the `DynamicTensorShape`.</param>
        /// <returns>The created `DynamicTensorShape`.</returns>
        public static DynamicTensorShape DynamicOfRank(int rank)
        {
            Logger.AssertIsTrue(0 <= rank && rank <= TensorShape.maxRank, "ValueError: DynamicTensorShape are capped to rank=8, cannot create empty shape of rank {0}", rank);
            var outShape = new DynamicTensorShape();
            outShape.m_IsRankDynamic = false;
            outShape.m_Rank = rank;
            return outShape;
        }

        /// <summary>
        /// Return the shape as an integer array, if the rank is dynamic this returns null.
        ///
        /// If a dimension is dynamic it is represented with a -1 in the array.
        /// </summary>
        /// <returns>The shape as an integer array.</returns>
        public int[] ToIntArray()
        {
            if (m_IsRankDynamic)
                return null;
            var array = new int[rank];
            for (var i = 0; i < array.Length; i++)
                array[i] = this[i].ToInt();
            return array;
        }

        /// <summary>
        /// Returns the dimension at a given axis as an integer.
        ///
        /// If a dimension is dynamic it is represented with a -1.
        /// </summary>
        /// <param name="axis">The axis to get the dimension of.</param>
        /// <returns>The integer dimension of an axis.</returns>
        public int Get(int axis)
        {
            return this[axis].ToInt();
        }

        /// <summary>
        /// Sets the dimension at a given axis.
        /// </summary>
        /// <param name="axis">The axis to set the dimension of.</param>
        /// <param name="dimension">The dimension of the axis. Use -1 for a dynamic dimension.</param>
        public void Set(int axis, int dimension)
        {
            this[axis] = DynamicTensorDim.FromInt(dimension);
        }

        /// <summary>
        /// Sets the dimension to be dynamic at a given axis.
        /// </summary>
        /// <param name="axis">The axis to set the dimension of.</param>
        public void SetDynamic(int axis)
        {
            this[axis] = DynamicTensorDim.Unknown;
        }

        /// <summary>
        /// DynamicTensorShape with same rank as other DynamicTensorShape and all dimensions dynamic
        /// </summary>
        internal static DynamicTensorShape DynamicOfRankLike(DynamicTensorShape other)
        {
            if (!other.hasRank)
                return DynamicRank;
            return DynamicOfRank(other.rank);
        }

        /// <summary>
        /// Asserts if this shape has a rank different from the given rank
        /// If tensor is dynamic rank then rank is set to given value
        /// </summary>
        internal void DeclareRank(int newRank)
        {
            if (hasRank)
            {
                Logger.AssertAreEqual(m_Rank, newRank, "RankError: incorrect rank, expecting {0}, got {1}", m_Rank, newRank);
                return;
            }

            m_IsRankDynamic = false;
            m_Rank = newRank;
        }

        internal void DeclareRank(DynamicTensorDim dim)
        {
            if (dim.isValue)
                DeclareRank(dim.value);
        }

        /// <summary>
        /// DynamicTensorShape with given rank and all dimensions 1.
        /// </summary>
        /// <param name="rank">The rank of the dynamic tensor shape.</param>
        /// <returns>The dynamic tensor shape of ones.</returns>
        public static DynamicTensorShape Ones(int rank)
        {
            Logger.AssertIsTrue(0 <= rank && rank <= TensorShape.maxRank, "ValueError: DynamicTensorShape are capped to rank=8, cannot create empty shape of rank {0}", rank);
            var outShape = new DynamicTensorShape();
            outShape.m_IsRankDynamic = false;
            outShape.m_Rank = rank;
            for (var i = 0; i < rank; i++)
            {
                outShape[i] = DynamicTensorDim.One;
            }

            return outShape;
        }

        internal static DynamicTensorShape OnesLike(DynamicTensorShape shape)
        {
            return shape.hasRank ? Ones(shape.rank) : DynamicRank;
        }

        /// <summary>
        /// Returns a string that represents the `DynamicTensorShape`.
        /// </summary>
        /// <returns>The string representation of the `DynamicTensorShape`.</returns>
        public override string ToString()
        {
            if (m_IsRankDynamic)
                return "Dynamic rank";

            StringBuilder sb = new StringBuilder();
            sb.Append("(");
            for (var i = 0; i < rank; i++)
            {
                if (i != 0)
                    sb.Append(", ");
                var dim = this[i];
                sb.Append(dim.ToString());
            }

            sb.Append(")");
            return sb.ToString();
        }

        /// <summary>
        /// Wraps axis to positive index between 0,rank
        /// (5,2,3,4)
        /// axis = -1 => axis_out = 3
        /// axis = 1 => axis_out = 1
        /// </summary>
        internal int Axis(int axis)
        {
            Logger.AssertIsTrue(axis >= -rank && axis < rank, "IndexError: axis {0} is out of bounds shape of rank, {1}", axis, rank);
            return axis >= 0 ? axis : rank + axis;
        }

        /// <summary>
        /// Removes axes of length 1. For example, if the `DynamicTensorShape` is (5, 1, 3, 1), the method returns (5, 3).
        /// </summary>
        internal DynamicTensorShape Squeeze()
        {
            if (!hasRank)
                return DynamicRank;
            Logger.AssertIsTrue(rank != 0, "ValueError: cannot squeeze scalar tensor {0}", this);

            var numAxes = 0;
            for (var i = 0; i < rank; i++)
            {
                if (!this[i].isValue)
                    return DynamicRank;
                if (this[i] == DynamicTensorDim.One)
                    numAxes += 1;
            }

            var shapeOut = DynamicOfRank(rank - numAxes);
            var index = 0;
            for (var i = 0; i < rank; i++)
            {
                if (this[i] != DynamicTensorDim.One)
                    shapeOut[index++] = this[i];
            }

            return shapeOut;
        }

        /// <summary>
        /// Removes the axis if its length is 1. For example, if `DynamicTensorShape` is (5, 1, 3, 1) and `axis` is 1, the method returns (5, 3, 1).
        /// </summary>
        internal DynamicTensorShape Squeeze(int axis)
        {
            if (!hasRank)
                return DynamicRank;

            axis = Axis(axis);
            var dim = this[axis];

            Logger.AssertIsTrue(!dim.isValue || dim == DynamicTensorDim.One, "ValueError: cannot squeeze axis with value != 1");

            var shapeOut = DynamicOfRank(rank - 1);
            var index = 0;
            for (var i = 0; i < rank; i++)
            {
                if (i != axis)
                    shapeOut[index++] = this[i];
            }

            return shapeOut;
        }

        /// <summary>
        /// Removes axes if their length is 1. For example, if `DynamicTensorShape` is (5, 1, 3, 1) and `axes` is {1, -1}, the method returns (5, 3).
        /// </summary>
        internal DynamicTensorShape Squeeze(PartialTensor axes)
        {
            if (axes == null)
                return Squeeze();

            if (!axes.IsStatic() || !hasRank)
                return DynamicRank;

            uint axesBitMask = 0;
            for (var i = 0; i < axes.length; i++)
            {
                var axis = Axis(axes[i].intValue);
                Logger.AssertIsTrue(((axesBitMask >> axis) & 1U) == 0, "ValueError: can't squeeze on same axis multiple times");
                axesBitMask |= 1U << axis;
                var dim = this[axis];
                Logger.AssertIsTrue(!dim.isValue || dim == DynamicTensorDim.One, "ValueError: cannot squeeze axis with value != 1");
            }

            var shapeOut = DynamicOfRank(rank - axes.length);
            var index = 0;
            for (var i = 0; i < rank; i++)
            {
                if (((axesBitMask >> i) & 1U) == 0)
                    shapeOut[index++] = this[i];
            }

            return shapeOut;
        }

        /// <summary>
        /// Inserts a new axis at `axis` position. For example if `DynamicTensorShape` is (2) and the value of `axis` is 0, the method returns (1, 2).
        /// </summary>
        internal DynamicTensorShape Unsqueeze(int axis)
        {
            if (!hasRank)
                return DynamicRank;

            Logger.AssertIsTrue(rank != TensorShape.maxRank, "ValueError: TensorShape are capped to rank=8, cannot unsqueeze rank 8 DynamicTensorShape {0}", this);

            var shapeOut = DynamicOfRank(rank + 1);

            axis = shapeOut.Axis(axis);
            var indexIn = 0;
            for (var indexOut = 0; indexOut < shapeOut.rank; indexOut++)
            {
                if (indexOut == axis)
                    shapeOut[indexOut] = DynamicTensorDim.One;
                else
                    shapeOut[indexOut] = this[indexIn++];
            }

            return shapeOut;
        }

        /// <summary>
        /// Inserts new axes at `axes` positions. For example if `DynamicTensorShape` is (2) and `axes` is {0, 1}, the method returns (1, 1, 2).
        /// </summary>
        internal DynamicTensorShape Unsqueeze(PartialTensor axes)
        {
            if (!hasRank || !axes.isPartiallyKnown)
                return DynamicRank;

            Logger.AssertIsTrue(rank + axes.length <= TensorShape.maxRank, "ValueError: TensorShape are capped to rank=8, cannot unsqueeze DynamicTensorShape {0} to rank greater than 8", this);

            var shapeOut = DynamicOfRank(rank + axes.length);

            if (!axes.IsStatic())
                return shapeOut;

            uint axesBitMask = 0;
            for (var i = 0; i < axes.length; i++)
            {
                var axis = shapeOut.Axis(axes[i].intValue);
                Logger.AssertIsTrue(((axesBitMask >> axis) & 1U) == 0, "ValueError: can't unsqueeze on same axis multiple times");
                axesBitMask |= 1U << axis;
            }

            var indexIn = 0;
            for (var indexOut = 0; indexOut < shapeOut.rank; indexOut++)
            {
                if (((axesBitMask >> indexOut) & 1U) == 1)
                    shapeOut[indexOut] = DynamicTensorDim.One;
                else
                    shapeOut[indexOut] = this[indexIn++];
            }

            return shapeOut;
        }

        /// <summary>
        /// Broadcasts the `DynamicTensorShape` with another `DynamicTensorShape`, according to numpy tensor broadcasting rules.
        /// </summary>
        internal DynamicTensorShape Broadcast(DynamicTensorShape other)
        {
            if (!hasRank || !other.hasRank)
                return DynamicRank;

            var outRank = Mathf.Max(rank, other.rank);
            var outShape = Ones(outRank);

            DynamicTensorDim* fixedOther = &other.m_D7;
            DynamicTensorDim* fixedOut = &outShape.m_D7;
            fixed (DynamicTensorDim* fixedThis = &m_D7)
            {
                for (var i = 0; i < outRank; i++)
                {
                    if (i < rank)
                        fixedOut[TensorShape.maxRank - i - 1] = fixedThis[TensorShape.maxRank - i - 1];
                    if (i < other.rank)
                        fixedOut[TensorShape.maxRank - i - 1] = DynamicTensorDim.Broadcast(fixedOut[TensorShape.maxRank - i - 1], fixedOther[TensorShape.maxRank - i - 1]);
                }
            }

            return outShape;
        }

        /// <summary>
        /// Multiplies two `DynamicTensorShape` objects.
        /// </summary>
        internal DynamicTensorShape MatMul(DynamicTensorShape other)
        {
            if (!hasRank || !other.hasRank)
                return DynamicRank;

            Assert.IsTrue(rank >= 1, "MatMul.ValueError: Rank of tensor must be at least 1");
            Assert.IsTrue(other.rank >= 1, "MatMul.ValueError: Rank of tensor must be at least 1");

            if (other.rank == 1)
                return MatMul(new DynamicTensorShape(other[0], DynamicTensorDim.One)).Squeeze(-1);
            if (rank == 1)
                return new DynamicTensorShape(DynamicTensorDim.One, this[0]).MatMul(other).Squeeze(-2);

            // broadcast along the dimensions not used in the matmul
            var outRank = Mathf.Max(rank, other.rank);
            var shapeOut = Ones(outRank);

            DynamicTensorDim* fixedOther = &other.m_D7;
            DynamicTensorDim* fixedOut = &shapeOut.m_D7;
            fixed (DynamicTensorDim* fixedThis = &m_D7)
            {
                for (var i = 2; i < shapeOut.rank; i++)
                {
                    if (i < rank)
                        fixedOut[TensorShape.maxRank - i - 1] = fixedThis[TensorShape.maxRank - i - 1];
                    if (i < other.rank)
                        fixedOut[TensorShape.maxRank - i - 1] = DynamicTensorDim.Broadcast(fixedOut[TensorShape.maxRank - i - 1], fixedOther[TensorShape.maxRank - i - 1]);
                }

                // Raise an error if the last dimension of a is not the same size as the second-to-last dimension of b.
                var mulThisDim = fixedThis[TensorShape.maxRank - 1];
                var mulOtherDim = fixedOther[TensorShape.maxRank - 2];
                Logger.AssertIsTrue(!mulThisDim.isValue || !mulOtherDim.isValue || mulThisDim == mulOtherDim, "MatMul2D.ValueError: mul dims not equal");

                fixedOut[TensorShape.maxRank - 2] = fixedThis[TensorShape.maxRank - 2];
                fixedOut[TensorShape.maxRank - 1] = fixedOther[TensorShape.maxRank - 1];
            }

            return shapeOut;
        }

        /// <summary>
        /// Creates two new shapes for input shapes 'a' and 'b' with value dims and optionally param dims divided through
        /// on both sides where possible.
        /// </summary>
        internal static void ReduceCommonFactors(DynamicTensorShape a, DynamicTensorShape b, out DynamicTensorShape reducedA, out DynamicTensorShape reducedB, bool reduceParams)
        {
            reducedA = new DynamicTensorShape(a);
            reducedB = new DynamicTensorShape(b);

            if (reducedA.m_IsRankDynamic || reducedB.m_IsRankDynamic)
                return;

            for (var i = 0; i < reducedA.rank; i++)
            {
                if (!reduceParams && reducedA[i].isParam)
                    continue;
                for (var j = 0; j < reducedB.rank && (reducedA[i].isParam || reducedA[i] > 1); j++)
                {
                    var gcd = DynamicTensorDim.GCD(reducedA[i], reducedB[j]);
                    if (gcd.isParam || gcd > 1)
                    {
                        reducedA[i] /= gcd;
                        reducedB[j] /= gcd;
                    }
                }
            }
        }

        /// <summary>
        /// Compares two `DynamicTensorShape` objects. Returns `true` if the two objects have the same rank, and all their dimensions are equal.
        /// </summary>
        /// <param name="a">The first `DynamicTensorShape` to compare.</param>
        /// <param name="b">The second `DynamicTensorShape` to compare.</param>
        /// <returns>Whether the two `DynamicTensorShape` objects are equal.</returns>
        public static bool operator ==(DynamicTensorShape a, DynamicTensorShape b)
        {
            if (!a.hasRank || !b.hasRank)
                return false;
            if (a.rank != b.rank)
                return false;
            if (a.m_D7 != b.m_D7)
                return false;
            if (a.m_D6 != b.m_D6)
                return false;
            if (a.m_D5 != b.m_D5)
                return false;
            if (a.m_D4 != b.m_D4)
                return false;
            if (a.m_D3 != b.m_D3)
                return false;
            if (a.m_D2 != b.m_D2)
                return false;
            if (a.m_D1 != b.m_D1)
                return false;
            if (a.m_D0 != b.m_D0)
                return false;
            return true;
        }

        /// <summary>
        /// Compares two `DynamicTensorShape` objects. Returns `true` if the two shapes have a different or dynamic rank, or at least one of their dimensions are not equal.
        /// </summary>
        /// <param name="a">The first `DynamicTensorShape` to compare.</param>
        /// <param name="b">The second `DynamicTensorShape` to compare.</param>
        /// <returns>Whether the two `DynamicTensorShape` objects are not equal.</returns>
        public static bool operator !=(DynamicTensorShape a, DynamicTensorShape b)
        {
            return !(a == b);
        }

        /// <summary>
        /// Determines whether the specified object is equal to the current `DynamicTensorShape`.
        /// </summary>
        /// <param name="obj">The object to compare.</param>
        /// <returns>Whether the object is equal to the current `DynamicTensorShape`.</returns>
        public override bool Equals(object obj)
        {
            // Check for null values and compare run-time types.
            if (obj == null || GetType() != obj.GetType())
                return false;

            return this == (DynamicTensorShape)obj;
        }

        /// <summary>
        /// Whether dynamic shapes a and b could be referring to the same underlying tensor shape
        /// </summary>
        internal static bool IsCompatible(DynamicTensorShape a, DynamicTensorShape b)
        {
            if (!a.hasRank || !b.hasRank)
                return true;
            if (a.rank != b.rank)
                return false;
            for (var i = 0; i < a.rank; i++)
            {
                if (a[i] != b[i])
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Whether a dynamic shape a and tensor shape b could be referring to the same underlying tensor shape
        /// </summary>
        internal static bool IsCompatible(DynamicTensorShape a, TensorShape b)
        {
            if (!a.hasRank)
                return true;
            if (a.rank != b.rank)
                return false;
            for (var i = 0; i < a.rank; i++)
            {
                if (a[i] != b[i])
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Returns a dynamic shape with the most known rank and dims from two
        /// given shapes that are known to be equal. Asserts if the shapes cannot be equal
        /// </summary>
        internal static DynamicTensorShape MaxDefinedShape(DynamicTensorShape a, DynamicTensorShape b)
        {
            if (!a.hasRank)
                return b;
            if (!b.hasRank)
                return a;
            Logger.AssertIsTrue(a.rank == b.rank, "InputError: incompatible tensor shapes");
            var shapeOut = DynamicOfRank(a.rank);
            for (var i = 0; i < shapeOut.rank; i++)
            {
                shapeOut[i] = DynamicTensorDim.MaxDefinedDim(a[i], b[i]);
            }

            return shapeOut;
        }

        /// <summary>
        /// Serves as the default hash function.
        /// </summary>
        /// <returns>The calculated hash code.</returns>
        public override int GetHashCode()
        {
            return m_IsRankDynamic.GetHashCode() ^ m_Rank.GetHashCode() ^ m_D7.GetHashCode() ^ m_D6.GetHashCode() ^ m_D5.GetHashCode()
                ^ m_D4.GetHashCode() ^ m_D3.GetHashCode() ^ m_D2.GetHashCode() ^ m_D1.GetHashCode() ^ m_D0.GetHashCode();
        }
    }
}

