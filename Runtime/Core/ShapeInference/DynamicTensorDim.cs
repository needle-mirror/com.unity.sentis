using System;
using System.ComponentModel;
using UnityEngine;

namespace Unity.Sentis
{
    /// <summary>
    /// Types of `DynamicTensorShape` dimension.
    /// </summary>
    [Serializable]
    enum DimType
    {
        /// <summary>
        /// The tensor dimension is unknown.
        /// </summary>
        Unknown = 0,

        /// <summary>
        /// The tensor dimension is fixed.
        /// </summary>
        Static,

        /// <summary>
        /// The tensor dimension is dynamic.
        /// </summary>
        Param
    }

    /// <summary>
    /// Represents a single dimension of a `DynamicTensorShape`.
    /// </summary>
    [Serializable]
    struct DynamicTensorDim
    {
        const string k_UnknownName = "?";

        DimType m_DimType;
        byte m_Param;
        int m_Value;

        public static DynamicTensorDim Unknown => new DynamicTensorDim();

        public static DynamicTensorDim FromInt(int value)
        {
            if (value == -1)
                return Unknown;
            Logger.AssertIsTrue(value >= 0, "Dim must be non-negative or equal to -1");
            return new DynamicTensorDim()
            {
                m_DimType = DimType.Static,
                m_Param = default,
                m_Value = value
            };
        }

        /// <summary>
        /// Initializes and returns an instance of `DynamicTensorDim` of fixed type, with an integer value.
        /// </summary>
        /// <param name="value">The size of the dim.</param>
        /// <returns>The dynamic tensor dim.</returns>
        public static DynamicTensorDim Int(int value)
        {
            Logger.AssertIsTrue(value >= 0, "Dim value cannot be negative");
            return new DynamicTensorDim()
            {
                m_DimType = DimType.Static,
                m_Param = default,
                m_Value = value
            };
        }

        /// <summary>
        /// Initializes and returns an instance of `DynamicTensorDim` of dynamic type, with a byte value.
        /// </summary>
        /// <param name="param">The byte value dynamic parameter.</param>
        /// <returns>The dynamic tensor dim.</returns>
        public static DynamicTensorDim Param(byte param)
        {
            return new DynamicTensorDim()
            {
                m_DimType = DimType.Param,
                m_Param = param,
                m_Value = default
            };
        }

        public bool isUnknown => m_DimType == DimType.Unknown;

        public bool isValue => m_DimType == DimType.Static;

        public bool isParam => m_DimType == DimType.Param;

        internal static DynamicTensorDim Zero => Int(0);
        internal static DynamicTensorDim One => Int(1);

        /// <summary>
        /// Return the dim as an integer, if the dim is dynamic this is -1.
        /// </summary>
        /// <returns>The dim as an integer.</returns>
        public int ToInt()
        {
            if (isValue)
                return value;
            return -1;
        }

        /// <summary>
        /// The value of the dimension. You can only call this method if `.isStatic` is true.
        /// </summary>
        public int value
        {
            get
            {
                Logger.AssertIsTrue(m_DimType == DimType.Static, "Cannot get value of dim which is not static");
                return m_Value;
            }
        }

        /// <summary>
        /// The value of the dimension. You can only call this method if `.isParam` is true.
        /// </summary>
        public byte param
        {
            get
            {
                //Logger.AssertIsTrue(m_DimType == DimType.Param, "Cannot get param of dim with type != DimType.Param");
                return m_Param;
            }
        }

        /// <summary>
        /// Returns a string that represents the `DynamicTensorDim`.
        /// </summary>
        /// <returns>The string representation of the `DynamicTensorDim`.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if the dim type is not implemented.</exception>
        public override string ToString()
        {
            return m_DimType switch
            {
                DimType.Unknown => k_UnknownName,
                DimType.Static => value.ToString(),
                DimType.Param => "d" + (int)param,
                _ => throw new ArgumentOutOfRangeException()
            };
        }

        public bool Equals(DynamicTensorDim other)
        {
            return m_DimType == other.m_DimType && m_Value == other.m_Value && m_Param == other.m_Param;
        }

        /// <summary>
        /// Determines whether the specified object is equal to the current `DynamicTensorDim`.
        /// </summary>
        /// <param name="obj">The object to compare against.</param>
        /// <returns>Whether the object is equal to the current `DynamicTensorDim`.</returns>
        public override bool Equals(object obj)
        {
            return obj is DynamicTensorDim other && Equals(other);
        }

        /// <summary>
        /// Whether the current 'DynamicTensorDim' is 'DimType.Value' and is equal to the specified dim.
        /// </summary>
        /// <param name="other">The 'DynamicTensorDim' to compare against.</param>
        /// <returns>Whether the other `DynamicTensorDim` is is a value and is equal to the current `DynamicTensorDim`.</returns>
        public bool EqualsValue(DynamicTensorDim other)
        {
            return m_DimType == DimType.Static && other.m_DimType == DimType.Static && m_Value == other.m_Value;
        }

        /// <summary>
        /// Whether the current 'DynamicTensorDim' is 'DimType.Param' and is equal to the specified dim.
        /// </summary>
        /// <param name="other">The 'DynamicTensorDim' to compare against.</param>
        /// <returns>Whether the other `DynamicTensorDim` is is a param and is equal to the current `DynamicTensorDim`.</returns>
        public bool EqualsParam(DynamicTensorDim other)
        {
            return m_DimType == DimType.Param && other.m_DimType == DimType.Param && m_Param == other.m_Param;
        }

        /// <summary>
        /// Determines whether two 'DynamicTensorDim' objects are equal.
        /// </summary>
        /// <param name="a">The first 'DynamicTensorDim' to compare.</param>
        /// <param name="b">The second 'DynamicTensorDim' to compare.</param>
        /// <returns>Whether the two 'DynamicTensorDim' objects are equal.</returns>
        public static bool operator ==(DynamicTensorDim a, DynamicTensorDim b)
        {
            if (a.m_DimType != b.m_DimType)
                return false;
            if (a.m_Value != b.m_Value)
                return false;
            if (a.m_Param != b.m_Param)
                return false;
            return true;
        }

        /// <summary>
        /// Determines whether two 'DynamicTensorDim' objects are not equal.
        /// </summary>
        /// <param name="a">The first 'DynamicTensorDim' to compare.</param>
        /// <param name="b">The second 'DynamicTensorDim' to compare.</param>
        /// <returns>Whether the two 'DynamicTensorDim' objects are not equal.</returns>
        public static bool operator !=(DynamicTensorDim a, DynamicTensorDim b)
        {
            return a.isValue && b.isValue && a.m_Value != b.m_Value;
        }

        /// <summary>
        /// Determines whether a 'DynamicTensorDim' is equal to a value.
        /// </summary>
        /// <param name="a">The 'DynamicTensorDim' to compare.</param>
        /// <param name="b">The integer value to compare.</param>
        /// <returns>Whether the 'DynamicTensorDim' object is equal to the value.</returns>
        public static bool operator ==(DynamicTensorDim a, int b)
        {
            return a.isValue && a.m_Value == b;
        }

        /// <summary>
        /// Determines whether a 'DynamicTensorDim' is not equal to a value.
        /// </summary>
        /// <param name="a">The 'DynamicTensorDim' to compare.</param>
        /// <param name="b">The integer value to compare.</param>
        /// <returns>Whether the 'DynamicTensorDim' object is not equal to the value.</returns>
        public static bool operator !=(DynamicTensorDim a, int b)
        {
            return a.isValue && a.m_Value != b;
        }

        /// <summary>
        /// Determines whether a 'DynamicTensorDim' is equal to a value.
        /// </summary>
        /// <param name="a">The integer value to compare.</param>
        /// <param name="b">The 'DynamicTensorDim' to compare.</param>
        /// <returns>Whether the 'DynamicTensorDim' object is equal to the value.</returns>
        public static bool operator ==(int a, DynamicTensorDim b)
        {
            return b.isValue && a == b.m_Value;
        }

        /// <summary>
        /// Determines whether a 'DynamicTensorDim' is not equal to a value.
        /// </summary>
        /// <param name="a">The integer value to compare.</param>
        /// <param name="b">The 'DynamicTensorDim' to compare.</param>
        /// <returns>Whether the 'DynamicTensorDim' object is not equal to the value.</returns>
        public static bool operator !=(int a, DynamicTensorDim b)
        {
            return b.isValue && a != b.m_Value;
        }

        /// <summary>
        /// Adds two `DynamicTensorDim` objects.
        /// </summary>
        /// <param name="a">The LHS 'DynamicTensorDim' of the operation.</param>
        /// <param name="b">The RHS 'DynamicTensorDim' of the operation.</param>
        /// <returns>The result of the add operation.</returns>
        public static DynamicTensorDim operator +(DynamicTensorDim a, DynamicTensorDim b)
        {
            if (a.isValue)
                return a.value + b;
            if (b.isValue)
                return b.value + a;
            return Unknown;
        }

        //   | 0   1   A   ?
        // --|-----------------
        // 0 | 0   4   A   ?
        // 3 | 3   4   ?   ?
        /// <summary>
        /// Adds a `DynamicTensorDim` to an `int`.
        /// </summary>
        /// <param name="a">The LHS integer of the operation.</param>
        /// <param name="b">The RHS 'DynamicTensorDim' of the operation.</param>
        /// <returns>The result of the add operation.</returns>
        public static DynamicTensorDim operator +(int a, DynamicTensorDim b)
        {
            if (b.isValue)
                return Int(a + b.value);
            if (a == 0)
                return b;
            return Unknown;
        }

        //   | 0   1
        // --|--------
        // 0 | 0   1
        // 3 | 3   4
        // A | A   ?
        // ? | ?   ?
        /// <summary>
        /// Adds an `int` to a `DynamicTensorDim`.
        /// </summary>
        /// <param name="a">The LHS 'DynamicTensorDim' of the operation.</param>
        /// <param name="b">The RHS integer of the operation.</param>
        /// <returns>The result of the add operation.</returns>
        public static DynamicTensorDim operator +(DynamicTensorDim a, int b)
        {
            return b + a;
        }

        //   | 0   1   A   B   ?
        // --|---------------------
        // 3 | 3   2   ?   ?   ?
        // A | A   ?   0   ?   ?
        // ? | ?   ?   ?   ?   ?
        /// <summary>
        /// Subtracts a `DynamicTensorDim` from another `DynamicTensorDim`.
        /// </summary>
        /// <param name="a">The LHS 'DynamicTensorDim' of the operation.</param>
        /// <param name="b">The RHS 'DynamicTensorDim' of the operation.</param>
        /// <returns>The result of the subtract operation.</returns>
        public static DynamicTensorDim operator -(DynamicTensorDim a, DynamicTensorDim b)
        {
            if (a.isValue)
                return a.value - b;
            if (b.isValue)
                return a - b.value;
            if (a.isParam && b.isParam && a.param == b.param)
                return Zero;
            return Unknown;
        }

        //   | 0   1   A   B   ?
        // --|---------------------
        // 3 | 3   2   ?   ?   ?
        /// <summary>
        /// Subtracts a `DynamicTensorDim` from an `int`.
        /// </summary>
        /// <param name="a">The LHS integer of the operation.</param>
        /// <param name="b">The RHS 'DynamicTensorDim' of the operation.</param>
        /// <returns>The result of the subtract operation.</returns>
        public static DynamicTensorDim operator -(int a, DynamicTensorDim b)
        {
            if (b.isValue)
                return Int(a - b.value);
            return Unknown;
        }

        //   | 0   1
        // --|---------
        // 3 | 3   2
        // A | A   ?
        // ? | ?   ?
        /// <summary>
        /// Subtracts an `int` from a `DynamicTensorDim`.
        /// </summary>
        /// <param name="a">The LHS 'DynamicTensorDim' of the operation.</param>
        /// <param name="b">The RHS integer of the operation.</param>
        /// <returns>The result of the subtract operation.</returns>
        public static DynamicTensorDim operator -(DynamicTensorDim a, int b)
        {
            if (a.isValue)
                return Int(a.value - b);
            if (b == 0)
                return a;
            return Unknown;
        }

        //   | 0   1   3   A   B   ?
        // --|-----------------------
        // 0 | 0   0   0   0   0   0
        // 2 | 0   2   6   ?   ?   ?
        // A | 0   A   ?   ?   ?   ?
        // ? | 0   ?   ?   ?   ?   ?
        /// <summary>
        /// Multiplies two `DynamicTensorDim` dimensions.
        /// </summary>
        /// <param name="a">The LHS 'DynamicTensorDim' of the operation.</param>
        /// <param name="b">The RHS 'DynamicTensorDim' of the operation.</param>
        /// <returns>The result of the multiply operation.</returns>
        public static DynamicTensorDim operator *(DynamicTensorDim a, DynamicTensorDim b)
        {
            if (a.isValue)
                return a.value * b;
            if (b.isValue)
                return b.value * a;
            return Unknown;
        }

        //   | 1   3   A   B   ?
        // --|--------------------
        // 0 | 0   0   0   0   0
        // 2 | 2   6   ?   ?   ?
        /// <summary>
        /// Multiplies an `int` by a `DynamicTensorDim`.
        /// </summary>
        /// <param name="a">The LHS integer of the operation.</param>
        /// <param name="b">The RHS 'DynamicTensorDim' of the operation.</param>
        /// <returns>The result of the multiply operation.</returns>
        public static DynamicTensorDim operator *(int a, DynamicTensorDim b)
        {
            if (b.isValue)
                return Int(a * b.value);
            if (a == 1)
                return b;
            if (a == 0)
                return Zero;
            return Unknown;
        }

        //   | 0   1   3
        // --|-----------
        // 2 | 0   2   6
        // A | 0   A   ?
        // ? | 0   ?   ?
        /// <summary>
        /// Multiplies a `DynamicTensorDim` by an `int`.
        /// </summary>
        /// <param name="a">The LHS 'DynamicTensorDim' of the operation.</param>
        /// <param name="b">The RHS integer of the operation.</param>
        /// <returns>The result of the multiply operation.</returns>
        public static DynamicTensorDim operator *(DynamicTensorDim a, int b)
        {
            return b * a;
        }

        //   | 1   2   3   A   B   ?
        // --|-----------------------
        // 0 | 0   0   0   0   0   0
        // 2 | 2   1  Err  ?   ?   ?
        // A | A   3   ?   1   ?   ?
        // ? | ?   ?   ?   ?   ?   ?
        /// <summary>
        /// Divides two `DynamicTensorDim` dimensions a whole number of times. The method throws an error if the result has a remainder.
        /// </summary>
        /// <param name="a">The LHS 'DynamicTensorDim' of the operation.</param>
        /// <param name="b">The RHS 'DynamicTensorDim' of the operation.</param>
        /// <returns>The result of the divide operation.</returns>
        public static DynamicTensorDim operator /(DynamicTensorDim a, DynamicTensorDim b)
        {
            if (a.isValue)
                return a.value / b;
            if (b.isValue)
                return a / b.value;
            if (a.isParam && b.isParam && a.param == b.param)
                return One;
            return Unknown;
        }

        //   | 1   2   3   A   ?
        // --|--------------------
        // 0 | 0   0   0   0   0
        // 2 | 2   1  Err  ?   ?
        /// <summary>
        /// Divides an `int` by a `DynamicTensorDim` a whole number of times. The method throws an error if the result has a remainder.
        /// </summary>
        /// <param name="a">The LHS integer of the operation.</param>
        /// <param name="b">The RHS 'DynamicTensorDim' of the operation.</param>
        /// <returns>The result of the divide operation.</returns>
        public static DynamicTensorDim operator /(int a, DynamicTensorDim b)
        {
            if (a == 0)
                return Zero;
            if (b.isValue)
            {
                Logger.AssertIsTrue(b.value != 0, "ValueError: cannot divide by dim of size 0");
                Logger.AssertIsTrue(a % b.value == 0, "ValueError: cannot divide DynamicTensorDims exactly");
                return Int(a / b.value);
            }

            return Unknown;
        }

        //   | 1   2   3
        // --|------------
        // 2 | 2   1  Err
        // A | A   3   ?
        // ? | ?   ?   ?
        /// <summary>
        /// Divides a `DynamicTensorDim` by an `int` a whole number of times. The method throws an error if the result has a remainder.
        /// </summary>
        /// <param name="a">The LHS 'DynamicTensorDim' of the operation.</param>
        /// <param name="b">The RHS integer of the operation.</param>
        /// <returns>The result of the divide operation.</returns>
        public static DynamicTensorDim operator /(DynamicTensorDim a, int b)
        {
            if (a.isValue)
            {
                Logger.AssertIsTrue(b != 0, "ValueError: cannot divide by dim of size 0");
                Logger.AssertIsTrue(a.value % b == 0, "ValueError: cannot divide DynamicTensorDims exactly");
                return Int(a.value / b);
            }
            if (b == 1)
                return a;
            return Unknown;
        }

        // with rounding direction = 1
        //   |  0   1   2
        // --|-------------
        // 0 | Err  0   0
        // 1 | Err  1   1
        // 2 | Err  2   1
        // A | Err  A   ?
        // ? | Err  ?   ?
        /// <summary>
        /// Divides a `DynamicTensorDim` by a `float` to return a rounded `DynamicTensorDim`.
        /// rounding direction greater than 0 = ceiling
        /// rounding direction less than 0 = floor
        /// rounding direction equals 0 = round
        /// </summary>
        public DynamicTensorDim DivideWithRounding(int b, int roundingDirection)
        {
            if (b == 1)
                return this;

            Logger.AssertIsTrue(b != 0, "ValueError: cannot divide by dim of size 0");

            if (!isValue)
                return Unknown;

            var v = value / (float)b;
            if (roundingDirection > 0)
                return Int(Mathf.CeilToInt(v));
            if (roundingDirection < 0)
                return Int(Mathf.FloorToInt(v));
            return Int(Mathf.RoundToInt(v));
        }

        /// <summary>
        /// Whether a `DynamicTensorDim` is known to be less than a given integer value.
        /// </summary>
        /// <param name="d">The `DynamicTensorDim` to compare.</param>
        /// <param name="v">The integer value to compare.</param>
        /// <returns>The result of the comparison</returns>
        public static bool operator <(DynamicTensorDim d, int v)
        {
            return d.m_DimType == DimType.Static && d.m_Value < v;
        }

        /// <summary>
        /// Whether a `DynamicTensorDim` is known to be greater than than a given integer value.
        /// </summary>
        /// <param name="d">The `DynamicTensorDim` to compare.</param>
        /// <param name="v">The integer value to compare.</param>
        /// <returns>The result of the comparison</returns>
        public static bool operator >(DynamicTensorDim d, int v)
        {
            return d.m_DimType == DimType.Static && d.m_Value > v;
        }

        /// <summary>
        /// Whether a `DynamicTensorDim` is known to be less than or equal to than a given integer value.
        /// </summary>
        /// <param name="d">The `DynamicTensorDim` to compare.</param>
        /// <param name="v">The integer value to compare.</param>
        /// <returns>The result of the comparison</returns>
        public static bool operator <=(DynamicTensorDim d, int v)
        {
            return d.m_DimType == DimType.Static && d.m_Value <= v;
        }

        /// <summary>
        /// Whether a `DynamicTensorDim` is known to be greater than or equal to than a given integer value.
        /// </summary>
        /// <param name="d">The `DynamicTensorDim` to compare.</param>
        /// <param name="v">The integer value to compare.</param>
        /// <returns>The result of the comparison</returns>
        public static bool operator >=(DynamicTensorDim d, int v)
        {
            return d.m_DimType == DimType.Static && d.m_Value >= v;
        }

        //   | 2   3   A   B   ?
        // --|-------------------
        // 2 | 2  Err  2   2   2
        // A | 2   3   A   A   A
        // ? | 2   3   A   B   ?
        /// <summary>
        /// Returns the better known of two `DynamicTensorDim` dimensions known to be equal. The method throws an error if both dimensions are values and not equal.
        /// </summary>
        /// <param name="a">The first `DynamicTensorDim`.</param>
        /// <param name="b">The second `DynamicTensorDim`.</param>
        /// <returns>The better known of the `DynamicTensorDim` objects.</returns>
        public static DynamicTensorDim MaxDefinedDim(DynamicTensorDim a, DynamicTensorDim b)
        {
            if (a.isUnknown)
                return b;
            if (b.isUnknown)
                return a;
            if (b.isValue)
            {
                Logger.AssertIsTrue(!a.isValue || b == a, "ValueError: value dims must be equal");
                return b;
            }

            return a;
        }

        //   | 1   3   A   B   ?
        // --|-----------------
        // 1 | 1   3   A   B   ?
        // 2 | 2  Err  2   2   2
        // A | A   3   A   ?   ?
        // ? | ?   3   ?   ?   ?
        /// <summary>
        /// Broadcasts two `DynamicTensorDim` dimensions using a broadcast rule where a dimension of size 1 can broadcast with any other dimension.
        /// </summary>
        public static DynamicTensorDim Broadcast(DynamicTensorDim a, DynamicTensorDim b)
        {
            if (a == One)
                return b;
            if (b == One)
                return a;
            if (a == b)
                return a;
            if (a.isValue && b.isValue)
                Logger.AssertIsTrue(a == b, "ValueError: broadcast dims must be equal or 1");
            if (a.isValue)
                return a;
            if (b.isValue)
                return b;
            return Unknown;
        }

        //   | 1   3   B   ?
        // --|-----------------
        // 1 | 1   3   B   ?
        // 2 | 2  Err  2   2
        // A | A   3   ?   ?
        // ? | ?   3   ?   ?
        /// <summary>
        /// Broadcasts the `DynamicTensorDim` with another `DynamicTensorDim` using a broadcast rule where a dimension of size 1 can broadcast with any other dimension.
        /// </summary>
        public DynamicTensorDim Broadcast(DynamicTensorDim other)
        {
            return Broadcast(this, other);
        }

        public DynamicTensorDim Pool(int kernel, int stride, int padding, int dilation, bool ceilMode, Layers.AutoPad autoPad)
        {
            switch (autoPad)
            {
                case Layers.AutoPad.Valid:
                    return (this - ((kernel - 1) * dilation + 1) + 1).DivideWithRounding(stride, 1);
                case Layers.AutoPad.SameLower:
                case Layers.AutoPad.SameUpper:
                    return DivideWithRounding(stride, 1);
                case Layers.AutoPad.NotSet:
                    return (this + padding - ((kernel - 1) * dilation + 1)).DivideWithRounding(stride, ceilMode ? 1 : -1) + 1;
                default:
                    throw new InvalidEnumArgumentException();
            }
        }

        public DynamicTensorDim Slice(PartialTensorElement start, PartialTensorElement end, PartialTensorElement step)
        {
            Logger.AssertIsTrue(!(step == 0), "Slice.InputError: Step cannot be 0");

            if (isValue && start.isIntValue && end.isIntValue && step.isIntValue)
            {
                if (value == 0)
                    return Zero;

                var clampAdjustDirection = step < 0 ? -1 : 0;

                var startValue = start.intValue < 0 ? value + start.intValue : start.intValue;
                startValue = Mathf.Clamp(startValue, 0, value + clampAdjustDirection);

                var endValue = end.intValue < 0 ? value + end.intValue : end.intValue;
                endValue = Mathf.Clamp(endValue, clampAdjustDirection, value);

                var outputDim = (int)Math.Ceiling((double)(endValue - startValue) / (double)step.intValue);
                return Int(Mathf.Max(outputDim, 0));
            }

            if (start.isUnknown || end.isUnknown)
                return Unknown;

            if (start == end)
                return Zero;

            var dimXElement = (PartialTensorElement)this;
            if (step > 0)
            {
                if (start == dimXElement || start == int.MaxValue || start >= this)
                    return Zero;
                if (end == 0 || end == int.MinValue || this >= -end)
                    return Zero;
                if (step == 1 && (start == 0 || start == int.MinValue) && (end == dimXElement || end == int.MaxValue))
                    return this;
            }
            else if (step < 0)
            {
                if (end == dimXElement || end == int.MaxValue || end >= this)
                    return Zero;
                if (start == 0 || start == int.MinValue || this >= -start)
                    return Zero;
                if (step == -1 && (end == -1 || end == int.MinValue) && (start == this || start == int.MaxValue))
                    return this;
            }

            return Unknown;
        }

        /// <summary>
        /// Calculates the greatest common divisor of two `DynamicTensorDim` objects.
        /// </summary>
        /// <param name="a">The first `DynamicTensorDim`.</param>
        /// <param name="b">The second `DynamicTensorDim`.</param>
        /// <returns>The greatest common divisor of the `DynamicTensorDim` objects.</returns>
        public static DynamicTensorDim GCD(DynamicTensorDim a, DynamicTensorDim b)
        {
            if (a == One || b == One)
                return One;
            if (a.isUnknown || b.isUnknown)
                return Unknown;
            if (a == b)
                return a;
            if (a == Zero)
                return b;
            if (b == Zero)
                return a;
            if (a.isParam || b.isParam)
                return Unknown;
            var x = a.value;
            var y = b.value;
            while (x != 0 && y != 0)
            {
                if (x > y)
                    x %= y;
                else
                    y %= x;
            }
            return Int(x | y);
        }

        public DynamicTensorDim Resize(PartialTensorElement e)
        {
            return !e.isFloatValue ? Unknown : Resize(e.floatValue);
        }

        public DynamicTensorDim Resize(float f)
        {
            if (f == 0)
                return Zero;
            // ReSharper disable once CompareOfFloatsByEqualityOperator
            if (f == 1)
                return this;
            if (!isValue)
                return Unknown;
            return Int(Mathf.RoundToInt(value * f));
        }

        /// <summary>
        /// Serves as the default hash function.
        /// </summary>
        /// <returns>The calculated hash code.</returns>
        public override int GetHashCode()
        {
            return m_DimType.GetHashCode() ^ m_Param.GetHashCode() ^ m_Value.GetHashCode();
        }
    }
}
