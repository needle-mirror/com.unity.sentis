using System;
using UnityEngine;

namespace Unity.Sentis
{
    [Serializable]
    enum ElementType
    {
        Unknown = 0,
        Value,
        Param
    }

    /// <summary>
    /// Represents a single element of a SymbolicTensorShape, can be an int value, char param or unknown
    /// </summary>
    [Serializable]
    struct PartialTensorElement
    {
        ElementType m_ElementType;
        char m_Param;
        int m_Value;

        public static PartialTensorElement Unknown => new PartialTensorElement();

        public PartialTensorElement(int value)
        {
            m_ElementType = ElementType.Value;
            m_Param = default;
            m_Value = value;
        }

        public PartialTensorElement(char param)
        {
            Logger.AssertIsTrue(param >= 0, "Element param cannot be negative");
            m_ElementType = ElementType.Param;
            m_Param = param;
            m_Value = default;
        }

        public bool isUnknown => m_ElementType == ElementType.Unknown;
        public bool isValue => m_ElementType == ElementType.Value;
        public bool isParam => m_ElementType == ElementType.Param;

        public static PartialTensorElement Zero => new PartialTensorElement(0);
        public static PartialTensorElement One => new PartialTensorElement(1);

        public int value
        {
            get
            {
                Logger.AssertIsTrue(m_ElementType == ElementType.Value, "Cannot get value of element with type != ElementType.Value");
                return m_Value;
            }
        }

        public char param
        {
            get
            {
                Logger.AssertIsTrue(m_ElementType == ElementType.Param, "Cannot get param of element with type != ElementType.Param");
                return m_Param;
            }
        }

        public SymbolicTensorDim ToSymbolicTensorDim()
        {
            switch (m_ElementType)
            {
                case ElementType.Unknown:
                    return SymbolicTensorDim.Unknown;
                case ElementType.Value:
                    return value < 0 ? SymbolicTensorDim.Unknown : new SymbolicTensorDim(value);
                case ElementType.Param:
                    return new SymbolicTensorDim(param);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static PartialTensorElement FromSymbolicTensorDim(SymbolicTensorDim dim)
        {
            if (dim.isParam)
                return new PartialTensorElement(dim.param);
            if (dim.isValue)
                return new PartialTensorElement(dim.value);
            return Unknown;
        }

        /// <summary>
        /// Returns a string that represents the `PartialTensorElement`.
        /// </summary>
        public override string ToString()
        {
            return m_ElementType switch
            {
                ElementType.Unknown => "?",
                ElementType.Value => value.ToString(),
                ElementType.Param => param.ToString(),
                _ => throw new ArgumentOutOfRangeException()
            };
        }

        public bool Equals(PartialTensorElement other)
        {
            return m_ElementType == other.m_ElementType && m_Value == other.m_Value && m_Param == other.m_Param;
        }

        /// <summary>
        /// Determines whether the specified object is equal to the current `PartialTensorElement`.
        /// </summary>
        public override bool Equals(object obj)
        {
            return obj is PartialTensorElement other && Equals(other);
        }

        /// <summary>
        ///
        /// Compares element to element
        /// ==
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | T   F   F   F   F
        /// 3 | F   F   F   F   F
        /// A | F   F   T   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator ==(PartialTensorElement a, PartialTensorElement b)
        {
            if (a.m_ElementType != b.m_ElementType)
                return false;
            if (a.m_Value != b.m_Value)
                return false;
            if (a.m_Param != b.m_Param)
                return false;
            return true;
        }

        /// <summary>
        ///
        /// Compares element to element
        /// !=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   T   T   T
        /// 3 | T   T   T   T   T
        /// A | T   T   F   T   T
        /// ? | T   T   T   T   T
        ///
        /// </summary>
        public static bool operator !=(PartialTensorElement a, PartialTensorElement b)
        {
            return !(a == b);
        }

        /// <summary>
        ///
        /// Compares element to int
        /// ==
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | T   F   F   F   F
        /// 3 | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator ==(PartialTensorElement a, int b)
        {
            if (!a.isValue)
                return false;
            return a.m_Value == b;
        }

        /// <summary>
        ///
        /// Compares element to int
        /// !=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   T   T   T
        /// 3 | T   T   T   T   T
        ///
        /// </summary>
        public static bool operator !=(PartialTensorElement a, int b)
        {
            return !(a == b);
        }

        /// <summary>
        ///
        /// Compares int to element
        /// ==
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | T   F   F   F   F
        /// 3 | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator ==(int a, PartialTensorElement b)
        {
            return b == a;
        }

        /// <summary>
        ///
        /// Compares int to element
        /// !=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   T   T   T
        /// 3 | T   T   T   T   T
        ///
        /// </summary>
        public static bool operator !=(int a, PartialTensorElement b)
        {
            return b != a;
        }

        /// <summary>
        ///
        /// Compares element to symbolic tensor dim
        /// ==
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | T   F   F   F   F
        /// 3 | F   F   F   F   F
        /// A | F   F   T   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator ==(PartialTensorElement a, SymbolicTensorDim b)
        {
            if (a.isParam && b.isParam)
                return a.param == b.param;
            if (a.isValue && b.isValue)
                return a.value == b.value;
            return false;
        }

        /// <summary>
        ///
        /// Compares element to symbolic tensor dim
        /// !=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   T   T   T
        /// 3 | T   T   T   T   T
        /// A | T   T   F   T   T
        /// ? | T   T   T   T   T
        ///
        /// </summary>
        public static bool operator !=(PartialTensorElement a, SymbolicTensorDim b)
        {
            return !(a == b);
        }

        /// <summary>
        ///
        /// Compares element to int
        /// >
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   F   F   F
        /// 3 | T   T   F   F   F
        ///
        /// </summary>
        public static bool operator >(PartialTensorElement a, int b)
        {
            if (!a.isValue)
                return false;
            return a.m_Value > b;
        }

        /// <summary>
        ///
        /// Compares element to int
        /// <
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   F   F   F
        /// 3 | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator <(PartialTensorElement a, int b)
        {
            if (!a.isValue)
                return false;
            return a.m_Value < b;
        }

        /// <summary>
        ///
        /// Compares element to symbolic tensor dim
        /// >
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   F   F   F
        /// 3 | T   T   F   F   F
        /// A | F   F   F   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator >(PartialTensorElement a, SymbolicTensorDim b)
        {
            if (!a.isValue || !b.isValue)
                return false;
            return a.m_Value > b.value;
        }

        /// <summary>
        ///
        /// Compares element to symbolic tensor dim
        /// <
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   F   F   F
        /// 3 | F   F   F   F   F
        /// A | F   F   F   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator <(PartialTensorElement a, SymbolicTensorDim b)
        {
            if (!a.isValue || !b.isValue)
                return false;
            return a.m_Value < b.value;
        }

        /// <summary>
        ///
        /// Compares element to symbolic tensor dim
        /// >=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   F   F   F   F
        /// 3 | T   T   F   F   F
        /// A | F   F   T   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator >=(PartialTensorElement a, SymbolicTensorDim b)
        {
            if (a.isParam && b.isParam)
                return a.param == b.param;
            if (a.isValue && b.isValue)
                return a.m_Value >= b.value;
            return false;
        }

        /// <summary>
        ///
        /// Compares element to symbolic tensor dim
        /// <=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | T   T   F   F   F
        /// 3 | F   F   F   F   F
        /// A | F   F   T   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator <=(PartialTensorElement a, SymbolicTensorDim b)
        {
            if (a.isParam && b.isParam)
                return a.param == b.param;
            if (a.isValue && b.isValue)
                return a.m_Value <= b.value;
            return false;
        }

        /// <summary>
        ///
        /// Compares symbolic tensor dim to element
        /// >=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   F   F   F   F
        /// 3 | T   T   F   F   F
        /// A | F   F   T   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator >=(SymbolicTensorDim a, PartialTensorElement b)
        {
            if (a.isParam && b.isParam)
                return a.param == b.param;
            if (a.isValue && b.isValue)
                return a.value >= b.value;
            return false;
        }

        /// <summary>
        ///
        /// Compares symbolic tensor dim to element
        /// <=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | T   T   F   F   F
        /// 3 | F   F   F   F   F
        /// A | F   F   T   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator <=(SymbolicTensorDim a, PartialTensorElement b)
        {
            if (a.isParam && b.isParam)
                return a.param == b.param;
            if (a.isValue && b.isValue)
                return a.value <= b.value;
            return false;
        }

        /// <summary>
        ///
        /// Subtracts element
        ///
        ///   | 0   1   A   B   ?
        /// --|---------------------
        ///   | 0   -1  ?   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator -(PartialTensorElement a)
        {
            if (a.isValue)
                return new PartialTensorElement(-a.value);
            return Unknown;
        }

        /// <summary>
        ///
        /// Adds element to element
        ///
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | 0   1   A   B   ?
        /// 3 | 3   4   ?   ?   ?
        /// A | A   ?   ?   ?   ?
        /// ? | ?   ?   ?   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator +(PartialTensorElement a, PartialTensorElement b)
        {
            if (a.isValue)
                return a.value + b;
            if (b.isValue)
                return b.value + a;
            return Unknown;
        }

        /// <summary>
        ///
        /// Adds element to int
        ///
        ///   | 0   1   A   ?
        /// --|-----------------
        /// 0 | 0   4   A   ?
        /// 3 | 3   4   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator +(int a, PartialTensorElement b)
        {
            if (b.isValue)
                return new PartialTensorElement(a + b.value);
            if (a == 0)
                return b;
            return Unknown;
        }

        /// <summary>
        ///
        /// Adds int to element
        ///
        ///   | 0   1
        /// --|--------
        /// 0 | 0   1
        /// 3 | 3   4
        /// A | A   ?
        /// ? | ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator +(PartialTensorElement a, int b)
        {
            return b + a;
        }

        /// <summary>
        ///
        /// Subtracts element from element
        ///
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 3 | 3   2   ?   ?   ?
        /// A | A   ?   0   ?   ?
        /// ? | ?   ?   ?   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator -(PartialTensorElement a, PartialTensorElement b)
        {
            if (a.isValue)
                return a.value - b;
            if (b.isValue)
                return a - b.value;
            if (a.isParam && b.isParam && a.param == b.param)
                return Zero;
            return Unknown;
        }

        /// <summary>
        /// Subtracts element from int
        ///
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 3 | 3   2   ?   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator -(int a, PartialTensorElement b)
        {
            if (b.isValue)
                return new PartialTensorElement(a - b.value);
            return Unknown;
        }

        /// <summary>
        ///
        /// Subtracts int from element
        ///
        ///   | 0   1
        /// --|---------
        /// 3 | 3   2
        /// A | A   ?
        /// ? | ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator -(PartialTensorElement a, int b)
        {
            if (a.isValue)
                return new PartialTensorElement(a.value - b);
            if (b == 0)
                return a;
            return Unknown;
        }

        /// <summary>
        /// Multiplies element by element
        ///
        ///   | 0   1   3   A   B   ?
        /// --|-----------------------
        /// 0 | 0   0   0   0   0   0
        /// 2 | 0   2   6   ?   ?   ?
        /// A | 0   A   ?   ?   ?   ?
        /// ? | 0   ?   ?   ?   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator *(PartialTensorElement a, PartialTensorElement b)
        {
            if (a.isValue)
                return a.value * b;
            if (b.isValue)
                return b.value * a;
            return Unknown;
        }

        /// <summary>
        /// Multiplies int by element
        ///
        ///   | 1   3   A   B   ?
        /// --|--------------------
        /// 0 | 0   0   0   0   0
        /// 2 | 2   6   ?   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator *(int a, PartialTensorElement b)
        {
            if (b.isValue)
                return new PartialTensorElement(a * b.value);
            if (a == 1)
                return b;
            if (a == 0)
                return Zero;
            return Unknown;
        }

        /// <summary>
        /// Multiplies element by int
        ///
        ///   | 0   1   3
        /// --|-----------
        /// 2 | 0   2   6
        /// A | 0   A   ?
        /// ? | 0   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator *(PartialTensorElement a, int b)
        {
            return b * a;
        }

        /// <summary>
        /// Returns the better known of two elements known to be equal, throws error if both elements are values and not equal
        ///
        ///   | 2   3   A   B   ?
        /// --|-------------------
        /// 2 | 2  Err  2   2   2
        /// A | 2   3   A   A   A
        /// ? | 2   3   A   B   ?
        ///
        /// </summary>
        public static PartialTensorElement MaxDefinedElement(PartialTensorElement a, PartialTensorElement b)
        {
            if (a.isUnknown)
                return b;
            if (b.isUnknown)
                return a;
            if (b.isValue)
            {
                Logger.AssertIsTrue(!a.isValue || b == a, "ValueError: value elements must be equal");
                return b;
            }

            return a;
        }

        /// <summary>
        /// Serves as the default hash function.
        /// </summary>
        public override int GetHashCode()
        {
            return m_ElementType.GetHashCode() ^ m_Param.GetHashCode() ^ m_Value.GetHashCode();
        }
    }
}
