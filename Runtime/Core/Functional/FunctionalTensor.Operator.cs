using System;
using UnityEngine;

namespace Unity.Sentis
{
    public partial class FunctionalTensor
    {
        /// <summary>
        /// Unary plus operator.
        /// </summary>
        /// <param name="a">The operand tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator +(FunctionalTensor a) => a;

        /// <summary>
        /// Unary negation operator.
        /// </summary>
        /// <param name="a">The operand tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator -(FunctionalTensor a) => Functional.Neg(a);

        /// <summary>
        /// Addition operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator +(FunctionalTensor a, FunctionalTensor b) => Functional.Add(a, b);

        /// <summary>
        /// Addition operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator +(FunctionalTensor a, int b) => ScalarMad(a, 1, b);

        /// <summary>
        /// Addition operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator +(int a, FunctionalTensor b) => b + a;

        /// <summary>
        /// Addition operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator +(FunctionalTensor a, float b) => ScalarMad(a, 1, b);

        /// <summary>
        /// Addition operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator +(float a, FunctionalTensor b) => b + a;

        /// <summary>
        /// Subtraction operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator -(FunctionalTensor a, FunctionalTensor b) => Functional.Sub(a, b);

        /// <summary>
        /// Subtraction operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator -(FunctionalTensor a, int b) => ScalarMad(a, 1, -b);

        /// <summary>
        /// Subtraction operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator -(int a, FunctionalTensor b) => ScalarMad(b, -1, a);

        /// <summary>
        /// Subtraction operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator -(FunctionalTensor a, float b) => ScalarMad(a, 1, -b);

        /// <summary>
        /// Subtraction operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator -(float a, FunctionalTensor b) => ScalarMad(b, -1, a);

        /// <summary>
        /// Multiply operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator *(FunctionalTensor a, FunctionalTensor b) => Functional.Mul(a, b);

        /// <summary>
        /// Multiply operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator *(int a, FunctionalTensor b) => ScalarMad(b, a, 0);

        /// <summary>
        /// Multiply operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator *(FunctionalTensor a, int b) => ScalarMad(a, b, 0);

        /// <summary>
        /// Multiply operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator *(float a, FunctionalTensor b) => ScalarMad(b, a, 0);

        /// <summary>
        /// Multiply operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator *(FunctionalTensor a, float b) => ScalarMad(a, b, 0);

        /// <summary>
        /// Division operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator /(FunctionalTensor a, FunctionalTensor b) => Functional.Div(a, b);

        /// <summary>
        /// Division operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator /(FunctionalTensor a, int b) => a.dataType == DataType.Float ? a / (float)b : Functional.Div(a, Functional.Constant(b));

        /// <summary>
        /// Division operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator /(int a, FunctionalTensor b) => b.dataType == DataType.Float ? (float)a / b : Functional.Div(Functional.Constant(a), b);

        /// <summary>
        /// Division operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator /(FunctionalTensor a, float b) => ScalarMad(a, 1 / b, 0);

        /// <summary>
        /// Division operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator /(float a, FunctionalTensor b) => a * Functional.Reciprocal(b);

        /// <summary>
        /// Remainder operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator %(FunctionalTensor a, FunctionalTensor b) => Functional.Remainder(a, b);

        /// <summary>
        /// Remainder operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator %(FunctionalTensor a, int b) => Functional.Remainder(a, Functional.Constant(b));

        /// <summary>
        /// Remainder operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator %(int a, FunctionalTensor b) => Functional.Remainder(Functional.Constant(a), b);

        /// <summary>
        /// Remainder operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator %(FunctionalTensor a, float b) => Functional.Remainder(a, Functional.Constant(b));

        /// <summary>
        /// Remainder operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator %(float a, FunctionalTensor b) => Functional.Remainder(Functional.Constant(a), b);

        /// <summary>
        /// Greater than operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator >(FunctionalTensor a, FunctionalTensor b) => Functional.Greater(a, b);

        /// <summary>
        /// Greater than operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator >(FunctionalTensor a, int b) => a.dataType == DataType.Float ? a > (float)b : a > Functional.Constant(b);

        /// <summary>
        /// Greater than operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator >(FunctionalTensor a, float b) => a.dataType == DataType.Int ? a > Mathf.FloorToInt(b) : a > Functional.Constant(b);

        /// <summary>
        /// Greater than operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator >(int a, FunctionalTensor b) => b < a;

        /// <summary>
        /// Greater than operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator >(float a, FunctionalTensor b) => b < a;

        /// <summary>
        /// Less than operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator <(FunctionalTensor a, FunctionalTensor b) => Functional.Less(a, b);

        /// <summary>
        /// Less than operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator <(FunctionalTensor a, int b) => a.dataType == DataType.Float ? a < (float)b : a < Functional.Constant(b);

        /// <summary>
        /// Less than operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator <(FunctionalTensor a, float b) => a.dataType == DataType.Int ? a < Mathf.CeilToInt(b) : a < Functional.Constant(b);

        /// <summary>
        /// Less than operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator <(int a, FunctionalTensor b) => b > a;

        /// <summary>
        /// Less than operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator <(float a, FunctionalTensor b) => b > a;

        /// <summary>
        /// Greater than or equal operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator >=(FunctionalTensor a, FunctionalTensor b) => Functional.GreaterEqual(a, b);

        /// <summary>
        /// Greater than or equal operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator >=(FunctionalTensor a, int b) => a.dataType == DataType.Float ? a >= (float)b : a >= Functional.Constant(b);

        /// <summary>
        /// Greater than or equal operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator >=(FunctionalTensor a, float b) => a.dataType == DataType.Int ? a >= Mathf.CeilToInt(b) : a >= Functional.Constant(b);

        /// <summary>
        /// Greater than or equal operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator >=(int a, FunctionalTensor b) => b <= a;

        /// <summary>
        /// Greater than or equal operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator >=(float a, FunctionalTensor b) => b <= a;

        /// <summary>
        /// Less than or equal operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator <=(FunctionalTensor a, FunctionalTensor b) => Functional.LessEqual(a, b);

        /// <summary>
        /// Less than or equal operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator <=(FunctionalTensor a, int b) => a.dataType == DataType.Float ? a <= (float)b : a <= Functional.Constant(b);

        /// <summary>
        /// Less than or equal operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator <=(FunctionalTensor a, float b) => a.dataType == DataType.Int ? a <= Mathf.FloorToInt(b) : a <= Functional.Constant(b);

        /// <summary>
        /// Less than or equal operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator <=(int a, FunctionalTensor b) => b >= a;

        /// <summary>
        /// Less than or equal operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator <=(float a, FunctionalTensor b) => b >= a;

        /// <summary>
        /// Unary not operator.
        /// </summary>
        /// <param name="a">The operand tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator ~(FunctionalTensor a) => Functional.LogicalNot(a);

        /// <summary>
        /// Xor operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator ^(FunctionalTensor a, FunctionalTensor b) => Functional.LogicalXor(a, b);

        /// <summary>
        /// Xor operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator ^(FunctionalTensor a, bool b) => a ^ Functional.Constant(b ? 1 : 0);

        /// <summary>
        /// Xor operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator ^(bool a, FunctionalTensor b) => b ^ a;

        /// <summary>
        /// And operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator &(FunctionalTensor a, FunctionalTensor b) => Functional.LogicalAnd(a, b);

        /// <summary>
        /// And operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator &(FunctionalTensor a, bool b) => a & Functional.Constant(b ? 1 : 0);

        /// <summary>
        /// And operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator &(bool a, FunctionalTensor b) => b & a;

        /// <summary>
        /// Or operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator |(FunctionalTensor a, FunctionalTensor b) => Functional.LogicalOr(a, b);

        /// <summary>
        /// Or operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator |(FunctionalTensor a, bool b) => a | Functional.Constant(b ? 1 : 0);

        /// <summary>
        /// Or operator.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor operator |(bool a, FunctionalTensor b) => b | a;

        // helper for operators with float values
        static FunctionalTensor ScalarMad(FunctionalTensor input, float s, float b)
        {
            input = input.Float();
            var output = Functional.FromLayer(new Layers.ScalarMad(-1, -1, s, b), DataType.Float, new[] { input });
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }

        // helper for operators with int values, type promotion to floats if needed
        static FunctionalTensor ScalarMad(FunctionalTensor input, int s, int b)
        {
            if (input.dataType == DataType.Float)
                return ScalarMad(input, (float)s, b);
            var output = Functional.FromLayer(new Layers.ScalarMad(-1, -1, s, b), DataType.Int, new[] { input });
            if (input.isShapeKnown)
                output.SetShape(input.shape);
            return output;
        }
    }
}
