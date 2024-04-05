using System;
using UnityEngine;

namespace Unity.Sentis
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns |input| element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Abs(FunctionalTensor input)
        {
            return FunctionalTensor.FromLayer(new Layers.Abs(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns acos(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Acos(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Acos(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns acosh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Acosh(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Acosh(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns input + other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Add(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FunctionalTensor.FromLayer(new Layers.Add(null, null, null), input.DataType, new[] { input, other });
        }

        /// <summary>
        /// Returns asin(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Asin(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Asin(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns asinh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Asinh(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Asinh(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns atan(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Atan(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Atan(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns atanh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Atanh(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Atanh(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns ⌈input⌉ element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Ceil(FunctionalTensor input)
        {
            if (input.DataType == DataType.Int)
                return input;
            return FunctionalTensor.FromLayer(new Layers.Ceil(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns input clamped to the range [min, max] element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="min">The min value.</param>
        /// <param name="max">The max value.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Clamp(FunctionalTensor input, float min, float max)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Clip(null, null, null, null), DataType.Float, new[] { input, Tensor(min), Tensor(max) });
        }

        /// <summary>
        /// Returns input clamped to the range [min, max] element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="min">The min value.</param>
        /// <param name="max">The max value.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Clamp(FunctionalTensor input, int min, int max)
        {
            if (input.DataType == DataType.Float)
                return Clamp(input, (float)min, max);
            return FunctionalTensor.FromLayer(new Layers.Clip(null, null, null, null), DataType.Int, new[] { input, Tensor(min), Tensor(max) });
        }

        /// <summary>
        /// Returns cos(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Cos(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Cos(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns cosh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Cosh(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Cosh(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns the input values converted from angles in degrees to radians element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Deg2Rad(FunctionalTensor input)
        {
            return Mathf.Deg2Rad * input;
        }

        /// <summary>
        /// Returns input / other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Div(FunctionalTensor input, FunctionalTensor other)
        {
            // TODO support rounding mode: string roundingMode
            (input, other) = PromoteTypes(input, other);
            return FunctionalTensor.FromLayer(new Layers.Div(null, null, null), input.DataType, new[] { input, other });
        }

        /// <summary>
        /// Returns the error function of input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Erf(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Erf(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns e^input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Exp(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Exp(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns input^exponent element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="exponent">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor FloatPower(FunctionalTensor input, FunctionalTensor exponent)
        {
            input = input.Float();
            exponent = exponent.Float();
            return FunctionalTensor.FromLayer(new Layers.Pow(null, null, null), DataType.Float, new[] { input, exponent });
        }

        /// <summary>
        /// Returns ⌊input⌋ element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Floor(FunctionalTensor input)
        {
            if (input.DataType == DataType.Int)
                return input;
            return FunctionalTensor.FromLayer(new Layers.Floor(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns ⌊input/other⌋ element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor FloorDivide(FunctionalTensor input, FunctionalTensor other)
        {
            return Floor(input / other);
        }

        /// <summary>
        /// Returns input % other element-wise. The sign of the output is the same as that of the dividend.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor FMod(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FunctionalTensor.FromLayer(new Layers.Mod(null, null, null, true), input.DataType, new[] { input, other });
        }

        /// <summary>
        /// Returns the fractional part of the input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Frac(FunctionalTensor input)
        {
            if (input.DataType == DataType.Int)
                return ZerosLike(input);
            // TODO add frac to backend and layers
            return input - Trunc(input);
        }

        /// <summary>
        /// Returns the linear interpolation input + weight * (end - input) element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="end">The second input tensor.</param>
        /// <param name="weight">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Lerp(FunctionalTensor input, FunctionalTensor end, float weight)
        {
            // TODO weight tensor
            // TODO add to layers and backend
            return input + weight * (end - input);
        }

        /// <summary>
        /// Returns log(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Log(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Log(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns log10(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Log10(FunctionalTensor input)
        {
            return Log(input) * 0.4342944819f;
        }

        /// <summary>
        /// Returns log(input + 1) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Log1P(FunctionalTensor input)
        {
            return Log(input + 1);
        }

        /// <summary>
        /// Returns log2(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Log2(FunctionalTensor input)
        {
            return Log(input) * 1.44269504089f;
        }

        /// <summary>
        /// Returns log(e^input + e^other) element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogAddExp(FunctionalTensor input, FunctionalTensor other)
        {
            return Log(Exp(input) + Exp(other));
        }

        /// <summary>
        /// Returns the logical AND input &#38; other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogicalAnd(FunctionalTensor input, FunctionalTensor other)
        {
            DeclareType(DataType.Int, input, other);
            return FunctionalTensor.FromLayer(new Layers.And(null, null, null), DataType.Int, new[] { input, other });
        }

        /// <summary>
        /// Returns the logical NOT ~input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogicalNot(FunctionalTensor input)
        {
            DeclareType(DataType.Int, input);
            return FunctionalTensor.FromLayer(new Layers.Not(null, null), DataType.Int, input);
        }

        /// <summary>
        /// Returns the logical OR input | other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogicalOr(FunctionalTensor input, FunctionalTensor other)
        {
            DeclareType(DataType.Int, input, other);
            return FunctionalTensor.FromLayer(new Layers.Or(null, null, null), DataType.Int, new[] { input, other });
        }

        /// <summary>
        /// Returns the logical XOR input | other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogicalXor(FunctionalTensor input, FunctionalTensor other)
        {
            DeclareType(DataType.Int, input, other);
            return FunctionalTensor.FromLayer(new Layers.Xor(null, null, null), DataType.Int, new[] { input, other });
        }

        /// <summary>
        /// Returns input * other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Mul(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FunctionalTensor.FromLayer(new Layers.Mul(null, null, null), input.DataType, new[] { input, other });
        }

        /// <summary>
        /// Returns -input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Neg(FunctionalTensor input)
        {
            return FunctionalTensor.FromLayer(new Layers.Neg(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Positive(FunctionalTensor input)
        {
            return input;
        }

        /// <summary>
        /// Returns input^exponent element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="exponent">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Pow(FunctionalTensor input, FunctionalTensor exponent)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Pow(null, null, null), DataType.Float, new[] { input, exponent });
        }

        /// <summary>
        /// Returns input^exponent element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="exponent">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Pow(FunctionalTensor input, float exponent)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Pow(null, null, null), DataType.Float, new[] { input, Tensor(exponent) });
        }

        /// <summary>
        /// Returns the input values converted from angles in radians to degrees element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Rad2Deg(FunctionalTensor input)
        {
            return Mathf.Rad2Deg * input;
        }

        /// <summary>
        /// Returns 1/input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Reciprocal(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Reciprocal(null, null), DataType.Float, input);
        }

        /// <summary>
        /// Returns input % other element-wise. The sign of the output is the same as that of the divider.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Remainder(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FunctionalTensor.FromLayer(new Layers.Mod(null, null, null), input.DataType, new[] { input, other });
        }

        /// <summary>
        /// Returns [input] element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Round(FunctionalTensor input)
        {
            // TODO implement 'decimals' arg
            if (input.DataType == DataType.Int)
                return input;
            return FunctionalTensor.FromLayer(new Layers.Round(null, null), DataType.Float, input);
        }

        /// <summary>
        /// Returns 1/√input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor RSqrt(FunctionalTensor input)
        {
            return Reciprocal(Sqrt(input));
        }

        /// <summary>
        /// Returns the sign of the input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sign(FunctionalTensor input)
        {
            return FunctionalTensor.FromLayer(new Layers.Sign(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns sin(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sin(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Sin(null, null), DataType.Float, input);
        }

        /// <summary>
        /// Returns sinh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sinh(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Sinh(null, null), DataType.Float, input);
        }

        /// <summary>
        /// Returns √(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sqrt(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Sqrt(null, null), DataType.Float, input);
        }

        /// <summary>
        /// Returns input*input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Square(FunctionalTensor input)
        {
            return FunctionalTensor.FromLayer(new Layers.Square(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns input - other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sub(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FunctionalTensor.FromLayer(new Layers.Sub(null, null, null), input.DataType, new[] { input, other });
        }

        /// <summary>
        /// Returns tan(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Tan(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Tan(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns tanh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Tanh(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalTensor.FromLayer(new Layers.Tanh(null, null), input.DataType, input);
        }

        /// <summary>
        /// Returns the truncated integer values of the elements of input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Trunc(FunctionalTensor input)
        {
            return Floor(Abs(input)) * Sign(input);
        }
    }
}
