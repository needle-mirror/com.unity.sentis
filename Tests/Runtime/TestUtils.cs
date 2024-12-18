using System;
using NUnit.Framework;
using UnityEngine;

namespace Unity.Sentis.Tests
{
    static class TestUtils
    {
        static void AssertEqual(ReadOnlySpan<float> a, ReadOnlySpan<float> b, float absoluteTolerance = 1e-3f, float relativeTolerance = 1e-3f)
        {
            for (var i = 0; i < a.Length; i++)
            {
                // https://web.mit.edu/10.001/Web/Tips/Converge.htm
                var delta = Mathf.Abs(a[i] - b[i]);
                var tolerance = (relativeTolerance / 1 - relativeTolerance) * Mathf.Abs(a[i]) + absoluteTolerance / (1 - relativeTolerance);
                Assert.IsTrue(delta < tolerance, "Values are not equal a[{0}]: {1}, b[{0}]: {2}", i, a[i], b[i]);
            }
        }

        static void AssertEqual(ReadOnlySpan<int> a, ReadOnlySpan<int> b)
        {
            for (var i = 0; i < a.Length; i++)
                Assert.IsTrue(a[i] == b[i], "Values are not equal a[{0}]: {1}, b[{0}]: {2}", i, a[i], b[i]);
        }

        public static void AssertEqual(Tensor a, Tensor b)
        {
            Assert.IsTrue(a.dataType == b.dataType);
            Assert.IsTrue(a.shape == b.shape);

            switch (a.dataType)
            {
                case DataType.Float:
                    AssertEqual((a as Tensor<float>).AsReadOnlySpan(), (b as Tensor<float>).AsReadOnlySpan());
                    break;
                case DataType.Int:
                    AssertEqual((a as Tensor<int>).AsReadOnlySpan(), (b as Tensor<int>).AsReadOnlySpan());
                    break;
            }
        }
    }
}

