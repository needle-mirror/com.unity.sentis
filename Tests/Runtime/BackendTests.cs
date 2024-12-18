using System;
using System.Runtime.CompilerServices;
using NUnit.Framework;
using Unity.Sentis.Layers;
using UnityEngine;
using UnityEngine.TestTools;

[assembly: InternalsVisibleTo("Unity.Sentis.RuntimeTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis.Tests
{
    class BackendTests
    {
        static FunctionalTensor RandomUniform(int[] shape, System.Random random, float low = -1f, float high = 1f)
        {
            return Functional.FromLayer(new RandomUniform(-1, shape, low, high, random.Next()), new[] { DataType.Float }, Array.Empty<FunctionalTensor>())[0];
        }

        static FunctionalTensor RandomInt(int[] shape, System.Random random, int low = byte.MinValue, int high = byte.MaxValue)
        {
            return Functional.RandInt(shape, low, high, random.Next());
        }

        static void TestFunctional(FunctionalTensor[] outputs, BackendType[] backendTypes)
        {
            var model = new FunctionalGraph().Compile(outputs);

            var worker = new Worker(model, BackendType.CPU);
            worker.Schedule();

            var expectedTensors = new Tensor[model.outputs.Count];
            for (var i = 0; i < expectedTensors.Length; i++)
                expectedTensors[i] = worker.PeekOutput(i).ReadbackAndClone();

            worker.Dispose();
            foreach (var backendType in backendTypes)
            {
                worker = new Worker(model, backendType);

                // TODO make random layers not contain any state so we don't need this
                foreach (var layer in model.layers)
                {
                    if (layer is RandomLayer randomLayer)
                        randomLayer.ResetSeed();
                }

                worker.Schedule();

                for (var i = 0; i < expectedTensors.Length; i++)
                {
                    using var receivedOutput = worker.PeekOutput(i).ReadbackAndClone();
                    TestUtils.AssertEqual(expectedTensors[i], receivedOutput);
                }

                worker.Dispose();
            }

            foreach (var expectedTensor in expectedTensors)
                expectedTensor.Dispose();
        }

        [Test]
        [UnityPlatform(include = new[] { RuntimePlatform.WindowsEditor, RuntimePlatform.WindowsPlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer, RuntimePlatform.LinuxEditor, RuntimePlatform.LinuxPlayer })] // TODO investigate iOS and Android fails
        public void Conv()
        {
            var rand = new System.Random("Conv".GetHashCode());

            void Test(int[] shape, int m, int[] kernelSize, bool bias, int group = 1, int[] strides = null, int[] pads = null, int[] dilations = null, AutoPad autoPad = AutoPad.NotSet, FusableActivation fusedActivation = FusableActivation.None)
            {
                var x = RandomUniform(shape, rand);
                var wShape = new int[kernelSize.Length + 2];
                wShape[0] = m;
                wShape[1] = shape[1] / group;
                for (var i = 2; i < wShape.Length; i++)
                    wShape[i] = kernelSize[i - 2];
                var w = RandomUniform(wShape, rand);
                var b = bias ? RandomUniform(new[] { m }, rand) : null;

                var outputs = Functional.FromLayer(new Conv(-1, -1, -1, -1, group, strides, pads, dilations, autoPad, kernelSize, fusedActivation), new[] { DataType.Float }, new[] { x, w, b });
                TestFunctional(outputs, new[] { BackendType.GPUCompute, BackendType.GPUPixel });
            }

            Test(new[] { 1, 3, 256, 256 }, 8, new[] { 3, 3 }, true);
            Test(new[] { 2, 3, 256, 256 }, 8, new[] { 3, 3 }, true);
            Test(new[] { 1, 3, 256, 256 }, 8, new[] { 3, 3 }, false);
            Test(new[] { 1, 3, 256, 256 }, 8, new[] { 3, 4 }, true);
            Test(new[] { 1, 3, 256, 256 }, 9, new[] { 3, 3 }, true, 3);
            Test(new[] { 1, 4, 256, 256 }, 8, new[] { 3, 3 }, true, 2);
            Test(new[] { 1, 3, 256, 256 }, 8, new[] { 3, 3 }, true, strides: new[] { 2, 3 });
            Test(new[] { 1, 3, 256, 256 }, 8, new[] { 3, 3 }, true, pads: new[] { 1, 2, 3, 4 });
            Test(new[] { 1, 3, 256, 256 }, 8, new[] { 3, 3 }, true, dilations: new[] { 2, 3 });
            Test(new[] { 1, 3, 256, 256 }, 8, new[] { 3, 3 }, true, fusedActivation: FusableActivation.Relu);
            Test(new[] { 1, 3, 32, 32, 32 }, 8, new[] { 3, 3, 3 }, true);
        }

        [Test]
        [UnityPlatform(include = new[] { RuntimePlatform.WindowsEditor, RuntimePlatform.WindowsPlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer, RuntimePlatform.LinuxEditor, RuntimePlatform.LinuxPlayer })] // TODO investigate iOS and Android fails
        public void Dense()
        {
            var rand = new System.Random("Dense".GetHashCode());

            void Test(int[] inputShape, int[] weightsShape, int[] biasShape, FusableActivation fusedActivation = FusableActivation.None)
            {
                var input = RandomUniform(inputShape, rand);
                var weights = RandomUniform(weightsShape, rand);
                var bias = RandomUniform(biasShape, rand);
                var outputs = Functional.FromLayer(new Dense(-1, -1, -1, -1, fusedActivation), new[] { DataType.Float }, new[] { input, weights, bias });
                TestFunctional(outputs, new[] { BackendType.GPUCompute, BackendType.GPUPixel });
            }

            Test(new[] { 1, 511, 513 }, new[] { 513, 515 }, new[] { 515 });
            Test(new[] { 1, 511, 513 }, new[] { 513, 515 }, new[] { 515 }, FusableActivation.Relu);
        }

        [Test]
        [UnityPlatform(include = new[] { RuntimePlatform.WindowsEditor, RuntimePlatform.WindowsPlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer, RuntimePlatform.LinuxEditor, RuntimePlatform.LinuxPlayer })] // TODO investigate iOS and Android fails
        public void MatMul()
        {
            var rand = new System.Random("MatMul".GetHashCode());

            void Test(int[] aShape, int[] bShape)
            {
                var a = RandomUniform(aShape, rand);
                var b = RandomUniform(bShape, rand);
                var outputs = Functional.FromLayer(new MatMul(-1, -1, -1), new[] { DataType.Float }, new[] { a, b });
                TestFunctional(outputs, new[] { BackendType.GPUCompute, BackendType.GPUPixel });
            }

            Test(new[] { 1, 512, 512 }, new[] { 1, 512, 512 });
            Test(new[] { 1, 1, 2048 }, new[] { 1, 2048, 32 });
            Test(new[] { 1, 32, 2048 }, new[] { 1, 2048, 1 });
            Test(new[] { 1, 511, 513 }, new[] { 1, 513, 515 });
            Test(new[] { 3, 4 }, new[] { 4, 5 });
            Test(new[] { 1, 3, 4 }, new[] { 1, 4, 5 });
            Test(new[] { 1, 3, 4 }, new[] { 2, 4, 5 });
            Test(new[] { 2, 3, 4 }, new[] { 2, 4, 5 });
            Test(new[] { 2, 1, 3, 4 }, new[] { 1, 2, 4, 5 });
            Test(new[] { 1, 1, 1, 3, 4 }, new[] { 2, 2, 2, 4, 5 });
        }

        [Test]
        [UnityPlatform(include = new[] { RuntimePlatform.WindowsEditor, RuntimePlatform.WindowsPlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer, RuntimePlatform.LinuxEditor, RuntimePlatform.LinuxPlayer })] // TODO investigate iOS and Android fails
        public void MatMul2D()
        {
            var rand = new System.Random("MatMul2D".GetHashCode());

            void Test(int[] aShape, bool transposeA, int[] bShape, bool transposeB)
            {
                var a = RandomUniform(aShape, rand);
                var b = RandomUniform(bShape, rand);
                var outputs = Functional.FromLayer(new MatMul2D(-1, -1, transposeA, -1, transposeB), new[] { DataType.Float }, new[] { a, b });
                TestFunctional(outputs, new[] { BackendType.GPUCompute, BackendType.GPUPixel });
            }

            Test(new[] { 511, 513 }, false, new[] { 513, 515 }, false);
            Test(new[] { 513, 511 }, true, new[] { 513, 515 }, false);
            Test(new[] { 511, 513 }, false, new[] { 515, 513 }, true);
            Test(new[] { 513, 511 }, true, new[] { 515, 513 }, true);
        }

        [Test]
        [UnityPlatform(include = new[] { RuntimePlatform.WindowsEditor, RuntimePlatform.WindowsPlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer, RuntimePlatform.LinuxEditor, RuntimePlatform.LinuxPlayer })] // TODO investigate iOS and Android fails
        public void ReduceL1()
        {
            var rand = new System.Random("ReduceL1".GetHashCode());

            void Test(FunctionalTensor input, int[] axes)
            {
                var outputs = Functional.FromLayer(new ReduceL1(-1, -1, -1), new[] { input.dataType }, new[] { input, Functional.Constant(axes) });
                TestFunctional(outputs, new[] { BackendType.GPUCompute, BackendType.GPUPixel });
            }

            Test(RandomUniform(new[] { 100 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 32, 32 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 32, 32 }, rand), new[] { 1 });
            Test(RandomUniform(new[] { 32, 32 }, rand), new[] { 0, 1 });

            Test(RandomInt(new[] { 100 }, rand, -16, 16), new[] { 0 });
            Test(RandomInt(new[] { 32, 32 }, rand, -16, 16), new[] { 0 });
            Test(RandomInt(new[] { 32, 32 }, rand, -16, 16), new[] { 1 });
            Test(RandomInt(new[] { 32, 32 }, rand, -16, 16), new[] { 0, 1 });
        }

        [Test]
        [UnityPlatform(include = new[] { RuntimePlatform.WindowsEditor, RuntimePlatform.WindowsPlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer, RuntimePlatform.LinuxEditor, RuntimePlatform.LinuxPlayer })] // TODO investigate iOS and Android fails
        public void ReduceL2()
        {
            var rand = new System.Random("ReduceL2".GetHashCode());

            void Test(FunctionalTensor input, int[] axes)
            {
                var outputs = Functional.FromLayer(new ReduceL2(-1, -1, -1), new[] { input.dataType }, new[] { input, Functional.Constant(axes) });
                TestFunctional(outputs, new[] { BackendType.GPUCompute, BackendType.GPUPixel });
            }

            Test(RandomUniform(new[] { 1000 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 1 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 0, 1 });
        }

        [Test]
        [UnityPlatform(include = new[] { RuntimePlatform.WindowsEditor, RuntimePlatform.WindowsPlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer, RuntimePlatform.LinuxEditor, RuntimePlatform.LinuxPlayer })] // TODO investigate iOS and Android fails
        public void ReduceLogSum()
        {
            var rand = new System.Random("ReduceLogSum".GetHashCode());

            void Test(FunctionalTensor input, int[] axes)
            {
                var outputs = Functional.FromLayer(new ReduceLogSum(-1, -1, -1), new[] { input.dataType }, new[] { input, Functional.Constant(axes) });
                TestFunctional(outputs, new[] { BackendType.GPUCompute, BackendType.GPUPixel });
            }

            Test(RandomUniform(new[] { 100 }, rand, 0, 1), new[] { 0 });
            Test(RandomUniform(new[] { 32, 32 }, rand, 0, 1), new[] { 0 });
            Test(RandomUniform(new[] { 32, 32 }, rand, 0, 1), new[] { 1 });
            Test(RandomUniform(new[] { 32, 32 }, rand, 0, 1), new[] { 0, 1 });
        }

        [Test]
        [UnityPlatform(include = new[] { RuntimePlatform.WindowsEditor, RuntimePlatform.WindowsPlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer, RuntimePlatform.LinuxEditor, RuntimePlatform.LinuxPlayer })] // TODO investigate iOS and Android fails
        public void ReduceLogSumExp()
        {
            var rand = new System.Random("ReduceLogSumExp".GetHashCode());

            void Test(FunctionalTensor input, int[] axes)
            {
                var outputs = Functional.FromLayer(new ReduceLogSumExp(-1, -1, -1), new[] { input.dataType }, new[] { input, Functional.Constant(axes) });
                TestFunctional(outputs, new[] { BackendType.GPUCompute, BackendType.GPUPixel });
            }

            Test(RandomUniform(new[] { 1000 }, rand, -40, 40), new[] { 0 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 1 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 0, 1 });
        }

        [Test]
        [UnityPlatform(include = new[] { RuntimePlatform.WindowsEditor, RuntimePlatform.WindowsPlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer, RuntimePlatform.LinuxEditor, RuntimePlatform.LinuxPlayer })] // TODO investigate iOS and Android fails
        public void ReduceMax()
        {
            var rand = new System.Random("ReduceMax".GetHashCode());

            void Test(FunctionalTensor input, int[] axes)
            {
                var outputs = Functional.FromLayer(new ReduceMax(-1, -1, -1), new[] { input.dataType }, new[] { input, Functional.Constant(axes) });
                TestFunctional(outputs, new[] { BackendType.GPUCompute, BackendType.GPUPixel });
            }

            Test(RandomUniform(new[] { 1000 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 1 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 0, 1 });

            Test(RandomInt(new[] { 1000 }, rand), new[] { 0 });
            Test(RandomInt(new[] { 256, 256 }, rand), new[] { 0 });
            Test(RandomInt(new[] { 256, 256 }, rand), new[] { 1 });
            Test(RandomInt(new[] { 256, 256 }, rand), new[] { 0, 1 });
        }

        [Test]
        [UnityPlatform(include = new[] { RuntimePlatform.WindowsEditor, RuntimePlatform.WindowsPlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer, RuntimePlatform.LinuxEditor, RuntimePlatform.LinuxPlayer })] // TODO investigate iOS and Android fails
        public void ReduceMean()
        {
            var rand = new System.Random("ReduceMean".GetHashCode());

            void Test(FunctionalTensor input, int[] axes)
            {
                var outputs = Functional.FromLayer(new ReduceMean(-1, -1, -1), new[] { input.dataType }, new[] { input, Functional.Constant(axes) });
                TestFunctional(outputs, new[] { BackendType.GPUCompute, BackendType.GPUPixel });
            }

            Test(RandomUniform(new[] { 1000 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 1 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 0, 1 });
        }

        [Test]
        [UnityPlatform(include = new[] { RuntimePlatform.WindowsEditor, RuntimePlatform.WindowsPlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer, RuntimePlatform.LinuxEditor, RuntimePlatform.LinuxPlayer })] // TODO investigate iOS and Android fails
        public void ReduceMin()
        {
            var rand = new System.Random("ReduceMin".GetHashCode());

            void Test(FunctionalTensor input, int[] axes)
            {
                var outputs = Functional.FromLayer(new ReduceMin(-1, -1, -1), new[] { input.dataType }, new[] { input, Functional.Constant(axes) });
                TestFunctional(outputs, new[] { BackendType.GPUCompute, BackendType.GPUPixel });
            }

            Test(RandomUniform(new[] { 1000 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 1 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 0, 1 });

            Test(RandomInt(new[] { 1000 }, rand), new[] { 0 });
            Test(RandomInt(new[] { 256, 256 }, rand), new[] { 0 });
            Test(RandomInt(new[] { 256, 256 }, rand), new[] { 1 });
            Test(RandomInt(new[] { 256, 256 }, rand), new[] { 0, 1 });
        }

        [Test]
        [UnityPlatform(include = new[] { RuntimePlatform.WindowsEditor, RuntimePlatform.WindowsPlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer, RuntimePlatform.LinuxEditor, RuntimePlatform.LinuxPlayer })] // TODO investigate iOS and Android fails
        public void ReduceProd()
        {
            var rand = new System.Random("ReduceProd".GetHashCode());

            void Test(FunctionalTensor input, int[] axes)
            {
                var outputs = Functional.FromLayer(new ReduceProd(-1, -1, -1), new[] { input.dataType }, new[] { input, Functional.Constant(axes) });
                TestFunctional(outputs, new[] { BackendType.GPUCompute, BackendType.GPUPixel });
            }

            Test(RandomUniform(new[] { 1000 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 1 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 0, 1 });

            Test(RandomInt(new[] { 6 }, rand, 1, 8), new[] { 0 });
            Test(RandomInt(new[] { 6, 6 }, rand, 1, 8), new[] { 0 });
            Test(RandomInt(new[] { 6, 6 }, rand, 1, 8), new[] { 1 });
            Test(RandomInt(new[] { 6, 6 }, rand, 1, 8), new[] { 0, 1 });
        }

        [Test]
        [UnityPlatform(include = new[] { RuntimePlatform.WindowsEditor, RuntimePlatform.WindowsPlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer, RuntimePlatform.LinuxEditor, RuntimePlatform.LinuxPlayer })] // TODO investigate iOS and Android fails
        public void ReduceSum()
        {
            var rand = new System.Random("ReduceSum".GetHashCode());

            void Test(FunctionalTensor input, int[] axes)
            {
                var outputs = Functional.FromLayer(new ReduceSum(-1, -1, -1), new[] { input.dataType }, new[] { input, Functional.Constant(axes) });
                TestFunctional(outputs, new[] { BackendType.GPUCompute, BackendType.GPUPixel });
            }

            Test(RandomUniform(new[] { 1000 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 1 });
            Test(RandomUniform(new[] { 256, 256 }, rand), new[] { 0, 1 });

            Test(RandomInt(new[] { 1000 }, rand), new[] { 0 });
            Test(RandomInt(new[] { 256, 256 }, rand), new[] { 0 });
            Test(RandomInt(new[] { 256, 256 }, rand), new[] { 1 });
            Test(RandomInt(new[] { 256, 256 }, rand), new[] { 0, 1 });
        }

        [Test]
        [UnityPlatform(include = new[] { RuntimePlatform.WindowsEditor, RuntimePlatform.WindowsPlayer, RuntimePlatform.OSXEditor, RuntimePlatform.OSXPlayer, RuntimePlatform.LinuxEditor, RuntimePlatform.LinuxPlayer })] // TODO investigate iOS and Android fails
        public void ReduceSumSquare()
        {
            var rand = new System.Random("ReduceSumSquare".GetHashCode());

            void Test(FunctionalTensor input, int[] axes)
            {
                var outputs = Functional.FromLayer(new ReduceSumSquare(-1, -1, -1), new[] { input.dataType }, new[] { input, Functional.Constant(axes) });
                TestFunctional(outputs, new[] { BackendType.GPUCompute, BackendType.GPUPixel });
            }

            Test(RandomUniform(new[] { 100 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 32, 32 }, rand), new[] { 0 });
            Test(RandomUniform(new[] { 32, 32 }, rand), new[] { 1 });
            Test(RandomUniform(new[] { 32, 32 }, rand), new[] { 0, 1 });

            Test(RandomInt(new[] { 100 }, rand, -16, 16), new[] { 0 });
            Test(RandomInt(new[] { 32, 32 }, rand, -16, 16), new[] { 0 });
            Test(RandomInt(new[] { 32, 32 }, rand, -16, 16), new[] { 1 });
            Test(RandomInt(new[] { 32, 32 }, rand, -16, 16), new[] { 0, 1 });
        }
    }
}

