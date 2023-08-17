using System;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Sentis
{
    public class CPUOps : Ops
    {
        public CPUOps(ITensorAllocator allocator = null)
            : base(BackendType.CPU, allocator) { }
    }

    public class GPUComputeOps : Ops
    {
        public GPUComputeOps(ITensorAllocator allocator = null)
            : base(BackendType.GPUCompute, allocator) { }
    }

    public class GPUCommandBufferOps : Ops
    {
        public GPUCommandBufferOps(ITensorAllocator allocator = null)
            : base(BackendType.GPUCommandBuffer, allocator) { }
    }

    public class GPUPixelOps : Ops
    {
        public GPUPixelOps(ITensorAllocator allocator = null)
            : base(BackendType.GPUPixel, allocator) { }
    }

    public abstract class Ops : IDisposable
    {
        ITensorAllocator m_Allocator;
        IBackend m_Backend;
        BackendType m_BackendType;

        public BackendType backendType => m_BackendType;

        protected Ops(BackendType backendType, ITensorAllocator allocator)
        {
            m_BackendType = backendType;
            m_Backend = BackendFactory.CreateBackend(backendType, allocator, false);
            m_Allocator = allocator ?? new TensorCachingAllocator();
        }

        public void Dispose()
        {
            m_Allocator?.Dispose();
            m_Backend?.Dispose();
        }

        /// <summary>
        /// Performs an element-wise `Add` math operation between a tensor and a float: f(a, b) = a + b.
        /// </summary>
        /// <param name="A">The first argument as a tensor.</param>
        /// <param name="b">The second argument as a float.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Add(TensorFloat A, float b)
        {
            using var B = m_Backend.ConstantOfShape(new TensorShape(), b);
            return m_Backend.Add(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Add` math operation between a float and a tensor: f(a, b) = a + b.
        /// </summary>
        /// <param name="a">The first argument as a float.</param>
        /// <param name="B">The second argument as a tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Add(float a, TensorFloat B)
        {
            using var A = m_Backend.ConstantOfShape(new TensorShape(), a);
            return m_Backend.Add(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Sub` math operation between a tensor and a float: f(a, b) = a - b.
        /// </summary>
        /// <param name="A">The first argument as a tensor.</param>
        /// <param name="b">The second argument as a float.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sub(TensorFloat A, float b)
        {
            using var B = m_Backend.ConstantOfShape(new TensorShape(), b);
            return m_Backend.Sub(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Sub` math operation between a float and a tensor: f(a, b) = a - b.
        /// </summary>
        /// <param name="a">The first argument as a float.</param>
        /// <param name="B">The second argument as a tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sub(float a, TensorFloat B)
        {
            using var A = m_Backend.ConstantOfShape(new TensorShape(), a);
            return m_Backend.Sub(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Mul` math operation between a tensor and a float: f(a, b) = a * b.
        /// </summary>
        /// <param name="A">The first argument as a tensor.</param>
        /// <param name="b">The second argument as a float.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Mul(TensorFloat A, float b)
        {
            using var B = m_Backend.ConstantOfShape(new TensorShape(), b);
            return m_Backend.Mul(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Mul` math operation between a float and a tensor: f(a, b) = a * b.
        /// </summary>
        /// <param name="a">The first argument as a float.</param>
        /// <param name="B">The second argument as a tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Mul(float a, TensorFloat B)
        {
            using var A = m_Backend.ConstantOfShape(new TensorShape(), a);
            return m_Backend.Mul(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Div` math operation between a tensor and a float: f(a, b) = a / b.
        /// </summary>
        /// <param name="A">The first argument as a tensor.</param>
        /// <param name="b">The second argument as a float.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Div(TensorFloat A, float b)
        {
            using var B = m_Backend.ConstantOfShape(new TensorShape(), b);
            return m_Backend.Div(A, B);
        }

        /// <summary>
        /// Updates values of A with values from B similar to setting a slice in numpy. A[..., start:end, ....] = B
        ///
        /// This returns a new tensor rather than working on A in-place.
        ///
        /// This supports numpy-style one-directional broadcasting of B into A.
        /// </summary>
        /// <param name="A">The first argument as a tensor.</param>
        /// <param name="B">The second argument as a tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Set(Tensor A, Tensor B, int axis, int start, int end)
        {
            var dim = A.shape[axis];
            start = start < 0 ? dim + start : start;
            start = Mathf.Clamp(start, 0, dim);
            end = end < 0 ? dim + end : end;
            end = Mathf.Clamp(end, 0, dim);
            var updatesShape = B.shape.Broadcast(TensorShape.Ones(A.shape.rank));
            Assert.IsTrue(end - start == updatesShape[axis] || updatesShape[axis] == 1, "ValueError: cannot broadcast B to A for Set between start and end.");
            updatesShape[axis] = 1;
            updatesShape = A.shape.Broadcast(updatesShape);
            updatesShape[axis] = end - start;
            using var updates = m_Backend.Expand(B, updatesShape);
            using var indicesRange = m_Backend.Range(start, end, 1);
            var indicesOnesShape = TensorShape.Ones(A.shape.rank);
            indicesOnesShape[axis] = updatesShape[axis];
            using var indicesOnes = m_Backend.Reshape(indicesRange, indicesOnesShape);
            using var indices = m_Backend.Expand(indicesOnes, updatesShape) as TensorInt;
            return m_Backend.ScatterElements(A, indices, updates, axis, Layers.ScatterReductionMode.None);
        }

        /// <summary>
        /// Performs a matrix multiplication operation with optional transposes: f(a, b) = a' x b'.
        /// </summary>
        /// <param name="X">The first input tensor.</param>
        /// <param name="xTranspose">Whether to transpose the first input tensor before performing the matrix multiplication.</param>
        /// <param name="y">The second input tensor.</param>
        /// <param name="yTranspose">Whether to transpose the second input tensor before performing the matrix multiplication.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat MatMul2D(TensorFloat X, bool xTranspose, TensorFloat y, bool yTranspose)
        {
            return m_Backend.MatMul2D(X, xTranspose, y, yTranspose);
        }

        /// <summary>
        /// Performs a multi-dimensional matrix multiplication operation: f(a, b) = a x b.
        /// </summary>
        /// <param name="X">The first input tensor.</param>
        /// <param name="Y">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat MatMul(TensorFloat X, TensorFloat Y)
        {
            return m_Backend.MatMul(X, Y);
        }

        /// <summary>
        /// Performs a matrix multiplication operation: f(x, w, b) = X x W + B.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="W">The weights tensor.</param>
        /// <param name="B">The bias tensor.</param>
        /// <param name="fusedActivation">The fused activation to apply to the output tensor after the dense operation.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Dense(TensorFloat X, TensorFloat W, TensorFloat B, Layers.FusableActivation fusedActivation)
        {
            return m_Backend.Dense(X, W, B, fusedActivation);
        }

        /// <summary>
        /// Computes the output tensor by retaining the lower triangular values from an input matrix or matrix batch and setting the other values to zero.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="k">The offset from the diagonal to keep.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Tril(Tensor X, int k = 0)
        {
            return m_Backend.Tril(X, k);
        }

        /// <summary>
        /// Computes the output tensor by retaining the upper triangular values from an input matrix or matrix batch and setting the other values to zero.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="k">The offset from the diagonal to exclude.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Triu(Tensor X, int k = 0)
        {
            return m_Backend.Triu(X, k);
        }

        /// <summary>
        /// Applies a convolution filter to an input tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="K">The filter tensor.</param>
        /// <param name="B">The optional bias tensor.</param>
        /// <param name="groups">The number of groups that input channels and output channels are divided into.</param>
        /// <param name="stride">The optional stride value for each spatial dimension of the filter.</param>
        /// <param name="pad">The optional lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="dilation">The optional dilation value of each spatial dimension of the filter.</param>
        /// <param name="fusedActivation">The fused activation type to apply after the convolution.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Conv(TensorFloat X, TensorFloat K, TensorFloat B, int groups, int[] stride, int[] pad, int[] dilation, Layers.FusableActivation fusedActivation)
        {
            return m_Backend.Conv(X, K, B, groups, stride, pad, dilation, fusedActivation);
        }

        /// <summary>
        /// Applies a transpose convolution filter to an input tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="K">The filter tensor.</param>
        /// <param name="B">The optional bias tensor.</param>
        /// <param name="stride">The optional stride value for each spatial dimension of the filter.</param>
        /// <param name="pad">The optional lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="outputAdjustment">The output padding value for each spatial dimension in the filter.</param>
        /// <param name="fusedActivation">The fused activation type to apply after the convolution.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Conv2DTrans(TensorFloat X, TensorFloat K, TensorFloat B, int[] stride, int[] pad, int[] outputAdjustment, Layers.FusableActivation fusedActivation)
        {
            return m_Backend.ConvTranspose(X, K, B, stride, pad, outputAdjustment, fusedActivation);
        }

        /// <summary>
        /// Calculates an output tensor by resampling the input tensor along the spatial dimensions with given scales.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="scale">The factor to scale each dimension by.</param>
        /// <param name="interpolationMode">The `InterpolationMode` to use for the operation.</param>
        /// <param name="nearestMode">The `NearestMode` to use for the operation when using `InterpolationMode.NearestMode`. The default is `NearestMode.RoundPreferFloor`.</param>
        /// <param name="coordTransformMode">The `CoordTransformMode` to use for the operation. The default is `CoordTransformMode.HalfPixel`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Resize(TensorFloat X, ReadOnlySpan<float> scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode = Layers.NearestMode.RoundPreferFloor, Layers.CoordTransformMode coordTransformMode = Layers.CoordTransformMode.HalfPixel)
        {
            return m_Backend.Resize(X, scale, interpolationMode, nearestMode, coordTransformMode);
        }

        /// <summary>
        /// Computes the output tensor by permuting data from depth into blocks of spatial data.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="blocksize">The size of the blocks to move the depth data into.</param>
        /// <param name="mode">The ordering of the data in the output tensor as a `DepthToSpaceMode`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat DepthToSpace(TensorFloat X, int blocksize, Layers.DepthToSpaceMode mode)
        {
            return m_Backend.DepthToSpace(X, blocksize, mode);
        }

        /// <summary>
        /// Computes the output tensor by permuting data from blocks of spatial data into depth.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="blocksize">The size of the blocks to move the depth data into.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat SpaceToDepth(TensorFloat x, int blocksize)
        {
            return m_Backend.SpaceToDepth(x, blocksize);
        }

        /// <summary>
        /// Calculates an output tensor by pooling the maximum values of the input tensor across its spatial dimensions according to the given pool and stride values.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="pool">The size of the kernel along each spatial axis.</param>
        /// <param name="stride">The stride along each spatial axis.</param>
        /// <param name="pad">The lower and upper padding values for each spatial dimension. For example, [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat MaxPool(TensorFloat X, int[] pool, int[] stride, int[] pad)
        {
            return m_Backend.MaxPool(X, pool, stride, pad);
        }

        /// <summary>
        /// Calculates an output tensor by pooling the mean values of the input tensor across its spatial dimensions according to the given pool and stride values.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="pool">The size of the kernel along each spatial axis.</param>
        /// <param name="stride">The stride along each spatial axis.</param>
        /// <param name="pad">The lower and upper padding values for each spatial dimension. For example, [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat AveragePool(TensorFloat X, int[] pool, int[] stride, int[] pad)
        {
            return m_Backend.AveragePool(X, pool, stride, pad);
        }

        /// <summary>
        /// Calculates an output tensor by pooling the maximum values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat GlobalMaxPool(TensorFloat X)
        {
            return m_Backend.GlobalMaxPool(X);
        }

        /// <summary>
        /// Calculates an output tensor by pooling the mean values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat GlobalAveragePool(TensorFloat X)
        {
            return m_Backend.GlobalAveragePool(X);
        }

        /// <summary>
        /// Calculates the output tensor by adding padding to the input tensor according to the given padding values and mode.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="pad">The lower and upper padding values for each dimension.</param>
        /// <param name="padMode">The `PadMode` to use when padding. The default value is `PadMode.Constant`.</param>
        /// <param name="constant">The constant value to fill with when using `PadMode.Constant`. The default value is 0.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Pad(TensorFloat X, ReadOnlySpan<int> pad, Layers.PadMode padMode = Layers.PadMode.Constant, float constant = 0.0f)
        {
            return m_Backend.Pad(X, pad, padMode, constant);
        }

        /// <summary>
        /// Computes the output tensor with an element-wise `ScaleBias` function: f(x, s, b) = x * s + b.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="S">The scale tensor.</param>
        /// <param name="B">The bias tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ScaleBias(TensorFloat X, TensorFloat S, TensorFloat B)
        {
            return m_Backend.ScaleBias(X, S, B);
        }

        /// <summary>
        /// Computes the mean variance on the spatial dimensions of the input tensor and normalizes them according to `scale` and `bias` tensors.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="S">The scale tensor.</param>
        /// <param name="B">The bias tensor.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat InstanceNormalization(TensorFloat X, TensorFloat S, TensorFloat B, float epsilon)
        {
            return m_Backend.InstanceNormalization(X, S, B, epsilon);
        }

        /// <summary>
        /// Computes the mean variance on the last dimension of the input tensor and normalizes it according to `scale` and `bias` tensors.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="S">The scale tensor.</param>
        /// <param name="B">The bias tensor.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat AxisNormalization(TensorFloat X, TensorFloat S, TensorFloat B, float epsilon)
        {
            return m_Backend.AxisNormalization(X, S, B, epsilon);
        }

        /// <summary>
        /// Normalizes the input tensor over local input regions.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="alpha">The scaling parameter to use for the normalization.</param>
        /// <param name="beta">The exponent to use for the normalization.</param>
        /// <param name="bias">The bias value to use for the normalization.</param>
        /// <param name="size">The number of channels to sum over.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat LRN(TensorFloat X, float alpha, float beta, float bias, int size)
        {
            return m_Backend.LRN(X, alpha, beta, bias, size);
        }

        /// <summary>
        /// Generates an output tensor of a given shape with random values in a normal distribution with given `mean` and `scale`, and an optional `seed` value.
        /// </summary>
        /// <param name="S">The shape to use for the output tensor.</param>
        /// <param name="mean">The mean of the normal distribution to use to generate the output.</param>
        /// <param name="scale">The standard deviation of the normal distribution to use to generate the output.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the operation generates a seed using `System.Random()`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat RandomNormal(TensorShape S, float mean, float scale, float? seed)
        {
            return m_Backend.RandomNormal(S, mean, scale, seed);
        }

        /// <summary>
        /// Generates an output tensor of a given shape with random values in a uniform distribution between a given `low` and `high`, and an optional `seed` value.
        /// </summary>
        /// <param name="S">The shape to use for the output tensor.</param>
        /// <param name="low">The lower end of the interval of the uniform distribution to use to generate the output.</param>
        /// <param name="high">The upper end of the interval of the uniform distribution to use to generate the output.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the operation generates a seed using `System.Random()`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat RandomUniform(TensorShape S, float low, float high, float? seed)
        {
            return m_Backend.RandomUniform(S, low, high, seed);
        }

        /// <summary>
        /// Generates an output tensor with values from a multinomial distribution according to the probabilities given by the input tensor.
        /// </summary>
        /// <param name="x">The probabilities input tensor.</param>
        /// <param name="count">The number of times to sample the input.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the operation generates a seed using `System.Random()`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Multinomial(TensorFloat x, int count, float? seed)
        {
            return m_Backend.Multinomial(x, count, seed);
        }

        /// <summary>
        /// Generates a one-hot tensor with a given `depth`, `indices` and on and off values.
        /// </summary>
        /// <param name="indices">The indices input tensor.</param>
        /// <param name="axis">The axis along which the operation adds the one-hot representation.</param>
        /// <param name="depth">The depth of the one-hot tensor.</param>
        /// <param name="offValue">The value to use for an off element.</param>
        /// <param name="onValue">The value to use for an on element.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt OneHot(TensorInt indices, int axis, int depth, int offValue, int onValue)
        {
            return m_Backend.OneHot(indices, axis, depth, offValue, onValue);
        }

        /// <summary>
        /// Calculates an output tensor by pooling the input tensor across each region of interest given by the `rois` tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="Rois">The region of interest input tensor.</param>
        /// <param name="Indices">The indices input tensor.</param>
        /// <param name="mode">The pooling mode of the operation as an `RoiPoolingMode`.</param>
        /// <param name="outputHeight">The height of the output tensor.</param>
        /// <param name="outputWidth">The width of the output tensor.</param>
        /// <param name="samplingRatio">The number of sampling points in the interpolation grid used to compute the output value of each pooled output bin.</param>
        /// <param name="spatialScale">The multiplicative spatial scale factor used to translate coordinates from their input spatial scale to the scale used when pooling.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat RoiAlign(TensorFloat X, TensorFloat Rois, TensorInt Indices, Layers.RoiPoolingMode mode, int outputHeight, int outputWidth, int samplingRatio, float spatialScale)
        {
            return m_Backend.RoiAlign(X, Rois, Indices, mode, outputHeight, outputWidth, samplingRatio, spatialScale);
        }

        /// <summary>
        /// Returns the indices of the elements of the input tensor that are not zero.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt NonZero(TensorFloat X)
        {
            return m_Backend.NonZero(X);
        }

        /// <summary>
        /// Returns the indices of the elements of the input tensor that are not zero.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt NonZero(TensorInt X)
        {
            return m_Backend.NonZero(X);
        }

        /// <summary>
        /// Generates a 1D output tensor where the values form an arithmetic progression defined by the `start`, `limit`, and `delta` values.
        /// </summary>
        /// <param name="start">The first value in the range.</param>
        /// <param name="limit">The limit of the range.</param>
        /// <param name="delta">The delta between subsequent values in the range.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Range(float start, float limit, float delta)
        {
            return m_Backend.Range(start, limit, delta);
        }

        /// <summary>
        /// Generates a 1D output tensor where the values form an arithmetic progression defined by the `start`, `limit`, and `delta` values.
        /// </summary>
        /// <param name="start">The first value in the range.</param>
        /// <param name="limit">The limit of the range.</param>
        /// <param name="delta">The delta between subsequent values in the range.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Range(int start, int limit, int delta)
        {
            return m_Backend.Range(start, limit, delta);
        }

        /// <summary>
        /// Generates an output tensor with values 0 or 1 from a Bernoulli distribution. The input tensor contains the probabilities to use for generating the output values.
        /// </summary>
        /// <param name="x">The probabilities input tensor.</param>
        /// <param name="dataType">The data type of the output tensor.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the operation generates a seed using `System.Random()`.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Bernoulli(TensorFloat x, DataType dataType, float? seed)
        {
            return m_Backend.Bernoulli(x, dataType, seed);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Relu` activation function: f(x) = max(0, x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Relu(TensorFloat x)
        {
            return m_Backend.Relu(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the `Softmax` activation function along an axis: f(x, axis) = exp(X) / ReduceSum(exp(X), axis).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to apply the `Softmax` activation function. The default value is -1.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Softmax(TensorFloat X, int axis = -1)
        {
            return m_Backend.Softmax(X, axis);
        }

        /// <summary>
        /// Computes an output tensor by applying the `LogSoftmax` activation function along an axis: f(x, axis) = log(Softmax(x, axis)).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to apply the `LogSoftmax` activation function. The default value is -1.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat LogSoftmax(TensorFloat X, int axis = -1)
        {
            return m_Backend.LogSoftmax(X, axis);
        }

        /// <summary>
        /// Computes an output tensor by applying the `Hardmax` activation function along an axis: f(x, axis) = 1 if x is the first maximum value along the specified axis, otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to apply the `Hardmax` activation function. The default value is -1.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Hardmax(TensorFloat X, int axis = -1)
        {
            return m_Backend.Hardmax(X, axis);
        }

        /// <summary>
        /// Performs the cumulative sum along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to apply the cumulative sum.</param>
        /// <param name="reverse">Whether to perform the cumulative sum from the end of the axis.</param>
        /// <param name="exclusive">Whether to include the respective input element in the cumulative sum.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat CumSum(TensorFloat X, int axis, bool reverse = false, bool exclusive = false)
        {
            return m_Backend.CumSum(X, axis, reverse, exclusive);
        }

        /// <summary>
        /// Performs the cumulative sum along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to apply the cumulative sum.</param>
        /// <param name="reverse">Whether to perform the cumulative sum from the end of the axis.</param>
        /// <param name="exclusive">Whether to include the respective input element in the cumulative sum.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt CumSum(TensorInt X, int axis, bool reverse = false, bool exclusive = false)
        {
            return m_Backend.CumSum(X, axis, reverse, exclusive);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Tanh` activation function: f(x) = tanh(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Tanh(TensorFloat X)
        {
            return m_Backend.Tanh(X);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Softplus` activation function: f(x) = ln(e^x + 1).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Softplus(TensorFloat X)
        {
            return m_Backend.Softplus(X);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Sigmoid` activation function: f(x) = 1/(1 + e^(-x)).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sigmoid(TensorFloat X)
        {
            return m_Backend.Sigmoid(X);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `HardSigmoid` activation function: f(x) = clamp(alpha * x + beta, 0, 1).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="alpha">The alpha value to use for the `HardSigmoid` activation function.</param>
        /// <param name="beta">The beta value to use for the `HardSigmoid` activation function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat HardSigmoid(TensorFloat x, float alpha, float beta)
        {
            return m_Backend.HardSigmoid(x, alpha, beta);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Elu` activation function: f(x) = x if x >= 0, otherwise f(x) = alpha * (e^x - 1).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="alpha">The alpha value to use for the `Elu` activation function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Elu(TensorFloat X, float alpha)
        {
            return m_Backend.Elu(X, alpha);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Gelu` activation function: f(x) = x / 2 * (1 + erf(x / sqrt(2))).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Gelu(TensorFloat X)
        {
            return m_Backend.Gelu(X);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Relu6` activation function: f(x) = clamp(x, 0, 6).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Relu6(TensorFloat X)
        {
            return m_Backend.Relu6(X);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `LeakyRelu` activation function: f(x) = x if x >= 0, otherwise f(x) = alpha * x.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="alpha">The alpha value to use for the `LeakyRelu` activation function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat LeakyRelu(TensorFloat x, float alpha)
        {
            return m_Backend.LeakyRelu(x, alpha);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Selu` activation function: f(x) = gamma * x if x >= 0, otherwise f(x) = (alpha * e^x - alpha).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="alpha">The alpha value to use for the `Selu` activation function.</param>
        /// <param name="gamma">The alpha value to use for the `Selu` activation function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Selu(TensorFloat x, float alpha, float gamma)
        {
            return m_Backend.Selu(x, alpha, gamma);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `PRelu` activation function: f(x) = x if x >= 0, otherwise f(x) = slope * x.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="slope">The slope tensor, must be unidirectional broadcastable to x.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat PRelu(TensorFloat x, TensorFloat slope)
        {
            return m_Backend.PRelu(x, slope);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Swish` activation function: f(x) = sigmoid(x) * x = x / (1 + e^{-x}).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Swish(TensorFloat x)
        {
            return m_Backend.Swish(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Abs` math function: f(x) = f(x) = |x|.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Abs(TensorFloat x)
        {
            return m_Backend.Abs(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Abs` math function: f(x) = f(x) = |x|.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Abs(TensorInt x)
        {
            return m_Backend.Abs(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Neg` math function: f(x) = -x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Neg(TensorFloat X)
        {
            return m_Backend.Neg(X);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Neg` math function: f(x) = -x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Neg(TensorInt X)
        {
            return m_Backend.Neg(X);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Ceil` math function: f(x) = ceil(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Ceil(TensorFloat X)
        {
            return m_Backend.Ceil(X);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Clip` math function: f(x) = clamp(x, min, max).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="min">The lower clip value.</param>
        /// <param name="max">The upper clip value.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Clip(TensorFloat x, float min, float max)
        {
            return m_Backend.Clip(x, min, max);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Floor` math function: f(x) = floor(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Floor(TensorFloat X)
        {
            return m_Backend.Floor(X);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Round` math function: f(x) = round(x).
        ///
        /// If the fractional part is equal to 0.5, rounds to the nearest even integer.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Round(TensorFloat X)
        {
            return m_Backend.Round(X);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Reciprocal` math function: f(x) = 1 / x.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Reciprocal(TensorFloat x)
        {
            return m_Backend.Reciprocal(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Square` math function: f(x) = x * x.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Square(TensorFloat x)
        {
            return m_Backend.Square(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Exp` math function: f(x) = exp(x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Exp(TensorFloat x)
        {
            return m_Backend.Exp(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Log` math function: f(x) = log(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Log(TensorFloat X)
        {
            return m_Backend.Log(X);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Sqrt` math function: f(x) = sqrt(x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sqrt(TensorFloat x)
        {
            return m_Backend.Sqrt(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Acos` trigonometric function: f(x) = acos(x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Acos(TensorFloat x)
        {
            return m_Backend.Acos(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Acosh` trigonometric function: f(x) = acosh(x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Acosh(TensorFloat x)
        {
            return m_Backend.Acosh(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Asin` trigonometric function: f(x) = asin(x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Asin(TensorFloat x)
        {
            return m_Backend.Asin(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Asinh` trigonometric function: f(x) = asinh(x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Asinh(TensorFloat x)
        {
            return m_Backend.Asinh(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Atan` trigonometric function: f(x) = atan(x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Atan(TensorFloat x)
        {
            return m_Backend.Atan(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Atanh` trigonometric function: f(x) = atanh(x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Atanh(TensorFloat x)
        {
            return m_Backend.Atanh(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Cos` trigonometric function: f(x) = cos(x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Cos(TensorFloat x)
        {
            return m_Backend.Cos(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Cosh` trigonometric function: f(x) = cosh(x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Cosh(TensorFloat x)
        {
            return m_Backend.Cosh(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Sin` trigonometric function: f(x) = sin(x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sin(TensorFloat x)
        {
            return m_Backend.Sin(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Sinh` trigonometric function: f(x) = sinh(x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sinh(TensorFloat x)
        {
            return m_Backend.Sinh(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Tan` trigonometric function: f(x) = tan(x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Tan(TensorFloat x)
        {
            return m_Backend.Tan(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Erf` activation function: f(x) = erf(x).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Erf(TensorFloat x)
        {
            return m_Backend.Erf(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Celu` activation function: f(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1)).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="alpha">The alpha value to use for the `Celu` activation function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Celu(TensorFloat x, float alpha)
        {
            return m_Backend.Celu(x, alpha);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `HardSwish` activation function: f(x) = x * max(0, min(1, 1/6 * x + 0.5)).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat HardSwish(TensorFloat x)
        {
            return m_Backend.HardSwish(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Shrink` activation function: f(x) = x + bias if x &lt; lambd. f(x) = x - bias if x &gt; lambd. Otherwise f(x) = 0.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="bias">The bias value to use for the `Shrink` activation function.</param>
        /// <param name="lambd">The lambda value to use for the `Shrink` activation function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Shrink(TensorFloat x, float bias, float lambd)
        {
            return m_Backend.Shrink(x, bias, lambd);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Softsign` activation function: f(x) = x/(|x| + 1).
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Softsign(TensorFloat x)
        {
            return m_Backend.Softsign(x);
        }

        /// <summary>
        /// Computes an output tensor by applying the element-wise `ThresholdedRelu` activation function: f(x) = x if x > alpha, otherwise f(x) = 0.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="alpha">The alpha value to use for the `ThresholdedRelu` activation function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ThresholdedRelu(TensorFloat x, float alpha)
        {
            return m_Backend.ThresholdedRelu(x, alpha);
        }

        /// <summary>
        /// Performs an element-wise `Sum` math operation: f(x1, x2 ... xn) = x1 + x2 ... xn.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sum(TensorFloat[] tensors)
        {
            return m_Backend.Sum(tensors);
        }

        /// <summary>
        /// Performs an element-wise `Add` math operation: f(a, b) = a + b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Add(TensorFloat A, TensorFloat B)
        {
            return m_Backend.Add(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Add` math operation: f(a, b) = a + b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Add(TensorInt A, TensorInt B)
        {
            return m_Backend.Add(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Sub` math operation: f(a, b) = a - b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sub(TensorFloat A, TensorFloat B)
        {
            return m_Backend.Sub(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Sub` math operation: f(a, b) = a - b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Sub(TensorInt A, TensorInt B)
        {
            return m_Backend.Sub(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Mul` math operation: f(a, b) = a * b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Mul(TensorFloat A, TensorFloat B)
        {
            return m_Backend.Mul(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Mul` math operation: f(a, b) = a * b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Mul(TensorInt A, TensorInt B)
        {
            return m_Backend.Mul(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Div` math operation: f(a, b) = a / b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Div(TensorFloat A, TensorFloat B)
        {
            return m_Backend.Div(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Div` math operation: f(a, b) = a / b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Div(TensorInt A, TensorInt B)
        {
            return m_Backend.Div(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Mod` math operation: f(a, b) = a % b.
        ///
        /// The sign of the remainder is the same as the sign of the divisor, as in Python.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Mod(TensorInt A, TensorInt B)
        {
            return m_Backend.Mod(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Mod` math operation: f(a, b) = a % b.
        ///
        /// The sign of the remainder is the same as the sign of the dividend, as in C#.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt FMod(TensorInt A, TensorInt B)
        {
            return m_Backend.FMod(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Mod` math operation: f(a, b) = a % b.
        ///
        /// The sign of the remainder is the same as the sign of the dividend, as in C#.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat FMod(TensorFloat A, TensorFloat B)
        {
            return m_Backend.FMod(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Pow` math operation: f(a, b) = pow(a, b).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Pow(TensorFloat A, TensorFloat B)
        {
            return m_Backend.Pow(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Pow` math operation: f(a, b) = pow(a, b).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Pow(TensorFloat A, TensorInt B)
        {
            return m_Backend.Pow(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Min` math operation: f(x1, x2 ... xn) = min(x1, x2 ... xn).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Min(TensorFloat[] tensors)
        {
            return m_Backend.Min(tensors);
        }

        /// <summary>
        /// Performs an element-wise `Min` math operation: f(x1, x2 ... xn) = min(x1, x2 ... xn).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Min(TensorInt[] tensors)
        {
            return m_Backend.Min(tensors);
        }

        /// <summary>
        /// Performs an element-wise `Max` math operation: f(x1, x2 ... xn) = max(x1, x2 ... xn).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Max(TensorFloat[] tensors)
        {
            return m_Backend.Max(tensors);
        }

        /// <summary>
        /// Performs an element-wise `Max` math operation: f(x1, x2 ... xn) = max(x1, x2 ... xn).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Max(TensorInt[] tensors)
        {
            return m_Backend.Max(tensors);
        }

        /// <summary>
        /// Performs an element-wise `Mean` math operation: f(x1, x2 ... xn) = (x1 + x2 ... xn) / n.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Mean(TensorFloat[] tensors)
        {
            return m_Backend.Mean(tensors);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMax` operation: f(x1, x2 ... xn) = max(x1, x2, ... , xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceMax(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceMax(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMean` operation: f(x1, x2 ... xn) = max(x1, x2, ... , xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ReduceMax(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceMax(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMean` operation: f(x1, x2 ... xn) = (x1 + x2 + ... + xn) / n.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceMean(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceMean(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMin` operation: f(x1, x2 ... xn) = min(x1, x2, ... , xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceMin(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceMin(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMin` operation: f(x1, x2 ... xn) = min(x1, x2, ... , xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ReduceMin(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceMin(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceProd` operation: f(x1, x2 ... xn) = x1 * x2 * ... * xn.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceProd(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceProd(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceProd` operation: f(x1, x2 ... xn) = x1 * x2 * ... * xn.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ReduceProd(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceProd(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceSum` operation: f(x1, x2 ... xn) = x1 + x2 + ... + xn.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceSum(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceSum(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceSum` operation: f(x1, x2 ... xn) = x1 + x2 + ... + xn.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ReduceSum(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceSum(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceSumSquare` operation: f(x1, x2 ... xn) = x1² + x2² + ... + xn².
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceSumSquare(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceSumSquare(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceSumSquare` operation: f(x1, x2 ... xn) = x1² + x2² + ... + xn².
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ReduceSumSquare(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceSumSquare(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceL1` operation: f(x1, x2 ... xn) = |x1| + |x2| + ... + |xn|.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceL1(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceL1(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceL1` operation: f(x1, x2 ... xn) = |x1| + |x2| + ... + |xn|.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ReduceL1(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceL1(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceL2` operation: f(x1, x2 ... xn) = sqrt(x1² + x2² + ... + xn²).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceL2(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceL2(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceLogSum` operation: f(x1, x2 ... xn) = log(x1 + x2 + ... + xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceLogSum(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceLogSum(X, axes, keepdim);
        }

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceLogSumExp` operation: f(x1, x2 ... xn) = log(e^x1 + e^x2 + ... + e^xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ReduceLogSumExp(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
        {
            return m_Backend.ReduceLogSumExp(X, axes, keepdim);
        }

        /// <summary>
        /// Computes the indices of the maximum elements of the input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ArgMax(TensorFloat X, int axis, bool keepdim, bool selectLastIndex = false)
        {
            return m_Backend.ArgMax(X, axis, keepdim, selectLastIndex);
        }

        /// <summary>
        /// Computes the indices of the maximum elements of the input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ArgMax(TensorInt X, int axis, bool keepdim, bool selectLastIndex = false)
        {
            return m_Backend.ArgMax(X, axis, keepdim, selectLastIndex);
        }

        /// <summary>
        /// Computes the indices of the minimum elements of the input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ArgMin(TensorFloat X, int axis, bool keepdim, bool selectLastIndex)
        {
            return m_Backend.ArgMin(X, axis, keepdim, selectLastIndex);
        }

        /// <summary>
        /// Computes the indices of the minimum elements of the input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ArgMin(TensorInt X, int axis, bool keepdim, bool selectLastIndex)
        {
            return m_Backend.ArgMin(X, axis, keepdim, selectLastIndex);
        }

        /// <summary>
        /// Performs an element-wise `Greater` logical comparison operation: f(a, b) = 1 if a > b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Greater(TensorFloat A, TensorFloat B)
        {
            return m_Backend.Greater(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Greater` logical comparison operation: f(a, b) = 1 if a > b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Greater(TensorInt A, TensorInt B)
        {
            return m_Backend.Greater(A, B);
        }

        /// <summary>
        /// Performs an element-wise `GreaterOrEqual` logical comparison operation: f(a, b) = 1 if a >= b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt GreaterOrEqual(TensorFloat A, TensorFloat B)
        {
            return m_Backend.GreaterOrEqual(A, B);
        }

        /// <summary>
        /// Performs an element-wise `GreaterOrEqual` logical comparison operation: f(a, b) = 1 if a >= b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt GreaterOrEqual(TensorInt A, TensorInt B)
        {
            return m_Backend.GreaterOrEqual(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Less` logical comparison operation: f(a, b) = 1 if a &lt; b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Less(TensorFloat A, TensorFloat B)
        {
            return m_Backend.Less(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Less` logical comparison operation: f(a, b) = 1 if a &lt; b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Less(TensorInt A, TensorInt B)
        {
            return m_Backend.Less(A, B);
        }

        /// <summary>
        /// Performs an element-wise `LessOrEqual` logical comparison operation: f(a, b) = 1 if a &lt;= b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt LessOrEqual(TensorFloat A, TensorFloat B)
        {
            return m_Backend.LessOrEqual(A, B);
        }

        /// <summary>
        /// Performs an element-wise `LessOrEqual` logical comparison operation: f(a, b) = 1 if a &lt;= b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt LessOrEqual(TensorInt A, TensorInt B)
        {
            return m_Backend.LessOrEqual(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Equal` logical comparison operation: f(a, b) = 1 if a == b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Equal(TensorFloat A, TensorFloat B)
        {
            return m_Backend.Equal(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Equal` logical comparison operation: f(a, b) = 1 if a == b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Equal(TensorInt A, TensorInt B)
        {
            return m_Backend.Equal(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Or` logical operation: f(a, b) = a | b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Or(TensorInt A, TensorInt B)
        {
            return m_Backend.Or(A, B);
        }

        /// <summary>
        /// Performs an element-wise `And` logical operation: f(a, b) = a & b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt And(TensorInt A, TensorInt B)
        {
            return m_Backend.And(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Xor` logical operation: f(a) = a ^ b.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Xor(TensorInt A, TensorInt B)
        {
            return m_Backend.Xor(A, B);
        }

        /// <summary>
        /// Performs an element-wise `Not` logical operation: f(x) = ~x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Not(TensorInt X)
        {
            return m_Backend.Not(X);
        }

        /// <summary>
        /// Performs an element-wise `Sign` math operation: f(x) = 1 if x > 0. f(x) = -1 if x &lt; 0. Otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Sign(TensorFloat X)
        {
            return m_Backend.Sign(X);
        }

        /// <summary>
        /// Performs an element-wise `Sign` math operation: f(x) = 1 if x > 0. f(x) = -1 if x &lt; 0. Otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Sign(TensorInt X)
        {
            return m_Backend.Sign(X);
        }

        /// <summary>
        /// Performs an element-wise `IsNaN` logical operation: f(x) = 1 if x is NaN, otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt IsNaN(TensorFloat X)
        {
            return m_Backend.IsNaN(X);
        }

        /// <summary>
        /// Performs an element-wise `IsInf` logical operation: f(x) = 1 elementwise if x is +Inf and `detectPositive` is `true`, or x is -Inf and `detectNegative` is `true`. Otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="detectNegative">Whether to detect negative infinities in the `IsInf` function.</param>
        /// <param name="detectPositive">Whether to detect positive infinities in the `IsInf` function.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt IsInf(TensorFloat X, bool detectNegative, bool detectPositive)
        {
            return m_Backend.IsInf(X, detectNegative, detectPositive);
        }

        /// <summary>
        /// Performs an element-wise `Where` logical operation: f(condition, a, b) = a if `condition` is `true`, otherwise f(condition, a, b) = b.
        /// </summary>
        /// <param name="C">The condition tensor.</param>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Where(TensorInt C, Tensor A, Tensor B)
        {
            return m_Backend.Where(C, A, B);
        }

        /// <summary>
        /// Calculates an output tensor by copying the data from the input tensor and using a given shape. The data from the input tensor is unchanged.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="shape">The shape of the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Reshape(Tensor X, TensorShape shape)
        {
            return m_Backend.Reshape(X, shape);
        }

        /// <summary>
        /// Calculates an output tensor by broadcasting the input tensor into a given shape.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="shape">The shape to broadcast the input shape together with to calculate the output tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Expand(Tensor X, TensorShape shape)
        {
            return m_Backend.Expand(X, shape);
        }

        /// <summary>
        /// Calculates an output tensor by reversing the dimensions of the input tensor.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Transpose(Tensor x)
        {
            return m_Backend.Transpose(x);
        }

        /// <summary>
        /// Calculates an output tensor by permuting the axes and data of the input tensor according to the given permutations.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="permutations">The axes to sample the output tensor from in the input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Transpose(Tensor x, int[] permutations)
        {
            return m_Backend.Transpose(x, permutations);
        }

        /// <summary>
        /// Calculates an output tensor by concatenating the input tensors along a given axis.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <param name="axis">The axis along which to concatenate the input tensors.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Concat(Tensor[] tensors, int axis)
        {
            return m_Backend.Concat(tensors, axis);
        }

        /// <summary>
        /// Calculates an output tensor by splitting the input tensor along a given axis between start and end.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="axis">The axis along which to split the input tensor.</param>
        /// <param name="start">The inclusive start value for the split.</param>
        /// <param name="end">The exclusive end value for the split.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Split(Tensor X, int axis, int start = 0, int end = int.MaxValue)
        {
            return m_Backend.Split(X, axis, start, end);
        }

        /// <summary>
        /// Calculates an output tensor by slicing the input tensor along given axes with given starts, ends, and steps.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="starts">The start index along each axis.</param>
        /// <param name="ends">The end index along each axis.</param>
        /// <param name="axes">The axes along which to slice. If this is `null`, the layer slices all axes.</param>
        /// <param name="steps">The step values for slicing. If this is `null`, the layer uses step size 1 throughout.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Slice(Tensor X, ReadOnlySpan<int> starts, ReadOnlySpan<int> ends, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps)
        {
            return m_Backend.Slice(X, starts, ends, axes, steps);
        }

        /// <summary>
        /// Calculates an output tensor by repeating the input layer a given number of times along each axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="repeats">The number of times to tile the input tensor along each axis.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Tile(Tensor X, ReadOnlySpan<int> repeats)
        {
            return m_Backend.Tile(X, repeats);
        }

        /// <summary>
        /// Selects slices of an input tensor along a given axis according to a condition tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="axis">The axis along which to compress.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Compress(Tensor X, TensorInt indices, int axis)
        {
            return m_Backend.Compress(X, indices, axis);
        }

        /// <summary>
        /// Takes values from the input tensor indexed by the indices tensor along a given axis and concatenates them.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="axis">The axis along which to gather.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Gather(Tensor X, TensorInt indices, int axis)
        {
            return m_Backend.Gather(X, indices, axis);
        }

        /// <summary>
        /// Takes values from the input tensor indexed by the indices tensor along a given axis and concatenates them.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="axis">The axis along which to gather.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor GatherElements(Tensor X, TensorInt indices, int axis)
        {
            return m_Backend.GatherElements(X, indices, axis);
        }

        /// <summary>
        /// Takes slices of values from the batched input tensor indexed by the `indices` tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="batchDims">The number of batch dimensions of the input tensor, the gather begins at the next dimension.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor GatherND(Tensor X, TensorInt indices, int batchDims)
        {
            return m_Backend.GatherND(X, indices, batchDims);
        }

        /// <summary>
        /// Copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor along a given axis.
        ///
        /// `ScatterElements` updates the values depending on the reduction mode used.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="updates">The updates tensor.</param>
        /// <param name="axis">The axis on which to perform the scatter.</param>
        /// <param name="reduction">The reduction mode used to update the values as a `ScatterReductionMode`.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor ScatterElements(Tensor X, TensorInt indices, Tensor updates, int axis, Layers.ScatterReductionMode reduction)
        {
            return m_Backend.ScatterElements(X, indices, updates, axis, reduction);
        }

        /// <summary>
        /// Copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor.
        ///
        /// `ScatterND` updates the values depending on the reduction mode used.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="updates">The updates tensor.</param>
        /// <param name="reduction">The reduction mode used to update the values as a `ScatterReductionMode`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ScatterND(TensorFloat X, TensorInt indices, TensorFloat updates, Layers.ScatterReductionMode reduction)
        {
            return m_Backend.ScatterND(X, indices, updates, reduction);
        }

        /// <summary>
        /// Copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor.
        ///
        /// `ScatterND` updates the values depending on the reduction mode used.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="updates">The updates tensor.</param>
        /// <param name="reduction">The reduction mode used to update the values as a `ScatterReductionMode`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ScatterND(TensorInt X, TensorInt indices, TensorInt updates, Layers.ScatterReductionMode reduction)
        {
            return m_Backend.ScatterND(X, indices, updates, reduction);
        }

        /// <summary>
        /// Generates an output tensor by computing a one-layer long short-term memory (LSTM) on an input tensor.
        /// </summary>
        /// <param name="X">The input sequences tensor.</param>
        /// <param name="W">The weights tensor for the gates of the LSTM.</param>
        /// <param name="R">The recurrent weights tensor for the gates of the LSTM.</param>
        /// <param name="B">The optional bias tensor for the input gate of the LSTM.</param>
        /// <param name="sequenceLens">The optional 1D tensor specifying the lengths of the sequences in a batch.</param>
        /// <param name="initialH">The optional initial values tensor of the hidden neurons of the LSTM. If this is `null`, the layer uses 0.</param>
        /// <param name="initialC">The optional initial values tensor of the cells of the LSTM. If this is `null`, the layer uses 0.</param>
        /// <param name="P">The optional weight tensor for the peepholes of the LSTM. If this is `null`, the layer uses 0.</param>
        /// <param name="direction">The direction of the LSTM as an `RnnDirection`.</param>
        /// <param name="activations">The activation functions of the LSTM as an array of `RnnActivation`.</param>
        /// <param name="activationAlpha">The alpha values of the activation functions of the LSTM.</param>
        /// <param name="activationBeta">The beta values of the activation functions of the LSTM.</param>
        /// <param name="inputForget">Whether to forget the input values in the LSTM. If this is `false`, the layer couples the input and forget gates.</param>
        /// <param name="clip">The cell clip threshold of the LSTM.</param>
        /// <param name="layout">The layout of the tensors as an `RnnLayout`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat[] LSTM(TensorFloat X, TensorFloat W, TensorFloat R, TensorFloat B, TensorInt sequenceLens, TensorFloat initialH, TensorFloat initialC, TensorFloat P, Layers.RnnDirection direction, Layers.RnnActivation[] activations, float[] activationAlpha, float[] activationBeta, bool inputForget, float clip, Layers.RnnLayout layout)
        {
            return m_Backend.LSTM(X, W, R, B, sequenceLens, initialH, initialC, P, direction, activations, activationAlpha, activationBeta, inputForget, clip, layout);
        }

        /// <summary>
        /// Calculates the top-K largest or smallest elements of an input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="k">The number of elements to calculate.</param>
        /// <param name="axis">The axis along which to perform the top-K operation.</param>
        /// <param name="largest">Whether to calculate the top-K largest elements. If this is `false`, the layer calculates the top-K smallest elements.</param>
        /// <param name="sorted">Whether to return the elements in sorted order.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor[] TopK(TensorFloat X, int k, int axis, bool largest, bool sorted)
        {
            return m_Backend.TopK(X, k, axis, largest, sorted);
        }

        /// <summary>
        /// Performs an `Einsum` math operation.
        /// </summary>
        /// <description>
        /// The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to an operand tensor, and the characters within the terms correspond to operands dimensions.
        /// This sequence may be followed by "->" to separate the left and right hand side of the equation. If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases, output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the equation.
        /// When a dimension character is repeated in the left-hand side, it represents summation along the dimension.
        /// The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions. Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions. The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the beginning of the output. The equation string may contain space (U+0020) character.
        /// </description>
        /// <param name="equation">The equation of the Einstein summation as a comma-separated list of subscript labels.</param>
        /// <param name="operands">The input tensors of the Einsum.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat Einsum(string equation, params TensorFloat[] operands)
        {
            return m_Backend.Einsum(equation, operands);
        }

        /// <summary>
        /// Calculates an output tensor of selected indices of boxes from input `boxes` and `scores` tensors where the indices are based on the scores and amount of intersection with previously selected boxes.
        /// </summary>
        /// <param name="boxes">The boxes input tensor.</param>
        /// <param name="scores">The scores input tensor.</param>
        /// <param name="maxOutputBoxesPerClass">The maximum number of boxes to return for each class.</param>
        /// <param name="iouThreshold">The threshold above which the intersect-over-union rejects a box.</param>
        /// <param name="scoreThreshold">The threshold below which the box score filters a box from the output.</param>
        /// <param name="centerPointBox">The format the `boxes` tensor uses to store the box data as a `CenterPointBox`.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt NonMaxSuppression(TensorFloat boxes, TensorFloat scores, int maxOutputBoxesPerClass, float iouThreshold, float scoreThreshold, Layers.CenterPointBox centerPointBox)
        {
            return m_Backend.NonMaxSuppression(boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox);
        }

        /// <summary>
        /// Calculates the shape of an input tensor as a 1D `TensorInt`.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="start">The inclusive start axis for slicing the shape of the input tensor. The default value is 0.</param>
        /// <param name="end">The exclusive end axis for slicing the shape of the input tensor. The default value is 8.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Shape(Tensor X, int start = 0, int end = TensorShape.maxRank)
        {
            return m_Backend.Shape(X, start, end);
        }

        /// <summary>
        /// Calculates the number of elements of an input tensor shape as a scalar `TensorInt`.
        /// </summary>
        /// <param name="shape">The input tensor shape.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt Size(TensorShape shape)
        {
            return m_Backend.Size(shape);
        }

        /// <summary>
        /// Generates a tensor with a given shape filled with a given value.
        /// </summary>
        /// <param name="X">The input tensor shape.</param>
        /// <param name="value">The fill value.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorFloat ConstantOfShape(TensorShape X, float value)
        {
            return m_Backend.ConstantOfShape(X, value);
        }

        /// <summary>
        /// Generates a tensor with a given shape filled with a given value.
        /// </summary>
        /// <param name="X">The input tensor shape.</param>
        /// <param name="value">The fill value.</param>
        /// <returns>The computed output tensor.</returns>
        public TensorInt ConstantOfShape(TensorShape X, int value)
        {
            return m_Backend.ConstantOfShape(X, value);
        }

        /// <summary>
        /// Creates a copy of a given input tensor with the same shape and values.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Copy(Tensor x)
        {
            return m_Backend.Copy(x);
        }

        /// <summary>
        /// Computes the output tensor using an element-wise `Cast` function: f(x) = (float)x or f(x) = (int)x depending on the value of `toType`.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <param name="toType">The data type to cast to as a `DataType`.</param>
        /// <returns>The computed output tensor.</returns>
        public Tensor Cast(Tensor x, DataType toType)
        {
            return m_Backend.Cast(x, toType);
        }
    }
}
