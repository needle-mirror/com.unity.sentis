using System;
using System.Collections.Generic;

namespace Unity.Sentis {

/// <summary>
/// An interface that provides methods for operations on tensors.
/// </summary>
public interface IOps : IDisposable
{
    /// <summary>
    /// Performs a matrix multiplication operation with optional transposes: f(a, b) = a' x b'.
    /// </summary>
    /// <param name="X">The first input tensor.</param>
    /// <param name="xTranspose">Whether to transpose the first input tensor before performing the matrix multiplication.</param>
    /// <param name="y">The second input tensor.</param>
    /// <param name="yTranspose">Whether to transpose the second input tensor before performing the matrix multiplication.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat MatMul2D(TensorFloat X, bool xTranspose, TensorFloat y, bool yTranspose);

    /// <summary>
    /// Performs a multi-dimensional matrix multiplication operation: f(a, b) = a x b.
    /// </summary>
    /// <param name="X">The first input tensor.</param>
    /// <param name="Y">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat MatMul(TensorFloat X, TensorFloat Y);

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
    TensorFloat Dense(TensorFloat X, TensorFloat W, TensorFloat B, Layers.FusableActivation fusedActivation);

    /// <summary>
    /// Computes the output tensor by retaining the lower triangular values from an input matrix or matrix batch and setting the other values to zero.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="k">The offset from the diagonal to keep.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Tril(Tensor X, int k = 0);

    /// <summary>
    /// Computes the output tensor by retaining the upper triangular values from an input matrix or matrix batch and setting the other values to zero.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="k">The offset from the diagonal to exclude.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Triu(Tensor X, int k = 0);

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
    TensorFloat Conv(TensorFloat X, TensorFloat K, TensorFloat B, int groups, int[] stride, int[] pad, int[] dilation, Layers.FusableActivation fusedActivation);

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
    TensorFloat Conv2DTrans(TensorFloat X, TensorFloat K, TensorFloat B, int[] stride, int[] pad, int[] outputAdjustment, Layers.FusableActivation fusedActivation);

    /// <summary>
    /// Calculates an output tensor by resampling the input tensor along the spatial dimensions with given scales.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="scale">The factor to scale each dimension by.</param>
    /// <param name="interpolationMode">The `InterpolationMode` to use for the operation.</param>
    /// <param name="nearestMode">The `NearestMode` to use for the operation when using `InterpolationMode.NearestMode`. The default is `NearestMode.RoundPreferFloor`.</param>
    /// <param name="coordTransformMode">The `CoordTransformMode` to use for the operation. The default is `CoordTransformMode.HalfPixel`.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Resize(TensorFloat X, float[] scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode = Layers.NearestMode.RoundPreferFloor, Layers.CoordTransformMode coordTransformMode = Layers.CoordTransformMode.HalfPixel);

    /// <summary>
    /// Computes the output tensor by permuting data from depth into blocks of spatial data.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="blocksize">The size of the blocks to move the depth data into.</param>
    /// <param name="mode">The ordering of the data in the output tensor as a `DepthToSpaceMode`.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat DepthToSpace(TensorFloat X, int blocksize, Layers.DepthToSpaceMode mode);

    /// <summary>
    /// Computes the output tensor by permuting data from blocks of spatial data into depth.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <param name="blocksize">The size of the blocks to move the depth data into.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat SpaceToDepth(TensorFloat x, int blocksize);

    /// <summary>
    /// Calculates an output tensor by pooling the maximum values of the input tensor across its spatial dimensions according to the given pool and stride values.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="pool">The size of the kernel along each spatial axis.</param>
    /// <param name="stride">The stride along each spatial axis.</param>
    /// <param name="pad">The lower and upper padding values for each spatial dimension. For example, [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat MaxPool(TensorFloat X, int[] pool, int[] stride, int[] pad);

    /// <summary>
    /// Calculates an output tensor by pooling the mean values of the input tensor across its spatial dimensions according to the given pool and stride values.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="pool">The size of the kernel along each spatial axis.</param>
    /// <param name="stride">The stride along each spatial axis.</param>
    /// <param name="pad">The lower and upper padding values for each spatial dimension. For example, [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat AveragePool(TensorFloat X, int[] pool, int[] stride, int[] pad);

    /// <summary>
    /// Calculates an output tensor by pooling the maximum values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat GlobalMaxPool(TensorFloat X);

    /// <summary>
    /// Calculates an output tensor by pooling the mean values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat GlobalAveragePool(TensorFloat X);

    /// <summary>
    /// Calculates the output tensor by adding padding to the input tensor according to the given padding values and mode.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="pad">The lower and upper padding values for each dimension.</param>
    /// <param name="padMode">The `PadMode` to use when padding. The default value is `PadMode.Constant`.</param>
    /// <param name="constant">The constant value to fill with when using `PadMode.Constant`. The default value is 0.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Pad(TensorFloat X, int[] pad, Layers.PadMode padMode = Layers.PadMode.Constant, float constant = 0.0f);

    /// <summary>
    /// Computes the output tensor with an element-wise `ScaleBias` function: f(x, s, b) = x * s + b.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="S">The scale tensor.</param>
    /// <param name="B">The bias tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat ScaleBias(TensorFloat X, TensorFloat S, TensorFloat B);

    /// <summary>
    /// Computes the mean variance on the spatial dimensions of the input tensor and normalizes them according to `scale` and `bias` tensors.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="S">The scale tensor.</param>
    /// <param name="B">The bias tensor.</param>
    /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat InstanceNormalization(TensorFloat X, TensorFloat S, TensorFloat B, float epsilon);

    /// <summary>
    /// Computes the mean variance on the spatial dimensions of the input tensor and normalizes them according to `scale` and `bias` tensors.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="S">The scale tensor.</param>
    /// <param name="B">The bias tensor.</param>
    /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat AxisNormalization(TensorFloat X, TensorFloat S, TensorFloat B, float epsilon);

    /// <summary>
    /// Normalizes the input tensor over local input regions.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="alpha">The scaling parameter to use for the normalization.</param>
    /// <param name="beta">The exponent to use for the normalization.</param>
    /// <param name="bias">The bias value to use for the normalization.</param>
    /// <param name="size">The number of channels to sum over.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat LRN(TensorFloat X, float alpha, float beta, float bias, int size);

    /// <summary>
    /// Generates an output tensor of a given shape with random values in a normal distribution with given `mean` and `scale`, and an optional `seed` value.
    /// </summary>
    /// <param name="S">The shape to use for the output tensor.</param>
    /// <param name="mean">The mean of the normal distribution to use to generate the output.</param>
    /// <param name="scale">The standard deviation of the normal distribution to use to generate the output.</param>
    /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the operation generates a seed using `System.Random()`.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat RandomNormal(TensorShape S, float mean, float scale, float? seed);

    /// <summary>
    /// Generates an output tensor of a given shape with random values in a uniform distribution between a given `low` and `high`, and an optional `seed` value.
    /// </summary>
    /// <param name="S">The shape to use for the output tensor.</param>
    /// <param name="low">The lower end of the interval of the uniform distribution to use to generate the output.</param>
    /// <param name="high">The upper end of the interval of the uniform distribution to use to generate the output.</param>
    /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the operation generates a seed using `System.Random()`.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat RandomUniform(TensorShape S, float low, float high, float? seed);

    /// <summary>
    /// Generates an output tensor with values from a multinomial distribution according to the probabilities given by the input tensor.
    /// </summary>
    /// <param name="x">The probabilities input tensor.</param>
    /// <param name="count">The number of times to sample the input.</param>
    /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the operation generates a seed using `System.Random()`.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Multinomial(TensorFloat x, int count, float? seed);

    /// <summary>
    /// Generates a one-hot tensor with a given `depth`, `indices` and on and off values.
    /// </summary>
    /// <param name="indices">The indices input tensor.</param>
    /// <param name="axis">The axis along which the operation adds the one-hot representation.</param>
    /// <param name="depth">The depth of the one-hot tensor.</param>
    /// <param name="offValue">The value to use for an off element.</param>
    /// <param name="onValue">The value to use for an on element.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt OneHot(TensorInt indices, int axis, int depth, int offValue, int onValue);

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
    TensorFloat RoiAlign(TensorFloat X, TensorFloat Rois, TensorInt Indices, Layers.RoiPoolingMode mode, int outputHeight, int outputWidth, int samplingRatio, float spatialScale);

    /// <summary>
    /// Returns the indices of the elements of the input tensor that are not zero.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt NonZero(TensorFloat X);

    /// <summary>
    /// Returns the indices of the elements of the input tensor that are not zero.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt NonZero(TensorInt X);

    /// <summary>
    /// Generates a 1D output tensor where the values form an arithmetic progression defined by the `start`, `limit`, and `delta` values.
    /// </summary>
    /// <param name="start">The first value in the range.</param>
    /// <param name="limit">The limit of the range.</param>
    /// <param name="delta">The delta between subsequent values in the range.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Range(float start, float limit, float delta);

    /// <summary>
    /// Generates a 1D output tensor where the values form an arithmetic progression defined by the `start`, `limit`, and `delta` values.
    /// </summary>
    /// <param name="start">The first value in the range.</param>
    /// <param name="limit">The limit of the range.</param>
    /// <param name="delta">The delta between subsequent values in the range.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Range(int start, int limit, int delta);

    /// <summary>
    /// Generates an output tensor with values 0 or 1 from a Bernoulli distribution. The input tensor contains the probabilities to use for generating the output values.
    /// </summary>
    /// <param name="x">The probabilities input tensor.</param>
    /// <param name="dataType">The data type of the output tensor.</param>
    /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the operation generates a seed using `System.Random()`.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Bernoulli(TensorFloat x, DataType dataType, float? seed);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Relu` activation function: f(x) = max(0, x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Relu(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the `Softmax` activation function along an axis: f(x, axis) = exp(X) / ReduceSum(exp(X), axis).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axis">The axis along which to apply the `Softmax` activation function. The default value is -1.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Softmax(TensorFloat X, int axis = -1);

    /// <summary>
    /// Computes an output tensor by applying the `LogSoftmax` activation function along an axis: f(x, axis) = log(Softmax(x, axis)).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axis">The axis along which to apply the `LogSoftmax` activation function. The default value is -1.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat LogSoftmax(TensorFloat X, int axis = -1);

    /// <summary>
    /// Computes an output tensor by applying the `Hardmax` activation function along an axis: f(x, axis) = 1 if x is the first maximum value along the specified axis, otherwise f(x) = 0.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axis">The axis along which to apply the `Hardmax` activation function. The default value is -1.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Hardmax(TensorFloat X, int axis = -1);

    /// <summary>
    /// Performs the cumulative sum along a given axis.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axis">The axis along which to apply the cumulative sum.</param>
    /// <param name="reverse">Whether to perform the cumulative sum from the end of the axis.</param>
    /// <param name="exclusive">Whether to include the respective input element in the cumulative sum.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat CumSum(TensorFloat X, int axis, bool reverse = false, bool exclusive = false);

    /// <summary>
    /// Performs the cumulative sum along a given axis.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axis">The axis along which to apply the cumulative sum.</param>
    /// <param name="reverse">Whether to perform the cumulative sum from the end of the axis.</param>
    /// <param name="exclusive">Whether to include the respective input element in the cumulative sum.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt CumSum(TensorInt X, int axis, bool reverse = false, bool exclusive = false);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Tanh` activation function: f(x) = tanh(x).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Tanh(TensorFloat X);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Softplus` activation function: f(x) = ln(e^x + 1).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Softplus(TensorFloat X);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Sigmoid` activation function: f(x) = 1/(1 + e^(-x)).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Sigmoid(TensorFloat X);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `HardSigmoid` activation function: f(x) = clamp(alpha * x + beta, 0, 1).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <param name="alpha">The alpha value to use for the `HardSigmoid` activation function.</param>
    /// <param name="beta">The beta value to use for the `HardSigmoid` activation function.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat HardSigmoid(TensorFloat x, float alpha, float beta);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Elu` activation function: f(x) = x if x >= 0, otherwise f(x) = alpha * (e^x - 1).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="alpha">The alpha value to use for the `Elu` activation function.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Elu(TensorFloat X, float alpha);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Gelu` activation function: f(x) = x / 2 * (1 + erf(x / sqrt(2))).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Gelu(TensorFloat X);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Relu6` activation function: f(x) = clamp(x, 0, 6).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Relu6(TensorFloat X);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `LeakyRelu` activation function: f(x) = x if x >= 0, otherwise f(x) = alpha * x.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <param name="alpha">The alpha value to use for the `LeakyRelu` activation function.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat LeakyRelu(TensorFloat x, float alpha);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Selu` activation function: f(x) = gamma * x if x >= 0, otherwise f(x) = (alpha * e^x - alpha).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <param name="alpha">The alpha value to use for the `Selu` activation function.</param>
    /// <param name="gamma">The alpha value to use for the `Selu` activation function.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Selu(TensorFloat x, float alpha, float gamma);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `PRelu` activation function: f(x) = x if x >= 0, otherwise f(x) = slope * x.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <param name="slope">The slope tensor, must be unidirectional broadcastable to x.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat PRelu(TensorFloat x, TensorFloat slope);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Swish` activation function: f(x) = sigmoid(x) * x = x / (1 + e^{-x}).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Swish(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Abs` math function: f(x) = f(x) = |x|.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Abs(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Abs` math function: f(x) = f(x) = |x|.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Abs(TensorInt x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Neg` math function: f(x) = -x.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Neg(TensorFloat X);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Neg` math function: f(x) = -x.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Neg(TensorInt X);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Ceil` math function: f(x) = ceil(x).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Ceil(TensorFloat X);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Clip` math function: f(x) = clamp(x, min, max).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <param name="min">The lower clip value.</param>
    /// <param name="max">The upper clip value.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Clip(TensorFloat x, float min, float max);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Floor` math function: f(x) = floor(x).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Floor(TensorFloat X);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Round` math function: f(x) = round(x).
    ///
    /// If the fractional part is equal to 0.5, rounds to the nearest even integer.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Round(TensorFloat X);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Reciprocal` math function: f(x) = 1 / x.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Reciprocal(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Square` math function: f(x) = x * x.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Square(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Exp` math function: f(x) = exp(x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Exp(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Log` math function: f(x) = log(x).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Log(TensorFloat X);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Sqrt` math function: f(x) = sqrt(x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Sqrt(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Acos` trigonometric function: f(x) = acos(x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Acos(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Acosh` trigonometric function: f(x) = acosh(x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Acosh(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Asin` trigonometric function: f(x) = asin(x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Asin(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Asinh` trigonometric function: f(x) = asinh(x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Asinh(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Atan` trigonometric function: f(x) = atan(x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Atan(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Atanh` trigonometric function: f(x) = atanh(x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Atanh(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Cos` trigonometric function: f(x) = cos(x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Cos(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Cosh` trigonometric function: f(x) = cosh(x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Cosh(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Sin` trigonometric function: f(x) = sin(x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Sin(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Sinh` trigonometric function: f(x) = sinh(x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Sinh(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Tan` trigonometric function: f(x) = tan(x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Tan(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Erf` activation function: f(x) = erf(x).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Erf(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Celu` activation function: f(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1)).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <param name="alpha">The alpha value to use for the `Celu` activation function.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Celu(TensorFloat x, float alpha);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `HardSwish` activation function: f(x) = x * max(0, min(1, 1/6 * x + 0.5)).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat HardSwish(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Shrink` activation function: f(x) = x + bias if x &lt; lambd. f(x) = x - bias if x &gt; lambd. Otherwise f(x) = 0.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <param name="bias">The bias value to use for the `Shrink` activation function.</param>
    /// <param name="lambd">The lambda value to use for the `Shrink` activation function.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Shrink(TensorFloat x, float bias, float lambd);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `Softsign` activation function: f(x) = x/(|x| + 1).
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Softsign(TensorFloat x);

    /// <summary>
    /// Computes an output tensor by applying the element-wise `ThresholdedRelu` activation function: f(x) = x if x > alpha, otherwise f(x) = 0.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <param name="alpha">The alpha value to use for the `ThresholdedRelu` activation function.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat ThresholdedRelu(TensorFloat x, float alpha);

    /// <summary>
    /// Performs an element-wise `Sum` math operation: f(x1, x2 ... xn) = x1 + x2 ... xn.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="tensors">The input tensors.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Sum(TensorFloat[] tensors);

    /// <summary>
    /// Performs an element-wise `Add` math operation: f(a, b) = a + b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Add(TensorFloat A, TensorFloat B);

    /// <summary>
    /// Performs an element-wise `Add` math operation: f(a, b) = a + b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Add(TensorInt A, TensorInt B);

    /// <summary>
    /// Performs an element-wise `Sub` math operation: f(a, b) = a - b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Sub(TensorFloat A, TensorFloat B);

    /// <summary>
    /// Performs an element-wise `Sub` math operation: f(a, b) = a - b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Sub(TensorInt A, TensorInt B);

    /// <summary>
    /// Performs an element-wise `Mul` math operation: f(a, b) = a * b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Mul(TensorFloat A, TensorFloat B);

    /// <summary>
    /// Performs an element-wise `Mul` math operation: f(a, b) = a * b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Mul(TensorInt A, TensorInt B);

    /// <summary>
    /// Performs an element-wise `Div` math operation: f(a, b) = a / b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Div(TensorFloat A, TensorFloat B);

    /// <summary>
    /// Performs an element-wise `Div` math operation: f(a, b) = a / b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Div(TensorInt A, TensorInt B);

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
    TensorInt Mod(TensorInt A, TensorInt B);

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
    TensorInt FMod(TensorInt A, TensorInt B);

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
    TensorFloat FMod(TensorFloat A, TensorFloat B);

    /// <summary>
    /// Performs an element-wise `Pow` math operation: f(a, b) = pow(a, b).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Pow(TensorFloat A, TensorFloat B);

    /// <summary>
    /// Performs an element-wise `Pow` math operation: f(a, b) = pow(a, b).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Pow(TensorFloat A, TensorInt B);

    /// <summary>
    /// Performs an element-wise `Min` math operation: f(x1, x2 ... xn) = min(x1, x2 ... xn).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="tensors">The input tensors.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Min(TensorFloat[] tensors);

    /// <summary>
    /// Performs an element-wise `Min` math operation: f(x1, x2 ... xn) = min(x1, x2 ... xn).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="tensors">The input tensors.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Min(TensorInt[] tensors);

    /// <summary>
    /// Performs an element-wise `Max` math operation: f(x1, x2 ... xn) = max(x1, x2 ... xn).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="tensors">The input tensors.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Max(TensorFloat[] tensors);

    /// <summary>
    /// Performs an element-wise `Max` math operation: f(x1, x2 ... xn) = max(x1, x2 ... xn).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="tensors">The input tensors.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Max(TensorInt[] tensors);

    /// <summary>
    /// Performs an element-wise `Mean` math operation: f(x1, x2 ... xn) = (x1 + x2 ... xn) / n.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="tensors">The input tensors.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Mean(TensorFloat[] tensors);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceMax` operation: f(x1, x2 ... xn) = max(x1, x2, ... , xn).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat ReduceMax(TensorFloat X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceMean` operation: f(x1, x2 ... xn) = max(x1, x2, ... , xn).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt ReduceMax(TensorInt X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceMean` operation: f(x1, x2 ... xn) = (x1 + x2 + ... + xn) / n.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat ReduceMean(TensorFloat X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceMin` operation: f(x1, x2 ... xn) = min(x1, x2, ... , xn).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat ReduceMin(TensorFloat X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceMin` operation: f(x1, x2 ... xn) = min(x1, x2, ... , xn).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt ReduceMin(TensorInt X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceProd` operation: f(x1, x2 ... xn) = x1 * x2 * ... * xn.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat ReduceProd(TensorFloat X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceProd` operation: f(x1, x2 ... xn) = x1 * x2 * ... * xn.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt ReduceProd(TensorInt X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceSum` operation: f(x1, x2 ... xn) = x1 + x2 + ... + xn.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat ReduceSum(TensorFloat X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceSum` operation: f(x1, x2 ... xn) = x1 + x2 + ... + xn.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt ReduceSum(TensorInt X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceSumSquare` operation: f(x1, x2 ... xn) = x1² + x2² + ... + xn².
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat ReduceSumSquare(TensorFloat X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceSumSquare` operation: f(x1, x2 ... xn) = x1² + x2² + ... + xn².
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt ReduceSumSquare(TensorInt X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceL1` operation: f(x1, x2 ... xn) = |x1| + |x2| + ... + |xn|.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat ReduceL1(TensorFloat X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceL1` operation: f(x1, x2 ... xn) = |x1| + |x2| + ... + |xn|.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt ReduceL1(TensorInt X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceL2` operation: f(x1, x2 ... xn) = sqrt(x1² + x2² + ... + xn²).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat ReduceL2(TensorFloat X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceLogSum` operation: f(x1, x2 ... xn) = log(x1 + x2 + ... + xn).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat ReduceLogSum(TensorFloat X, int[] axes, bool keepdim);

    /// <summary>
    /// Reduces an input tensor along the given axes using the `ReduceLogSumExp` operation: f(x1, x2 ... xn) = log(e^x1 + e^x2 + ... + e^xn).
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axes">The axes along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat ReduceLogSumExp(TensorFloat X, int[] axes, bool keepdim);

    /// <summary>
    /// Computes the indices of the maximum elements of the input tensor along a given axis.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axis">The axis along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt ArgMax(TensorFloat X, int axis, bool keepdim, bool selectLastIndex = false);

    /// <summary>
    /// Computes the indices of the maximum elements of the input tensor along a given axis.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axis">The axis along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt ArgMax(TensorInt X, int axis, bool keepdim, bool selectLastIndex = false);

    /// <summary>
    /// Computes the indices of the minimum elements of the input tensor along a given axis.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axis">The axis along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt ArgMin(TensorFloat X, int axis, bool keepdim, bool selectLastIndex);

    /// <summary>
    /// Computes the indices of the minimum elements of the input tensor along a given axis.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axis">The axis along which to reduce.</param>
    /// <param name="keepdim">Whether to keep the reduced axes in the output tensor.</param>
    /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt ArgMin(TensorInt X, int axis, bool keepdim, bool selectLastIndex);

    /// <summary>
    /// Performs an element-wise `Greater` logical comparison operation: f(a, b) = 1 if a > b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Greater(TensorFloat A, TensorFloat B);

    /// <summary>
    /// Performs an element-wise `Greater` logical comparison operation: f(a, b) = 1 if a > b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Greater(TensorInt A, TensorInt B);

    /// <summary>
    /// Performs an element-wise `GreaterOrEqual` logical comparison operation: f(a, b) = 1 if a >= b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt GreaterOrEqual(TensorFloat A, TensorFloat B);

    /// <summary>
    /// Performs an element-wise `GreaterOrEqual` logical comparison operation: f(a, b) = 1 if a >= b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt GreaterOrEqual(TensorInt A, TensorInt B);

    /// <summary>
    /// Performs an element-wise `Less` logical comparison operation: f(a, b) = 1 if a &lt; b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Less(TensorFloat A, TensorFloat B);

    /// <summary>
    /// Performs an element-wise `Less` logical comparison operation: f(a, b) = 1 if a &lt; b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Less(TensorInt A, TensorInt B);

    /// <summary>
    /// Performs an element-wise `LessOrEqual` logical comparison operation: f(a, b) = 1 if a &lt;= b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt LessOrEqual(TensorFloat A, TensorFloat B);

    /// <summary>
    /// Performs an element-wise `LessOrEqual` logical comparison operation: f(a, b) = 1 if a &lt;= b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt LessOrEqual(TensorInt A, TensorInt B);

    /// <summary>
    /// Performs an element-wise `Equal` logical comparison operation: f(a, b) = 1 if a == b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Equal(TensorFloat A, TensorFloat B);

    /// <summary>
    /// Performs an element-wise `Equal` logical comparison operation: f(a, b) = 1 if a == b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Equal(TensorInt A, TensorInt B);

    /// <summary>
    /// Performs an element-wise `Or` logical operation: f(a, b) = a | b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Or(TensorInt A, TensorInt B);

    /// <summary>
    /// Performs an element-wise `And` logical operation: f(a, b) = a & b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt And(TensorInt A, TensorInt B);

    /// <summary>
    /// Performs an element-wise `Xor` logical operation: f(a) = a ^ b.
    /// </summary>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Xor(TensorInt A, TensorInt B);

    /// <summary>
    /// Performs an element-wise `Not` logical operation: f(x) = ~x.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Not(TensorInt X);

    /// <summary>
    /// Performs an element-wise `Sign` math operation: f(x) = 1 if x > 0. f(x) = -1 if x &lt; 0. Otherwise f(x) = 0.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat Sign(TensorFloat X);

    /// <summary>
    /// Performs an element-wise `Sign` math operation: f(x) = 1 if x > 0. f(x) = -1 if x &lt; 0. Otherwise f(x) = 0.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Sign(TensorInt X);

    /// <summary>
    /// Performs an element-wise `IsNaN` logical operation: f(x) = 1 if x is NaN, otherwise f(x) = 0.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt IsNaN(TensorFloat X);

    /// <summary>
    /// Performs an element-wise `IsInf` logical operation: f(x) = 1 elementwise if x is +Inf and `detectPositive` is `true`, or x is -Inf and `detectNegative` is `true`. Otherwise f(x) = 0.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="detectNegative">Whether to detect negative infinities in the `IsInf` function.</param>
    /// <param name="detectPositive">Whether to detect positive infinities in the `IsInf` function.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt IsInf(TensorFloat X, bool detectNegative, bool detectPositive);

    /// <summary>
    /// Performs an element-wise `Where` logical operation: f(condition, a, b) = a if `condition` is `true`, otherwise f(condition, a, b) = b.
    /// </summary>
    /// <param name="C">The condition tensor.</param>
    /// <param name="A">The first input tensor.</param>
    /// <param name="B">The second input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Where(TensorInt C, Tensor A, Tensor B);

    /// <summary>
    /// Calculates an output tensor by copying the data from the input tensor and using a given shape. The data from the input tensor is unchanged.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="shape">The shape of the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Reshape(Tensor X, TensorShape shape);

    /// <summary>
    /// Calculates an output tensor by broadcasting the input tensor into a given shape.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="shape">The shape to broadcast the input shape together with to calculate the output tensor.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Expand(Tensor X, TensorShape shape);

    /// <summary>
    /// Calculates an output tensor by reversing the dimensions of the input tensor.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Transpose(Tensor x);

    /// <summary>
    /// Calculates an output tensor by permuting the axes and data of the input tensor according to the given permutations.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <param name="permutations">The axes to sample the output tensor from in the input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Transpose(Tensor x, int[] permutations);

    /// <summary>
    /// Calculates an output tensor by concatenating the input tensors along a given axis.
    /// </summary>
    /// <param name="tensors">The input tensors.</param>
    /// <param name="axis">The axis along which to concatenate the input tensors.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Concat(Tensor[] tensors, int axis);

    /// <summary>
    /// Calculates an output tensor by splitting the input tensor along a given axis between start and end.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="axis">The axis along which to split the input tensor.</param>
    /// <param name="start">The inclusive start value for the split.</param>
    /// <param name="end">The exclusive end value for the split.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Split(Tensor X, int axis, int start, int end);

    /// <summary>
    /// Calculates an output tensor by slicing the input tensor along given axes with given starts, ends, and steps.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="starts">The start index along each axis.</param>
    /// <param name="ends">The end index along each axis.</param>
    /// <param name="axes">The axes along which to slice. If this is `null`, the layer slices all axes.</param>
    /// <param name="steps">The step values for slicing. If this is `null`, the layer uses step size 1 throughout.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Slice(Tensor X, int[] starts, int[] ends, int[] axes, int[] steps);

    /// <summary>
    /// Calculates an output tensor by repeating the input layer a given number of times along each axis.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="repeats">The number of times to tile the input tensor along each axis.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Tile(Tensor X, int[] repeats);

    /// <summary>
    /// Selects slices of an input tensor along a given axis according to a condition tensor.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="indices">The indices tensor.</param>
    /// <param name="axis">The axis along which to compress.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Compress(Tensor X, TensorInt indices, int axis);

    /// <summary>
    /// Takes values from the input tensor indexed by the indices tensor along a given axis and concatenates them.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="indices">The indices tensor.</param>
    /// <param name="axis">The axis along which to gather.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Gather(Tensor X, TensorInt indices, int axis);

    /// <summary>
    /// Takes values from the input tensor indexed by the indices tensor along a given axis and concatenates them.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="indices">The indices tensor.</param>
    /// <param name="axis">The axis along which to gather.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor GatherElements(Tensor X, TensorInt indices, int axis);

    /// <summary>
    /// Takes slices of values from the batched input tensor indexed by the `indices` tensor.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="indices">The indices tensor.</param>
    /// <param name="batchDims">The number of batch dimensions of the input tensor, the gather begins at the next dimension.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor GatherND(Tensor X, TensorInt indices, int batchDims);

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
    Tensor ScatterElements(Tensor X, TensorInt indices, Tensor updates, int axis, Layers.ScatterReductionMode reduction);

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
    TensorFloat ScatterND(TensorFloat X, TensorInt indices, TensorFloat updates, Layers.ScatterReductionMode reduction);

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
    TensorInt ScatterND(TensorInt X, TensorInt indices, TensorInt updates, Layers.ScatterReductionMode reduction);

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
    TensorFloat[] LSTM(TensorFloat X, TensorFloat W, TensorFloat R, TensorFloat B, TensorInt sequenceLens, TensorFloat initialH, TensorFloat initialC, TensorFloat P, Layers.RnnDirection direction, Layers.RnnActivation[] activations, float[] activationAlpha, float[] activationBeta, bool inputForget, float clip, Layers.RnnLayout layout);

    /// <summary>
    /// Calculates the top-K largest or smallest elements of an input tensor along a given axis.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="k">The number of elements to calculate.</param>
    /// <param name="axis">The axis along which to perform the top-K operation.</param>
    /// <param name="largest">Whether to calculate the top-K largest elements. If this is `false`, the layer calculates the top-K smallest elements.</param>
    /// <param name="sorted">Whether to return the elements in sorted order.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor[] TopK(TensorFloat X, int k, int axis, bool largest, bool sorted);

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
    TensorFloat Einsum(string equation, params TensorFloat[] operands);

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
    TensorInt NonMaxSuppression(TensorFloat boxes, TensorFloat scores, int maxOutputBoxesPerClass, float iouThreshold, float scoreThreshold, Layers.CenterPointBox centerPointBox);

    /// <summary>
    /// Calculates the shape of an input tensor as a 1D `TensorInt`.
    /// </summary>
    /// <param name="X">The input tensor.</param>
    /// <param name="start">The inclusive start axis for slicing the shape of the input tensor. The default value is 0.</param>
    /// <param name="end">The exclusive end axis for slicing the shape of the input tensor. The default value is 8.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Shape(Tensor X, int start = 0, int end = TensorShape.maxRank);

    /// <summary>
    /// Calculates the number of elements of an input tensor shape as a scalar `TensorInt`.
    /// </summary>
    /// <param name="shape">The input tensor shape.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt Size(TensorShape shape);

    /// <summary>
    /// Generates a tensor with a given shape filled with a given value.
    /// </summary>
    /// <param name="X">The input tensor shape.</param>
    /// <param name="value">The fill value.</param>
    /// <returns>The computed output tensor.</returns>
    TensorFloat ConstantOfShape(TensorShape X, float value);

    /// <summary>
    /// Generates a tensor with a given shape filled with a given value.
    /// </summary>
    /// <param name="X">The input tensor shape.</param>
    /// <param name="value">The fill value.</param>
    /// <returns>The computed output tensor.</returns>
    TensorInt ConstantOfShape(TensorShape X, int value);

    /// <summary>
    /// Creates a copy of a given input tensor with the same shape and values.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Copy(Tensor x);

    /// <summary>
    /// Computes the output tensor using an element-wise `Cast` function: f(x) = (float)x or f(x) = (int)x depending on the value of `toType`.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <param name="toType">The data type to cast to as a `DataType`.</param>
    /// <returns>The computed output tensor.</returns>
    Tensor Cast(Tensor x, DataType toType);

    /// <summary>
    /// Pins and returns a tensor using this backend.
    /// </summary>
    /// <param name="x">The input tensor.</param>
    /// <param name="uploadCache">Whether to also move the elements of the tensor to the device.</param>
    /// <returns>The pinned input tensor.</returns>
    Tensor PinToDevice(Tensor x, bool uploadCache = true);

    /// <summary>
    /// Allocates a new tensor with the internal allocator.
    /// </summary>
    Tensor NewTensor(TensorShape shape, DataType dataType, AllocScope scope);

    /// <summary>
    /// Resets the internal allocator.
    /// </summary>
    void ResetAllocator(bool keepCachedMemory = true);

    /// <summary>
    /// Called after every layer execution. It allows IOps to run cleanup operations
    /// such as clearing temporary buffers only used in the scope of the last layer
    /// executed.
    /// </summary>
    void PostLayerCleanup();

    /// <summary>
    /// Returns the `DeviceType` for the ops.
    /// </summary>
    DeviceType deviceType { get; }
}

/// <summary>
/// An interface that provides methods for compiling models.
/// </summary>
interface IModelCompiler
{
    /// <summary>
    /// Prepares a model for execution, allocating required intermediate tensors.
    /// </summary>
    void PrepareModel(Model model, IDictionary<string, TensorShape> inputShapes, IVars vars);

    /// <summary>
    /// Prepares a layer for execution.
    /// </summary>
    void PreExecuteLayer(Layers.Layer layer, Tensor[] inputs);
}

/// <summary>
/// An interface that provides methods for storing variables.
/// </summary>
public interface IVars : IDisposable
{
    /// <summary>
    /// Sets a given input with a tensor.
    /// </summary>
    void SetInput(string name, Tensor x);

    /// <summary>
    /// Prepares storage for a given model.
    /// </summary>
    void PrepareStorage(Model model, IOps optionalOpsToPrepareTensors = null, IDictionary<string, TensorShape> optionalInputShapes = null, bool takeoverWeights = false);

    /// <summary>
    /// Gathers the input tensors for a given layer.
    /// </summary>
    Tensor[] GatherInputs(Layers.Layer forLayer);

    /// <summary>
    /// Prepares storage for a given layer.
    /// </summary>
    void PrepareStorage(Layers.Layer forLayer);

    /// <summary>
    /// Disposes storage that can be deleted after executing a given layer.
    /// </summary>
    void DisposeAfterLayer(Layers.Layer forLayer);

    /// <summary>
    /// Stores the result of execution for a given layer.
    /// </summary>
    void Store(Layers.Layer fromLayer, Tensor result);

    /// <summary>
    /// Stores the result of execution for a given tensor name.
    /// </summary>
    void Store(string fromLayer, Tensor result);

    /// <summary>
    /// Peeks the output tensor of a given name.
    /// </summary>
    Tensor PeekOutput(string name);

    /// <summary>
    /// Returns the current allocator.
    /// </summary>
    ITensorAllocator GetAllocator();
}

/// <summary>
/// Options for the lifetime of an allocation.
/// </summary>
public enum AllocScope
{
    /// <summary>
    /// Use this tensor in other layers or as the output of a model.
    /// </summary>
    LayerOutput,

    /// <summary>
    /// Use this tensor only temporarily as an intermediate step when executing a layer.
    /// </summary>
    InternalToLayer
}

/// <summary>
/// An interface that provides methods for allocating tensors.
/// </summary>
public interface ITensorAllocator : IDisposable
{
    /// <summary>
    /// Allocates a tensor of a given shape, data type on a given device type, and given scope.
    /// </summary>
    Tensor Alloc(TensorShape shape, DataType dataType, DeviceType deviceType, AllocScope scope = AllocScope.LayerOutput);

    /// <summary>
    /// Allocates a tensor of a given shape, data type, and a given scope from an existing `ITensorData` buffer.
    /// </summary>
    Tensor Alloc(TensorShape shape, DataType dataType, ITensorData buffer, AllocScope scope = AllocScope.LayerOutput);

    /// <summary>
    /// Allows ITensorAllocator to run cleanup operations such as clearing
    /// temporary buffers only used in the scope of the last layer executed.
    /// </summary>
    void PostLayerCleanup();

    // MoveToDevice() callback is called from the following Tensor methods:
    // UploadToDevice(), AttachToDevice() and DetachFromDevice()
    /// <summary>
    /// Moves a tensor to a device.
    /// </summary>
    void MoveToDevice(Tensor x, ITensorData newBuffer, ITensorData oldBuffer, bool disposeDetachedBufferHint);

    // NOTE: Release() should be ready to handle edge-case situation when
    //  externally created new Tensor instance is passed with
    //  ITensorData (tensorOnDevice) that is already owned by the allocator
    /// <summary>
    /// Releases a tensor.
    /// </summary>
    void Release(Tensor x, bool calledFromTensorDispose);

    /// <summary>
    /// Waives ownership of a tensor.
    /// </summary>
    void WaiveOwnership(Tensor x);

    /// <summary>
    /// Resets the allocator.
    /// </summary>
    void Reset(bool keepCachedMemory); // end-of-frame
}

/// <summary>
/// Represents a context object that holds the model operations and variables for layer execution.
/// </summary>
public class ExecutionContext
{
    /// <summary>
    /// The `IOps` used for execution.
    /// </summary>
    public IOps ops;

    /// <summary>
    /// The `IVars` used for execution
    /// </summary>
    public IVars vars;
}

} // namespace Unity.Sentis

