using System;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Unity.Sentis.TestsUtils")]

namespace Unity.Sentis
{
    /// <summary>
    /// An interface that provides methods for operations on tensors.
    /// </summary>
    interface IBackend : IDisposable
    {
        /// <summary>
        /// Performs a matrix multiplication operation with optional transposes: f(a, b) = a' x b'.
        /// </summary>
        /// <param name="X">The first input tensor.</param>
        /// <param name="Y">The second input tensor.</param>
        /// <param name="O">The output tensor.</param>
        /// <param name="xTranspose">Whether to transpose the first input tensor before performing the matrix multiplication.</param>
        /// <param name="yTranspose">Whether to transpose the second input tensor before performing the matrix multiplication.</param>
        void MatMul2D(Tensor<float> X, Tensor<float> Y, Tensor<float> O, bool xTranspose, bool yTranspose);

        /// <summary>
        /// Performs a multi-dimensional matrix multiplication operation: f(a, b) = a x b.
        /// </summary>
        /// <param name="X">The first input tensor.</param>
        /// <param name="Y">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void MatMul(Tensor<float> X, Tensor<float> Y, Tensor<float> O);

        /// <summary>
        /// Performs a matrix multiplication operation: f(x, w, b) = X x W + B.
        /// input tensor must be of shape
        /// B : (N)
        /// W : (K, N)
        /// X : (..B.. M, K)
        /// O : (..B.., M, N)
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="W">The weights tensor.</param>
        /// <param name="B">The bias tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="fusedActivation">The fused activation to apply to the output tensor after the dense operation.</param>
        void Dense(Tensor<float> X, Tensor<float> W, Tensor<float> B, Tensor<float> O, Layers.FusableActivation fusedActivation);

        /// <summary>
        /// Performs a batched-matrix multiplication operation: f(x, w, b) = X x W + B.
        /// input tensor must be of shape
        /// B : (..B.., N)
        /// W : (..B..,K, N)
        /// X : (..B.., M, N)
        /// O : (..B.., M, N)
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="W">The weights tensor.</param>
        /// <param name="B">The bias tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="fusedActivation">The fused activation to apply to the output tensor after the dense operation.</param>
        void DenseBatched(Tensor<float> X, Tensor<float> W, Tensor<float> B, Tensor<float> O, Layers.FusableActivation fusedActivation);

        /// <summary>
        /// Computes the output tensor by retaining the lower triangular values from an input matrix or matrix batch and setting the other values to zero.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="k">The offset from the diagonal to keep.</param>
        void Tril(Tensor X, Tensor O, int k);

        /// <summary>
        /// Computes the output tensor by retaining the upper triangular values from an input matrix or matrix batch and setting the other values to zero.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="k">The offset from the diagonal to exclude.</param>
        void Triu(Tensor X, Tensor O, int k);

        /// <summary>
        /// Applies a convolution filter to an input tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="K">The filter tensor.</param>
        /// <param name="B">The optional bias tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="groups">The number of groups that input channels and output channels are divided into.</param>
        /// <param name="strides">The stride value for each spatial dimension of the filter.</param>
        /// <param name="pads">The lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="dilations">The dilation value of each spatial dimension of the filter.</param>
        /// <param name="fusedActivation">The fused activation type to apply after the convolution.</param>
        void Conv(Tensor<float> X, Tensor<float> K, Tensor<float> B, Tensor<float> O, int groups, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation);

        /// <summary>
        /// Applies a transpose convolution filter to an input tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="W">The filter tensor.</param>
        /// <param name="B">The optional bias tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="strides">The stride value for each spatial dimension of the filter.</param>
        /// <param name="pads">The lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="outputPadding">The output padding value for each spatial dimension in the filter.</param>
        /// <param name="fusedActivation">The fused activation type to apply after the convolution.</param>
        void ConvTranspose(Tensor<float> X, Tensor<float> W, Tensor<float> B, Tensor<float> O, Span<int> strides, Span<int> pads, Span<int> outputPadding, Layers.FusableActivation fusedActivation);

        /// <summary>
        /// Calculates an output tensor by resampling the input tensor along the spatial dimensions with given scales.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="scale">The factor to scale each dimension by.</param>
        /// <param name="interpolationMode">The `InterpolationMode` to use for the operation.</param>
        /// <param name="nearestMode">The `NearestMode` to use for the operation when using `InterpolationMode.NearestMode`.</param>
        /// <param name="coordTransformMode">The `CoordTransformMode` to use for the operation.</param>
        void Resize(Tensor<float> X, Tensor<float> O, ReadOnlySpan<float> scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode, Layers.CoordTransformMode coordTransformMode);

        /// <summary>
        /// Calculates an output tensor by sampling the input tensor by coordinates given by the grid tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="grid">The grid tensor containing the spatial coordinates per output pixel.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="mode">The `InterpolationMode` to use for the operation.</param>
        /// <param name="paddingMode">The `PaddingMode` to use for the operation.</param>
        /// <param name="alignCorners">Whether to map the extreme values in the coordinates 0 and 1 to the centre of the corner pixels rather than the outer corners.</param>
        void GridSample(Tensor<float> X, Tensor<float> grid, Tensor<float> O, Layers.InterpolationMode mode, Layers.PaddingMode paddingMode, bool alignCorners);

        /// <summary>
        /// Computes the output tensor by permuting data from depth into blocks of spatial data.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="blocksize">The size of the blocks to move the depth data into.</param>
        /// <param name="mode">The ordering of the data in the output tensor as a `DepthToSpaceMode`.</param>
        void DepthToSpace(Tensor<float> X, Tensor<float> O, int blocksize, Layers.DepthToSpaceMode mode);

        /// <summary>
        /// Computes the output tensor by permuting data from blocks of spatial data into depth.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="blocksize">The size of the blocks to move the depth data into.</param>
        void SpaceToDepth(Tensor<float> X, Tensor<float> O, int blocksize);

        /// <summary>
        /// Calculates an output tensor by pooling the maximum values of the input tensor across its spatial dimensions according to the given pool and stride values.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="kernelShape">The size of the kernel along each spatial axis.</param>
        /// <param name="strides">The stride along each spatial axis.</param>
        /// <param name="pads">The lower and upper padding values for each spatial dimension. For example, [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.</param>
        void MaxPool(Tensor<float> X, Tensor<float> O, int[] kernelShape, int[] strides, int[] pads);

        /// <summary>
        /// Calculates an output tensor by pooling the mean values of the input tensor across its spatial dimensions according to the given pool and stride values.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="kernelShape">The size of the kernel along each spatial axis.</param>
        /// <param name="strides">The stride along each spatial axis.</param>
        /// <param name="pads">The lower and upper padding values for each spatial dimension. For example, [pad_left, pad_right] for 1D, or [pad_top, pad_bottom, pad_left, pad_right] for 2D.</param>
        void AveragePool(Tensor<float> X, Tensor<float> O, int[] kernelShape, int[] strides, int[] pads);

        /// <summary>
        /// Calculates an output tensor by pooling the maximum values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void GlobalMaxPool(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Calculates an output tensor by pooling the mean values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void GlobalAveragePool(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Calculates the output tensor by adding padding to the input tensor according to the given padding values, mode and constant value.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="pad">The lower and upper padding values for each dimension.</param>
        /// <param name="padMode">The `PadMode` to use when padding.</param>
        /// <param name="constant">The constant value to fill with.</param>
        void Pad(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> pad, Layers.PadMode padMode, float constant);

        /// <summary>
        /// Calculates the output tensor by adding padding to the input tensor according to the given padding values, mode and constant value.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="pad">The lower and upper padding values for each dimension.</param>
        /// <param name="padMode">The `PadMode` to use when padding.</param>
        /// <param name="constant">The constant value to fill with.</param>
        void Pad(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> pad, Layers.PadMode padMode, int constant);

        /// <summary>
        /// Computes the output tensor with an element-wise `ScaleBias` function: f(x, s, b) = x * s + b.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="S">The scale tensor.</param>
        /// <param name="B">The bias tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void ScaleBias(Tensor<float> X, Tensor<float> S, Tensor<float> B, Tensor<float> O);

        /// <summary>
        /// Computes the mean variance on the spatial dimensions of the input tensor and normalizes them according to `scale` and `bias` tensors.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="S">The scale tensor.</param>
        /// <param name="B">The bias tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero.</param>
        void InstanceNormalization(Tensor<float> X, Tensor<float> S, Tensor<float> B, Tensor<float> O, float epsilon);

        /// <summary>
        /// Computes the mean variance on the last dimension of the input tensor and normalizes it according to `scale` and `bias` tensors.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="S">The scale tensor.</param>
        /// <param name="B">The bias tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero.</param>
        void LayerNormalization(Tensor<float> X, Tensor<float> S, Tensor<float> B, Tensor<float> O, float epsilon);

        /// <summary>
        /// Reduces an input tensor along the final axis using the `ReduceMeanSquare` operation: f(x1, x2 ... xn) = (x1^2 + x2^2 + ... + xn^2) / n.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="S">The scale tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero.</param>
        void RMSNormalization(Tensor<float> X, Tensor<float> S, Tensor<float> O, float epsilon);

        /// <summary>
        /// Computes the mean variance on the last dimension of the input tensor and normalizes it according to `scale` and `bias` tensors.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="S">The scale tensor.</param>
        /// <param name="B">The bias tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="mean">The mean tensor.</param>
        /// <param name="variance">The variance tensor.</param>
        /// <param name="epsilon">The epsilon value the layer uses to avoid division by zero.</param>
        void BatchNormalization(Tensor<float> X, Tensor<float> S, Tensor<float> B, Tensor<float> mean, Tensor<float> variance, Tensor<float> O, float epsilon);

        /// <summary>
        /// Computes the index of the element which cumulative sum until said element is >= than a random value
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="random">The probability values used for the exit criteria.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void TopP(Tensor<float> X, Tensor<float> random, Tensor<int> O);

        /// <summary>
        /// Generates an output tensor of a given shape with random values in a normal distribution with given `mean` and `scale`, and an optional `seed` value.
        /// </summary>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="mean">The mean of the normal distribution to use to generate the output.</param>
        /// <param name="scale">The standard deviation of the normal distribution to use to generate the output.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the operation generates a seed using `System.Random()`.</param>
        void RandomNormal(Tensor<float> O, float mean, float scale, int? seed);

        /// <summary>
        /// Generates an output tensor of a given shape with random values in a uniform distribution between a given `low` and `high`, and an optional `seed` value.
        /// </summary>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="low">The lower end of the interval of the uniform distribution to use to generate the output.</param>
        /// <param name="high">The upper end of the interval of the uniform distribution to use to generate the output.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the operation generates a seed using `System.Random()`.</param>
        void RandomUniform(Tensor<float> O, float low, float high, int? seed);

        /// <summary>
        /// Generates a one-hot tensor with a given `depth`, `indices` and on and off values.
        /// </summary>
        /// <param name="indices">The indices input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis along which the operation adds the one-hot representation.</param>
        /// <param name="depth">The depth of the one-hot tensor.</param>
        /// <param name="offValue">The value to use for an off element.</param>
        /// <param name="onValue">The value to use for an on element.</param>
        void OneHot(Tensor<int> indices, Tensor<int> O, int axis, int depth, int offValue, int onValue);

        /// <summary>
        /// Generates a one-hot tensor with a given `depth`, `indices` and on and off values.
        /// </summary>
        /// <param name="indices">The indices input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis along which the operation adds the one-hot representation.</param>
        /// <param name="depth">The depth of the one-hot tensor.</param>
        /// <param name="offValue">The value to use for an off element.</param>
        /// <param name="onValue">The value to use for an on element.</param>
        void OneHot(Tensor<int> indices, Tensor<float> O, int axis, int depth, float offValue, float onValue);

        /// <summary>
        /// Calculates an output tensor by pooling the input tensor across each region of interest given by the `rois` tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="rois">The region of interest input tensor.</param>
        /// <param name="indices">The indices input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="mode">The pooling mode of the operation as an `RoiPoolingMode`.</param>
        /// <param name="outputHeight">The height of the output tensor.</param>
        /// <param name="outputWidth">The width of the output tensor.</param>
        /// <param name="samplingRatio">The number of sampling points in the interpolation grid used to compute the output value of each pooled output bin.</param>
        /// <param name="spatialScale">The multiplicative spatial scale factor used to translate coordinates from their input spatial scale to the scale used when pooling.</param>
        void RoiAlign(Tensor<float> X, Tensor<float> rois, Tensor<int> indices, Tensor<float> O, Layers.RoiPoolingMode mode, int outputHeight, int outputWidth, int samplingRatio, float spatialScale);

        /// <summary>
        /// Generates a 1D output tensor where the values form an arithmetic progression defined by the `start` and `delta` values.
        /// </summary>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="start">The first value in the range.</param>
        /// <param name="delta">The delta between subsequent values in the range.</param>
        void Range(Tensor<float> O, float start, float delta);

        /// <summary>
        /// Generates a 1D output tensor where the values form an arithmetic progression defined by the `start` and `delta` values.
        /// </summary>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="start">The first value in the range.</param>
        /// <param name="delta">The delta between subsequent values in the range.</param>
        void Range(Tensor<int> O, int start, int delta);

        /// <summary>
        /// Generates an output tensor with values 0 or 1 from a Bernoulli distribution. The input tensor contains the probabilities to use for generating the output values.
        /// </summary>
        /// <param name="X">The probabilities input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the operation generates a seed using `System.Random()`.</param>
        void Bernoulli(Tensor<float> X, Tensor O, int? seed);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Relu` activation function: f(x) = max(0, x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Relu(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the `Softmax` activation function along an axis: f(x, axis) = exp(X) / ReduceSum(exp(X), axis).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis along which to apply the `Softmax` activation function.</param>
        void Softmax(Tensor<float> X, Tensor<float> O, int axis);

        /// <summary>
        /// Computes an output tensor by applying the `LogSoftmax` activation function along an axis: f(x, axis) = log(Softmax(x, axis)).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis along which to apply the `LogSoftmax` activation function.</param>
        void LogSoftmax(Tensor<float> X, Tensor<float> O, int axis);

        /// <summary>
        /// Computes an output tensor by applying the `Hardmax` activation function along an axis: f(x, axis) = 1 if x is the first maximum value along the specified axis, otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis along which to apply the `Hardmax` activation function.</param>
        void Hardmax(Tensor<float> X, Tensor<float> O, int axis);

        /// <summary>
        /// Performs the cumulative sum along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis along which to apply the cumulative sum.</param>
        /// <param name="reverse">Whether to perform the cumulative sum from the end of the axis.</param>
        /// <param name="exclusive">Whether to include the respective input element in the cumulative sum.</param>
        void CumSum(Tensor<float> X, Tensor<float> O, int axis, bool reverse, bool exclusive);

        /// <summary>
        /// Performs the cumulative sum along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis along which to apply the cumulative sum.</param>
        /// <param name="reverse">Whether to perform the cumulative sum from the end of the axis.</param>
        /// <param name="exclusive">Whether to include the respective input element in the cumulative sum.</param>
        void CumSum(Tensor<int> X, Tensor<int> O, int axis, bool reverse, bool exclusive);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Tanh` activation function: f(x) = tanh(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Tanh(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Softplus` activation function: f(x) = ln(e^x + 1).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Softplus(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Sigmoid` activation function: f(x) = 1/(1 + e^(-x)).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Sigmoid(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `HardSigmoid` activation function: f(x) = clamp(alpha * x + beta, 0, 1).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="alpha">The alpha value to use for the `HardSigmoid` activation function.</param>
        /// <param name="beta">The beta value to use for the `HardSigmoid` activation function.</param>
        void HardSigmoid(Tensor<float> X, Tensor<float> O, float alpha, float beta);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Elu` activation function: f(x) = x if x >= 0, otherwise f(x) = alpha * (e^x - 1).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="alpha">The alpha value to use for the `Elu` activation function.</param>
        void Elu(Tensor<float> X, Tensor<float> O, float alpha);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Gelu` activation function: f(x) = x / 2 * (1 + erf(x / sqrt(2))).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Gelu(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Gelu` aproximate but fast gelu function: f(x) = (x / 2) * (tanh(x + x^3 * 0.04472) * 0.7978) + 1.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void GeluFast(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Relu6` activation function: f(x) = clamp(x, 0, 6).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Relu6(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `LeakyRelu` activation function: f(x) = x if x >= 0, otherwise f(x) = alpha * x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="alpha">The alpha value to use for the `LeakyRelu` activation function.</param>
        void LeakyRelu(Tensor<float> X, Tensor<float> O, float alpha);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Selu` activation function: f(x) = gamma * x if x >= 0, otherwise f(x) = (alpha * e^x - alpha).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="alpha">The alpha value to use for the `Selu` activation function.</param>
        /// <param name="gamma">The alpha value to use for the `Selu` activation function.</param>
        void Selu(Tensor<float> X, Tensor<float> O, float alpha, float gamma);

        /// <summary>
        /// Performs an element-wise `Mad` math operation: multiplies and adds bias to a tensor: f(T, s, b) = s * T + b.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="s">Input scalar for multiplication.</param>
        /// <param name="b">Input bias for addition.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void ScalarMad(Tensor<float> X, Tensor<float> O, float s, float b);

        /// <summary>
        /// Performs an element-wise `Mad` math operation: multiplies and adds bias to a tensor: f(T, s, b) = s * T + b.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="s">Input scalar for multiplication.</param>
        /// <param name="b">Input bias for addition.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void ScalarMad(Tensor<int> X, Tensor<int> O, int s, int b);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `PRelu` activation function: f(x) = x if x >= 0, otherwise f(x) = slope * x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="slope">The slope tensor, must be unidirectional broadcastable to x.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void PRelu(Tensor<float> X, Tensor<float> slope, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Swish` activation function: f(x) = sigmoid(x) * x = x / (1 + e^{-x}).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Swish(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Abs` math function: f(x) = f(x) = |x|.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Abs(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Abs` math function: f(x) = f(x) = |x|.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Abs(Tensor<int> X, Tensor<int> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Neg` math function: f(x) = -x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Neg(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Neg` math function: f(x) = -x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Neg(Tensor<int> X, Tensor<int> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Ceil` math function: f(x) = ceil(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Ceil(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Clip` math function: f(x) = clamp(x, min, max).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="min">The lower clip value.</param>
        /// <param name="max">The upper clip value.</param>
        void Clip(Tensor<float> X, Tensor<float> O, float min, float max);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Clip` math function: f(x) = clamp(x, min, max).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="min">The lower clip value.</param>
        /// <param name="max">The upper clip value.</param>
        void Clip(Tensor<int> X, Tensor<int> O, int min, int max);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Floor` math function: f(x) = floor(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Floor(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Round` math function: f(x) = round(x).
        ///
        /// If the fractional part is equal to 0.5, rounds to the nearest even integer.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Round(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Reciprocal` math function: f(x) = 1 / x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Reciprocal(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Square` math function: f(x) = x * x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Square(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Square` math function: f(x) = x * x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Square(Tensor<int> X, Tensor<int> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Exp` math function: f(x) = exp(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Exp(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Log` math function: f(x) = log(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Log(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Sqrt` math function: f(x) = sqrt(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Sqrt(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Acos` trigonometric function: f(x) = acos(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Acos(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Acosh` trigonometric function: f(x) = acosh(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Acosh(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Asin` trigonometric function: f(x) = asin(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Asin(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Asinh` trigonometric function: f(x) = asinh(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Asinh(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Atan` trigonometric function: f(x) = atan(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Atan(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Atanh` trigonometric function: f(x) = atanh(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Atanh(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Cos` trigonometric function: f(x) = cos(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Cos(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Cosh` trigonometric function: f(x) = cosh(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Cosh(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Sin` trigonometric function: f(x) = sin(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Sin(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Sinh` trigonometric function: f(x) = sinh(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Sinh(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Tan` trigonometric function: f(x) = tan(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Tan(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Erf` activation function: f(x) = erf(x).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Erf(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Celu` activation function: f(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1)).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="alpha">The alpha value to use for the `Celu` activation function.</param>
        void Celu(Tensor<float> X, Tensor<float> O, float alpha);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `HardSwish` activation function: f(x) = x * max(0, min(1, 1/6 * x + 0.5)).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void HardSwish(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Shrink` activation function: f(x) = x + bias if x &lt; lambd. f(x) = x - bias if x &gt; lambd. Otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="bias">The bias value to use for the `Shrink` activation function.</param>
        /// <param name="lambd">The lambda value to use for the `Shrink` activation function.</param>
        void Shrink(Tensor<float> X, Tensor<float> O, float bias, float lambd);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `Softsign` activation function: f(x) = x/(|x| + 1).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Softsign(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Computes an output tensor by applying the element-wise `ThresholdedRelu` activation function: f(x) = x if x > alpha, otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="alpha">The alpha value to use for the `ThresholdedRelu` activation function.</param>
        void ThresholdedRelu(Tensor<float> X, Tensor<float> O, float alpha);

        /// <summary>
        /// Performs an element-wise `Add` math operation: f(a, b) = a + b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Add(Tensor<float> A, Tensor<float> B, Tensor<float> O);

        /// <summary>
        /// Performs an element-wise `Add` math operation: f(a, b) = a + b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Add(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Sub` math operation: f(a, b) = a - b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Sub(Tensor<float> A, Tensor<float> B, Tensor<float> O);

        /// <summary>
        /// Performs an element-wise `Sub` math operation: f(a, b) = a - b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Sub(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Mul` math operation: f(a, b) = a * b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Mul(Tensor<float> A, Tensor<float> B, Tensor<float> O);

        /// <summary>
        /// Performs an element-wise `Mul` math operation: f(a, b) = a * b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Mul(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Div` math operation: f(a, b) = a / b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Div(Tensor<float> A, Tensor<float> B, Tensor<float> O);

        /// <summary>
        /// Performs an element-wise `Div` math operation: f(a, b) = a / b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Div(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Mod` math operation: f(a, b) = a % b.
        ///
        /// The sign of the remainder is the same as the sign of the divisor, as in Python.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Mod(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Mod` math operation: f(a, b) = a % b.
        ///
        /// The sign of the remainder is the same as the sign of the divisor, as in Python.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Mod(Tensor<float> A, Tensor<float> B, Tensor<float> O);

        /// <summary>
        /// Performs an element-wise `Mod` math operation: f(a, b) = a % b.
        ///
        /// The sign of the remainder is the same as the sign of the dividend, as in C#.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void FMod(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Mod` math operation: f(a, b) = a % b.
        ///
        /// The sign of the remainder is the same as the sign of the dividend, as in C#.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void FMod(Tensor<float> A, Tensor<float> B, Tensor<float> O);

        /// <summary>
        /// Performs an element-wise `Pow` math operation: f(a, b) = pow(a, b).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Pow(Tensor<float> A, Tensor<float> B, Tensor<float> O);

        /// <summary>
        /// Performs an element-wise `Pow` math operation: f(a, b) = pow(a, b).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Pow(Tensor<float> A, Tensor<int> B, Tensor<float> O);

        /// <summary>
        /// Performs an element-wise `Min` math operation: f(a, b) = min(a, b).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Min(Tensor<float> A, Tensor<float> B, Tensor<float> O);

        /// <summary>
        /// Performs an element-wise `Min` math operation: f(a, b) = min(a, b).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Min(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Max` math operation: f(a, b) = max(a, b).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Max(Tensor<float> A, Tensor<float> B, Tensor<float> O);

        /// <summary>
        /// Performs an element-wise `Max` math operation: f(a, b) = max(a, b).
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Max(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMax` operation: f(x1, x2 ... xn) = max(x1, x2, ... , xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceMax(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMax` operation: f(x1, x2 ... xn) = max(x1, x2, ... , xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceMax(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMean` operation: f(x1, x2 ... xn) = (x1 + x2 + ... + xn) / n.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceMean(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMin` operation: f(x1, x2 ... xn) = min(x1, x2, ... , xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceMin(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceMin` operation: f(x1, x2 ... xn) = min(x1, x2, ... , xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceMin(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceProd` operation: f(x1, x2 ... xn) = x1 * x2 * ... * xn.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceProd(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceProd` operation: f(x1, x2 ... xn) = x1 * x2 * ... * xn.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceProd(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceSum` operation: f(x1, x2 ... xn) = x1 + x2 + ... + xn.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceSum(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceSum` operation: f(x1, x2 ... xn) = x1 + x2 + ... + xn.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceSum(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceSumSquare` operation: f(x1, x2 ... xn) = x1² + x2² + ... + xn².
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceSumSquare(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceSumSquare` operation: f(x1, x2 ... xn) = x1² + x2² + ... + xn².
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceSumSquare(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceL1` operation: f(x1, x2 ... xn) = |x1| + |x2| + ... + |xn|.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceL1(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceL1` operation: f(x1, x2 ... xn) = |x1| + |x2| + ... + |xn|.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceL1(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceL2` operation: f(x1, x2 ... xn) = sqrt(x1² + x2² + ... + xn²).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceL2(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceLogSum` operation: f(x1, x2 ... xn) = log(x1 + x2 + ... + xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceLogSum(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Reduces an input tensor along the given axes using the `ReduceLogSumExp` operation: f(x1, x2 ... xn) = log(e^x1 + e^x2 + ... + e^xn).
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axes">The axes along which to reduce.</param>
        void ReduceLogSumExp(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes);

        /// <summary>
        /// Computes the indices of the maximum elements of the input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis along which to reduce.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
        void ArgMax(Tensor<float> X, Tensor<int> O, int axis, bool selectLastIndex);

        /// <summary>
        /// Computes the indices of the maximum elements of the input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis along which to reduce.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
        void ArgMax(Tensor<int> X, Tensor<int> O, int axis, bool selectLastIndex);

        /// <summary>
        /// Computes the indices of the minimum elements of the input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis along which to reduce.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
        void ArgMin(Tensor<float> X, Tensor<int> O, int axis, bool selectLastIndex);

        /// <summary>
        /// Computes the indices of the minimum elements of the input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis along which to reduce.</param>
        /// <param name="selectLastIndex">Whether to perform the operation from the back of the axis.</param>
        void ArgMin(Tensor<int> X, Tensor<int> O, int axis, bool selectLastIndex);

        /// <summary>
        /// Performs an element-wise `Greater` logical comparison operation: f(a, b) = 1 if a > b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Greater(Tensor<float> A, Tensor<float> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Greater` logical comparison operation: f(a, b) = 1 if a > b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Greater(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `GreaterOrEqual` logical comparison operation: f(a, b) = 1 if a >= b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void GreaterOrEqual(Tensor<float> A, Tensor<float> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `GreaterOrEqual` logical comparison operation: f(a, b) = 1 if a >= b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void GreaterOrEqual(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Less` logical comparison operation: f(a, b) = 1 if a &lt; b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Less(Tensor<float> A, Tensor<float> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Less` logical comparison operation: f(a, b) = 1 if a &lt; b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Less(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `LessOrEqual` logical comparison operation: f(a, b) = 1 if a &lt;= b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void LessOrEqual(Tensor<float> A, Tensor<float> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `LessOrEqual` logical comparison operation: f(a, b) = 1 if a &lt;= b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void LessOrEqual(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Equal` logical comparison operation: f(a, b) = 1 if a == b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Equal(Tensor<float> A, Tensor<float> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Equal` logical comparison operation: f(a, b) = 1 if a == b, otherwise f(x) = 0.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Equal(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Or` logical operation: f(a, b) = a | b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Or(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `And` logical operation: f(a, b) = a &amp; b.
        ///
        /// This supports numpy-style broadcasting of input tensors.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void And(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Xor` logical operation: f(a) = a ^ b.
        /// </summary>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Xor(Tensor<int> A, Tensor<int> B, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Not` logical operation: f(x) = ~x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Not(Tensor<int> X, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `Sign` math operation: f(x) = 1 if x > 0. f(x) = -1 if x &lt; 0. Otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Sign(Tensor<float> X, Tensor<float> O);

        /// <summary>
        /// Performs an element-wise `Sign` math operation: f(x) = 1 if x > 0. f(x) = -1 if x &lt; 0. Otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Sign(Tensor<int> X, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `IsNaN` logical operation: f(x) = 1 if x is NaN, otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void IsNaN(Tensor<float> X, Tensor<int> O);

        /// <summary>
        /// Performs an element-wise `IsInf` logical operation: f(x) = 1 elementwise if x is +Inf and `detectPositive` is `true`, or x is -Inf and `detectNegative` is `true`. Otherwise f(x) = 0.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="detectNegative">Whether to detect negative infinities in the `IsInf` function.</param>
        /// <param name="detectPositive">Whether to detect positive infinities in the `IsInf` function.</param>
        void IsInf(Tensor<float> X, Tensor<int> O, bool detectNegative, bool detectPositive);

        /// <summary>
        /// Performs an element-wise `Where` logical operation: f(condition, a, b) = a if `condition` is `true`, otherwise f(condition, a, b) = b.
        /// </summary>
        /// <param name="C">The condition tensor.</param>
        /// <param name="A">The first input tensor.</param>
        /// <param name="B">The second input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Where(Tensor<int> C, Tensor A, Tensor B, Tensor O);

        /// <summary>
        /// Calculates an output tensor by copying the data from the input tensor and using a given shape. The data from the input tensor is unchanged.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Reshape(Tensor X, Tensor O);

        /// <summary>
        /// Calculates an output tensor by broadcasting the input tensor into a given shape.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Expand(Tensor X, Tensor O);

        /// <summary>
        /// Calculates an output tensor by reversing the dimensions of the input tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Transpose(Tensor X, Tensor O);

        /// <summary>
        /// Calculates an output tensor by permuting the axes and data of the input tensor according to the given permutations.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="permutations">The axes to sample the output tensor from in the input tensor.</param>
        void Transpose(Tensor X, Tensor O, ReadOnlySpan<int> permutations);

        /// <summary>
        /// Calculates an output tensor by splitting the input tensor along a given axis between start and end.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis along which to split the input tensor.</param>
        /// <param name="start">The inclusive start value for the split.</param>
        void Split(Tensor X, Tensor O, int axis, int start);

        /// <summary>
        /// Calculates an output tensor by slicing the input tensor along given axes with given starts, ends, and steps.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="starts">The start index along each axis.</param>
        /// <param name="axes">The axes along which to slice.</param>
        /// <param name="steps">The step values for slicing.</param>
        void Slice(Tensor X, Tensor O, ReadOnlySpan<int> starts, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps);

        /// <summary>
        /// Copies the input tensor and updates values at indexes specified by the slices defined by axes, starts, ends, and steps.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="values">The values tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="starts">The start index along each axis.</param>
        /// <param name="axes">The axes along which to slice.</param>
        /// <param name="steps">The step values for slicing.</param>
        void SliceSet(Tensor X, Tensor values, Tensor O, ReadOnlySpan<int> starts, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps);

        /// <summary>
        /// Updates values at indexes specified by the slices defined by axis, start and step.
        /// </summary>
        /// <param name="values">The values tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axes along which to slice.</param>
        /// <param name="start">The start index along the axis.</param>
        /// <param name="step">The step value for slicing.</param>
        void SliceSet(Tensor values, Tensor O, int axis, int start, int step);

        /// <summary>
        /// Calculates an output tensor by repeating the input layer a given number of times along each axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="repeats">The number of times to tile the input tensor along each axis.</param>
        void Tile(Tensor X, Tensor O, ReadOnlySpan<int> repeats);

        /// <summary>
        /// Takes values from the input tensor indexed by the indices tensor along a given axis and concatenates them.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis along which to gather.</param>
        void Gather(Tensor X, Tensor<int> indices, Tensor O, int axis);

        /// <summary>
        /// Takes values from the input tensor indexed by the indices tensor along a given axis and concatenates them.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis along which to gather.</param>
        void GatherElements(Tensor X, Tensor<int> indices, Tensor O, int axis);

        /// <summary>
        /// Takes slices of values from the batched input tensor indexed by the `indices` tensor.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="batchDims">The number of batch dimensions of the input tensor, the gather begins at the next dimension.</param>
        void GatherND(Tensor X, Tensor<int> indices, Tensor O, int batchDims);

        /// <summary>
        /// Copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor along a given axis.
        ///
        /// `ScatterElements` updates the values depending on the reduction mode used.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="updates">The updates tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="axis">The axis on which to perform the scatter.</param>
        /// <param name="reduction">The reduction mode used to update the values as a `ScatterReductionMode`.</param>
        void ScatterElements(Tensor X, Tensor<int> indices, Tensor updates, Tensor O, int axis, Layers.ScatterReductionMode reduction);

        /// <summary>
        /// Copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor.
        ///
        /// `ScatterND` updates the values depending on the reduction mode used.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="updates">The updates tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="reduction">The reduction mode used to update the values as a `ScatterReductionMode`.</param>
        void ScatterND(Tensor<float> X, Tensor<int> indices, Tensor<float> updates, Tensor<float> O, Layers.ScatterReductionMode reduction);

        /// <summary>
        /// Copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor.
        ///
        /// `ScatterND` updates the values depending on the reduction mode used.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="updates">The updates tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="reduction">The reduction mode used to update the values as a `ScatterReductionMode`.</param>
        void ScatterND(Tensor<int> X, Tensor<int> indices, Tensor<int> updates, Tensor<int> O, Layers.ScatterReductionMode reduction);

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
        /// <param name="Y">The output tensor to be computed and filled with the concatenated intermediate output values of the hidden.</param>
        /// <param name="Yh">The output tensor to be computed and filled with the last output value of the hidden.</param>
        /// <param name="Yc">The output tensor to be computed and filled with the last output value of the cell.</param>
        /// <param name="direction">The direction of the LSTM as an `RnnDirection`.</param>
        /// <param name="activations">The activation functions of the LSTM as an array of `RnnActivation`.</param>
        /// <param name="activationAlpha">The alpha values of the activation functions of the LSTM.</param>
        /// <param name="activationBeta">The beta values of the activation functions of the LSTM.</param>
        /// <param name="inputForget">Whether to forget the input values in the LSTM. If this is `false`, the layer couples the input and forget gates.</param>
        /// <param name="clip">The cell clip threshold of the LSTM.</param>
        /// <param name="layout">The layout of the tensors as an `RnnLayout`.</param>
        void LSTM(Tensor<float> X, Tensor<float> W, Tensor<float> R, Tensor<float> B, Tensor<int> sequenceLens, Tensor<float> initialH, Tensor<float> initialC, Tensor<float> P, Tensor<float> Y, Tensor<float> Yh, Tensor<float> Yc, Layers.RnnDirection direction, Layers.RnnActivation[] activations, float[] activationAlpha, float[] activationBeta, bool inputForget, float clip, Layers.RnnLayout layout);

        /// <summary>
        /// Calculates the top-K largest or smallest elements of an input tensor along a given axis.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="values">The output tensor to be computed and filled with the top K values from the input tensor.</param>
        /// <param name="indices">The output tensor to be computed and filled with the corresponding input tensor indices for the top K values from the input tensor.</param>
        /// <param name="k">The number of elements to calculate.</param>
        /// <param name="axis">The axis along which to perform the top-K operation.</param>
        /// <param name="largest">Whether to calculate the top-K largest elements. If this is `false`, the layer calculates the top-K smallest elements.</param>
        void TopK(Tensor<float> X, Tensor<float> values, Tensor<int> indices, int k, int axis, bool largest);

        /// <summary>
        /// Performs an `Einsum` math operation.
        /// The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to an operand tensor, and the characters within the terms correspond to operands dimensions.
        /// This sequence may be followed by "->" to separate the left and right hand side of the equation. If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases, output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the equation.
        /// When a dimension character is repeated in the left-hand side, it represents summation along the dimension.
        /// The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions. Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions. The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the beginning of the output. The equation string may contain space (U+0020) character.
        /// </summary>
        /// <param name="inputTensors">The input tensors.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="operandIndices">The operand indices for each input tensor.</param>
        /// <param name="outputIndices">The output indices for each input tensor.</param>
        /// <param name="sumIndices">The indices along which to sum.</param>
        /// <param name="sumShape">The shape along which to sum.</param>
        void Einsum(Tensor<float>[] inputTensors, Tensor<float> O, TensorIndex[] operandIndices, TensorIndex outputIndices, TensorIndex sumIndices, TensorShape sumShape);

        /// <summary>
        /// Sets the entries of a tensor to 0.
        /// </summary>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void MemClear(Tensor O);

        /// <summary>
        /// Sets the entries of a tensor to a given fill value.
        /// </summary>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="value">The fill value.</param>
        void MemSet(Tensor<float> O, float value);

        /// <summary>
        /// Sets the entries of a tensor to a given fill value.
        /// </summary>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="value">The fill value.</param>
        void MemSet(Tensor<int> O, int value);

        /// <summary>
        /// Creates a copy of a given input tensor with the same shape and values.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void MemCopy(Tensor X, Tensor O);

        /// <summary>
        /// Copy blocks of values from X to O, we copy 'count' blocks each of length 'length' values with initial offsets
        /// given by 'offsetX', 'offsetO' and with strides given by 'strideX', 'strideO'
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="strideX">The stride of the blocks in the input tensor.</param>
        /// <param name="strideO">The stride of the blocks in the output tensor.</param>
        /// <param name="length">The number of elements in each block.</param>
        /// <param name="count">The number of blocks to copy.</param>
        /// <param name="offsetX">The first index to copy from in the input tensor.</param>
        /// <param name="offsetO">The first index to copy to in the output tensor.</param>
        void MemCopyStride(Tensor X, Tensor O, int strideX, int strideO, int length, int count, int offsetX, int offsetO);

        /// <summary>
        /// Computes the output tensor using an element-wise `Cast` function: f(x) = (float)x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Cast(Tensor<int> X, Tensor<float> O);

        /// <summary>
        /// Computes the output tensor using an element-wise `Cast` function: f(x) = (int)x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Cast(Tensor<float> X, Tensor<int> O);

        /// <summary>
        /// Computes the output tensor using an element-wise `Cast` function: f(x) = (float)x.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        void Cast(Tensor<short> X, Tensor<float> O);

        /// <summary>
        /// Computes the output tensor by selecting slices from an input tensor according to the 'indices' tensor along an 'axis'.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="numIndices">The number of indices.</param>
        /// <param name="axis">The axis along which to compress.</param>
        void CompressWithIndices(Tensor X, Tensor<int> indices, Tensor O, int numIndices, int axis);

        /// <summary>
        /// Computes the output tensor by unpacking four uint8 values from each int value and scaling to floats.
        /// </summary>
        /// <param name="X">The input tensor.</param>
        /// <param name="O">The output tensor to be computed and filled.</param>
        /// <param name="scale">The scale value to use for dequantization.</param>
        /// <param name="zeroPoint">The zero point value to use for dequantization.</param>
        void DequantizeLinear(Tensor<byte> X, Tensor<float> O, float scale, byte zeroPoint);

        /// <summary>
        /// Returns the `BackendType` for the ops.
        /// </summary>
        BackendType backendType { get; }
    }

    /// <summary>
    /// An interface that provides methods for storing variables.
    /// </summary>
    interface IModelStorage : IDisposable
    {
        /// <summary>
        /// Sets a given input with a tensor.
        /// </summary>
        /// <param name="index">The name of the input.</param>
        /// <param name="X">The tensor for the input.</param>
        void SetInput(int index, Tensor X);

        /// <summary>
        /// Prepares storage for a given model.
        /// </summary>
        /// <param name="model">The model to prepare the storage of.</param>
        /// <param name="takeoverWeights">Whether the execution can take ownership of the weights of the model.</param>
        void PrepareStorage(Model model, bool takeoverWeights = false);

        /// <summary>
        /// Retrieves Tensor for given index.
        /// </summary>
        /// <param name="index">The index for which to retrieve its underlying Tensor.</param>
        /// <returns>The retrieved tensor.</returns>
        Tensor GetTensor(int index);

        /// <summary>
        /// Retrieves shape of tensor for given index.
        /// </summary>
        /// <param name="tensorIndex">The tensor index.</param>
        /// <returns>The retrieved tensor shape.</returns>
        TensorShape GetTensorShape(int tensorIndex);

        /// <summary>
        /// Retrieves data type of tensor for given index.
        /// </summary>
        /// <param name="tensorIndex">The tensor index.</param>
        /// <returns>The retrieved data type.</returns>
        DataType GetDataType(int tensorIndex);

        /// <summary>
        /// Reads single value of integer tensor.
        /// </summary>
        /// <param name="tensorIndex">The tensor index.</param>
        /// <param name="defaultValue">The default value to return if the tensor is null.</param>
        /// <returns>The retrieved integer value.</returns>
        int GetInt(int tensorIndex, int defaultValue = 0);

        /// <summary>
        /// Reads single value of float tensor.
        /// </summary>
        /// <param name="tensorIndex">The tensor index.</param>
        /// <param name="defaultValue">The default value to return if the tensor is null.</param>
        /// <returns>The retrieved float value.</returns>
        float GetFloat(int tensorIndex, float defaultValue = 0f);

        /// <summary>
        /// Reads values of 1D integer tensor.
        /// </summary>
        /// <param name="tensorIndex">The tensor index.</param>
        /// <param name="defaultValue">The default value to return if the tensor is null.</param>
        /// <returns>The retrieved integer values as a read only span.</returns>
        ReadOnlySpan<int> GetInts(int tensorIndex, ReadOnlySpan<int> defaultValue = default);

        /// <summary>
        /// Reads values of 1D float tensor.
        /// </summary>
        /// <param name="tensorIndex">The tensor index.</param>
        /// <param name="defaultValue">The default value to return if the tensor is null.</param>
        /// <returns>The retrieved float values as a read only span.</returns>
        ReadOnlySpan<float> GetFloats(int tensorIndex, ReadOnlySpan<float> defaultValue = default);

        /// <summary>
        /// Allocates a new Tensor and stores the result of execution for a given tensor index.
        /// Reuses a tensor from a memory pool if possible or creates a new one.
        /// </summary>
        /// <param name="index">The index of the output.</param>
        /// <param name="shape">The desired TensorShape.</param>
        /// <param name="dataType">The desired DataType.</param>
        /// <param name="backendType">The desired BackendType.</param>
        /// <returns>The allocated tensor.</returns>
        Tensor AllocateTensorAndStore(int index, TensorShape shape, DataType dataType, BackendType backendType);

        /// <summary>
        /// Allocates a new Tensor.
        /// First tries to get a old one from the memory pool, if not create a new one
        /// </summary>
        /// <param name="shape">The desired TensorShape.</param>
        /// <param name="dataType">The desired DataType.</param>
        /// <param name="backendType">The desired BackendType.</param>
        /// <returns>The allocated tensor.</returns>
        Tensor AllocateTensor(TensorShape shape, DataType dataType, BackendType backendType);

        /// <summary>
        /// Disposes storage that can be deleted after executing a given layer.
        /// </summary>
        /// <param name="forLayer">The layer to dispose the temporary storage of.</param>
        void DisposeAfterLayer(Layer forLayer);

        /// <summary>
        /// Disposes storage that can be deleted before executing the whole model.
        /// </summary>
        void DisposeOnExecute();

        /// <summary>
        /// Release tensor to allocator to backend's internal pool.
        /// </summary>
        /// <param name="tensor">The tensor to dispose.</param>
        void Dispose(Tensor tensor);

        /// <summary>
        /// Stores the result of execution for a given tensor index.
        /// </summary>
        /// <param name="index">The index of the tensor to store.</param>
        /// <param name="result">The tensor result of execution.</param>
        void Store(int index, Tensor result);

        /// <summary>
        /// Returns a reference to default output tensor of a given index.
        /// This is non-blocking.
        ///
        /// The reference is valid only until you call `Execute()` or `Dispose()` on the worker.
        /// </summary>
        /// <param name="index">The index of the tensor to peek.</param>
        /// <returns>The output tensor.</returns>
        Tensor PeekTensor(int index);

        /// <summary>
        /// Take ownership the output tensor of a given index.
        /// This is non-blocking.
        ///
        /// Remove all reference to tensor in storage.
        /// </summary>
        /// <param name="index">The index of the tensor to gain ownership.</param>
        /// <returns>The tensor.</returns>
        Tensor TakeTensorOwnership(int index);
    }

    /// <summary>
    /// Represents a context object that holds the model operations and variables for layer execution.
    /// </summary>
    struct ExecutionContext
    {
        /// <summary>
        /// The `IBackend` used for execution.
        /// </summary>
        public IBackend backend;

        /// <summary>
        /// The CPU backend used for fallback cpu execution.
        /// </summary>
        public CPUBackend cpuBackend;

        /// <summary>
        /// The `IModelStorage` used for execution
        /// </summary>
        public IModelStorage storage;
    }
}

