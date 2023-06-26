using UnityEngine;
using System.Runtime.CompilerServices;
using Unity.Sentis;
using UnityEngine.Assertions;
using System;
using static Unity.Sentis.ShaderPropertyID;
using Unity.Mathematics;

[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis {

readonly struct PixelFunc
{
    readonly Material m_Material;

    public PixelFunc(string name)
    {
        m_Material = PixelShaderSingleton.Instance.FindMaterial(name);
    }

    public void SetInt(int nameID, int value)
    {
        m_Material.SetInt(nameID, value);
    }

    public void SetFloatArray(int nameID, float[] values)
    {
        m_Material.SetFloatArray(nameID, values);
    }

    public void SetFloat(int nameID, float value)
    {
        m_Material.SetFloat(nameID, value);
    }

    public void SetVector(int nameID, Vector4 value)
    {
        m_Material.SetVector(nameID, value);
    }

    public void SetTexture(int nameID, Texture value)
    {
        m_Material.SetTexture(nameID, value);
    }

    public void EnableKeyword(string keyword)
    {
        m_Material.EnableKeyword(keyword);
    }

    internal void SetTensor(TensorProperties tensorProperties, TextureTensorData pinX)
    {
        m_Material.SetTexture(tensorProperties.k_ID_Ptr, pinX.bufferAsTexture);
        m_Material.SetInt(tensorProperties.k_ID_WidthMask, pinX.widthMask);
        m_Material.SetInt(tensorProperties.k_ID_WidthShift, pinX.widthShift);
    }

    internal void SetTensorBlockStride(TensorProperties tensorProperties, TextureTensorData pinX)
    {
        m_Material.SetInt(tensorProperties.k_ID_StrideAxis, pinX.strideAxis);
        m_Material.SetInt(tensorProperties.k_ID_DimAxis, pinX.dimAxis);
        m_Material.SetInt(tensorProperties.k_ID_DimBlocked, pinX.dimAxisDiv4);
    }

    public void Dispatch(TextureTensorData pinO)
    {
        m_Material.SetInt(k_TensorPropertiesO.k_ID_WidthShift, pinO.widthShift);
        m_Material.SetInt(k_ID_LengthO, pinO.shape.length);
        Graphics.Blit(null, pinO.bufferAsTexture, m_Material);
    }

    public void Dispatch(RenderTexture renderTexture)
    {
        Graphics.Blit(null, renderTexture, m_Material);
    }
}

static class PixelShaderHelper
{
    static readonly float[] k_ScratchPadFloat8 = new float[8];

    public static unsafe void SetInt8(this PixelFunc func, int nameID, int* ptr)
    {
        for (var i = 0; i < 8; i++)
            k_ScratchPadFloat8[i] = ptr[i];

        func.SetFloatArray(nameID, k_ScratchPadFloat8);
    }

    public static void SetShape(this PixelFunc func, int nameID, TensorShape shape)
    {
        for (var i = 0; i < 8; i++)
        {
            k_ScratchPadFloat8[i] = i < shape.rank ? shape[-1 - i] : 1;
        }

        func.SetFloatArray(nameID, k_ScratchPadFloat8);
    }

    public static void SetStrides(this PixelFunc func, int nameID, TensorShape shape)
    {
        var stride = 1;
        var rank = shape.rank;
        for (var i = 0; i < rank; i++)
        {
            var dim = shape[rank - 1 - i];
            k_ScratchPadFloat8[i] = dim == 1 ? 0 : stride;
            stride *= dim;
        }

        Array.Clear(k_ScratchPadFloat8, rank, 8 - rank);

        func.SetFloatArray(nameID, k_ScratchPadFloat8);
    }
}

/// <summary>
/// Represents a GPUPixel backend ops.
/// </summary>
public class GPUPixelOps : CPUOps
{
    /// <summary>
    /// Initializes and returns an instance of `GPUPixelOps`.
    /// </summary>
    public GPUPixelOps(ITensorAllocator allocator = null)
        : base(allocator) { }

    /// <inheritdoc/>
    public override DeviceType deviceType => DeviceType.CPU;

    /// <summary>
    /// Pins the tensor as `TextureTensorData` on any axis (choose last).
    /// </summary>
    static TextureTensorData PinBlockAny(Tensor X, bool uploadCache = true)
    {
        if (X.tensorOnDevice is TextureTensorData textureTensorData)
            return textureTensorData;
        return TextureTensorData.Pin(X, X.shape.rank - 1, uploadCache);
    }

    /// <summary>
    /// Pins the tensor as TextureTensorData on any axis except `nonBlockAxis`. (Choose last unless avoid, else one before last.)
    /// </summary>
    static TextureTensorData PinBlockOther(Tensor X, int nonBlockAxis, bool uploadCache = true)
    {
        if (X.tensorOnDevice is TextureTensorData textureTensorData)
            if (textureTensorData.blockAxis != nonBlockAxis)
                return textureTensorData;
        var axis = nonBlockAxis == X.shape.rank - 1 ? X.shape.rank - 2 : X.shape.rank - 1;
        return TextureTensorData.Pin(X, axis, uploadCache);
    }

    /// <summary>
    /// Pins the tensor X blocking along the same axis as a given other TextureTensorData
    /// This can be used to block an output tensor along the same axis as an input tensor for an op
    /// </summary>
    static TextureTensorData PinAsSame(Tensor X, TextureTensorData other, bool uploadCache = true)
    {
        return TextureTensorData.Pin(X, X.shape.rank - other.shape.rank + other.blockAxis, uploadCache);
    }

    /// <summary>
    /// Pin tensors A and B along the same axis, the blocking for A takes priority in case neither tensor is pinned or
    /// both tensors are pinned
    /// </summary>
    static void PinBothSame(Tensor A, Tensor B)
    {
        var pinA = A.tensorOnDevice as TextureTensorData;
        var pinB = B.tensorOnDevice as TextureTensorData;
        if (pinA == null == pinB is null)
            pinA = PinBlockAny(A);
        else if (pinB != null)
            pinA = PinAsSame(A, pinB);
        PinAsSame(B, pinA);
    }

    /// <inheritdoc/>
    protected override void MemSet(Tensor O, int value, int length = -1, int offsetO = 0)
    {
        if (O.dataType != DataType.Float || (length > 0 && length != O.shape.length) || offsetO != 0)
        {
            base.MemSet(O, value, length, offsetO);
            return;
        }

        var func = new PixelFunc("Hidden/Sentis/MemSet");
        func.SetFloat(k_ID_memValue, math.asfloat(value));
        var pinO = PinBlockAny(O, false);
        func.Dispatch(pinO);
    }

    /// <inheritdoc/>
    public override TensorFloat ConstantOfShape(TensorShape X, float value)
    {
        var O = NewOutputTensorFloat(X);
        if (O.shape.HasZeroDims())
            return O;
        MemSet(O, math.asint(value));
        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat MatMul(TensorFloat X, TensorFloat Y)
    {
        if (X.shape.Length(0, -2) == 1 && Y.shape.Length(0, -2) == 1)
            return Gemm(X, Y, null, Layers.FusableActivation.None);
        if (X.shape.rank == 2 && Y.shape.rank == 2)
            return MatMul2D(X, false, Y, false);

        return base.MatMul(X, Y);
    }

    /// <inheritdoc/>
    public override TensorFloat MatMul2D(TensorFloat X, bool xTranspose, TensorFloat Y, bool yTranspose)
    {
        var Oshape = ShapeInference.MatMul2D(X.shape, false, Y.shape, false);
        if (X.shape.HasZeroDims() || Y.shape.HasZeroDims())
            return ConstantOfShape(Oshape, 0.0f);

        if (xTranspose || yTranspose)
            return base.MatMul2D(X, xTranspose, Y, yTranspose);

        return Gemm(X, Y, null, Layers.FusableActivation.None);
    }

    /// <inheritdoc/>
    public override TensorFloat Dense(TensorFloat X, TensorFloat W, TensorFloat B, Layers.FusableActivation fusedActivation)
    {
        return Gemm(X, W, B, fusedActivation);
    }

    TensorFloat Gemm(TensorFloat X, TensorFloat W, TensorFloat B, Layers.FusableActivation fusedActivation)
    {
        bool isDense = B == null ? false : true;
        // TODO: Support transpose X, W
        var Oshape = isDense ? ShapeInference.Dense(X.shape, W.shape, B.shape) : X.shape.MatMul(W.shape);
        if (Oshape.HasZeroDims())
            return NewOutputTensorFloat(Oshape);

        var O = (fusedActivation == Layers.FusableActivation.None || fusedActivation == Layers.FusableActivation.Relu) ? NewOutputTensorFloat(Oshape) : NewTempTensorFloat(Oshape);

        var func = new PixelFunc("Hidden/Sentis/Dense");

        var pinO = TextureTensorData.Pin(O, O.shape.rank - 1, uploadCache: false);
        var pinX = TextureTensorData.Pin(X, X.shape.rank - 1);
        var pinW = TextureTensorData.Pin(W, W.shape.rank - 1);
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensor(k_TensorPropertiesW, pinW);

        if (isDense)
        {
            var pinB = TextureTensorData.Pin(B, B.shape.rank - 1);
            func.SetTensor(k_TensorPropertiesB, pinB);
            func.EnableKeyword("Dense");
        }

        func.SetInt(k_TensorPropertiesO.k_ID_DimBlocked, pinO.dimAxisDiv4);
        func.SetInt(k_TensorPropertiesX.k_ID_DimBlocked, pinX.dimAxisDiv4);
        func.SetInt(k_TensorPropertiesW.k_ID_DimBlocked, pinW.dimAxisDiv4);

        if (fusedActivation == Layers.FusableActivation.Relu)
            func.EnableKeyword("Relu");

        func.Dispatch(pinO);

        if (!(fusedActivation == Layers.FusableActivation.None || fusedActivation == Layers.FusableActivation.Relu))
            O = ApplyFusedActivation(O, fusedActivation);

        return O;
    }

    TensorFloat ApplyFusedActivation(TensorFloat X, Layers.FusableActivation fusedActivation)
    {
        switch (fusedActivation)
        {
            case Layers.FusableActivation.None:
                return Copy(X) as TensorFloat;
            case Layers.FusableActivation.Relu:
                return Relu(X);
            default:
                throw new NotImplementedException();
        }
    }

    /// <inheritdoc/>
    public override TensorFloat Conv(TensorFloat X, TensorFloat K, TensorFloat B, int groups, int[] stride, int[] pad, int[] dilation, Layers.FusableActivation fusedActivation)
    {
        if (X.shape.rank != 4)
            return base.Conv(X, K, B, groups, stride, pad, dilation, fusedActivation);

        var Oshape = ShapeInference.Conv(X.shape, K.shape, B.shape, groups, stride, pad, dilation);

        if (Oshape.HasZeroDims())
            return NewOutputTensorFloat(Oshape);

        var O = (fusedActivation == Layers.FusableActivation.None || fusedActivation == Layers.FusableActivation.Relu) ? NewOutputTensorFloat(Oshape) : NewTempTensorFloat(Oshape);

        PixelFunc func;
        var isDepthwise = K.shape[0] == groups && K.shape[1] == 1;

        var pinX = TextureTensorData.Pin(X, 1);
        var pinK = TextureTensorData.Pin(K, isDepthwise ? 0 : 1);
        var pinB = TextureTensorData.Pin(B, 0);
        var pinO = TextureTensorData.Pin(O, 1, uploadCache: false);

        if (isDepthwise)
        {
            func = new PixelFunc("Hidden/Sentis/DepthwiseConv2D");
        }
        else if (groups > 1)
        {
            func = new PixelFunc("Hidden/Sentis/GroupedConv2D");
            func.SetInt(k_ID_O_channels, pinO.shape[1]);
            func.SetInt(k_ID_K_channelsDivGroupDiv4, pinK.dimAxisDiv4);
            func.SetInt(k_ID_X_channels, pinX.shape[1]);
        }
        else
        {
            func = new PixelFunc("Hidden/Sentis/Conv2D");
            func.SetInt(k_ID_X_channels, pinX.shape[1]);
        }

        func.SetInt(k_ID_O_width, pinO.shape[3]);
        func.SetInt(k_ID_O_height, pinO.shape[2]);
        func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);

        func.SetTensor(k_TensorPropertiesX, pinX);

        func.SetInt(k_ID_X_width, pinX.shape[3]);
        func.SetInt(k_ID_X_height, pinX.shape[2]);
        func.SetInt(k_ID_X_channelsDiv4, pinX.dimAxisDiv4);

        func.SetTensor(k_TensorPropertiesK, pinK);

        func.SetInt(k_ID_K_width, pinK.shape[3]);
        func.SetInt(k_ID_K_height, pinK.shape[2]);

        func.SetTensor(k_TensorPropertiesB, pinB);

        func.SetInt(k_ID_StrideY, stride[0]);
        func.SetInt(k_ID_StrideX, stride[1]);
        func.SetInt(k_ID_PadY, pad[0]);
        func.SetInt(k_ID_PadX, pad[1]);
        func.SetInt(k_ID_DilationY, dilation[0]);
        func.SetInt(k_ID_DilationX, dilation[1]);
        func.SetInt(k_ID_Groups, groups);

        if (fusedActivation == Layers.FusableActivation.Relu)
            func.EnableKeyword("RELU");

        func.Dispatch(pinO);

        if (!(fusedActivation == Layers.FusableActivation.None || fusedActivation == Layers.FusableActivation.Relu))
            O = ApplyFusedActivation(O, fusedActivation);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat Conv2DTrans(TensorFloat X, TensorFloat K, TensorFloat B, int[] stride, int[] pad, int[] outputAdjustment, Layers.FusableActivation fusedActivation)
    {
        var Oshape = ShapeInference.ConvTranspose(X.shape, K.shape, B.shape, stride, pad, outputAdjustment);
        if (Oshape.HasZeroDims())
            return NewOutputTensorFloat(Oshape);

        var O = (fusedActivation != Layers.FusableActivation.None) ? NewTempTensorFloat(Oshape) : NewOutputTensorFloat(Oshape);

        var func = new PixelFunc("Hidden/Sentis/Conv2DTrans");

        var pinX = TextureTensorData.Pin(X, 1);
        var pinK = TextureTensorData.Pin(K, 0);
        var pinB = TextureTensorData.Pin(B, 0);
        var pinO = TextureTensorData.Pin(O, 1, uploadCache: false);

        func.SetInt(k_ID_O_width, pinO.shape[3]);
        func.SetInt(k_ID_O_height, pinO.shape[2]);
        func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);

        func.SetTensor(k_TensorPropertiesX, pinX);

        func.SetInt(k_ID_X_width, pinX.shape[3]);
        func.SetInt(k_ID_X_height, pinX.shape[2]);
        func.SetInt(k_ID_X_channelsDiv4, pinX.dimAxisDiv4);

        func.SetTensor(k_TensorPropertiesK, pinK);

        func.SetInt(k_ID_K_width, pinK.shape[3]);
        func.SetInt(k_ID_K_height, pinK.shape[2]);
        func.SetInt(k_ID_K_mDivGroup, pinK.shape[1]);

        func.SetTensor(k_TensorPropertiesB, pinB);

        func.SetInt(k_ID_StrideY, stride[0]);
        func.SetInt(k_ID_StrideX, stride[1]);

        func.SetInt(k_ID_PadY, K.shape[2] - pad[0] - 1);
        func.SetInt(k_ID_PadX, K.shape[3] - pad[1] - 1);

        if (fusedActivation == Layers.FusableActivation.Relu)
            func.EnableKeyword("RELU");

        func.Dispatch(pinO);

        if (!(fusedActivation == Layers.FusableActivation.None || fusedActivation == Layers.FusableActivation.Relu))
            O = ApplyFusedActivation(O, fusedActivation);

        return O;
    }

    TensorFloat Activation(TensorFloat X, string kernelName, float alpha = 0f, float beta = 0f)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var func = new PixelFunc("Hidden/Sentis/Activation");

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, uploadCache: false);

        func.SetFloat(k_ID_Alpha, alpha);
        func.SetFloat(k_ID_Beta, beta);

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensorBlockStride(k_TensorPropertiesO, pinO);

        func.EnableKeyword(kernelName);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat Relu(TensorFloat X)
    {
        return Activation(X, "Relu");
    }

    /// <inheritdoc/>
    public override TensorFloat Relu6(TensorFloat X)
    {
        return Activation(X, "Relu6");
    }

    /// <inheritdoc/>
    public override TensorFloat LeakyRelu(TensorFloat X, float alpha)
    {
        Logger.AssertIsTrue(alpha <= 1, "LeakyRelu.ValueError: alpha is supposed to be <= 1, got {0}", alpha);
        return Activation(X, "LeakyRelu", alpha);
    }

    /// <inheritdoc/>
    public override TensorFloat Tanh(TensorFloat X)
    {
        return Activation(X, "Tanh");
    }

    /// <inheritdoc/>
    public override TensorFloat Softplus(TensorFloat X)
    {
        return Activation(X, "Softplus");
    }

    /// <inheritdoc/>
    public override TensorFloat Sigmoid(TensorFloat X)
    {
        return Activation(X, "Sigmoid");
    }

    /// <inheritdoc/>
    public override TensorFloat HardSigmoid(TensorFloat X, float alpha, float beta)
    {
        return Activation(X, "HardSigmoid", alpha, beta);
    }

    /// <inheritdoc/>
    public override TensorFloat Elu(TensorFloat X, float alpha)
    {
        return Activation(X, "Elu", alpha);
    }

    /// <inheritdoc/>
    public override TensorFloat Gelu(TensorFloat X)
    {
        return Activation(X, "Gelu");
    }

    /// <inheritdoc/>
    public override TensorFloat Selu(TensorFloat X, float alpha, float gamma)
    {
        return Activation(X, "Selu", alpha, gamma);
    }

    /// <inheritdoc/>
    public override TensorFloat Swish(TensorFloat X)
    {
        return Activation(X, "Swish");
    }

    /// <inheritdoc/>
    public override TensorFloat Abs(TensorFloat X)
    {
        return Activation(X, "Abs");
    }

    /// <inheritdoc/>
    public override TensorFloat Neg(TensorFloat X)
    {
        return Activation(X, "Neg");
    }

    /// <inheritdoc/>
    public override TensorFloat Ceil(TensorFloat X)
    {
        return Activation(X, "Ceil");
    }

    /// <inheritdoc/>
    public override TensorFloat Clip(TensorFloat X, float min, float max)
    {
        return Activation(X, "Clip", min, max);
    }

    /// <inheritdoc/>
    public override TensorFloat Floor(TensorFloat X)
    {
        return Activation(X, "Floor");
    }

    /// <inheritdoc/>
    public override TensorFloat Round(TensorFloat X)
    {
        return Activation(X, "Round");
    }

    /// <inheritdoc/>
    public override TensorFloat Reciprocal(TensorFloat X)
    {
        return Activation(X, "Reciprocal");
    }

    /// <inheritdoc/>
    public override TensorFloat Square(TensorFloat X)
    {
        return Activation(X, "Square");
    }

    /// <inheritdoc/>
    public override TensorFloat Exp(TensorFloat X)
    {
        return Activation(X, "Exp");
    }

    /// <inheritdoc/>
    public override TensorFloat Log(TensorFloat X)
    {
        return Activation(X, "Log");
    }

    /// <inheritdoc/>
    public override TensorFloat Sqrt(TensorFloat X)
    {
        return Activation(X, "Sqrt");
    }

    /// <inheritdoc/>
    public override TensorFloat Celu(TensorFloat X, float alpha)
    {
        return Activation(X, "Celu", alpha);
    }

    /// <inheritdoc/>
    public override TensorFloat HardSwish(TensorFloat X)
    {
        return Activation(X, "HardSwish");
    }

    /// <inheritdoc/>
    public override TensorFloat Softsign(TensorFloat X)
    {
        return Activation(X, "Softsign");
    }

    /// <inheritdoc/>
    public override TensorFloat ThresholdedRelu(TensorFloat X, float alpha)
    {
        return Activation(X, "ThresholdedRelu", alpha);
    }

    /// <inheritdoc/>
    public override TensorFloat Acos(TensorFloat X)
    {
        return Activation(X, "Acos");
    }

    /// <inheritdoc/>
    public override TensorFloat Acosh(TensorFloat X)
    {
        return Activation(X, "Acosh");
    }

    /// <inheritdoc/>
    public override TensorFloat Asin(TensorFloat X)
    {
        return Activation(X, "Asin");
    }

    /// <inheritdoc/>
    public override TensorFloat Asinh(TensorFloat X)
    {
        return Activation(X, "Asinh");
    }

    /// <inheritdoc/>
    public override TensorFloat Atan(TensorFloat X)
    {
        return Activation(X, "Atan");
    }

    /// <inheritdoc/>
    public override TensorFloat Atanh(TensorFloat X)
    {
        return Activation(X, "Atanh");
    }

    /// <inheritdoc/>
    public override TensorFloat Cos(TensorFloat X)
    {
        return Activation(X, "Cos");
    }

    /// <inheritdoc/>
    public override TensorFloat Cosh(TensorFloat X)
    {
        return Activation(X, "Cosh");
    }

    /// <inheritdoc/>
    public override TensorFloat Sin(TensorFloat X)
    {
        return Activation(X, "Sin");
    }

    /// <inheritdoc/>
    public override TensorFloat Sinh(TensorFloat X)
    {
        return Activation(X, "Sinh");
    }

    /// <inheritdoc/>
    public override TensorFloat Tan(TensorFloat X)
    {
        return Activation(X, "Tan");
    }

    /// <inheritdoc/>
    public override TensorFloat Erf(TensorFloat X)
    {
        return Activation(X, "Erf");
    }

    /// <inheritdoc/>
    public override TensorFloat Add(TensorFloat A, TensorFloat B)
    {
        return BroadcastBinary(A, B, "Add");
    }

    /// <inheritdoc/>
    public override TensorFloat Sub(TensorFloat A, TensorFloat B)
    {
        return BroadcastBinary(A, B, "Sub");
    }

    /// <inheritdoc/>
    public override TensorFloat Div(TensorFloat A, TensorFloat B)
    {
        return BroadcastBinary(A, B, "Div");
    }

    /// <inheritdoc/>
    public override TensorFloat Pow(TensorFloat A, TensorFloat B)
    {
        return BroadcastBinary(A, B, "Pow");
    }

    /// <inheritdoc/>
    public override TensorFloat FMod(TensorFloat A, TensorFloat B)
    {
        return BroadcastBinary(A, B, "FMod");
    }

    /// <inheritdoc/>
    public override TensorFloat Mul(TensorFloat A, TensorFloat B)
    {
        return BroadcastBinary(A, B, "Mul");
    }

    /// <inheritdoc/>
    public override TensorFloat Sum(TensorFloat[] tensors)
    {
        return Broadcast(tensors, "Add");
    }

    /// <inheritdoc/>
    public override TensorFloat Min(TensorFloat[] tensors)
    {
        return Broadcast(tensors, "Min");
    }

    /// <inheritdoc/>
    public override TensorFloat Max(TensorFloat[] tensors)
    {
        return Broadcast(tensors, "Max");
    }

    /// <inheritdoc/>
    public override TensorFloat Mean(TensorFloat[] tensors)
    {
        return Broadcast(tensors, "Mean");
    }

    /// <inheritdoc/>
    public override Tensor Expand(Tensor X, TensorShape newShape)
    {
        if (X.dataType != DataType.Float)
            return base.Expand(X, newShape);

        var O = NewOutputTensorFloat(X.shape.Broadcast(newShape));
        if (O.shape.HasZeroDims())
            return O;

        var func = new PixelFunc("Hidden/Sentis/Expand");

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, false);

        func.SetTensor(k_TensorPropertiesX, pinX);

        func.SetShape(k_ID_DimO, pinO.blockedShape);
        func.SetStrides(k_TensorPropertiesX.k_ID_Strides, pinX.blockedShape);

        if (pinX.dimAxis == 1)
            func.EnableKeyword("BLOCKEDDIM_RANK1");
        func.Dispatch(pinO);

        return O;
    }

    TensorFloat BroadcastBinary(TensorFloat A, TensorFloat B, string kernelName)
    {
        var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;
        Broadcast(O, A, B, kernelName, 0, 0);
        return O;
    }

    TensorFloat Broadcast(TensorFloat[] tensors, string kernelName)
    {
        var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(tensors));
        if (O.shape.HasZeroDims())
            return O;

        var curX = tensors[0];
        var normalization = 1.0f / tensors.Length;
        for (var t = 1; t < tensors.Length; t++)
        {
            var nextX = tensors[t];
            var Otmp = t == tensors.Length - 1 ? O : NewTempTensorFloat(TensorShapeHelper.BroadcastShape(curX, nextX));
            Broadcast(Otmp, curX, tensors[t], kernelName, t == 1 ? normalization : 1.0f, normalization);
            curX = Otmp;
        }

        return O;
    }

    void Broadcast(TensorFloat O, TensorFloat A, TensorFloat B, string kernelName, float normalizationX, float normalizationY)
    {
        var isALarger = A.shape.length > B.shape.length;
        PinBothSame(isALarger ? A : B, isALarger ? B : A);
        var pinA = A.tensorOnDevice as TextureTensorData;
        var pinB = B.tensorOnDevice as TextureTensorData;
        var pinO = PinAsSame(O, pinA, false);

        var func = new PixelFunc("Hidden/Sentis/Broadcast");
        func.EnableKeyword(kernelName);

        func.SetTensor(k_TensorPropertiesA, pinA);
        func.SetTensor(k_TensorPropertiesB, pinB);

        func.SetShape(k_ID_DimO, pinO.blockedShape);
        func.SetStrides(k_TensorPropertiesA.k_ID_Strides, pinA.blockedShape);
        func.SetStrides(k_TensorPropertiesB.k_ID_Strides, pinB.blockedShape);

        if (pinA.dimAxis == 1)
            func.EnableKeyword("BLOCKEDDIM_RANK1_A");
        if (pinB.dimAxis == 1)
            func.EnableKeyword("BLOCKEDDIM_RANK1_B");

        func.SetFloat(k_ID_alpha, normalizationX);
        func.SetFloat(k_ID_beta, normalizationY);

        func.Dispatch(pinO);
    }

    /// <inheritdoc/>
    public override Tensor Concat(Tensor[] tensors, int axis)
    {
        if (tensors[0].dataType != DataType.Float)
            return base.Concat(tensors, axis);

        var O = NewOutputTensor(TensorShapeHelper.ConcatShape(tensors, axis), tensors[0].dataType);
        if (O.shape.HasZeroDims())
            return O;

        axis = O.shape.Axis(axis);

        var func = new PixelFunc("Hidden/Sentis/Concat");
        var strideAxis = O.shape.Strides(axis);
        func.SetInt(k_ID_StrideAxis, strideAxis);
        var oShape = O.shape;
        oShape[axis] = 0;
        TextureTensorData pinA = null;
        TextureTensorData pinB = null;
        foreach (var tensor in tensors)
        {
            if (tensor.shape.length == 0)
                continue;
            if (pinA == null)
            {
                pinA = PinBlockAny(tensor);
                func.SetInt(k_ID_StrideAxis, pinA.blockedShape.Strides(axis));
                if (axis != pinA.blockAxis)
                    func.EnableKeyword("BLOCKWISE");
                oShape[axis] += pinA.shape[axis];
                continue;
            }

            pinB = PinAsSame(tensor, pinA);
            oShape[axis] += pinB.shape[axis];
            var pinO = PinAsSame(oShape == O.shape ? O : NewTempTensorFloat(oShape), pinA, false);

            func.SetTensor(k_TensorPropertiesA, pinA);
            func.SetTensor(k_TensorPropertiesB, pinB);
            func.SetInt(k_ID_ConcatLengthA, pinA.shape[axis]);

            if (axis == pinO.blockAxis)
                func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
            else
                func.SetInt(k_TensorPropertiesO.k_ID_DimAxis, pinO.blockedShape[axis]);
            func.SetInt(k_TensorPropertiesA.k_ID_DimAxis, pinA.blockedShape[axis]);
            func.SetInt(k_TensorPropertiesB.k_ID_DimAxis, pinB.blockedShape[axis]);
            func.Dispatch(pinO);
            pinA = pinO;
        }

        if (pinB == null)
            Graphics.Blit(pinA.bufferAsTexture, PinAsSame(O, pinA, false).bufferAsTexture);

        return O;
    }

    unsafe void Slice(TensorFloat X, TensorFloat O, int* startsLocal, int* stepsLocal)
    {
        if (!(X.tensorOnDevice is TextureTensorData))
        {
            // find axis that isn't sliced along
            for (var axis = X.shape.rank - 1; axis >= 0; axis--)
            {
                if (X.shape[axis] == O.shape[axis] && startsLocal[axis] == 1 && stepsLocal[axis] == 1)
                {
                    TextureTensorData.Pin(X, axis);
                    break;
                }
            }
        }

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, false);

        var func = new PixelFunc("Hidden/Sentis/Slice");

        func.SetTensor(k_TensorPropertiesX, pinX);

        TensorShape xShape;

        if (pinX.dimAxis == pinO.dimAxis && startsLocal[pinX.blockAxis] == 1 && stepsLocal[pinX.blockAxis] == 1)
        {
            func.EnableKeyword("BLOCKWISE");
            func.SetShape(k_ID_DimO, pinO.blockedShape);
            xShape = pinX.blockedShape;
        }
        else
        {
            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
            func.SetTensorBlockStride(k_TensorPropertiesX, pinX);
            func.SetShape(k_ID_DimO, pinO.shape);
            xShape = pinX.shape;
        }

        var offsetX = 0;
        var strideX = 1;
        var stridesX = stackalloc int[8];
        for (var i = 0; i < pinX.shape.rank; i++)
        {
            var axis = pinO.shape.rank - 1 - i;
            offsetX += startsLocal[axis] * strideX;
            stridesX[i] = strideX * stepsLocal[axis];
            strideX *= xShape[axis];
        }

        func.SetInt8(k_TensorPropertiesX.k_ID_Strides, stridesX);
        func.SetInt(k_ID_OffsetX, offsetX);

        func.Dispatch(pinO);
    }

    /// <inheritdoc/>
    public override Tensor Slice(Tensor X, int[] starts, int[] ends, int[] axes, int[] steps)
    {
        if (X.dataType != DataType.Float)
            return base.Slice(X, starts, ends, axes, steps);

        var O = NewOutputTensor(X.shape.Slice(starts, ends, axes, steps), X.dataType);

        if (O.shape.HasZeroDims())
            return O;

        unsafe
        {
            var startsLocal = stackalloc int[TensorShape.maxRank];
            var stepsLocal = stackalloc int[TensorShape.maxRank];

            for (var i = 0; i < 8; i++)
            {
                stepsLocal[i] = 1;
            }

            for (var i = 0; i < starts.Length; i++)
            {
                var axis = axes == null ? i : X.shape.Axis(axes[i]);
                var step = steps != null ? steps[i] : 1;
                var dim = X.shape[axis];

                var clampAdjustDirection = step < 0 ? -1 : 0;

                var start = starts[i];
                start = start < 0 ? dim + start : start;
                start = Mathf.Clamp(start, 0, dim + clampAdjustDirection);

                startsLocal[axis] = start;
                stepsLocal[axis] = step;
            }

            Slice(X as TensorFloat, O as TensorFloat, startsLocal, stepsLocal);
        }

        return O;
    }

    TensorFloat SoftmaxActivation(Tensor X, int reduceAxis, string endKernelName)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        //Allocate temp tensors
        var reduceOpShape = X.shape.Reduce(reduceAxis);
        var B = NewTempTensorFloat(reduceOpShape);
        var S = NewTempTensorFloat(reduceOpShape);

        reduceAxis = X.shape.Axis(reduceAxis);

        var pinX = PinBlockOther(X, nonBlockAxis: reduceAxis);
        var pinO = PinAsSame(O, pinX, false);
        var pinB = PinAsSame(B, pinX, false);
        var pinS = PinAsSame(S, pinX, false);

        var dimAxis = pinX.blockedShape[reduceAxis];
        var strideAxis = pinX.blockedShape.Strides(reduceAxis);

        // x_max = X.max(axis=1)
        {
            var func = new PixelFunc("Hidden/Sentis/Reduce");
            func.EnableKeyword("REDUCEMAX");
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetInt(k_TensorPropertiesX.k_ID_StrideAxis, strideAxis);
            func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, dimAxis);
            func.Dispatch(pinB);
        }
        // e_x_sum = Sum[exp(x[:,c] - x_max[:]), c]
        {
            var func = new PixelFunc("Hidden/Sentis/ReduceExpBias");
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesB, pinB);
            func.SetInt(k_TensorPropertiesX.k_ID_StrideAxis, strideAxis);
            func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, dimAxis);
            func.Dispatch(pinS);
        }
        {
            var func = new PixelFunc("Hidden/Sentis/Softmax");
            func.EnableKeyword(endKernelName);
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesB, pinB);
            func.SetTensor(k_TensorPropertiesS, pinS);
            func.SetInt(k_TensorPropertiesX.k_ID_StrideAxis, strideAxis);
            func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, dimAxis);
            func.Dispatch(pinO);
        }

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat Softmax(TensorFloat X, int axis)
    {
        return SoftmaxActivation(X, axis, "SOFTMAXEND");
    }

    /// <inheritdoc/>
    public override TensorFloat LogSoftmax(TensorFloat X, int axis)
    {
        return SoftmaxActivation(X, axis, "LOGSOFTMAXEND");
    }

    void Reduce(Tensor X, Tensor O, int reduceAxis, string kernelName)
    {
        reduceAxis = X.shape.Axis(reduceAxis);

        var pinX = PinBlockOther(X, nonBlockAxis: reduceAxis);
        var pinO = PinAsSame(O, pinX, false);

        var func = new PixelFunc("Hidden/Sentis/Reduce");
        func.EnableKeyword(kernelName);

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetInt(k_TensorPropertiesX.k_ID_StrideAxis, pinX.blockedShape.Strides(reduceAxis));
        func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, pinX.blockedShape[reduceAxis]);
        func.SetFloat(k_ID_Normalization, 1.0f / X.shape[reduceAxis]);
        func.Dispatch(pinO);
    }

    TensorFloat Reduce(TensorFloat X, int[] axes, bool keepdim, string fullKernelName, string startKernelName = null, string middleKernelName = null, string endKernelName = null)
    {
        var shapeO = X.shape.Reduce(axes, keepdim);
        if (shapeO.HasZeroDims())
            return NewOutputTensorFloat(shapeO);

        var O = keepdim ? NewOutputTensorFloat(shapeO) : NewOutputTensorFloat(X.shape.Reduce(axes, true));

        startKernelName ??= fullKernelName;
        middleKernelName ??= fullKernelName;
        endKernelName ??= fullKernelName;

        var allAxes = (axes == null) || (axes.Length == 0);
        var axesDim = allAxes ? X.shape.rank : axes.Length;
        var shapeXReduced = X.shape;
        var isInitial = true;

        for (var i = 0; i < axesDim - 1; i++)
        {
            var axis = allAxes ? i : X.shape.Axis(axes[i]);
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            shapeXReduced[axis] = 1;
            var Otmp = NewTempTensorFloat(shapeXReduced);
            Reduce(X, Otmp, axis, isInitial ? startKernelName : middleKernelName);

            X = Otmp;

            isInitial = false;
        }

        {
            var axis = allAxes ? axesDim - 1 : X.shape.Axis(axes[axesDim - 1]);
            Reduce(X, O, axis, isInitial ? fullKernelName : endKernelName);
        }

        if (!keepdim)
            O = Reshape(O, shapeO) as TensorFloat;
        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceMax(TensorFloat X, int[] axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "REDUCEMAX");
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceMean(TensorFloat X, int[] axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "REDUCEMEAN");
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceMin(TensorFloat X, int[] axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "REDUCEMIN");
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceProd(TensorFloat X, int[] axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "REDUCEPROD");
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceSum(TensorFloat X, int[] axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "REDUCESUM");
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceL1(TensorFloat X, int[] axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "REDUCEL1", "REDUCEL1", "REDUCESUM", "REDUCESUM");
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceL2(TensorFloat X, int[] axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "REDUCEL2", "REDUCESUMSQUARE", "REDUCESUM", "REDUCESQRT");
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceLogSum(TensorFloat X, int[] axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "REDUCELOGSUM", "REDUCESUM", "REDUCESUM", "REDUCELOGSUM");
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceLogSumExp(TensorFloat X, int[] axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "REDUCELOGSUMEXP");
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceSumSquare(TensorFloat X, int[] axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "REDUCESUMSQUARE", "REDUCESUMSQUARE", "REDUCESUM", "REDUCESUM");
    }

    /// <inheritdoc/>
    public override Tensor Transpose(Tensor X, int[] permutations)
    {
        if (X.dataType != DataType.Float)
            return base.Transpose(X, permutations);

        var O = NewOutputTensorFloat(X.shape.Transpose(permutations));
        if (O.shape.HasZeroDims())
            return O;

        var pinX = PinBlockAny(X);
        var oAxis = pinX.blockAxis;
        for (var i = 0; i < permutations.Length; i++)
        {
            if (permutations[i] == pinX.blockAxis)
            {
                oAxis = i;
                break;
            }
        }

        // pin O so that the transposed blocked axis matches
        var pinO = TextureTensorData.Pin(O, oAxis, uploadCache: false);

        var func = new PixelFunc("Hidden/Sentis/Transpose");

        func.SetTensor(k_TensorPropertiesX, pinX);

        var rank = pinX.shape.rank;
        unsafe
        {
            var stridesX = stackalloc int[TensorShape.maxRank];
            var strideX = 1;
            for (var i = 0; i < rank; i++)
            {
                var dim = pinX.blockedShape[-1 - i];
                stridesX[i] = dim > 1 ? strideX : 0;
                strideX *= dim;
            }

            var permutedStridesX = stackalloc int[TensorShape.maxRank];
            for (var i = 0; i < rank; i++)
            {
                permutedStridesX[i] = stridesX[rank - 1 - permutations[rank - 1 - i]];
            }

            func.SetInt8(k_TensorPropertiesX.k_ID_Strides, permutedStridesX);
        }

        func.SetShape(k_ID_DimO, pinO.blockedShape);

        func.Dispatch(pinO);

        return O;
    }

    TensorFloat GlobalPool(TensorFloat X, string kernelName)
    {
        var O = NewOutputTensorFloat(ShapeInference.GlobalPool(X.shape));
        if (O.shape.HasZeroDims())
            return O;

        var pinX = TextureTensorData.Pin(X, 1);
        var pinO = TextureTensorData.Pin(O, 1, uploadCache: false);

        var func = new PixelFunc("Hidden/Sentis/GlobalPool");
        func.EnableKeyword(kernelName);

        func.SetTensor(k_TensorPropertiesX, pinX);
        var spatialSize = X.shape.Strides(1);
        func.SetInt(k_ID_SpatialSizeX, spatialSize);
        func.SetInt(k_ID_DimAxis, pinX.blockedShape[1]);
        func.SetFloat(k_ID_Normalization, 1.0f / spatialSize);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat GlobalAveragePool(TensorFloat X)
    {
        return GlobalPool(X, "AVGPOOL");
    }

    /// <inheritdoc/>
    public override TensorFloat GlobalMaxPool(TensorFloat X)
    {
        return GlobalPool(X, "MAXPOOL");
    }

    TensorFloat LocalPool2D(TensorFloat X, int[] pool, int[] stride, int[] pad, string kernelName)
    {
        var O = NewOutputTensorFloat(ShapeInference.ApplyPool(X.shape, pool, stride, pad));
        if (O.shape.HasZeroDims())
            return O;

        var pinX = TextureTensorData.Pin(X, 1);
        var pinO = TextureTensorData.Pin(O, 1, uploadCache: false);

        var func = new PixelFunc("Hidden/Sentis/LocalPool");
        func.EnableKeyword(kernelName);

        func.SetInt(k_ID_O_width, pinO.shape[3]);
        func.SetInt(k_ID_O_height, pinO.shape[2]);
        func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);

        func.SetTensor(k_TensorPropertiesX, pinX);

        func.SetInt(k_ID_X_width, pinX.shape[3]);
        func.SetInt(k_ID_X_height, pinX.shape[2]);
        func.SetInt(k_ID_X_channelsDiv4, pinX.dimAxisDiv4);

        func.SetInt(k_ID_StrideY, stride[0]);
        func.SetInt(k_ID_StrideX, stride[1]);
        func.SetInt(k_ID_PadY, pad[0]);
        func.SetInt(k_ID_PadX, pad[1]);
        func.SetInt(k_ID_PoolY, pool[0]);
        func.SetInt(k_ID_PoolX, pool[1]);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat MaxPool(TensorFloat X, int[] pool, int[] stride, int[] pad)
    {
        if (X.shape.rank == 4)
            return LocalPool2D(X, pool, stride, pad, "MAXPOOL");
        else
            return base.MaxPool(X, pool, stride, pad);
    }

    /// <inheritdoc/>
    public override TensorFloat AveragePool(TensorFloat X, int[] pool, int[] stride, int[] pad)
    {
        if (X.shape.rank == 4)
            return LocalPool2D(X, pool, stride, pad, "AVGPOOL");
        else
            return base.AveragePool(X, pool, stride, pad);
    }

    /// <inheritdoc/>
    public override TensorFloat DepthToSpace(TensorFloat X, int blocksize, Layers.DepthToSpaceMode mode)
    {
        var O = NewOutputTensorFloat(ShapeInference.DepthToSpace(X.shape, blocksize));
        if (O.shape.HasZeroDims())
            return O;

        var func = new PixelFunc("Hidden/Sentis/DepthToSpace");

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, false);

        func.SetInt(k_ID_O_width, pinO.shape[3]);
        func.SetInt(k_ID_O_height, pinO.shape[2]);
        func.SetInt(k_ID_O_channels, pinO.shape[1]);

        func.SetTensor(k_TensorPropertiesX, pinX);

        func.SetInt(k_ID_X_width, pinX.shape[3]);
        func.SetInt(k_ID_X_height, pinX.shape[2]);
        func.SetInt(k_ID_X_channels, pinX.shape[1]);

        func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
        func.SetTensorBlockStride(k_TensorPropertiesX, pinX);

        func.SetInt(k_ID_BlockSize, blocksize);

        if (mode == Layers.DepthToSpaceMode.ColumnRowDepth)
            func.EnableKeyword("COLUMNROWDEPTH");
        else if (mode == Layers.DepthToSpaceMode.DepthColumnRow)
            func.EnableKeyword("DEPTHCOLUMNROW");

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat SpaceToDepth(TensorFloat X, int blocksize)
    {
        var O = NewOutputTensorFloat(ShapeInference.SpaceToDepth(X.shape, blocksize));
        if (O.shape.HasZeroDims())
            return O;

        var func = new PixelFunc("Hidden/Sentis/SpaceToDepth");

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, false);

        func.SetInt(k_ID_O_width, pinO.shape[3]);
        func.SetInt(k_ID_O_height, pinO.shape[2]);
        func.SetInt(k_ID_O_channels, pinO.shape[1]);

        func.SetTensor(k_TensorPropertiesX, pinX);

        func.SetInt(k_ID_X_width, pinX.shape[3]);
        func.SetInt(k_ID_X_height, pinX.shape[2]);
        func.SetInt(k_ID_X_channels, pinX.shape[1]);

        func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
        func.SetTensorBlockStride(k_TensorPropertiesX, pinX);

        func.SetInt(k_ID_BlockSize, blocksize);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Split(Tensor X, int axis, int start, int end)
    {
        if (X.dataType != DataType.Float)
            return base.Split(X, axis, start, end);
        axis = X.shape.Axis(axis);
        var O = NewOutputTensor(X.shape.Split(axis, start, end), X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, false);

        var func = new PixelFunc("Hidden/Sentis/Split");

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetInt(k_ID_StrideAxis, pinO.blockedShape.Strides(axis));
        func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, pinX.blockedShape[axis]);
        func.SetInt(k_ID_SplitStart, start);
        func.SetInt(k_ID_SplitLength, pinO.blockedShape[axis]);
        if (pinX.blockAxis != axis)
            func.EnableKeyword("BLOCKWISE");

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat Pad(TensorFloat X, int[] pad, Layers.PadMode padMode, float constant)
    {
        if (padMode != Layers.PadMode.Constant)
            Assert.IsFalse(X.shape.HasZeroDims(), "ValueError: zero dimensions input for Pad operator is not supported");

        var Oshape = X.shape.Pad(pad);
        if (X.shape.HasZeroDims())
            return ConstantOfShape(Oshape, constant);
        var O = NewOutputTensorFloat(Oshape);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = X.tensorOnDevice as TextureTensorData;
        var xRank = X.shape.rank;
        var blockAxis = pinX?.blockAxis ?? 0;

        if (pinX == null || (blockAxis >= 0 && pad[blockAxis] + pad[blockAxis + xRank] > 0))
        {
            // repin X again if pad on blocked axis
            blockAxis = xRank - 1;
            for (; blockAxis >= 0; blockAxis--)
            {
                if (X.shape[blockAxis] > 1 && pad[blockAxis] + pad[blockAxis + xRank] == 0)
                    break;
            }

            pinX = TextureTensorData.Pin(X, blockAxis);
        }

        var pinO = PinAsSame(O, pinX, false);

        var func = new PixelFunc("Hidden/Sentis/Pad");

        switch (padMode)
        {
            case Layers.PadMode.Constant:
                func.EnableKeyword("CONSTANT");
                break;
            case Layers.PadMode.Reflect:
                func.EnableKeyword("REFLECT");
                break;
            case Layers.PadMode.Edge:
                func.EnableKeyword("EDGE");
                break;
            case Layers.PadMode.Symmetric:
                func.EnableKeyword("SYMMETRIC");
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(padMode), padMode, null);
        }

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetInt(k_ID_MaxBlockIndexX, pinX.blockedShape.length - 1);

        unsafe
        {
            var padArray = stackalloc int[8];
            for (var i = 0; i < xRank; i++)
            {
                padArray[i] = pad[xRank - 1 - i];
            }

            func.SetInt8(k_ID_Pad, padArray);
        }

        func.SetShape(k_ID_DimO, pinO.blockedShape);
        func.SetShape(k_ID_DimX, pinX.blockedShape);
        func.SetStrides(k_TensorPropertiesX.k_ID_Strides, pinX.blockedShape);

        func.SetFloat(k_ID_Beta, constant);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat Resize(TensorFloat X, float[] scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode = Layers.NearestMode.RoundPreferFloor, Layers.CoordTransformMode coordTransformMode = Layers.CoordTransformMode.HalfPixel)
    {
        if (X.shape.rank != 4)
            return base.Resize(X, scale, interpolationMode, nearestMode, coordTransformMode);

        var O = NewOutputTensorFloat(ShapeInference.Resize(X.shape, scale));
        if (O.shape.HasZeroDims())
            return O;

        Vector4 scaleXY = Vector4.zero;
        Vector4 biasXY = Vector4.zero;
        for (int i = 0; i < 2; i++)
        {
            OpsUtils.GetScaleAndBias(X.shape[2 + i], O.shape[2 + i], scale[2 + i], coordTransformMode, interpolationMode, nearestMode, out float outputScale, out float outputBias);
            scaleXY[i] = outputScale;
            biasXY[i] = outputBias;
        }

        var func = new PixelFunc("Hidden/Sentis/Upsample2D");

        if (interpolationMode == Layers.InterpolationMode.Nearest)
        {
            if (nearestMode == Layers.NearestMode.RoundPreferFloor || nearestMode == Layers.NearestMode.Ceil)
                func.EnableKeyword("NEAREST_CEIL");
            else if (nearestMode == Layers.NearestMode.RoundPreferCeil || nearestMode == Layers.NearestMode.Floor)
                func.EnableKeyword("NEAREST_FLOOR");
            else
                throw new NotImplementedException();
        }
        else //if (interpolationMode == Layers.InterpolationMode.Linear)
        {
            func.EnableKeyword("LINEAR");
        }

        var pinX = TextureTensorData.Pin(X, 1);
        var pinO = TextureTensorData.Pin(O, 1, uploadCache: false);

        func.SetInt(k_ID_O_width, pinO.shape[3]);
        func.SetInt(k_ID_O_height, pinO.shape[2]);
        func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);

        func.SetTensor(k_TensorPropertiesX, pinX);

        func.SetInt(k_ID_X_width, pinX.shape[3]);
        func.SetInt(k_ID_X_height, pinX.shape[2]);
        func.SetInt(k_ID_X_channelsDiv4, pinX.dimAxisDiv4);

        func.SetVector(k_ID_Scale, scaleXY);
        func.SetVector(k_ID_Bias, biasXY);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat ScaleBias(TensorFloat X, TensorFloat S, TensorFloat B)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var func = new PixelFunc("Hidden/Sentis/ScaleBias");

        var pinX = X.tensorOnDevice as TextureTensorData;
        pinX ??= TextureTensorData.Pin(X, X.shape.rank - 2);
        var pinS = TextureTensorData.Pin(S, 0);
        var pinB = TextureTensorData.Pin(B, 0);
        var pinO = PinAsSame(O, pinX, false);

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensor(k_TensorPropertiesS, pinS);
        func.SetTensor(k_TensorPropertiesB, pinB);
        if (pinX.blockAxis == 1)
        {
            func.EnableKeyword("BLOCK_C");
            func.SetInt(k_ID_StrideAxis, pinO.strideAxis);
            func.SetInt(k_TensorPropertiesO.k_ID_DimBlocked, pinO.dimAxisDiv4);
        }
        else
        {
            func.SetInt(k_ID_StrideC, pinO.blockedShape.Strides(1));
            func.SetInt(k_ID_DimC, pinO.blockedShape[1]);
        }

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Copy(Tensor X)
    {
        if (X.dataType != DataType.Float)
            return base.Copy(X);

        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, uploadCache: false);
        Graphics.Blit(pinX.bufferAsTexture, pinO.bufferAsTexture);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Reshape(Tensor X, TensorShape newShape)
    {
        if (X.dataType != DataType.Float)
            return base.Reshape(X, newShape);

        Logger.AssertAreEqual(X.shape.length, newShape.length, "Reshape.LengthError: in/out tensorshape must have the same # of elements : ({0}, {1})", X.shape.length, newShape.length);

        var O = NewOutputTensor(newShape, X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        // TODO shallow reshape if X and newShape have compatible shapes on blockedReshape

        var pinX = X.tensorOnDevice as TextureTensorData;
        TextureTensorData pinO;

        if (pinX != null)
        {
            // try and pin O in a layout that can be read in float4 blocks from x
            var blockAxis = O.shape.rank - 1;
            var strideO = 1;
            for (; blockAxis >= 0; blockAxis--)
            {
                if (strideO >= pinX.strideAxis)
                    break;
                strideO *= O.shape[blockAxis];
            }

            pinO = TextureTensorData.Pin(O, blockAxis, false);
        }
        else
        {
            pinX = PinBlockAny(X);
            pinO = PinAsSame(O, pinX, false);
        }

        var func = new PixelFunc("Hidden/Sentis/Reshape");
        func.SetTensor(k_TensorPropertiesX, pinX);
        if (pinX.strideAxis == pinO.strideAxis && (pinX.dimAxis == pinO.dimAxis) || (pinX.dimAxis % 4 == 0 && pinO.dimAxis % 4 == 0))
        {
            func.EnableKeyword("BLOCKWISE");
        }
        else
        {
            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
            func.SetTensorBlockStride(k_TensorPropertiesX, pinX);
        }

        func.Dispatch(pinO);

        return O;
    }
}
} // namespace Unity.Sentis
