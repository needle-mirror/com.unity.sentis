using UnityEngine;
using System.Runtime.CompilerServices;
using UnityEngine.Assertions;
using System;
using static Unity.Sentis.ShaderPropertyID;

[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis {

readonly struct PixelFunc
{
    readonly Material m_Material;

    public PixelFunc(string name)
    {
        m_Material = PixelShaderSingleton.Instance.FindMaterial(name);
    }

    public void SetBool(int nameID, bool value)
    {
        m_Material.SetInt(nameID, value ? 1 : 0);
    }

    public void SetInt(int nameID, int value)
    {
        m_Material.SetInteger(nameID, value);
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
        m_Material.SetInteger(tensorProperties.k_ID_WidthMask, pinX.widthMask);
        m_Material.SetInteger(tensorProperties.k_ID_WidthShift, pinX.widthShift);
    }

    internal void SetTensorBlockStride(TensorProperties tensorProperties, TextureTensorData pinX)
    {
        m_Material.SetInteger(tensorProperties.k_ID_StrideAxis, pinX.strideAxis);
        m_Material.SetInteger(tensorProperties.k_ID_DimAxis, pinX.dimAxis);
        m_Material.SetInteger(tensorProperties.k_ID_DimBlocked, pinX.dimAxisDiv4);
    }

    public void Dispatch(TextureTensorData pinO)
    {
        m_Material.SetInteger(k_TensorPropertiesO.k_ID_WidthShift, pinO.widthShift);
        m_Material.SetInteger(k_ID_LengthO, pinO.shape.length);
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

    public static void SetShapeStrides(this PixelFunc func, TensorProperties properties, TensorShape shape)
    {
        unsafe
        {
            var pShape = stackalloc int[TensorShape.maxRank];
            var pStrides = stackalloc int[TensorShape.maxRank];
            OpsUtils.PinTensorShapeStrides(shape, pShape, pStrides);
            func.SetInt8(properties.k_ID_Shape, pShape);
            func.SetInt8(properties.k_ID_Strides, pStrides);
        }
        func.SetInt(properties.k_ID_Rank, shape.rank);
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
public class GPUPixelBackend : CPUBackend
{
    /// <summary>
    /// Initializes and returns an instance of `GPUPixelBackend`.
    /// </summary>
    public GPUPixelBackend(ITensorAllocator allocator = null)
        : base(allocator) { }

    /// <inheritdoc/>
    public override DeviceType deviceType => DeviceType.CPU;

    /// <summary>
    /// Pins the tensor as `TextureTensorData` on any axis (choose last).
    /// </summary>
    static TextureTensorData PinBlockAny(Tensor X, bool clearOnInit = true)
    {
        if (X.tensorOnDevice is TextureTensorData textureTensorData)
            return textureTensorData;
        return TextureTensorData.Pin(X, X.shape.rank - 1, clearOnInit);
    }

    /// <summary>
    /// Pins the tensor as TextureTensorData on any axis except `nonBlockAxis`. (Choose last unless avoid, else one before last.)
    /// </summary>
    static TextureTensorData PinBlockOther(Tensor X, int nonBlockAxis, bool clearOnInit = true)
    {
        if (X.tensorOnDevice is TextureTensorData textureTensorData)
            if (textureTensorData.blockAxis != nonBlockAxis)
                return textureTensorData;
        var axis = nonBlockAxis == X.shape.rank - 1 ? X.shape.rank - 2 : X.shape.rank - 1;
        return TextureTensorData.Pin(X, axis, clearOnInit);
    }

    /// <summary>
    /// Pins the tensor X blocking along the same axis as a given other TextureTensorData
    /// This can be used to block an output tensor along the same axis as an input tensor for an op
    /// </summary>
    static TextureTensorData PinAsSame(Tensor X, TextureTensorData other, bool clearOnInit = true)
    {
        return TextureTensorData.Pin(X, X.shape.rank - other.shape.rank + other.blockAxis, clearOnInit);
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
    public override Tensor Cast(Tensor X, DataType toType)
    {
        if (X.dataType == toType)
            return Copy(X);

        var O = NewOutputTensor(X.shape, toType);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, false);

        var func = new PixelFunc("Hidden/Sentis/Cast");
        func.EnableKeyword(X.dataType == DataType.Int ? "IntToFloat" : "FloatToInt");
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat ConstantOfShape(TensorShape X, float value)
    {
        var O = NewOutputTensorFloat(X);
        if (O.shape.HasZeroDims())
            return O;
        var func = new PixelFunc("Hidden/Sentis/ConstantOfShape");
        var pinO = PinBlockAny(O, false);
        func.EnableKeyword("Float");
        func.SetFloat(k_ID_memValueFloat, value);
        func.Dispatch(pinO);
        return O;
    }

    /// <inheritdoc/>
    public override TensorInt ConstantOfShape(TensorShape X, int value)
    {
        var O = NewOutputTensorInt(X);
        if (O.shape.HasZeroDims())
            return O;
        var func = new PixelFunc("Hidden/Sentis/ConstantOfShape");
        var pinO = PinBlockAny(O, false);
        func.EnableKeyword("Int");
        func.SetInt(k_ID_memValueInt, value);
        func.Dispatch(pinO);
        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat MatMul(TensorFloat X, TensorFloat Y)
    {
        var Oshape = X.shape.MatMul(Y.shape);
        if (X.shape.HasZeroDims() || X.shape.HasZeroDims())
            return ConstantOfShape(Oshape, 0.0f);

        var xShape = X.shape.rank == 1 ? new TensorShape(1, X.shape[0]) : X.shape;
        var yShape = Y.shape.rank == 1 ? new TensorShape(Y.shape[0], 1) : Y.shape;
        var oShape = X.shape.rank > 1 && Y.shape.rank > 1 ? Oshape : xShape.MatMul(yShape);

        var O = NewOutputTensorFloat(Oshape);

        var func = new PixelFunc("Hidden/Sentis/MatMul");

        var pinO = TextureTensorData.Pin(O, Y.shape.rank == 1 ? -1 : O.shape.rank - 1, clearOnInit: false);
        var pinA = TextureTensorData.Pin(X, X.shape.rank - 1);
        var pinB = TextureTensorData.Pin(Y, Y.shape.rank == 1 ? -1 : Y.shape.rank - 1);
        if (xShape != pinA.shape)
            pinA.SetShape(xShape, xShape.rank - 1);
        if (yShape != pinB.shape)
            pinB.SetShape(yShape, yShape.rank - 1);
        if (oShape != pinO.shape)
            pinO.SetShape(oShape, oShape.rank - 1);

        func.SetTensor(k_TensorPropertiesA, pinA);
        func.SetTensor(k_TensorPropertiesB, pinB);

        func.SetShape(k_ID_DimO, pinO.blockedShape);
        func.SetStrides(k_TensorPropertiesA.k_ID_Strides, pinA.blockedShape);
        func.SetStrides(k_TensorPropertiesB.k_ID_Strides, pinB.blockedShape);
        func.SetInt(k_TensorPropertiesA.k_ID_DimAxis, pinA.dimAxis);
        func.SetInt(k_ID_Kdiv4, pinA.dimAxisDiv4);

        func.Dispatch(pinO);

        if (X.shape != pinA.shape)
            pinA.SetShape(X.shape, X.shape.rank - 1);
        if (Y.shape != pinB.shape)
            pinB.SetShape(Y.shape, Y.shape.rank == 1 ? -1 : Y.shape.rank - 1);
        if (O.shape != pinO.shape)
            pinO.SetShape(O.shape, Y.shape.rank == 1 ? -1 : O.shape.rank - 1);
        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat MatMul2D(TensorFloat X, bool xTranspose, TensorFloat Y, bool yTranspose)
    {
        var Oshape = ShapeInference.Gemm(X.shape, Y.shape, xTranspose, yTranspose);
        if (X.shape.HasZeroDims() || Y.shape.HasZeroDims())
            return ConstantOfShape(Oshape, 0.0f);

        var O = NewOutputTensorFloat(Oshape);

        var func = new PixelFunc("Hidden/Sentis/Gemm");

        var pinO = TextureTensorData.Pin(O, 1, clearOnInit: false);
        var pinX = TextureTensorData.Pin(X, xTranspose ? 0 : 1);
        var pinW = TextureTensorData.Pin(Y, yTranspose ? 0 : 1);
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensor(k_TensorPropertiesW, pinW);

        if (xTranspose)
            func.EnableKeyword("TRANSPOSE_X");
        if (yTranspose)
            func.EnableKeyword("TRANSPOSE_W");
        func.SetInt(k_ID_M, pinO.blockedShape[0]);
        func.SetInt(k_ID_K, pinX.dimAxis);
        func.SetInt(k_ID_Kdiv4, pinX.dimAxisDiv4);
        func.SetInt(k_ID_Ndiv4, pinO.dimAxisDiv4);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat Dense(TensorFloat X, TensorFloat W, TensorFloat B, Layers.FusableActivation fusedActivation)
    {
        return Gemm(X, W, B, fusedActivation);
    }

    TensorFloat Gemm(TensorFloat X, TensorFloat W, TensorFloat B, Layers.FusableActivation fusedActivation)
    {
        var isDense = B != null;
        var Oshape = isDense ? ShapeInference.Dense(X.shape, W.shape, B.shape) : X.shape.MatMul(W.shape);
        if (Oshape.HasZeroDims())
            return NewOutputTensorFloat(Oshape);

        var O = (fusedActivation == Layers.FusableActivation.None || fusedActivation == Layers.FusableActivation.Relu) ? NewOutputTensorFloat(Oshape) : NewTempTensorFloat(Oshape);

        var func = new PixelFunc("Hidden/Sentis/Dense");

        var pinO = TextureTensorData.Pin(O, O.shape.rank - 1, clearOnInit: false);
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
        func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, pinX.dimAxis);
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
    public override TensorFloat Conv(TensorFloat X, TensorFloat K, TensorFloat B, int groups, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
    {
        if (X.shape.rank > 5)
            return base.Conv(X, K, B, groups, strides, pads, dilations, fusedActivation);

        var Oshape = ShapeInference.Conv(X.shape, K.shape, groups, strides, pads, dilations);

        if (Oshape.HasZeroDims())
            return NewOutputTensorFloat(Oshape);

        var O = NewOutputTensorFloat(Oshape);

        var isDepthwise = K.shape[0] == groups && K.shape[1] == 1;

        var pinX = TextureTensorData.Pin(X, 1);
        var pinK = TextureTensorData.Pin(K, isDepthwise ? 0 : 1);
        var pinO = TextureTensorData.Pin(O, 1, clearOnInit: false);

        var numSpatialDims = X.shape.rank - 2;

        PixelFunc func;

        if (isDepthwise)
        {
            func = new PixelFunc("Hidden/Sentis/DepthwiseConv");
        }
        else if (groups > 1)
        {
            func = new PixelFunc("Hidden/Sentis/GroupedConv");
            func.SetInt(k_ID_X_channels, pinX.shape[1]);
            func.SetInt(k_ID_O_channels, pinO.shape[1]);
            func.SetInt(k_ID_K_channelsDivGroupDiv4, pinK.dimAxisDiv4);
        }
        else
        {
            func = new PixelFunc("Hidden/Sentis/Conv");
            func.SetInt(k_ID_X_channels, pinX.shape[1]);
        }

        if (numSpatialDims == 1)
            func.EnableKeyword("CONV1D");
        else if (numSpatialDims == 2)
            func.EnableKeyword("CONV2D");
        else
            func.EnableKeyword("CONV3D");

        func.SetInt(k_ID_O_width, pinO.shape[-1]);
        func.SetInt(k_ID_X_width, pinX.shape[-1]);
        func.SetInt(k_ID_K_width, pinK.shape[-1]);
        func.SetInt(k_ID_StrideX, strides[numSpatialDims - 1]);
        func.SetInt(k_ID_PadX, pads[numSpatialDims - 1]);
        func.SetInt(k_ID_DilationX, dilations[numSpatialDims - 1]);
        func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetInt(k_ID_X_channelsDiv4, pinX.dimAxisDiv4);
        func.SetTensor(k_TensorPropertiesK, pinK);
        if (B != null)
        {
            func.EnableKeyword("USEBIAS");
            var pinB = TextureTensorData.Pin(B, 0);
            func.SetTensor(k_TensorPropertiesB, pinB);
        }
        func.SetInt(k_ID_Groups, groups);

        if (numSpatialDims > 1)
        {
            func.SetInt(k_ID_O_height, pinO.shape[-2]);
            func.SetInt(k_ID_X_height, pinX.shape[-2]);
            func.SetInt(k_ID_K_height, pinK.shape[-2]);
            func.SetInt(k_ID_StrideY, strides[numSpatialDims - 2]);
            func.SetInt(k_ID_PadY, pads[numSpatialDims - 2]);
            func.SetInt(k_ID_DilationY, dilations[numSpatialDims - 2]);
        }

        if (numSpatialDims > 2)
        {
            func.SetInt(k_ID_O_depth, pinO.shape[-3]);
            func.SetInt(k_ID_X_depth, pinX.shape[-3]);
            func.SetInt(k_ID_K_depth, pinK.shape[-3]);
            func.SetInt(k_ID_StrideZ, strides[numSpatialDims - 3]);
            func.SetInt(k_ID_PadZ, pads[numSpatialDims - 3]);
            func.SetInt(k_ID_DilationZ, dilations[numSpatialDims - 3]);
        }

        if (fusedActivation == Layers.FusableActivation.Relu)
            func.EnableKeyword("RELU");

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat ConvTranspose(TensorFloat X, TensorFloat W, TensorFloat B, Span<int> strides, Span<int> pads, Span<int> outputPadding, Layers.FusableActivation fusedActivation)
    {
        if (X.shape.rank > 5)
            return base.ConvTranspose(X, W, B, strides, pads, outputPadding, fusedActivation);

        var Oshape = ShapeInference.ConvTranspose(X.shape, W.shape, strides, pads, outputPadding);
        if (Oshape.HasZeroDims())
            return NewOutputTensorFloat(Oshape);

        var O = (fusedActivation != Layers.FusableActivation.None) ? NewTempTensorFloat(Oshape) : NewOutputTensorFloat(Oshape);

        var func = new PixelFunc("Hidden/Sentis/ConvTranspose");

        var pinX = TextureTensorData.Pin(X, 1);
        var pinK = TextureTensorData.Pin(W, 0);
        if (B != null)
        {
            var pinB = TextureTensorData.Pin(B, 0);
            func.SetTensor(k_TensorPropertiesB, pinB);
            func.EnableKeyword("USEBIAS");
        }
        var pinO = TextureTensorData.Pin(O, 1, clearOnInit: false);

        var numSpatialDims = X.shape.rank - 2;

        if (numSpatialDims == 1)
            func.EnableKeyword("CONVTRANSPOSE1D");
        else if (numSpatialDims == 2)
            func.EnableKeyword("CONVTRANSPOSE2D");
        else
            func.EnableKeyword("CONVTRANSPOSE3D");

        func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetInt(k_ID_X_channels, pinX.dimAxis);
        func.SetInt(k_ID_X_channelsDiv4, pinX.dimAxisDiv4);
        func.SetTensor(k_TensorPropertiesK, pinK);
        func.SetInt(k_ID_K_mDivGroup, pinK.shape[1]);

        func.SetInt(k_ID_K_width, pinK.shape[-1]);
        func.SetInt(k_ID_O_width, pinO.shape[-1]);
        func.SetInt(k_ID_X_width, pinX.shape[-1]);
        func.SetInt(k_ID_PadX, W.shape[-1] - pads[numSpatialDims - 1] - 1);
        func.SetInt(k_ID_StrideX, strides[numSpatialDims - 1]);

        if (numSpatialDims > 1)
        {
            func.SetInt(k_ID_K_height, pinK.shape[-2]);
            func.SetInt(k_ID_X_height, pinX.shape[-2]);
            func.SetInt(k_ID_O_height, pinO.shape[-2]);
            func.SetInt(k_ID_StrideY, strides[numSpatialDims - 2]);
            func.SetInt(k_ID_PadY, W.shape[-2] - pads[numSpatialDims - 2] - 1);
        }

        if (numSpatialDims > 2)
        {
            func.SetInt(k_ID_K_depth, pinK.shape[-3]);
            func.SetInt(k_ID_X_depth, pinX.shape[-3]);
            func.SetInt(k_ID_O_depth, pinO.shape[-3]);
            func.SetInt(k_ID_StrideZ, strides[numSpatialDims - 3]);
            func.SetInt(k_ID_PadZ, W.shape[-3] - pads[numSpatialDims - 3] - 1);
        }

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
        var pinO = PinAsSame(O, pinX, clearOnInit: false);

        func.SetFloat(k_ID_Alpha, alpha);
        func.SetFloat(k_ID_Beta, beta);

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensorBlockStride(k_TensorPropertiesO, pinO);

        func.EnableKeyword(kernelName);

        func.Dispatch(pinO);

        return O;
    }

    TensorInt Activation(TensorInt X, string kernelName)
    {
        var O = NewOutputTensorInt(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var func = new PixelFunc("Hidden/Sentis/ActivationInt");

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, clearOnInit: false);

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
    public override TensorFloat Shrink(TensorFloat X, float bias, float lambd)
    {
        return Activation(X, "Shrink", bias, lambd);
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
    public override TensorInt Abs(TensorInt X)
    {
        return Activation(X, "Abs");
    }

    /// <inheritdoc/>
    public override TensorFloat Neg(TensorFloat X)
    {
        return Activation(X, "Neg");
    }

    /// <inheritdoc/>
    public override TensorInt Neg(TensorInt X)
    {
        return Activation(X, "Neg");
    }

    /// <inheritdoc/>
    public override TensorInt Not(TensorInt X)
    {
        return Activation(X, "Not");
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
    public override TensorInt Sign(TensorInt X)
    {
        return Activation(X, "Sign");
    }

    /// <inheritdoc/>
    public override TensorFloat Sign(TensorFloat X)
    {
        return Activation(X, "Sign");
    }

    /// <inheritdoc/>
    public override TensorFloat ThresholdedRelu(TensorFloat X, float alpha)
    {
        return Activation(X, "ThresholdedRelu", alpha);
    }

    /// <inheritdoc/>
    public override TensorFloat PRelu(TensorFloat X, TensorFloat S)
    {
        return Broadcast(DataType.Float, X, S, "PRelu") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt And(TensorInt A, TensorInt B)
    {
        return Broadcast(DataType.Int, A, B, "And") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorInt Equal(TensorFloat A, TensorFloat B)
    {
        return Broadcast(DataType.Int, A, B, "Equal") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorInt Equal(TensorInt A, TensorInt B)
    {
        return Broadcast(DataType.Int, A, B, "EqualInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorInt Greater(TensorFloat A, TensorFloat B)
    {
        return Broadcast(DataType.Int, A, B, "Greater") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorInt Greater(TensorInt A, TensorInt B)
    {
        return Broadcast(DataType.Int, A, B, "GreaterInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorInt GreaterOrEqual(TensorFloat A, TensorFloat B)
    {
        return Broadcast(DataType.Int, A, B, "GreaterOrEqual") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorInt GreaterOrEqual(TensorInt A, TensorInt B)
    {
        return Broadcast(DataType.Int, A, B, "GreaterOrEqualInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorInt Less(TensorFloat A, TensorFloat B)
    {
        return Broadcast(DataType.Int, A, B, "Less") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorInt Less(TensorInt A, TensorInt B)
    {
        return Broadcast(DataType.Int, A, B, "LessInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorInt LessOrEqual(TensorFloat A, TensorFloat B)
    {
        return Broadcast(DataType.Int, A, B, "LessOrEqual") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorInt LessOrEqual(TensorInt A, TensorInt B)
    {
        return Broadcast(DataType.Int, A, B, "LessOrEqualInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorInt Or(TensorInt A, TensorInt B)
    {
        return Broadcast(DataType.Int, A, B, "Or") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorInt Xor(TensorInt A, TensorInt B)
    {
        return Broadcast(DataType.Int, A, B, "Xor") as TensorInt;
    }

    /// <inheritdoc/>
    public override Tensor Where(TensorInt C, Tensor A, Tensor B)
    {
        var dataType = A.dataType;
        var O = NewOutputTensor(A.shape.Broadcast(B.shape.Broadcast(C.shape)), dataType);
        if (O.shape.HasZeroDims())
            return O;

        PinBothSame(A, B);
        PinBothSame(A, C);
        var pinX = C.tensorOnDevice as TextureTensorData;
        var pinA = A.tensorOnDevice as TextureTensorData;
        var pinB = B.tensorOnDevice as TextureTensorData;
        var pinO = PinAsSame(O, pinA, false);

        var func = new PixelFunc("Hidden/Sentis/Where");
        func.EnableKeyword(dataType == DataType.Int ? "WhereInt" : "WhereFloat");

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensor(k_TensorPropertiesA, pinA);
        func.SetTensor(k_TensorPropertiesB, pinB);

        func.SetShape(k_ID_DimO, pinO.blockedShape);
        func.SetStrides(k_TensorPropertiesX.k_ID_Strides, pinX.blockedShape);
        func.SetStrides(k_TensorPropertiesA.k_ID_Strides, pinA.blockedShape);
        func.SetStrides(k_TensorPropertiesB.k_ID_Strides, pinB.blockedShape);

        func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, pinX.dimAxis);
        func.SetInt(k_TensorPropertiesA.k_ID_DimAxis, pinA.dimAxis);
        func.SetInt(k_TensorPropertiesB.k_ID_DimAxis, pinB.dimAxis);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorInt IsInf(TensorFloat X, bool detectNegative, bool detectPositive)
    {
        var O = NewOutputTensorInt(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var func = new PixelFunc("Hidden/Sentis/IsInfNaN");
        func.EnableKeyword("IsInf");

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, clearOnInit: false);

        func.SetInt(k_ID_detectNegative, detectNegative ? 1 : 0);
        func.SetInt(k_ID_detectPositive, detectPositive ? 1 : 0);

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorInt IsNaN(TensorFloat X)
    {
        var O = NewOutputTensorInt(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var func = new PixelFunc("Hidden/Sentis/IsInfNaN");
        func.EnableKeyword("IsNaN");

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, clearOnInit: false);

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.Dispatch(pinO);

        return O;
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
        return Broadcast(DataType.Float, A, B, "Add") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt Add(TensorInt A, TensorInt B)
    {
        return Broadcast(DataType.Int, A, B, "AddInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorFloat Sub(TensorFloat A, TensorFloat B)
    {
        return Broadcast(DataType.Float, A, B, "Sub") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt Sub(TensorInt A, TensorInt B)
    {
        return Broadcast(DataType.Int, A, B, "SubInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorFloat Div(TensorFloat A, TensorFloat B)
    {
        return Broadcast(DataType.Float, A, B, "Div") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt Div(TensorInt A, TensorInt B)
    {
        return Broadcast(DataType.Int, A, B, "DivInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorFloat Pow(TensorFloat A, TensorFloat B)
    {
        return Broadcast(DataType.Float, A, B, "Pow") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorFloat Pow(TensorFloat A, TensorInt B)
    {
        return Broadcast(DataType.Float, A, B, "PowInt") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorFloat FMod(TensorFloat A, TensorFloat B)
    {
        return Broadcast(DataType.Float, A, B, "FMod") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt FMod(TensorInt A, TensorInt B)
    {
        return Broadcast(DataType.Int, A, B, "FModInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorInt Mod(TensorInt A, TensorInt B)
    {
        return Broadcast(DataType.Int, A, B, "ModInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorFloat Mul(TensorFloat A, TensorFloat B)
    {
        return Broadcast(DataType.Float, A, B, "Mul") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt Mul(TensorInt A, TensorInt B)
    {
        return Broadcast(DataType.Int, A, B, "MulInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorFloat Sum(TensorFloat[] tensors)
    {
        return Broadcast(tensors, "Add") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt Sum(TensorInt[] tensors)
    {
        return Broadcast(tensors, "AddInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorFloat Min(TensorFloat[] tensors)
    {
        return Broadcast(tensors, "Min") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt Min(TensorInt[] tensors)
    {
        return Broadcast(tensors, "MinInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorFloat Max(TensorFloat[] tensors)
    {
        return Broadcast(tensors, "Max") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt Max(TensorInt[] tensors)
    {
        return Broadcast(tensors, "MaxInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorFloat Mean(TensorFloat[] tensors)
    {
        return Broadcast(tensors, "Mean") as TensorFloat;
    }

    /// <inheritdoc/>
    public override Tensor Expand(Tensor X, TensorShape newShape)
    {
        var dataType = X.dataType;
        var O = NewOutputTensor(X.shape.Broadcast(newShape), dataType);
        if (O.shape.HasZeroDims())
            return O;

        var func = new PixelFunc("Hidden/Sentis/Expand");
        if (X.dataType == DataType.Int)
            func.EnableKeyword("INT");

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, false);

        func.SetTensor(k_TensorPropertiesX, pinX);

        func.SetShape(k_ID_DimO, pinO.blockedShape);
        func.SetStrides(k_TensorPropertiesX.k_ID_Strides, pinX.blockedShape);

        func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, pinX.dimAxis);
        func.Dispatch(pinO);

        return O;
    }

    Tensor Broadcast(DataType dataType, Tensor A, Tensor B, string kernelName)
    {
        var O = NewOutputTensor(TensorShapeHelper.BroadcastShape(A, B), dataType);
        if (O.shape.HasZeroDims())
            return O;
        Broadcast(O, A, B, kernelName, 0, 0);
        return O;
    }

    Tensor Broadcast(Tensor[] tensors, string kernelName)
    {
        var dataType = tensors[0].dataType;
        var O = NewOutputTensor(TensorShapeHelper.BroadcastShape(tensors), dataType);
        if (O.shape.HasZeroDims())
            return O;

        var curX = tensors[0];
        var normalization = 1.0f / tensors.Length;
        for (var t = 1; t < tensors.Length; t++)
        {
            var nextX = tensors[t];
            var Otmp = t == tensors.Length - 1 ? O : NewTensor(TensorShapeHelper.BroadcastShape(curX, nextX), dataType, AllocScope.InternalToLayer);
            Broadcast(Otmp, curX, tensors[t], kernelName, t == 1 ? normalization : 1.0f, normalization);
            curX = Otmp;
        }

        return O;
    }

    void Broadcast(Tensor O, Tensor A, Tensor B, string kernelName, float normalizationX, float normalizationY)
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

        func.SetInt(k_TensorPropertiesA.k_ID_DimAxis, pinA.dimAxis);
        func.SetInt(k_TensorPropertiesB.k_ID_DimAxis, pinB.dimAxis);

        func.SetFloat(k_ID_alpha, normalizationX);
        func.SetFloat(k_ID_beta, normalizationY);

        func.Dispatch(pinO);
    }

    /// <inheritdoc/>
    public override Tensor Concat(Tensor[] tensors, int axis)
    {
        var dataType = tensors[0].dataType;
        var O = NewOutputTensor(TensorShapeHelper.ConcatShape(tensors, axis), dataType);
        if (O.shape.HasZeroDims())
            return O;

        axis = O.shape.Axis(axis);

        var oShape = O.shape;
        oShape[axis] = 0;
        TextureTensorData pinA = null;
        TextureTensorData pinB = null;

        var func = new PixelFunc("Hidden/Sentis/Concat");
        if (dataType == DataType.Int)
            func.EnableKeyword("INT");
        var strideAxis = O.shape.Strides(axis);
        func.SetInt(k_ID_StrideAxis, strideAxis);
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
            var pinO = PinAsSame(oShape == O.shape ? O : NewTensor(oShape, dataType, AllocScope.InternalToLayer), pinA, false);

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
        {
            func = new PixelFunc("Hidden/Sentis/Copy");
            if (dataType == DataType.Int)
                func.EnableKeyword("INT");
            func.SetTensor(k_TensorPropertiesX, pinA);
            func.Dispatch(PinAsSame(O, pinA, false));
        }

        return O;
    }

    unsafe void Slice(Tensor X, Tensor O, int* startsLocal, int* stepsLocal)
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
        if (X.dataType == DataType.Int)
            func.EnableKeyword("INT");

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
    public override Tensor Slice(Tensor X, ReadOnlySpan<int> starts, ReadOnlySpan<int> ends, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps)
    {
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

            Slice(X, O, startsLocal, stepsLocal);
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
            func.EnableKeyword("ReduceMax");
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

    /// <inheritdoc/>
    public override TensorFloat Hardmax(TensorFloat X, int axis)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        axis = X.shape.Axis(axis);

        // Allocate temp tensors
        var reduceOpShape = X.shape.Reduce(axis);
        var argMax = NewTempTensorInt(reduceOpShape);

        // argmax
        ReduceIndices(X, argMax, "ArgMax", axis, false);

        // one hot from argmax
        var pinArgMax = PinBlockAny(argMax);
        var pinO = PinAsSame(O, pinArgMax);

        var func = new PixelFunc("Hidden/Sentis/HardmaxEnd");

        func.SetTensor(k_TensorPropertiesX, pinArgMax);
        func.SetInt(k_ID_StrideAxis, pinO.blockedShape.Strides(axis));
        func.SetInt(k_TensorPropertiesO.k_ID_DimAxis, pinO.blockedShape[axis]);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorInt OneHot(TensorInt X, int axis, int depth, int offValue, int onValue)
    {
        var O = NewOutputTensorInt(ShapeInference.OneHot(X.shape, axis, depth));
        if (O.shape.HasZeroDims())
            return O;

        axis = O.shape.Axis(axis);

        var pinX = PinBlockAny(X);
        var pinO = TextureTensorData.Pin(O, axis > pinX.blockAxis ? pinX.blockAxis : pinX.blockAxis + 1, false);

        var func = new PixelFunc("Hidden/Sentis/OneHot");
        func.SetInt(k_ID_offValue, offValue);
        func.SetInt(k_ID_onValue, onValue);

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetInt(k_ID_StrideAxis, pinO.blockedShape.Strides(axis));
        func.SetInt(k_TensorPropertiesO.k_ID_DimAxis, pinO.blockedShape[axis]);

        func.Dispatch(pinO);

        return O;
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

    Tensor Reduce(Tensor X, ReadOnlySpan<int> axes, bool keepdim, string fullKernelName, string startKernelName = null, string middleKernelName = null, string endKernelName = null)
    {
        var dataType = X.dataType;
        var shapeO = X.shape.Reduce(axes, keepdim);
        if (shapeO.HasZeroDims())
            return NewOutputTensor(shapeO, dataType);

        var O = keepdim ? NewOutputTensor(shapeO, dataType) : NewOutputTensor(X.shape.Reduce(axes, true), dataType);

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
            var Otmp = NewTensor(shapeXReduced, dataType, AllocScope.InternalToLayer);
            Reduce(X, Otmp, axis, isInitial ? startKernelName : middleKernelName);

            X = Otmp;

            isInitial = false;
        }

        {
            var axis = allAxes ? axesDim - 1 : X.shape.Axis(axes[axesDim - 1]);
            Reduce(X, O, axis, isInitial ? fullKernelName : endKernelName);
        }

        if (!keepdim)
            O = Reshape(O, shapeO);
        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceMax(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceMax") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt ReduceMax(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceMaxInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceMean(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceMean") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceMin(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceMin") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt ReduceMin(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceMinInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceProd(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceProd") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt ReduceProd(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceProdInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceSum(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceSum") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt ReduceSum(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceSumInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceL1(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceL1", "ReduceL1", "ReduceSum", "ReduceSum") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt ReduceL1(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceL1Int", "ReduceL1Int", "ReduceSumInt", "ReduceSumInt") as TensorInt;
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceL2(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceL2", "ReduceSumSquare", "ReduceSum", "ReduceSqrt") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceLogSum(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceLogSum", "ReduceSum", "ReduceSum", "ReduceLogSum") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceLogSumExp(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceLogSumExp") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorFloat ReduceSumSquare(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceSumSquare", "ReduceSumSquare", "ReduceSum", "ReduceSum") as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt ReduceSumSquare(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
    {
        return Reduce(X, axes, keepdim, "ReduceSumSquareInt", "ReduceSumSquareInt", "ReduceSumInt", "ReduceSumInt") as TensorInt;
    }

    void ReduceIndices(Tensor X, Tensor O, string kernelName, int axis, bool selectLastIndex)
    {
        var pinX = PinBlockOther(X, nonBlockAxis: axis);
        var pinO = PinAsSame(O, pinX, false);

        var func = new PixelFunc("Hidden/Sentis/ReduceIndices");
        func.EnableKeyword(kernelName);
        if (X.dataType == DataType.Int)
            func.EnableKeyword("X_INT");
        func.EnableKeyword(selectLastIndex ? "Last" : "First");

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetInt(k_TensorPropertiesX.k_ID_StrideAxis, pinX.blockedShape.Strides(axis));
        func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, pinX.blockedShape[axis]);
        func.Dispatch(pinO);
    }

    TensorInt ReduceIndices(Tensor X, string kernelName, int axis, bool keepdim, bool selectLastIndex)
    {
        axis = X.shape.Axis(axis);
        var shapeO = X.shape.Reduce(axis, keepdim);
        if (shapeO.HasZeroDims())
            return NewOutputTensorInt(shapeO);

        var O = keepdim ? NewOutputTensorInt(shapeO) : NewOutputTensorInt(X.shape.Reduce(axis, true));

        ReduceIndices(X, O, kernelName, axis, selectLastIndex);

        if (!keepdim)
            O = Reshape(O, shapeO) as TensorInt;
        return O;
    }

    /// <inheritdoc/>
    public override TensorInt ArgMax(TensorFloat X, int axis, bool keepdim, bool selectLastIndex = false)
    {
        return ReduceIndices(X, "ArgMax", axis, keepdim, selectLastIndex);
    }

    /// <inheritdoc/>
    public override TensorInt ArgMax(TensorInt X, int axis, bool keepdim, bool selectLastIndex = false)
    {
        return ReduceIndices(X, "ArgMax", axis, keepdim, selectLastIndex);
    }

    /// <inheritdoc/>
    public override TensorInt ArgMin(TensorFloat X, int axis, bool keepdim, bool selectLastIndex = false)
    {
        return ReduceIndices(X, "ArgMin", axis, keepdim, selectLastIndex);
    }

    /// <inheritdoc/>
    public override TensorInt ArgMin(TensorInt X, int axis, bool keepdim, bool selectLastIndex = false)
    {
        return ReduceIndices(X, "ArgMin", axis, keepdim, selectLastIndex);
    }

    /// <inheritdoc/>
    public override Tensor Gather(Tensor X, TensorInt indices, int axis)
    {
        var dataType = X.dataType;
        var O = NewOutputTensor(ShapeInference.Gather(X.shape, indices.shape, axis), dataType);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = PinBlockAny(X);
        var pinB = PinBlockAny(indices);
        var pinO = PinBlockAny(O, false);

        var func = new PixelFunc("Hidden/Sentis/Gather");
        if (dataType == DataType.Int)
            func.EnableKeyword("GatherInt");
        func.SetInt(k_ID_endLength, X.shape.Strides(axis));
        func.SetInt(k_ID_indicesLength, indices.shape.length);
        func.SetInt(k_ID_axisDim, X.shape[axis]);
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensorBlockStride(k_TensorPropertiesX, pinX);
        func.SetTensor(k_TensorPropertiesB, pinB);
        func.SetTensorBlockStride(k_TensorPropertiesB, pinB);
        func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor GatherElements(Tensor X, TensorInt indices, int axis)
    {
        var dataType = X.dataType;
        var O = NewOutputTensor(indices.shape, dataType);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = PinBlockAny(X);
        var pinB = PinBlockAny(indices);
        var pinO = PinBlockAny(O, false);

        var func = new PixelFunc("Hidden/Sentis/GatherElements");
        if (dataType == DataType.Int)
            func.EnableKeyword("GatherInt");
        func.SetInt(k_ID_endLength, X.shape.Strides(axis));
        func.SetInt(k_ID_axisDim, X.shape[axis]);
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensorBlockStride(k_TensorPropertiesX, pinX);
        func.SetTensor(k_TensorPropertiesB, pinB);
        func.SetTensorBlockStride(k_TensorPropertiesB, pinB);
        func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor GatherND(Tensor X, TensorInt indices, int batchDims)
    {
        var O = NewOutputTensor(ShapeInference.GatherND(X.shape, indices.shape, batchDims), X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = PinBlockAny(X);
        var pinB = PinBlockAny(indices);
        var pinO = PinBlockAny(O, false);

        var func = new PixelFunc("Hidden/Sentis/GatherND");
        func.SetInt(k_ID_iStart, TensorShape.maxRank - pinO.shape.rank);
        func.SetInt(k_ID_iEndIndices, TensorShape.maxRank - pinO.shape.rank + pinB.shape.rank - 1);
        func.SetInt(k_ID_iEndX, TensorShape.maxRank - pinO.shape.rank + batchDims);
        func.SetInt(k_ID_iEndMin, TensorShape.maxRank - pinO.shape.rank + Math.Min(batchDims, pinB.shape.rank - 1));
        func.SetInt(k_ID_iStartB, TensorShape.maxRank - pinX.shape.rank + batchDims);
        func.SetInt(k_ID_iEndB, TensorShape.maxRank - pinX.shape.rank + batchDims + pinB.shape[-1]);
        func.SetShapeStrides(k_TensorPropertiesX, pinX.shape);
        func.SetShapeStrides(k_TensorPropertiesO, pinO.shape);
        func.SetShapeStrides(k_TensorPropertiesB, pinB.shape);
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensorBlockStride(k_TensorPropertiesX, pinX);
        func.SetTensor(k_TensorPropertiesB, pinB);
        func.SetTensorBlockStride(k_TensorPropertiesB, pinB);
        func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor ScatterElements(Tensor X, TensorInt indices, Tensor updates, int axis, Layers.ScatterReductionMode reduction)
    {
        if (indices.shape.HasZeroDims())
            return Copy(X);
        var dataType = X.dataType;
        var O = NewOutputTensor(X.shape, dataType);
        if (O.shape.HasZeroDims())
            return O;

        axis = X.shape.Axis(axis);
        var pinX = PinBlockOther(X, axis);
        var pinB = PinAsSame(indices, pinX);
        var pinW = PinAsSame(updates, pinX);
        var pinO = PinAsSame(O, pinX, false);

        var func = new PixelFunc("Hidden/Sentis/ScatterElements");
        if (dataType == DataType.Int)
            func.EnableKeyword("ScatterInt");
        switch (reduction)
        {
            case Layers.ScatterReductionMode.None:
                func.EnableKeyword("ReduceNone");
                break;
            case Layers.ScatterReductionMode.Add:
                func.EnableKeyword("ReduceAdd");
                break;
            case Layers.ScatterReductionMode.Mul:
                func.EnableKeyword("ReduceMul");
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(reduction), reduction, null);
        }
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensorBlockStride(k_TensorPropertiesX, pinX);
        func.SetTensor(k_TensorPropertiesB, pinB);
        func.SetTensorBlockStride(k_TensorPropertiesB, pinB);
        func.SetTensor(k_TensorPropertiesW, pinW);
        func.SetTensorBlockStride(k_TensorPropertiesW, pinW);
        func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
        func.SetInt(k_ID_DimAxis, pinO.blockedShape[axis]);
        func.SetInt(k_ID_NumIndices, pinB.blockedShape[axis]);
        func.SetInt(k_ID_StrideAxis, pinO.blockedShape.Strides(axis));
        func.Dispatch(pinO);

        return O;
    }

    Tensor ScatterND(Tensor X, TensorInt indices, Tensor updates, Layers.ScatterReductionMode reduction)
    {
        var dataType = X.dataType;
        var O = NewOutputTensor(X.shape, dataType);
        if (O.shape.HasZeroDims())
            return O;

        var func = new PixelFunc("Hidden/Sentis/ScatterND");
        if (dataType == DataType.Int)
            func.EnableKeyword("ScatterInt");

        var K = indices.shape[-1];
        var pinX = PinBlockAny(X);
        if (pinX.dimAxis >= X.shape.rank - K)
            pinX = TextureTensorData.Pin(X, X.shape.rank - K - 1);
        var pinB = TextureTensorData.Pin(indices, indices.shape.rank - 1);
        var pinW = PinAsSame(updates, pinX);
        var pinO = PinAsSame(O, pinX, false);

        switch (reduction)
        {
            case Layers.ScatterReductionMode.None:
                func.EnableKeyword("ReduceNone");
                break;
            case Layers.ScatterReductionMode.Add:
                func.EnableKeyword("ReduceAdd");
                break;
            case Layers.ScatterReductionMode.Mul:
                func.EnableKeyword("ReduceMul");
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(reduction), reduction, null);
        }
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensorBlockStride(k_TensorPropertiesX, pinX);
        func.SetTensor(k_TensorPropertiesB, pinB);
        func.SetTensorBlockStride(k_TensorPropertiesB, pinB);
        func.SetTensor(k_TensorPropertiesW, pinW);
        func.SetTensorBlockStride(k_TensorPropertiesW, pinW);
        func.SetTensorBlockStride(k_TensorPropertiesO, pinO);

        var Kdiv4 = pinB.blockedShape[-1];
        if (Kdiv4 > 1)
            func.EnableKeyword("K_LARGE");
        func.SetShape(k_TensorPropertiesX.k_ID_Shape, X.shape);
        func.SetInt(k_ID_SliceLength, pinX.blockedShape.Length(K));
        func.SetInt(k_ID_NumIndices, pinB.blockedShape.length / Kdiv4);
        func.SetInt(k_ID_K, K);
        func.SetInt(k_ID_Kdiv4, Kdiv4);
        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat ScatterND(TensorFloat X, TensorInt indices, TensorFloat updates, Layers.ScatterReductionMode reduction)
    {
        return ScatterND(X, indices, updates, reduction) as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt ScatterND(TensorInt X, TensorInt indices, TensorInt updates, Layers.ScatterReductionMode reduction)
    {
        return ScatterND(X, indices, updates, reduction) as TensorInt;
    }

    /// <inheritdoc/>
    public override Tensor Transpose(Tensor X)
    {
        var dataType = X.dataType;
        var O = NewOutputTensor(X.shape.Transpose(), X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = PinBlockAny(X);
        var oAxis = pinX.blockAxis < 0 ? -1 : X.shape.rank - 1 - pinX.blockAxis;
        var pinO = TextureTensorData.Pin(O, oAxis, clearOnInit: false);

        var func = new PixelFunc("Hidden/Sentis/Transpose");
        if (dataType == DataType.Int)
            func.EnableKeyword("INT");

        func.SetTensor(k_TensorPropertiesX, pinX);

        var rank = pinX.shape.rank;
        unsafe
        {
            var permutedStridesX = stackalloc int[TensorShape.maxRank];
            var strideX = 1;
            for (var i = 0; i < rank; i++)
            {
                var dim = pinX.blockedShape[-1 - i];
                permutedStridesX[rank - 1 - i] = dim > 1 ? strideX : 0;
                strideX *= dim;
            }

            func.SetInt8(k_TensorPropertiesX.k_ID_Strides, permutedStridesX);
        }

        func.SetShape(k_ID_DimO, pinO.blockedShape);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Transpose(Tensor X, int[] permutations)
    {
        var dataType = X.dataType;
        var O = NewOutputTensor(X.shape.Transpose(permutations), dataType);
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
        var pinO = TextureTensorData.Pin(O, oAxis, clearOnInit: false);

        var func = new PixelFunc("Hidden/Sentis/Transpose");
        if (dataType == DataType.Int)
            func.EnableKeyword("INT");

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
        var pinO = TextureTensorData.Pin(O, 1, clearOnInit: false);

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

    TensorFloat LocalPool(TensorFloat X, int[] pool, int[] stride, int[] pad, string kernelName)
    {
        var O = NewOutputTensorFloat(ShapeInference.ApplyPool(X.shape, pool, stride, pad));
        if (O.shape.HasZeroDims())
            return O;

        var pinX = TextureTensorData.Pin(X, 1);
        var pinO = TextureTensorData.Pin(O, 1, clearOnInit: false);

        var numSpatialDims = X.shape.rank - 2;

        var func = new PixelFunc("Hidden/Sentis/LocalPool");
        func.EnableKeyword(numSpatialDims == 2 ? "POOL2D" : "POOL1D");
        func.EnableKeyword(kernelName);

        func.SetInt(k_ID_O_width, pinO.shape[-1]);
        func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetInt(k_ID_X_width, pinX.shape[-1]);
        func.SetInt(k_ID_X_channelsDiv4, pinX.dimAxisDiv4);

        func.SetInt(k_ID_StrideX, stride[numSpatialDims - 1]);
        func.SetInt(k_ID_PadX, pad[numSpatialDims - 1]);
        func.SetInt(k_ID_PoolX, pool[numSpatialDims - 1]);

        if (numSpatialDims > 1)
        {
            func.SetInt(k_ID_StrideY, stride[numSpatialDims - 2]);
            func.SetInt(k_ID_PadY, pad[numSpatialDims - 2]);
            func.SetInt(k_ID_PoolY, pool[numSpatialDims - 2]);
            func.SetInt(k_ID_X_height, pinX.shape[-2]);
            func.SetInt(k_ID_O_height, pinO.shape[-2]);
        }

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat MaxPool(TensorFloat X, int[] pool, int[] stride, int[] pad)
    {
        if (X.shape.rank > 4)
            return base.MaxPool(X, pool, stride, pad);

        return LocalPool(X, pool, stride, pad, "MAXPOOL");
    }

    /// <inheritdoc/>
    public override TensorFloat AveragePool(TensorFloat X, int[] pool, int[] stride, int[] pad)
    {
        if (X.shape.rank > 4)
            return base.AveragePool(X, pool, stride, pad);

        return LocalPool(X, pool, stride, pad, "AVGPOOL");
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
        axis = X.shape.Axis(axis);
        var O = NewOutputTensor(X.shape.Split(axis, start, end), X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, false);

        var func = new PixelFunc("Hidden/Sentis/Split");
        if (X.dataType == DataType.Int)
            func.EnableKeyword("INT");

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
    public override Tensor Tile(Tensor X, ReadOnlySpan<int> repeats)
    {
        var dataType = X.dataType;
        var O = NewOutputTensor(X.shape.Tile(repeats), dataType);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = X.tensorOnDevice as TextureTensorData;
        var xRank = X.shape.rank;
        var blockAxis = pinX?.blockAxis ?? 0;

        if (pinX == null || (blockAxis >= 0 && repeats[blockAxis] > 1))
        {
            // repin X again if repeat on blocked axis
            blockAxis = xRank - 1;
            for (; blockAxis >= 0; blockAxis--)
            {
                if (X.shape[blockAxis] > 1 && repeats[blockAxis] == 1)
                    break;
            }

            pinX = TextureTensorData.Pin(X, blockAxis);
        }

        var pinO = PinAsSame(O, pinX, false);

        var func = new PixelFunc("Hidden/Sentis/Tile");
        if (dataType == DataType.Int)
            func.EnableKeyword("INT");
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetShape(k_ID_DimO, pinO.blockedShape);
        func.SetShape(k_ID_DimX, pinX.blockedShape);
        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat Pad(TensorFloat X, ReadOnlySpan<int> pad, Layers.PadMode padMode, float constant)
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
    public override TensorFloat Resize(TensorFloat X, ReadOnlySpan<float> scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode = Layers.NearestMode.RoundPreferFloor, Layers.CoordTransformMode coordTransformMode = Layers.CoordTransformMode.HalfPixel)
    {
        if (X.shape.rank < 4 || X.shape.rank > 5)
            return base.Resize(X, scale, interpolationMode, nearestMode, coordTransformMode);

        var O = NewOutputTensorFloat(ShapeInference.Resize(X.shape, scale));
        if (O.shape.HasZeroDims())
            return O;

        var numSpatialDims = X.shape.rank - 2;

        var pinX = TextureTensorData.Pin(X, 1);
        var pinO = TextureTensorData.Pin(O, 1, clearOnInit: false);

        var func = new PixelFunc("Hidden/Sentis/Upsample");
        func.EnableKeyword(numSpatialDims == 3 ? "Upsample3D" : "Upsample2D");

        var scaleXY = Vector4.zero;
        var biasXY = Vector4.zero;
        for (var i = 0; i < numSpatialDims; i++)
        {
            OpsUtils.GetScaleAndBias(X.shape[2 + i], O.shape[2 + i], scale[2 + i], coordTransformMode, interpolationMode, nearestMode, out float outputScale, out float outputBias);
            scaleXY[i] = outputScale;
            biasXY[i] = outputBias;
        }
        func.SetVector(k_ID_Scale, scaleXY);
        func.SetVector(k_ID_Bias, biasXY);

        if (interpolationMode == Layers.InterpolationMode.Nearest)
        {
            switch (nearestMode)
            {
                case Layers.NearestMode.RoundPreferFloor:
                case Layers.NearestMode.Ceil:
                    func.EnableKeyword("NEAREST_CEIL");
                    break;
                case Layers.NearestMode.RoundPreferCeil:
                case Layers.NearestMode.Floor:
                    func.EnableKeyword("NEAREST_FLOOR");
                    break;
                default:
                    throw new NotImplementedException();
            }
        }
        else //if (interpolationMode == Layers.InterpolationMode.Linear)
        {
            func.EnableKeyword("LINEAR");
        }

        func.SetInt(k_ID_O_width, pinO.shape[-1]);
        func.SetInt(k_ID_O_height, pinO.shape[-2]);
        func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetInt(k_ID_X_width, pinX.shape[-1]);
        func.SetInt(k_ID_X_height, pinX.shape[-2]);
        func.SetInt(k_ID_X_channelsDiv4, pinX.dimAxisDiv4);

        if (numSpatialDims > 2)
        {
            func.SetInt(k_ID_O_depth, pinO.shape[-3]);
            func.SetInt(k_ID_X_depth, pinX.shape[-3]);
        }

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
    public override TensorFloat BatchNormalization(TensorFloat X, TensorFloat S, TensorFloat B, TensorFloat mean, TensorFloat variance, float epsilon)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var func = new PixelFunc("Hidden/Sentis/BatchNormalization");

        var pinX = TextureTensorData.Pin(X, 1);
        var pinS = TextureTensorData.Pin(S, 0);
        var pinB = TextureTensorData.Pin(B, 0);
        var pinM = TextureTensorData.Pin(mean, 0);
        var pinV = TextureTensorData.Pin(variance, 0);
        var pinO = PinAsSame(O, pinX, false);

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensor(k_TensorPropertiesS, pinS);
        func.SetTensor(k_TensorPropertiesB, pinB);
        func.SetTensor(k_TensorPropertiesM, pinM);
        func.SetTensor(k_TensorPropertiesV, pinV);

        func.SetInt(k_ID_O_channels, pinO.dimAxis);
        func.SetInt(k_ID_O_width, pinO.strideAxis);
        func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);

        func.SetFloat(k_ID_epsilon, epsilon);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat InstanceNormalization(TensorFloat X, TensorFloat S, TensorFloat B, float epsilon)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var spatialSize = X.shape.Strides(1);
        var pinX = TextureTensorData.Pin(X, 1);
        var pooledShape = ShapeInference.GlobalPool(X.shape);
        var A = NewTempTensorFloat(pooledShape);
        var K = NewTempTensorFloat(pooledShape);
        var pinA = PinAsSame(A, pinX, false);
        var pinK = PinAsSame(K, pinX, false);

        {
            var func = new PixelFunc("Hidden/Sentis/GlobalPool");
            func.EnableKeyword("AVGPOOL");

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetInt(k_ID_SpatialSizeX, spatialSize);
            func.SetInt(k_ID_DimAxis, pinX.blockedShape[1]);
            func.SetFloat(k_ID_Normalization, 1.0f / spatialSize);

            func.Dispatch(pinA);
        }

        {
            var func = new PixelFunc("Hidden/Sentis/GlobalPool");
            func.EnableKeyword("AVGSQUAREPOOL");

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetInt(k_ID_SpatialSizeX, spatialSize);
            func.SetInt(k_ID_DimAxis, pinX.blockedShape[1]);
            func.SetFloat(k_ID_Normalization, 1.0f / spatialSize);

            func.Dispatch(pinK);
        }

        {
            var pinO = TextureTensorData.Pin(O, 1, clearOnInit: false);
            var pinS = TextureTensorData.Pin(S, 0);
            var pinB = TextureTensorData.Pin(B, 0);
            var func = new PixelFunc("Hidden/Sentis/InstanceNormalizationTail");

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesS, pinS);
            func.SetTensor(k_TensorPropertiesA, pinA);
            func.SetTensor(k_TensorPropertiesB, pinB);
            func.SetTensor(k_TensorPropertiesK, pinK);
            func.SetInt(k_ID_StrideAxis, spatialSize);
            func.SetInt(k_ID_O_channelsDiv4, pinO.blockedShape[1]);
            func.SetFloat(k_ID_epsilon, epsilon);

            func.Dispatch(pinO);
        }

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat AxisNormalization(TensorFloat X, TensorFloat S, TensorFloat B, float epsilon)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var axis = X.shape.Axis(-1);
        var reducedShape = X.shape.Reduce(axis);
        var A = NewTempTensorFloat(reducedShape);
        var K = NewTempTensorFloat(reducedShape);

        Reduce(X, A, axis, "ReduceMean");
        Reduce(X, K, axis, "ReduceMeanSquare");

        var pinX = PinBlockAny(X);
        var pinA = PinAsSame(A, pinX);
        var pinK = PinAsSame(K, pinX);
        var pinS = TextureTensorData.Pin(S, -1);
        var pinB = TextureTensorData.Pin(B, -1);
        var pinO = PinAsSame(O, pinX, clearOnInit: false);
        var func = new PixelFunc("Hidden/Sentis/AxisNormalizationTail");

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensor(k_TensorPropertiesS, pinS);
        func.SetTensor(k_TensorPropertiesA, pinA);
        func.SetTensor(k_TensorPropertiesB, pinB);
        func.SetTensor(k_TensorPropertiesK, pinK);
        func.SetInt(k_ID_reduceLength, pinO.shape[-1]);
        func.SetFloat(k_ID_epsilon, epsilon);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat RoiAlign(TensorFloat X, TensorFloat Rois, TensorInt Indices, Layers.RoiPoolingMode mode, int outputHeight, int outputWidth, int samplingRatio, float spatialScale)
    {
        var O = NewOutputTensorFloat(ShapeInference.RoiAlign(X.shape, Rois.shape, Indices.shape, outputHeight, outputWidth));
        if (O.shape.HasZeroDims())
            return O;

        var pinX = TextureTensorData.Pin(X, 1);
        var pinB = PinBlockAny(Indices);
        var pinS = TextureTensorData.Pin(Rois, 1);
        var pinO = PinAsSame(O, pinX, false);

        var func = new PixelFunc("Hidden/Sentis/RoiAlign");
        func.EnableKeyword(mode == Layers.RoiPoolingMode.Avg ? "RoiAlignAvg" : "RoiAlignMax");

        func.SetFloat(k_ID_spatialScale, spatialScale);
        func.SetInt(k_ID_numRois, Rois.shape[0]);
        func.SetFloat(k_ID_normalizeOHeight, 1.0f / outputHeight);
        func.SetFloat(k_ID_normalizeOWidth, 1.0f / outputWidth);
        func.SetInt(k_ID_samplingRatio, samplingRatio);

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensor(k_TensorPropertiesB, pinB);
        func.SetTensorBlockStride(k_TensorPropertiesB, pinB);
        func.SetTensor(k_TensorPropertiesS, pinS);

        func.SetInt(k_ID_O_width, pinO.shape[-1]);
        func.SetInt(k_ID_O_height, pinO.shape[-2]);
        func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);
        func.SetInt(k_ID_X_width, pinX.shape[-1]);
        func.SetInt(k_ID_X_height, pinX.shape[-2]);
        func.SetInt(k_ID_X_channelsDiv4, pinX.dimAxisDiv4);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat RandomUniform(TensorShape tensorShape, float low, float high, float? seed)
    {
        var O = NewOutputTensorFloat(tensorShape);

        if (O.shape.HasZeroDims())
            return O;

        var pinO = PinBlockAny(O, false);

        var func = new PixelFunc("Hidden/Sentis/Random");
        func.EnableKeyword("RandomUniform");
        func.SetInt(k_ID_seed, (int)Random.GetOpSeed(seed));
        func.SetFloat(k_ID_low, low);
        func.SetFloat(k_ID_high, high);
        func.SetTensorBlockStride(k_TensorPropertiesO, pinO);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat RandomNormal(TensorShape tensorShape, float mean, float scale, float? seed)
    {
        var O = NewOutputTensorFloat(tensorShape);

        if (O.shape.HasZeroDims())
            return O;

        var pinO = PinBlockAny(O, false);

        var func = new PixelFunc("Hidden/Sentis/Random");
        func.EnableKeyword("RandomNormal");
        func.SetInt(k_ID_seed, (int)Random.GetOpSeed(seed));
        func.SetFloat(k_ID_mean, mean);
        func.SetFloat(k_ID_scale, scale);
        func.SetTensorBlockStride(k_TensorPropertiesO, pinO);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Bernoulli(TensorFloat X, DataType dataType, float? seed)
    {
        var O = NewOutputTensor(X.shape, dataType);
        if (O.shape.HasZeroDims())
            return O;

        var func = new PixelFunc("Hidden/Sentis/Random");
        func.EnableKeyword(dataType == DataType.Int ? "BernoulliInt" : "Bernoulli");

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, clearOnInit: false);

        func.SetInt(k_ID_seed, (int)Random.GetOpSeed(seed));
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensorBlockStride(k_TensorPropertiesO, pinO);

        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat Range(float start, float limit, float delta)
    {
        var O = NewOutputTensorFloat(ShapeInference.Range(start, limit, delta));
        if (O.shape.HasZeroDims())
            return O;

        var pinO = PinBlockAny(O, clearOnInit: false);

        var func = new PixelFunc("Hidden/Sentis/Range");
        func.SetFloat(k_ID_rangeStartFloat, start);
        func.SetFloat(k_ID_rangeDeltaFloat, delta);
        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorInt Range(int start, int limit, int delta)
    {
        var O = NewOutputTensorInt(ShapeInference.Range(start, limit, delta));
        if (O.shape.HasZeroDims())
            return O;

        var pinO = PinBlockAny(O, clearOnInit: false);

        var func = new PixelFunc("Hidden/Sentis/Range");
        func.EnableKeyword("INT");
        func.SetInt(k_ID_rangeStartInt, start);
        func.SetInt(k_ID_rangeDeltaInt, delta);
        func.Dispatch(pinO);

        return O;
    }

    Tensor Trilu(Tensor X, int k, bool upper)
    {
        var O = NewOutputTensor(X.shape, X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, false);

        var func = new PixelFunc("Hidden/Sentis/Trilu");
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
        func.SetInt(k_ID_width, X.shape[-1]);
        func.SetInt(k_ID_height, X.shape[-2]);
        func.SetInt(k_ID_direction, upper ? 1 : -1);
        func.SetInt(k_ID_offset, k);
        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Tril(Tensor X, int k)
    {
        return Trilu(X, k, false);
    }

    /// <inheritdoc/>
    public override Tensor Triu(Tensor X, int k)
    {
        return Trilu(X, k, true);
    }

    Tensor CumSum(Tensor X, int axis, bool reverse, bool exclusive)
    {
        var dataType = X.dataType;
        var O = NewOutputTensor(X.shape, dataType);
        if (O.shape.HasZeroDims())
            return O;

        axis = X.shape.Axis(axis);

        var pinX = PinBlockOther(X, nonBlockAxis: axis);
        var pinO = PinAsSame(O, pinX, false);

        var func = new PixelFunc("Hidden/Sentis/CumSum");
        if (dataType == DataType.Int)
            func.EnableKeyword("INT");
        func.EnableKeyword(reverse ? "REVERSE" : "FORWARD");
        func.EnableKeyword(exclusive ? "EXCLUSIVE" : "INCLUSIVE");

        func.SetTensor(k_TensorPropertiesX, pinX);
        func.SetInt(k_ID_StrideAxis, pinX.blockedShape.Strides(axis));
        func.SetInt(k_ID_DimAxis, pinX.blockedShape[axis]);
        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override TensorFloat CumSum(TensorFloat X, int axis, bool reverse, bool exclusive)
    {
        return CumSum(X, axis, reverse, exclusive) as TensorFloat;
    }

    /// <inheritdoc/>
    public override TensorInt CumSum(TensorInt X, int axis, bool reverse, bool exclusive)
    {
        return CumSum(X, axis, reverse, exclusive) as TensorInt;
    }

    /// <inheritdoc/>
    public override Tensor Copy(Tensor X)
    {
        var O = NewOutputTensor(X.shape, X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = PinBlockAny(X);
        var pinO = PinAsSame(O, pinX, clearOnInit: false);
        var func = new PixelFunc("Hidden/Sentis/Copy");
        if (X.dataType == DataType.Int)
            func.EnableKeyword("INT");
        func.SetTensor(k_TensorPropertiesX, pinX);
        func.Dispatch(pinO);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Reshape(Tensor X, TensorShape newShape)
    {
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
        if (X.dataType == DataType.Int)
            func.EnableKeyword("INT");
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
