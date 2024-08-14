using UnityEngine;
using Unity.Sentis;
using UnityEngine.Assertions;

public class TextureToTensor : MonoBehaviour
{
    // 8x8 red texture
    [SerializeField]
    Texture2D texture;

    void Start()
    {
        // tensor dimensions are taken from texture
        using Tensor<float> tensor = TextureConverter.ToTensor(texture);
        Assert.AreEqual(tensor.shape, new TensorShape(1, 4, 8, 8));

        // specifying channel number truncates the channels from the texture
        using Tensor<float> tensorRGB = TextureConverter.ToTensor(texture, channels: 3);
        Assert.AreEqual(tensorRGB.shape, new TensorShape(1, 3, 8, 8));

        // specifying width and/or height resamples the texture linearly
        using Tensor<float> tensor16X24 = TextureConverter.ToTensor(texture, width: 4, height: 12);
        Assert.AreEqual(tensor16X24.shape, new TensorShape(1, 4, 12, 4));

        // for more complex conversions use a TextureTransform, the defaults are as above
        using Tensor<float> tensorT = TextureConverter.ToTensor(texture, new TextureTransform());
        Assert.AreEqual(tensorT.shape, tensor.shape);

        // RGB tensor from transform
        using Tensor<float> tensorTRGB = TextureConverter.ToTensor(texture, new TextureTransform().SetDimensions(channels: 3));
        Assert.AreEqual(tensorTRGB.shape, tensorRGB.shape);

        // linear sampled tensor from texture
        using Tensor<float> tensorT16X24 = TextureConverter.ToTensor(texture, new TextureTransform().SetDimensions(width: 4, height: 12));
        Assert.AreEqual(tensorT16X24.shape, tensor16X24.shape);

        // alternative tensor layout
        using Tensor<float> tensorTNHWC = TextureConverter.ToTensor(texture, new TextureTransform().SetTensorLayout(TensorLayout.NHWC));
        Assert.AreEqual(tensorTNHWC.shape, new TensorShape(1, 8, 8, 4));

        // explicit alternative tensor layout
        using Tensor<float> tensorTNHCW = TextureConverter.ToTensor(texture, new TextureTransform().SetTensorLayout(0, 2, 1, 3));
        Assert.AreEqual(tensorTNHCW.shape, new TensorShape(1, 8, 4, 8));

        // set tensor 0, 0 from bottom left of texture rather than default top left
        using Tensor<float> tensorTBottomLeft = TextureConverter.ToTensor(texture, new TextureTransform().SetCoordOrigin(CoordOrigin.BottomLeft));

        // swizzle color channels of texture using preset
        using Tensor<float> tensorTBGRA = TextureConverter.ToTensor(texture, new TextureTransform().SetChannelSwizzle(ChannelSwizzle.BGRA));
        // make the tensors readable (move to CPU) before accessing with indices
        Assert.AreEqual(tensorTBGRA.ReadbackAndClone()[0, 2, 0, 0], tensor.ReadbackAndClone()[0, 0, 0, 0]);

        // swizzle color channels of texture explicitly to all sample from Red channel in texture
        using Tensor<float> tensorTRRRR = TextureConverter.ToTensor(texture, new TextureTransform().SetChannelSwizzle(0, 0, 0, 0));
        // make the tensors readable (move to CPU) before accessing with indices
        Assert.AreEqual(tensorTRRRR.ReadbackAndClone()[0, 3, 0, 0], tensor.ReadbackAndClone()[0, 0, 0, 0]);

        // chain transform settings together
        using Tensor<float> tensorTChained = TextureConverter.ToTensor(texture, new TextureTransform().SetDimensions(channels: 3).SetCoordOrigin(CoordOrigin.BottomLeft).SetChannelSwizzle(ChannelSwizzle.BGRA));
    }
}
