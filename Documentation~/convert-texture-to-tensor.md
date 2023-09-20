# Convert a texture to a tensor

Use [`TextureConverter.ToTensor`](xref:Unity.Sentis.TextureConverter.ToTensor(UnityEngine.Texture,Unity.Sentis.TextureTransform)) to convert a [Texture2D](https://docs.unity3d.com/ScriptReference/Texture2D.html) or a [render texture](https://docs.unity3d.com/ScriptReference/RenderTexture.html) to a tensor.

```
using UnityEngine;
using Unity.Sentis;

public class ConvertTextureToTensor : MonoBehaviour
{
    Texture2D inputTexture;

    void Start()
    {
        TensorFloat inputTensor = TextureConverter.ToTensor(inputTexture);
    }
}
```

By default, the tensor has the following properties:

- The same height, width and number of channels as the texture.
- A tensor layout of batch, channels, height, width (NCHW), for example 1 × 3 × 24 × 32 for a single RGB texture with a height of 24 and a width of 32.
- A data type of float, with values from 0 to 1.
- Uses mipmap level 0 if there's a mipmap. Refer to [Mipmaps introduction](https://docs.unity3d.com/Documentation/Manual/texture-mipmaps-introduction.html) for more information.

Make sure the format of the texture matches what your model needs. If you need to change the format of the texture, for example to change the number of channels, you can use the settings in [Texture Import Settings window](https://docs.unity3d.com/Documentation/Manual/class-TextureImporter.html).

Depending on the input tensor your model needs, you might also need to scale the values in the tensor before you run the model. For example, if your model needs values from 0 to 255 instead of from 0 to 1. You can use the `WorkerFactory.CreateOps` API to scale a tensor. Refer to [Do operations on a tensor](do-complex-tensor-operations.md) for more information.

Refer to the `Convert textures to tensors` example in the [sample scripts](package-samples.md) for an example.

### Override texture shape and layout

You can use parameters in `TextureConverter.ToTensor` to override the width, height and number of channels of a texture. For example:

```
// Set a property to -1 to use the default value
TensorFloat inputTensor = TextureConverter.ToTensor(inputTexture, width: 4, height: 12, channels: -1);
```

You can also use a `TextureTransform` object to override the properties of a texture. For example, the following code changes or "swizzles" the order of the texture channels to blue, green, red, alpha:

```
// Create a TextureTransform that swizzles the order of the channels of the texture
TextureTransform swizzleChannels = new TextureTransform().SetChannelSwizzle(ChannelSwizzle.BGRA);

// Convert the texture to a tensor using the TextureTransform object
TensorFloat inputTensor = TextureConverter.ToTensor(inputTexture, swizzleChannels);
``` 

You can also chain operations together.

```
// Create a TextureTransform that swizzles the order of the channels of the texture and changes the size
TextureTransform swizzleChannelsAndChangeSize = new TextureTransform().SetChannelSwizzle(ChannelSwizzle.BGRA).SetDimensions(width: 4, height: 12);

// Convert the texture to a tensor using the TextureTransform object
TensorFloat inputTensor = TextureConverter.ToTensor(inputTexture, swizzleChannelsAndChangeSize);
```

If the width and height of the texture doesn't match the width and height of the tensor, Sentis applies linear resampling to upsample or downsample the texture.

Refer to the [TextureTransform](xref:Unity.Sentis.TextureTransform) API reference for more information.

### Set a tensor to the correct format

When you convert a texture to a tensor, Sentis defaults to the batch size, channels, height, width (NCHW) layout.

If your model needs a different layout, use [`TextureTransform.SetTensorLayout`](xref:Unity.Sentis.TextureTransform.SetTensorLayout(Unity.Sentis.TensorLayout)) to set the layout of the converted tensor.

Refer to [Tensor fundamentals in Sentis](tensor-fundamentals.md) for more information about tensor formats.

## Additional resources

- [Tensor fundamentals in Sentis](tensor-fundamentals.md)
- [Do operations on a tensor](do-complex-tensor-operations.md)
- [Use output data](use-model-output.md)
