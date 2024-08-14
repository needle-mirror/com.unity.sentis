# Convert a texture to a tensor

Use [`TextureConverter.ToTensor`](Unity.Sentis.TextureConverter.ToTensor*) to convert a [`Texture2D`](xref:UnityEngine.Texture2D) or a [`RenderTexture`](xref:UnityEngine.RenderTexture) to a tensor.

```
using UnityEngine;
using Unity.Sentis;

public class ConvertTextureToTensor : MonoBehaviour
{
    Texture2D inputTexture;

    void Start()
    {
        Tensor<float> inputTensor = TextureConverter.ToTensor(inputTexture);
    }
}
```

By default, the tensor has the following properties:

- It matches the texture's height, width, and number of channels.
- It has a data type of float.
- It follows the tensor layout of batch, channels, height, width (NCHW), for example 1 × 3 × 24 × 32 for a single RGB texture with a height of 24 and a width of 32.
- It uses mipmap level 0 if there's a mipmap. Refer to [Mipmaps introduction](https://docs.unity3d.com/Documentation/Manual/texture-mipmaps-introduction.html) for more information.

Make sure the format of the texture matches the requirements of your model. To change the format of the texture, such as adjusting the number of channels, use the settings in [Texture Import Settings window](https://docs.unity3d.com/Documentation/Manual/class-TextureImporter.html).

Depending on the input tensor your model needs, you might also need to scale the values in the tensor before you run the model. For example, if your model needs values from 0 to 255 instead of from 0 to 1. You can edit the model using the functional API to scale a tensor input. Refer to [Edit a model](edit-a-model.md) for more information.

Refer to the `Convert textures to tensors` example in the [sample scripts](package-samples.md) for an example.

### Override texture shape and layout

You can use parameters in [`TextureConverter.ToTensor`](Unity.Sentis.TextureConverter.ToTensor*) to override the width, height, and number of channels of a texture. For example:

```
// Set a property to -1 to use the default value
Tensor<float> inputTensor = TextureConverter.ToTensor(inputTexture, width: 4, height: 12, channels: -1);
```

You can also use a [`TextureTransform`](xref:Unity.Sentis.TextureTransform) object to override the properties of a texture. For example, the following code changes or "swizzles" the order of the texture channels to blue, green, red, alpha:

```
// Create a TextureTransform that swizzles the order of the channels of the texture
TextureTransform swizzleChannels = new TextureTransform().SetChannelSwizzle(ChannelSwizzle.BGRA);

// Convert the texture to a tensor using the TextureTransform object
Tensor<float> inputTensor = TextureConverter.ToTensor(inputTexture, swizzleChannels);
```

You can also chain operations together.

```
// Create a TextureTransform that swizzles the order of the channels of the texture and changes the size
TextureTransform swizzleChannelsAndChangeSize = new TextureTransform().SetChannelSwizzle(ChannelSwizzle.BGRA).SetDimensions(width: 4, height: 12);

// Convert the texture to a tensor using the TextureTransform object
Tensor<float> inputTensor = TextureConverter.ToTensor(inputTexture, swizzleChannelsAndChangeSize);
```

If the width and height of the texture doesn't match the width and height of the tensor, Sentis applies linear resampling to upsample or downsample the texture.

Refer to the [`TextureTransform`](xref:Unity.Sentis.TextureTransform) API reference for more information.

### Set a tensor to the correct format

When you convert a texture to a tensor, Sentis defaults to the NCHW layout.

If your model needs a different layout, use [`SetTensorLayout`](xref:Unity.Sentis.TextureTransform.SetTensorLayout*) to set the layout of the converted tensor.

Refer to [Tensor fundamentals in Sentis](tensor-fundamentals.md) for more information about tensor formats.

### Avoid Tensor and Texture creation

Allocating memory affects performance. If you can, try to allocate all needed memory on startup. You can use [`TextureConverter`](Unity.Sentis.TextureConverter) methods to directly operate on pre-allocated tensor/textures.

For example, to read data from a webcam every frame, copy the webcam texture content into the input tensor. Don't create a new tensor every frame.
```
Tensor<float> inputTensor;
Texture webcamTexture;

// Allocate resources on startup
void Start()
{
    inputTensor = new Tensor<float>(new TensorShape(1, 3, webcamTexture.height, webcamTexture.width));
}
void Update()
{
    // Copy webcamTexture into inputTensor : no memory allocations!
    TextureConverter.ToTensor(webcamTexture, inputTensor, new TextureTransform());

    // Run inference
}
```

## Additional resources

- [Tensor fundamentals in Sentis](tensor-fundamentals.md)
- [Edit a model](edit-a-model.md)
- [Use output data](use-model-output.md)
