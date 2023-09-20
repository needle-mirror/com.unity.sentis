# Use output data

After you [get the output from a model](get-the-output.md) as a tensor, you can post-process the data to use it in your project.

You can also use [WorkerFactory.CreateOps](do-complex-tensor-operations.md) to do calculations on tensor outputs.

## Convert to a flattened 1D array

Use `MakeReadable` to move a tensor on the GPU to the CPU before you read it. Use `ToReadOnlyArray` to convert tensor data to a flattened 1D array of floats or ints.

For example:

```
TensorFloat outputTensor = worker.Execute(inputTensor).PeekOutput() as TensorFloat;
outputTensor.MakeReadable();
float[] outputData = outputTensor.ToReadOnlyArray();
```

## Convert to a render texture

To convert a tensor to a render texture, use the following APIs:

- `TextureConverter.ToTexture` to output tensor data to a render texture. Sentis creates a new render texture to do this.
- `TextureConverter.RenderToTexture` to write tensor data to an existing render texture you provide.

If you use `ToTexture`, Sentis uses the tensor to set the size and channels of the render texture. Sentis makes the following changes if the tensor doesn't match the render texture: 

- Linearly samples the tensor if the dimensions don't match.
- Removes channels from the end, if the render texture has fewer channels than the tensor.
- Sets values in RGB channels to 0 and values in the alpha channel to 1, if the render texture has more channels than the tensor.

Refer to the `Convert tensors to textures` example in the [sample scripts](package-samples.md) for working examples.

### ToTexture example

```
// Define an empty render texture
public RenderTexture rt;

void Start()
{
    ...
    
    // Get the output of the model as a tensor
    TensorFloat outputTensor = worker.Execute(inputTensor).PeekOutput() as TensorFloat;

    // Convert the tensor to a texture and store it in the uninstantiated render texture
    rt = TextureConverter.ToTexture(outputTensor);
}
```

You can use parameters in `ToTexture` to override the width, height and number of channels of a texture. 

For example:

```
    // Set a property to -1 to use the default value
    rt = TextureConverter.ToTexture(outputTensor, width: 4, height: 12, channels: -1);
```

### RenderToTexture example

```
// Define an empty render texture
public RenderTexture rt;

void Start()
{
    ...
    
    // Instantiate the render texture
    rt = new RenderTexture(24, 32, 0, RenderTextureFormat.ARGB32);

    // Get the output of the model as a tensor
    TensorFloat outputTensor = worker.Execute(inputTensor).PeekOutput() as TensorFloat;

    // Convert the tensor to a texture and store it in the render texture
    TextureConverter.RenderToTexture(outputTensor, rt);
}
```

## Copy to the screen

To copy an output tensor to the screen, follow these steps:

1. Set the `Camera.targetTexture` property of `Camera.main` to null.
2. Create a script and attach it to the Camera.
3. In the script, use `TextureConverter.RenderToScreen` in an event function such as `OnRenderImage`.

If the image is too bright, the output tensor might be using values from 0 to 255 instead of values from 0 to 1. You can use [WorkerFactory.CreateOps](do-complex-tensor-operations.md) to remap the values in the output tensor before you call `RenderToScreen`.

The following script uses a model to change a texture, then copies the result to the screen. You can set `modelAsset` to one of the [style transfer models](https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style) from ONNX, and `inputImage` to a texture. [Check the Texture import settings](convert-texture-to-tensor.md) to make sure the texture matches the shape and layout the model needs.

```
using UnityEngine;
using Unity.Sentis;

public class StyleTransfer : MonoBehaviour
{
    public ModelAsset modelAsset;
    public Model runtimeModel;
    private IWorker worker;
    public Texture2D inputImage;
    public RenderTexture outputTexture;

    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        // Create the input tensor from the texture
        TensorFloat inputTensor = TextureConverter.ToTensor(inputImage);

        // Run the model and get the output as a tensor
        worker.Execute(inputTensor);
        TensorFloat outputTensor = worker.PeekOutput() as TensorFloat;

        // Create a tensor operation to divide the output values by 255, to remap to the (0-1) color range
        Ops ops = WorkerFactory.CreateOps(BackendType.GPUCompute, new TensorCachingAllocator());
        TensorFloat tensorScaled = ops.Div(outputTensor, 255f);

        // Copy the rescaled tensor to the screen as a texture
        TextureConverter.RenderToScreen(tensorScaled);
    }

    void OnDisable()
    {
        worker.Dispose();
    }
}
```


If you use the Universal Render Pipeline (URP) or the High-Definition Render Pipeline (HDRP), you must call `RenderToScreen` in the `RenderPipelineManager.endFrameRendering` or `RenderPipelineManager.endContextRendering` callbacks. Refer to [Rendering.RenderPipelineManager](https://docs.unity3d.com/ScriptReference/Rendering.RenderPipelineManager.html) for more information.

Refer to the `Copy a texture tensor to the screen` example in the [sample scripts](package-samples.md) for an example.

## Override shape and layout

You can use a `TextureTransform` object to override the properties of the texture. For example, the following code changes or "swizzles" the order of the texture channels to blue, green, red, alpha:

```
// Create a TextureTransform that swizzles the order of the channels of the texture
TextureTransform swizzleChannels = new TextureTransform().SetChannelSwizzle(ChannelSwizzle.BGRA);

// Convert the tensor to a texture using the TextureTransform object
TextureConverter.RenderToTexture(outputTensor, rt, swizzleChannels);
``` 

You can also chain operations together.

```
// Create a TextureTransform that swizzles the order of the channels of the texture and changes the size
TextureTransform swizzleChannelsAndChangeSize = new TextureTransform().SetChannelSwizzle(ChannelSwizzle.BGRA).SetDimensions(width: 256, height: 320);

// Convert the tensor to a texture using the TextureTransform object
TextureConverter.RenderToTexture(outputTensor, rt, swizzleChannelsAndChangeSize);
```

Refer to the [TextureTransform](xref:Unity.Sentis.TextureTransform) API for more information.

### Read a tensor in the correct format

When you convert a tensor to a texture, Sentis reads the tensor with a batch size, channels, height, width (NCHW) layout.

If the tensor is a different layout, use [`TextureTransform.SetTensorLayout`](xref:Unity.Sentis.TextureTransform.SetTensorLayout(Unity.Sentis.TensorLayout)) to make sure Sentis reads the tensor correctly.

Refer to [Tensor fundamentals in Sentis](tensor-fundamentals.md) for more information about tensor formats.

## Additional resources

- [Get output from a model](get-the-output.md)
- [Create and modify tensors](do-basic-tensor-operations.md)
- [Do operations on tensors](do-complex-tensor-operations.md)

