## Use a command buffer

You can use a command buffer to create a queue of Sentis commands, then run the commands on the GPU later.

Follow these steps:

1. Create or get an empty command buffer.
2. [Create a worker](create-an-engine.md) with the `BackendType.GPUCommandBuffer` back end type.
3. In a Unity event function, for example [`OnRenderImage`](https://docs.unity3d.com/ScriptReference/MonoBehaviour.OnRenderImage.html), add Sentis methods to the command buffer.
4. Use `Graphics.ExecuteCommandBuffer` to execute the command buffer.

To add commands to the command buffer, do any of the following:

- Use a Sentis API that takes the command buffer as a parameter, for example `TextureConverter.ToTensor`.
- Use the `ExecuteWorker` method on the command buffer to add a command that runs the model.

For example, the following code creates and execute a command buffer queue that converts a render texture to an input tensor, runs the model, then converts the output tensor to a render texture.

```
using UnityEngine;
using Unity.Sentis;
using UnityEngine.Rendering;

public class UseCommandBuffer : MonoBehaviour
{
    public ModelAsset modelAsset;
    IWorker worker;

    void Start()
    {
        // Create a worker that uses the GPUCommandBuffer back end type
        worker = WorkerFactory.CreateWorker(BackendType.GPUCommandBuffer, ModelLoader.Load(modelAsset));
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        // Create a new command buffer
        CommandBuffer commandBuffer = new CommandBuffer();

        // Add a command to the command buffer that creates an input tensor from the source texture data
        // you can also call TextureConverter.ToTensor(commandBuffer, source, inputTensor) if you allready have the tensor allocated
        Tensor inputTensor = TextureConverter.ToTensor(commandBuffer, source);

        // Add a command to the command buffer that runs the model using the input tensor
        commandBuffer.ExecuteWorker(worker, inputTensor);

        // Add a command to the command buffer that blits the output to the destination render texture
        TextureConverter.RenderToTexture(commandBuffer, worker.PeekOutput() as TensorFloat, destination);

        // Execute all the commands in the command buffer
        Graphics.ExecuteCommandBuffer(commandBuffer);

        // Clear the input tensor to free its memory.
        inputTensor.Dispose();
    }

    void OnDisable()
    {
        worker.Dispose();
    }
}
```

For more information, refer to:

- [Extending the Built-in Render Pipeline with CommandBuffers](https://docs.unity3d.com/Documentation/Manual/GraphicsCommandBuffers.html)
- The [Unity Discussions group for the Sentis beta](https://discussions.unity.com/c/10), which has a full sample project that uses command buffers.
