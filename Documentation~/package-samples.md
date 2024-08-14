# Samples

Samples are provided with the Sentis package to illustrate how to use the API and facilitate the learning process of using Sentis.

The available samples are:
- [Sample projects](#sample-projects) from the Sentis GitHub
- [Sample scripts](#sample-scripts) from the Package Manager

Validated models are also available to use in your project. To understand and download available models, refer to [Supported models](supported-models.md).

## Sample projects

Full sample projects are available to demonstrate various Sentis use cases.

To access complete sample projects, visit the [Sentis samples](https://github.com/Unity-Technologies/sentis-samples) GitHub repository. This repository will be updated over time with more samples. Some samples contain a helpful video overview linked in the `readme` file.

## Sample scripts

Use the sample scripts to implement specific features in your own project.

To find the sample scripts:

1. Go to **Window** > **Package Manager**, and select **Sentis** from the package list.
2. Select **Samples**.

To import a sample folder into your project, select **Import**. Unity creates a `Samples` folder in your project, and imports the sample folder you selected.

The following table describes the available samples:

| Sample folder                           | Description                                                                                                                                                                                      |
|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| |
| **Convert tensors to textures**         | Examples of converting tensors to textures. Refer to [Use output data](use-model-output.md) for more information.                                                                                |
| **Convert textures to tensors**         | Examples of converting textures to tensors. Refer to [Create input for a model](create-an-input-tensor.md) for more information.                                                                 |
| **Copy a texture tensor to the screen** | An example of using [`TextureConverter.RenderToScreen`](xref:Unity.Sentis.TextureConverter.RenderToScreen*) to copy a texture tensor to the screen. Refer to [Use output data](use-model-output.md) for more information.                              |
| **Encrypt a model**                     | Example of serializing an encrypted model to disk using a custom editor window and loading that encrypted model at runtime. Refer to [Encrypt a model](encrypt-a-model.md) for more information. |
| **Quantize a model**                    | Example of serializing a quantized model to disk using a custom editor window and loading that quantized model at runtime. Refer to [Quantize a model](quantize-a-model.md) for more information.                 |
| **Read output asynchronously**          | Examples of reading the output from a model asynchronously using compute shaders. Refer to [Read output from a model asynchronously](read-output-async.md) for more information.                 |
| **Run a model a layer at a time**       | An example of using [`ScheduleIterable`](xref:Unity.Sentis.Worker.ScheduleIterable*) to run a model a layer a time. Refer to [Run a model](run-a-model.md) for more information.                                                            |
| **Run a model**                         | Examples of running models with different numbers of inputs and outputs. Refer to [Run a model](run-a-model.md) for more information.                                                            |
| **Use the functional API with an existing model**               | An example of using the functional API to extend an existing model. Refer to [Edit a model](edit-a-model.md) for more information.  |
| **Use a compute buffer**                | An example of using a compute shader to write data to a tensor on the GPU.                                                                                                                       |
| **Use Burst to write data**             | An example of using Burst to write data to a tensor in the Job system.                                                                                                                           |
| **Use tensor indexing methods**         | Examples of using tensor indexing methods to get and set tensor values.                                                                                                                          |

## Additional resources

- [Unity Discussions group for the Sentis beta](https://discussions.unity.com/c/10)
- [Understand the Sentis workflow](understand-sentis-workflow.md)
- [Understand models in Sentis](models-concept.md)
- [Tensor fundamentals in Sentis](tensor-fundamentals.md)
- [Supported models](supported-models.md)
