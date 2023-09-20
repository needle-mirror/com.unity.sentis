# Samples

The Sentis package comes with a set of samples that show examples of using the API. There are both smaller scripts as well as full sample projects.

To find the full sample projects:

- Go to the [Sentis samples github repository](https://github.com/Unity-Technologies/sentis-samples), which will be updated over time with more samples. Some samples contain a helpful video overview linked in the readme file.

To find the sample scripts:

- Go to **Windows > Package Manager**, and select **Sentis** from the package list.
- Select **Samples**.
- To import a sample folder into your Project, select **Import**. Unity creates a `Samples` folder in your Project, and imports the sample folder you selected.

|Sample folder|Description|
|-|-|
|**Add a custom layer**|An example of adding a custom layer to implement a custom ONNX operator.|
|**Check the metadata of a model**|An example of checking the metadata of a model.|
|**Convert tensors to textures**|Examples of converting tensors to textures. Refer to [Use output data](use-model-output.md) for more information. | 
|**Convert textures to tensors**|Examples of converting textures to tensors. Refer to [Create input for a model](create-an-input-tensor.md) for more information. |
|**Copy a texture tensor to the screen**|An example of using `TextureConverter.RenderToScreen` to copy a texture tensor to the screen. Refer to [Use output data](use-model-output.md) for more information. |
|**Do an operation on a tensor**|An example of using `Ops` to do an operation on a tensor. Refer to [Do operations on tensors](do-complex-tensor-operations.md) for more information. |
|**Read output asynchronously**|Examples of reading the output from a model asynchronously using compute shaders. Refer to [Read output from a model asynchronously](read-output-async.md) for more information. |
|**Run a model a layer at a time**|An example of using `StartManualSchedule` to run a model a layer a time. Refer to [Run a model](run-a-model.md) for more information.|`
|**Run a model**|Examples of running models with different numbers of inputs and outputs. Refer to [Run a model](run-a-model.md) for more information.|
|**Use a compute buffer**|An example of using a compute shader to write data to a tensor on the GPU. |
|**Use Burst to write data**|An example of using Burst to write data to a tensor in the Job system. |
|**Use tensor indexing methods**|Examples of using tensor indexing methods to get and set tensor values. |

## Additional resources

- [Unity Discussions group for the Sentis beta](https://discussions.unity.com/c/10)
- [Understand the Sentis workflow](understand-sentis-workflow.md)
- [Understand models in Sentis](models-concept.md)
- [Tensor fundamentals in Sentis](tensor-fundamentals.md)
