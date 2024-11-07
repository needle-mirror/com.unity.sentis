# Workflow example

The following example includes a simple script that takes an image of a handwritten digit and predicts the likelihood of the image representing a digit.

Use this example to [Understand the sentis workflow](understand-sentis-workflow.md).

## Use the example

To use the example, follow these steps:

1. Attach the following script to a GameObject in your scene:

```
using UnityEngine;
using Unity.Sentis;

public class ClassifyHandwrittenDigit : MonoBehaviour
{
    public Texture2D inputTexture;
    public ModelAsset modelAsset;

    Model runtimeModel;
    Worker worker;
    public float[] results;

    void Start()
    {
        Model sourceModel = ModelLoader.Load(modelAsset);

        // Create a functional graph that runs the input model and then applies softmax to the output.
        FunctionalGraph graph = new FunctionalGraph();
        FunctionalTensor[] inputs = graph.AddInputs(sourceModel);
        FunctionalTensor[] outputs = Functional.Forward(sourceModel, inputs);
        FunctionalTensor softmax = Functional.Softmax(outputs[0]);

        // Create a model with softmax by compiling the functional graph.
        runtimeModel = graph.Compile(softmax);

        // Create input data as a tensor
        using Tensor inputTensor = TextureConverter.ToTensor(inputTexture, width: 28, height: 28, channels: 1);

        // Create an engine
        worker = new Worker(runtimeModel, BackendType.GPUCompute);

        // Run the model with the input data
        worker.Schedule(inputTensor);

        // Get the result
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;

        // outputTensor is still pending
        // Either read back the results asynchronously or do a blocking download call
        results = outputTensor.DownloadToArray();
    }

    void OnDisable()
    {
        // Tell the GPU we're finished with the memory the engine used
        worker.Dispose();
    }
}
```

2. Download a handwriting recognition ONNX model file from the [ONNX Model Zoo](https://github.com/onnx/models) (GitHub). For example, the [MNIST Handwritten Digit Recognition model](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist) mnist-8.onnx.
3. Drag the downloaded model file into the `Assets` folder of the **Project** window.
4. Open the **Inspector** window of the GameObject and drag the model asset into the **Model Asset** field.
5. Download the `digit.png` image below and drag it into the `Assets` folder of the **Project** window.

   ![A handwritten number 7](images/digit.png)

6. Open the **Inspector** window of the imported image.
7. In **Import Settings**, expand the **Advanced** section to reveal more settings.
8. Set **Non-Power of 2** to `None` and click **Apply**.
9. Open the **Inspector** window of the GameObject and drag the **digit** asset into the **Input Texture** field.
10. Click **Play** to run the project.

In the **Inspector** window of the GameObject, each item of the **Results** array shows how highly the model predicts the image to be a digit. For example, item 0 of the array represents how highly the model predicts the image being a handwritten zero.

## Additional resources
- [Samples](package-samples.md)
- [Understand the sentis workflow](understand-sentis-workflow.md)
- [Create a model](create-a-model.md)

