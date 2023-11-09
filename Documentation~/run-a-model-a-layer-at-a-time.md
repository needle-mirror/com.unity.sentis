# Run a model a layer at a time

To run a model a layer at a time, use the `StartManualSchedule` method of the worker. The method creates an `IEnumerator` object.

For example a model may take 50 milliseconds to execute. Running execution in a single frame would cause low or stuttering framerates in gameplay. Splitting the model to run across 10 frames could ideally spend 5 milliseconds of execution per frame allowing for smooth framerates.

The following code sample runs the model one layer per frame, and executes the rest of the `Update` method only after the model finishes.

```
using UnityEngine;
using Unity.Sentis;
using System.Collections;

public class StartManualSchedule : MonoBehaviour
{
    public ModelAsset modelAsset;
    IWorker worker;
    Tensor inputTensor;

    public Texture2D inputTexture;
    public RenderTexture outputTexture;

    IEnumerator modelEnumerator;
    bool started = false;
    bool hasMoreModelToRun = true;

    void Start()
    {
        Model runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);
        inputTensor = TextureConverter.ToTensor(inputTexture);
    }

    void Update()
    {
        if (!started)
        {
            modelEnumerator = worker.StartManualSchedule(inputTensor);
            started = true;
        }

        // Iterate running the model once per frame
        // In each iteration of the do-while loop, use the IEnumerator object to run the next layer of the model
        if (hasMoreModelToRun)
        {
            // Use MoveNext() to run the next layer of the model. MoveNext() returns true if there's more work to do
            hasMoreModelToRun = modelEnumerator.MoveNext();

            // Log the progress so far as a float value. 0 means no progress, 1 means complete
            Debug.Log(worker.scheduleProgress);
        }

        else
        {
            // Get the output tensor
            var outputTensor = worker.PeekOutput() as TensorFloat;
            // Move the output tensor to CPU before reading the data
            outputTensor.MakeReadable();
            float[] outputArray = outputTensor.ToReadOnlyArray();

            // turn off the component once the tensor is obtained
            enabled = false;
        }
    }

    void OnDisable()
    {
        worker.Dispose();
        inputTensor.Dispose();
    }
}
```

Refer to the `Run a model a layer at a time` example in the [sample scripts](package-samples.md) for an example.

## Additional resources

- [Run a model](run-a-model.md)
- [Understand models in Sentis](models-concept.md)
- [Create an engine to run a model](create-an-engine.md)
- [Profile a model](profile-a-model.md)
