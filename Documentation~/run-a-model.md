# Run a model

After you [create a worker](create-an-engine.md), use `Execute` to run the model.

```
worker.Execute(inputTensor);
```

You can enable verbose mode when you create a worker. Sentis logs execution to the Console window when you run the model.
```
worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel, verbose: true);
```

When you run a model for the first time in the Unity Editor, it might be slow because Sentis needs to compile code and shaders. Later runs are faster.

Refer to the `Run a model` example in the [sample scripts](package-samples.md) for an example.

## Run a model a layer at a time

To run a model a layer at a time, use the `StartManualSchedule` method of the worker. The method creates an `IEnumerator` object.

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

- [Understand models in Sentis](models-concept.md)
- [Create an engine to run a model](create-an-engine.md)
- [Profile a model](profile-a-model.md)
