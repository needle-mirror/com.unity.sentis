# Run a model a layer at a time

To run a model a layer at a time, use the `StartManualSchedule` method of the worker. The method creates an `IEnumerator` object.

For example a model may take 50 milliseconds to execute. Running execution in a single frame would cause low or stuttering framerates in gameplay. Splitting the model to run across 10 frames could ideally spend 5 milliseconds of execution per frame allowing for smooth framerates.

The following code sample runs the model one layer per frame, and executes the rest of the `Update` method only after the model finishes.

```
using UnityEngine;
using Unity.Sentis;
using System;
using System.Collections;

public class ModelExecutionInParts : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;
    IWorker m_Engine;
    Tensor m_Input;

    // Set this number higher for faster GPUs
    const int k_LayersPerFrame = 20;

    IEnumerator m_Schedule;
    bool m_Started = false;

    void OnEnable()
    {
        var model = ModelLoader.Load(modelAsset);
        m_Engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        m_Input = TensorFloat.Zeros(new TensorShape(1024));
    }

    void Update()
    {
        if (!m_Started)
        {
            // StartManualSchedule starts the scheduling of the model
            // it returns a IEnumerator to iterate over the model layers, scheduling each layer sequentially
            m_Schedule = m_Engine.StartManualSchedule(m_Input);
            m_Started = true;
        }

        int it = 0;
        while (m_Schedule.MoveNext())
        {
            if (++it % k_LayersPerFrame == 0)
                return;
        }

        var outputTensor = m_Engine.PeekOutput() as TensorFloat;
        outputTensor.MakeReadable();

        // Set this flag to false if we want to run the network again
        m_Started = false;
    }

    void OnDisable()
    {
        // Clean up Sentis resources.
        m_Engine.Dispose();
        m_Input.Dispose();
    }
}
```

Refer to the `Run a model a layer at a time` example in the [sample scripts](package-samples.md) for an example.

## Additional resources

- [Run a model](run-a-model.md)
- [Understand models in Sentis](models-concept.md)
- [Create an engine to run a model](create-an-engine.md)
- [Profile a model](profile-a-model.md)
