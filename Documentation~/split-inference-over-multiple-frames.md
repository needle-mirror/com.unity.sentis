# Split inference over multiple frames

To run a model one layer at a time, use the [`StartManualSchedule`](xref:Unity.Sentis.GenericWorker.StartManualSchedule) method of the worker. This method creates an `IEnumerator` object.

For example, if a model takes 50 milliseconds to execute, running it in one frame might cause stuttering or low frame rates in gameplay. Instead, by spreading the execution over 10 frames, you can allocate 5 milliseconds per frame, ensuring smoother performance.

The following code sample runs the model one layer per frame and executes the rest of the `Update` method only after the model finishes.

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
        m_Input = TensorFloat.AllocZeros(new TensorShape(1024));
    }

    void Update()
    {
        if (!m_Started)
        {
            // StartManualSchedule starts the scheduling of the model
            // it returns a IEnumerator to iterate over the model layers, scheduling each layer sequentially
            m_Schedule = m_Engine.ExecuteLayerByLayer(m_Input);
            m_Started = true;
        }

        int it = 0;
        while (m_Schedule.MoveNext())
        {
            if (++it % k_LayersPerFrame == 0)
                return;
        }

        var outputTensor = m_Engine.PeekOutput() as TensorFloat;
        var cpuCopyTensor = outputTensor.ReadbackAndClone();
        // cpuCopyTensor is a CPU copy of the output tensor, you can access it and modify it

        // Set this flag to false if we want to run the network again
        m_Started = false;
        cpuCopyTensor.Dispose();
    }

    void OnDisable()
    {
        // Clean up Sentis resources.
        m_Engine.Dispose();
        m_Input.Dispose();
    }
}
```

For an example, refer to the `Run a model a layer at a time` example in the [sample scripts](package-samples.md).

## Additional resources

- [Run a model](run-a-model.md)
- [Understand models in Sentis](models-concept.md)
- [Create an engine to run a model](create-an-engine.md)
- [Profile a model](profile-a-model.md)
