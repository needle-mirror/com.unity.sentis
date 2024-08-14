# Split inference over multiple frames

To run a model one layer at a time, use the [`ScheduleIterable`](xref:Unity.Sentis.Worker.ScheduleIterable) method of the worker. This method creates an `IEnumerator` object.

For example, if a model takes 50 milliseconds to execute, running it in one frame might cause stuttering or low frame rates in gameplay. Instead, by spreading the execution over 10 frames, you can allocate 5 milliseconds per frame, ensuring smoother performance.

The following code sample runs the model one layer per frame and executes the rest of the `Update` method only after the model finishes.

```
// Set a larger number for faster GPUs
const int k_LayersPerFrame = 20;

IEnumerator m_Schedule;
bool m_Started = false;

void Update()
{
    if (!m_Started)
    {
        // ExecuteLayerByLayer starts the scheduling of the model
        // It returns an IEnumerator to iterate over the model layers and schedule each layer sequentially
        m_Schedule = m_Worker.ScheduleIterable(m_Input);
        m_Started = true;
    }

    int it = 0;
    while (m_Schedule.MoveNext())
    {
        if (++it % k_LayersPerFrame == 0)
            return;
    }

    var outputTensor = m_Worker.PeekOutput() as Tensor<float>;
    var cpuCopyTensor = outputTensor.ReadbackAndClone();
    // cpuCopyTensor is a CPU copy of the output tensor. You can access it and modify it

    // Set this flag to false to run the network again
    m_Started = false;
    cpuCopyTensor.Dispose();
}
```

For an example, refer to the `Run a model a layer at a time` example in the [sample scripts](package-samples.md).

## Additional resources

- [Run a model](run-a-model.md)
- [Understand models in Sentis](models-concept.md)
- [Create an engine to run a model](create-an-engine.md)
- [Profile a model](profile-a-model.md)
