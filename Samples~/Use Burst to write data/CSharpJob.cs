using Unity.Sentis;
using UnityEngine;
using Unity.Burst;
using Unity.Jobs;
using Unity.Collections;

public class CSharpJob : MonoBehaviour
{
    [SerializeField]
    ModelAsset modelAsset;

    Tensor m_Input;
    IWorker m_Engine;

    void OnEnable()
    {
        // Everything that can be statically assigned is setup during Start to avoid memory churn.
        var model = ModelLoader.Load(modelAsset);
        m_Input = TensorFloat.Zeros(new TensorShape(1024));
        m_Engine = WorkerFactory.CreateWorker(BackendType.CPU, model);
    }

    void OnDisable()
    {
        m_Input.Dispose();
        m_Engine.Dispose();
    }

    [BurstCompile]
    struct SimpleJob : IJobParallelFor
    {
        [Unity.Collections.LowLevel.Unsafe.NativeDisableUnsafePtrRestriction]
        public NativeArray<float> data;
        public void Execute(int i)
        {
            data[i] = 43.0f;
        }
    }

    void Update()
    {
        var burstTensorDataX = BurstTensorData.Pin(m_Input);
        SimpleJob job = new SimpleJob() { data = burstTensorDataX.array.GetNativeArrayHandle<float>() };

        // Set the fence on the input so Sentis doesn't execute until the job is complete.
        burstTensorDataX.fence = job.Schedule(burstTensorDataX.array.Length, 64);

        m_Engine.Execute(m_Input);

        // Peek the value from Sentis, without taking ownership of the Tensor (see PeekOutput docs for details).
        var outputTensor = m_Engine.PeekOutput() as TensorFloat;

        // Note that accessing the data via [] operator will block until all work is complete.
        // See the AsyncReadback samples for details on how to avoid blocking.
        Debug.Assert(outputTensor[1023] == 42.0f);
        Debug.Log($"Burst {outputTensor[1023]}");
    }
}

