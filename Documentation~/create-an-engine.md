# Create an engine to run a model

To run a model, you need to create a worker. A worker is the engine that breaks the model down into executable tasks and schedules the tasks to run on a backend, usually the GPU or CPU.

## Create a Worker

Use [`new Worker(...)`](xref:Unity.Sentis.Worker.#ctor*) to create a worker. You must specify a backend type, which tells Sentis where to run the worker and a [runtime model](import-a-model-file.md#create-a-runtime-model).

For example, the following code creates a worker that runs on the GPU using Sentis compute shaders.

```
using UnityEngine;
using Unity.Sentis;

public class CreateWorker : MonoBehaviour
{
    ModelAsset modelAsset;
    Model runtimeModel;
    Worker worker;

    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, BackendType.GPUCompute);
    }
}
```

## Backend types

Sentis provides CPU and GPU backend types. To understand how Sentis executes operations using the different backends, refer to [How Sentis runs a model](how-sentis-runs-a-model.md).

If a backend type doesn't support a Sentis layer in a model, the worker will assert. Refer to [Supported ONNX operators](supported-operators.md) for more information.

Among the backend types, [`BackendType.GPUCompute`](xref:Unity.Sentis.BackendType.GPUCompute) and [`BackendType.CPU`](xref:Unity.Sentis.BackendType.CPU) are the fastest. Use [BackendType.GPUPixel](xref:Unity.Sentis.BackendType.GPUPixel) only if your platform does not support compute shaders. To check if your runtime platform supports compute shaders, use [SystemInfo.supportsComputeShaders](xref:UnityEngine.SystemInfo.supportsComputeShaders).

If you use [`BackendType.GPUCompute`](xref:Unity.Sentis.BackendType.GPUCompute) with the DirectX12 Graphics API on a supported platform, Sentis uses [DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/dml) to accelerate inference. Refer to [Supported ONNX operators](supported-operators.md) for information.

If you use [`BackendType.CPU`](xref:Unity.Sentis.BackendType.CPU) with WebGL, Burst compiles to WebAssembly code which might be slow. For more information, refer to [Getting started with WebGL development](https://docs.unity3d.com/Documentation/Manual/webgl-gettingstarted.html).

The speed of model execution depends on the platform's support for multithreading in Burst or its full support for compute shaders. You can [profile a model](profile-a-model.md) to understand the performance of a model.

## Additional resources

- [Create a runtime model](import-a-model-file.md#create-a-runtime-model)
- [How Sentis runs a model](how-sentis-runs-a-model.md)
- [Supported ONNX operators](supported-operators.md)
- [Run a model](run-a-model.md)
