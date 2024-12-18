# Changelog
All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [2.1.2] - 2024-12-18

### Added
- Support for accelerated inference using [DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/dml) in future Unity versions
- Support for RMSNormalization subgraph optimization

### Changed
- Some methods in the TextureConverter have been made obsolete

### Fixed
- Improved CPU backend performance by optimizing Burst job count
- Reduced memory usage when serializing and deserializing large models
- Shader issues on XBOX hardware
- GPUPixel now reuses RenderTextures reducing garbage collection

## [2.1.1] - 2024-11-07

### Fixed
- Ordering of inputs now matches ONNX model consistently
- Loading of models with constants over 500mb
- Serialization of models over 2GB in size
- Issues in example code in documentation
- Shape inference for `Split` and `Select` operators
- Profiler tags for operators are now consistent with lower overhead
- `CopyOutput` can now be used on GPUPixel backend if the tensor is the correct shape

## [2.1.0] - 2024-09-09

### Updated
- Fast activation fusing for MatMul
- Batched MatMul optimization
- Faster Upload for CPUTensorData

### Fixed
- CPU Tensors were not properly disposed.
- Reshaping of empty tensor
- Exposed internal fields of TextureTensorData

## [2.0.0] - 2024-08-14

### Added
- Callbacks for retrieving the ONNX metadata at import time.
- `Upload` method to `Tensor` to allow for direct upload of data.
- `IndexSelect` functional operator.
- `scoreThreshold` parameter for `NMS` functional method.
- `Tensor.Reshape` accepts tensor shapes of different lengths, provided the capacity of the tensor data is large enough.
- `Worker.CopyOutput` method for copying outputs into allocated tensors.
- `FunctionalTensor.ToString` shows the data type and shape (if known) for debugging when using the functional API.

### Changed
- Made a generic `Tensor` class so that you can declare `Tensor<float>` and `Tensor<int>` rather than `TensorFloat` and `TensorInt`.
- Reworked and unified the `Worker` methods for setting inputs and scheduling.
- `IWorker` and `GenericWorker` to be an instantiable `Worker` class.
- Renamed `Execute` to `Schedule` throughout for clarity.
- Renamed `BurstTensorData` to `CPUTensorData` for consistency.
- Renamed and reworked `SymbolicTensorShape` as `DynamicTensorShape` for clarity and ease of use.
- Reworked the `Functional.Compile` method as methods on a `FunctionalGraph` object, for clearer declaration of inputs and outputs.

### Fixed
- Issues with the `Slice` operator on zero-length dimensions.
- Compilation of the `GridSample` shaders on consoles.

### Removed
- `GPUCommandBuffer` backend, in favour of a unified GPUCompute backend using command buffers.
- `IBackend` methods, use the functional API to create and edit models instead.
- Many methods and classes from the public API to keep it clear and maintainable.

## [1.6.0-pre.1] - 2024-07-18

### Added
- SliceSet backend method for Concat and other operations.
- Support for GridSample ONNX operator, Functional API and backend method, PaddingMode Enum for use with GridSample.
- Methods to IModelStorage to allow retrieval of data types, tensor shapes and CPU values.
- CPU backend to ExecutionContext for CPU fallback execution.
- Optimization that replaces a Gather with a single index by a Split or Narrow.
- Functional RandomChoice methods similar to numpy.
- BitonicSort method for fast GPU sorting.

### Changed
- Model Input, Output, Layer and Constant indices are stored as integers rather than strings.
- Min and Max no longer take more than 2 inputs in the backend, for more inputs use repeated application of the backend calls.
- Optimized loading of compute functions for performance.
- Reduced CPU allocations inside operations to avoid GC.
- CPU fallback pass runs at worker instantiation rather than being serialized to the model.

### Fixed
- Many issues with Tensors of zero length and uploading/downloading data. They no longer have null backendData.
- Issue where multiple Random layers were sometimes incorrectly collapsed to single layer by optimization pass.
- NonMaxSuppression to have fast inference on CPU and GPUCompute backends.
- Error messages for model deserialization.
- Slice inference issues for slices of length 0.

### Removed
- Mean, Sum, Concat from the backend. Add, ScalarMad and SliceSet operations are used instead.
- Unnecessary 'keepdim' argument from Reduce backend ops.
- Constructor for Constant with a tensor argument.
- CompleteOperationsAndDownload and similar methods from Tensor.

## [1.5.0-pre.3] - 2024-06-10

## [1.5.0-pre.2] - 2024-05-21
### Fixed
- Fix linker error when doing a build

## [1.5.0-pre.1] - 2024-05-08

### Changed
- Better main thread cpu performance
- Faster unary pointwise operations
- Faster model import

### Fixed
- Reworked project dependencies to reduce build size
- Unconnected input tensors are handled properly
- Better ONNX opset 18 support
- Multinomial randomness is coherent across frames
- Fixed shader compilation errors on XBOX and Switch

## [1.4.0-pre.3] - 2024-04-09

### Fixed
- Outdated documentation.
- Error message for importing incompatible Sentis models.
- NonMaxSuppression output tensor has correct shape.
- Output of Size operator.

### Removed
- Non-working NonMaxSuppression GPUCompute optimizations.

## [1.4.0-pre.2] - 2024-04-06

### Fixed
- Broken tests.
- Outdated documentation.

## [1.4.0-pre.1] - 2024-04-05

### Added
- Fast path for Scatter and Gather ops.
- Quantization API for quantizing model weights to Float16 or Uint8.
- TensorByte and TensorShort tensor types for quantization.
- Pad operator to support 'axes' input, 'TensorInt' input and 'wrap' mode.
- GeluFast op for tiny stories optimization.
- Tensor 'Reshape' method for changing shape without backend.
- Functional API for compiling models with torch-like syntax.
- Analytics reporting on model import.

### Changed
- Reworked async API with awaitable methods on tensor readback.
- Reworked tensor allocation scheme.
- Renamed tensor fields and methods.
- Reworked model serialization to use flatbuffers.
- Random layers accept integer seed rather than float.
- Improved NonMaxSuppression inference drastically and added GPUCompute backend.

### Fixed
- Dispatch limit issues on Split operator.
- Optimization pass where Dense has transposed weights.
- Import settings for Resize op on opset 10.
- Links to external sources in docs.
- Edge cases in ScatterElements and GatherElements infer correctly.
- Conv operator going out of bounds on GPUCompute and GPUCommandBuffer.

### Removed
- Broken optimization pass for Dense > ScaleBias.
- ArrayTensorData class and methods.
- 'CreateWorker' method from model extensions.
- Mapping of param symbolic tensor dims to original names.
- Model API for adding inputs, constants, layers to model (in favour of Functional API).
- Ops for tensor operations (in favour of Functional API).


## [1.3.0-pre.3] - 2024-01-17

### Added
- LoadModelDesc method in ModelLoader now public
- TensorInt data type for Clip
- Docs page for exporting models to ONNX format
- Model sources in documentation
- Editor menu with links to documentation and project submission form

### Changed
- clearOnInit default to false in tensor data pinning
- NonZero asserts on command buffer as unsupported
- Improved inspector for custom layers and allow horizontal scrolling
- Rewrote docs and sample code for model execution in parts
- Package description with links to project submission and email

### Fixed
- Thread group count for Split
- Importer no longer fails when inspector is out of focus
- Docstrings for accuracy
- Broken links in documentation to onnx model zoo
- Dense import step with transpose
- Warnings for async methods in tensor data classes
- Resize opset 10 imports with correct settings

### Removed
- Broken Dense to ScaleBias optimization


## [1.3.0-pre.2] - 2023-11-21

### Fixed
- Dependency not declared in package manifest


## [1.3.0-pre.1] - 2023-11-09

### Added
- Mad (multiply add) method added to Ops
- Multinomial method added to Ops
- TextureConverter ToTensor methods can accept allocated tensors as parameters
- AddInput method with TensorShape parameter added to Model
- UnknownOfRank method added to SymbolicTensorShape
- Save method with Stream added to ModelWriter
- SingleHeadAttention fused layer
- Importer can handle external weight files for large models
- Encrypted model sample project

### Changed
- Many Ops methods now accept generic Tensor types and infer output types
- Added kernel shape to Conv layer
- ONNXModelConverter takes model file path when instantiating
- Concat parameter ordering in Ops method

### Fixed
- GatherElements inference fixed when not gathering on innermost axis
- Optimizer pass for Einsum no longer breaks model
- Reduce operator no longer incorrectly optimizes to Identity in edge case
- ArgMax and ArgMin output data types correctly inferred as int
- Pad import with empty constant pad fixed
- Optimizer pass for removing duplicate layers and constants
- Internal dlls no longer visible from outside package and no longer cause conflicts

### Removed
- Dependencies on experimental code


## [1.2.0-exp.2] - 2023-09-20

### Added
- Fast inference path for 1D Resize
- OneHot operator now supports float tensors properly
- Import for LayerNormalization operator
- Import for Gelu operator
- Optimized ScalarMad layer with optimizer passes

### Removed
- Watermark on Unity scenes referencing Sentis
- License key requirement for Sentis

### Changed
- Moved CPU fallback pass to mandatory optimizer passes and remove MakeReadable calls in layer inference
- Binary search for allocator speedup
- Cached compute func instances for inference speedup
- Refactored AxisNormalization to be LayerNormalization

### Fixed
- Import settings for Upsample operator
- Condition for fast inference path for 2D Resize on CPU
- Inspector no longer crashes when model can't be imported
- Serialize to streaming assets now correctly uses optimal path and doesn't crash the editor with large models
- Serialize to streaming assets creates directory if it doesn't exist and refreshes asset database


## [1.1.1-exp.2] - 2023-09-04

### Fixed
- Fixed inference with multiple Conv layers with different attributes in same model
- Fixed inference for AveragePool and MaxPool with large channel counts on GPUPixel
- Fixed batched MatMul inference on GPUCompute and GPUCommandBuffer
- Fixed Conv inference with no bias on GPUCompute and GPUCommandBuffer
- Fixed inspector view when model can't be loaded
- Made Mode.Load(FileStream) added


## [1.1.1-exp.1] - 2023-08-31

### Fixed
- Fixed faulty Transpose optimization
- Fixed documentation titles
- Fixed problem in BatchNormalization optimization pass
- Fixed ReduceMean inference error
- Fixed Reduce issues on Unity 2021.3.30f on iOS devices


## [1.1.0-exp.2] - 2023-08-17

### Changed
- Documentation was updated to 1.1.0


## [1.1.0-exp.1] - 2023-08-14

### Added
- Added AsyncReadbackRequest and IsAsyncReadbackRequestDone for asynchronous readback from GPU
- Added Shrink, Abs (int), Neg (int), Not, Sign, PRelu, Hardmax, OneHot to GPUPixel backend
- Added And, Equal, Greater, GreaterOrEqual, IsInf, IsNaN, Less, LessOrEqual, Or, Xor, Where to GPUPixel backend
- Added Add (int), Sub (int), Div (int), Pow (int), FMod (int), Mod (int), FMod (int), Mul(int), Sum (int), Min (int), Max (int) to GPUPixel backend
- Added Expand (int), Concat (int), Slice (int), Transpose (int), Transpose (no permutations), Split (int), Tile, Resize (3D), Reshape (int) to GPUPixel backend
- Added ReduceMax (int), ReduceMin (int), ReduceProd (int), ReduceSum (int), ReduceL1 (int), ReduceSumSquare (int), ArgMax, ArgMin to GPUPixel backend
- Added Gather, GatherElements, GatherND, ScatterElements, ScatterND to GPUPixel backend
- Added InstanceNormalization, AxisNormalization, RoiAlign, RandomUniform, RandomNormal, Bernoulli, Range, Trilu, CumSum to GPUPixel backend
- Added MaxPool (1D), AveragePool (1D) to GPUPixel backend
- Added Conv (1D and 3D) to GPUPixel backend
- Added BatchNormalization layer and op
- Added CreateBackend method to BackendFactory
- Added SimplifyReshapeInputPass to optimizer to reduce shapes to constants when they are inputs to Reshape layer
- Added data type inference and added data types to outputs in inspector
- Added optimization passes to simplify and/or remove trivial Reshape, Expand, Tile, Reduce, Cast and CastLike layers
- Added CustomLayer class to inherit custom layers from
- Added 'Serialize To StreamingAssets' option for model asset
- Added method to ModelLoader to Load from path
- Added methods to ModelWriter to SaveModelDesc and SaveModelWeights to memory streams
- Added 'Add', 'Div', 'Sub' and 'Mul' utility methods to Ops for operations between tensors and floats
- Added 'Set' utility method to Ops to set a slice of a tensor from another tensor similar to NumPy

### Removed
- Removed ScheduleAsyncDownload for asynchronous readback from GPU
- Removed internal CPU cache for Tensors, Tensor on GPU must call MakeReadable to access tensor with indexes

### Changed
- Split IOps interface to Ops class with helper functions and IBackend interface with backend methods
- Improved Documentation for style, accuracy and clarity
- Renamed uploadCache to clearOnInit for the Pin method
- Changed Ops with int[] and float[] inputs to accept Span and ReadOnlySpan to reduce allocation
- ConvTranspose now imports with default pads and strides when they aren't provided
- Optimized Broadcast ops speed (Add, Div, Sub, Mul, Pow...) on GPUCompute and GPUCommandBuffer
- Optimized Reduce ops speed (ReduceMax, ReduseSum....) on GPUCompute and GPUCommandBuffer
- Optimized Transpose ops speed on GPUCompute and GPUCommandBuffer
- Optimized LSTM speed on GPUCompute and GPUCommandBuffer
- Made RoundDenormalWeightsPass part of optimization passes and removed option from inspector
- Unified shape inference and partial tensor inference to improve import time and better reduce number of layers in optimized model
- Duplicated constants when used on both GPU and CPU to avoid unnecessary uploads and readbacks
- Reduced memory and CPU overhead in model importer
- Changed Conv and ConvTranspose to accept dynamic weight tensor and use "kernel_shape" parameter for shape inference
- PeekOutput in IWorker no longer accepts 'prepareCacheForAccess' parameter (use MakeReadable on returned tensor instead)

### Fixed
- Fixed MatMul for 1D input tensors
- Fixed ConvTranspose to work with non 2D cases
- Fixed Conv to accept null bias tensor
- Fixed GPUPixel backend to work with TensorInts
- Fixed import crash for model constants with zero length


## [1.0.0-exp.6] - 2023-06-26

### Added
- First beta build.

#Contributors
- Alexandre Ribard
- Aurimas Petrovas
- Bob Donovan
- Giles Coope
- Jeffrey Rainy
- Mark Green
