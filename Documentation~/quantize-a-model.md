# Quantize a Model

Sentis imports model constants and weights as 32-bit values. If you wish to reduce the storage space required by your model on disk and in memory, you may consider model quantization.

Quantization works by representing the weight values in a lower precision format, and then casting back to a high precision format before executing the operation.

## Quantization types

Sentis currently supports the following quantization types.

| Quantization type | Bits per value | Description                                                                                                |
|-------------------|----------------|------------------------------------------------------------------------------------------------------------|
| None              | 32 bit         | The values are stored in full precision.                                                                   |
| Float16           | 16 bit         | The values are cast to a 16 bit floating point value, model accuracy is often close to the original model. |
| Uint8             | 8 bit          | The values are quantized linearly as fixed point values between the highest and lowest value in the data set, model accuracy may degrade severely |

Reducing the number of bits per value decreases the disk and memory usage of your model, while the inference speed remains almost unchanged.

Sentis only quantizes float weights used as inputs to specific operations (Dense, MatMul, Conv, etc.), leaving integer constants unchanged.

The impact on model accuracy varies depending on the model type. The most effective way to assess whether model quantization is appropriate for your needs is through testing.

## Quantizing a loaded model

Use the [`ModelQuantizer`](xref:Unity.Sentis.Quantization.ModelQuantizer) API to quantize a model and the [`ModelWriter`](xref:Unity.Sentis.ModelWriter) API to save the quantized model to disk.

```
using Unity.Sentis;

void QuantizeAndSerializeModel(Model model, string path)
{
    // Sentis destructively edits the source model in memory when quantizing.
    ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model);

    // Serialize the quantized model to a file.
    ModelWriter.Save(path, model);
}
```

