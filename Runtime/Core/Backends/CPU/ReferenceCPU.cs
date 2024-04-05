using System;
using UnityEngine;

namespace Unity.Sentis {

/// <summary>
/// Represents a CPU backend ops.
/// </summary>
public partial class CPUBackend : IBackend
{
    /// <inheritdoc/>
    public BackendType backendType => BackendType.CPU;

    /// <summary>
    /// Initializes and returns an instance of `CPUBackend`.
    /// </summary>
    public CPUBackend() {}

    /// <summary>
    /// Disposes of the ops and any associated memory.
    /// </summary>
    public void Dispose()
    {
        m_MemoryPool?.Dispose();
        m_MemoryPool = null;
    }

    void ConvND(TensorFloat X, TensorFloat K, TensorFloat B, TensorFloat O, int groups, Span<int> stride, Span<int> pad, Span<int> dilation, Layers.FusableActivation fusedActivation)
    {
        var Otmp = (fusedActivation != Layers.FusableActivation.None) ? AllocTensorFloat(O.shape) : O;

        BurstTensorData.Pin(X);
        BurstTensorData.Pin(K);
        BurstTensorData.Pin(B);
        BurstTensorData.Pin(Otmp);

        int inputGroupedChannels = X.shape[1] / groups;
        int outputGroupedChannels = Otmp.shape[1] / groups;

        var itK = new TensorNDIterator(K.shape);
        itK = itK.RemoveDim(0);
        itK = itK.RemoveDim(0);

        var itX = new TensorNDIterator(X.shape);
        for (var itO = new TensorNDIterator(Otmp.shape); itO.HasNext(); itO.MoveNext())
        {
            int n = itO[0];
            int k = itO[1];
            float v = B[k];

            itX[0] = n;

            for (var c = 0; c < inputGroupedChannels; ++c)
            {
                itX[1] = (k / outputGroupedChannels) * inputGroupedChannels + c;

                itK.Reset();
                for (; itK.HasNext(); itK.MoveNext())
                {
                    bool outOfBounds = false;
                    for (int i = 0; i < stride.Length; i++)
                    {
                        int dx = itK[i];
                        int ox = itO[2 + i] * stride[i] + dilation[i] * dx - pad[i];

                        if ((ox < 0) || (ox >= X.shape[2 + i]))
                        {
                            outOfBounds = true;
                            break;
                        }

                        itX[2 + i] = ox;
                    }

                    if (outOfBounds)
                        continue;

                    float xv = X[itX.index];
                    float kv = K[k * K.shape[1] * itK.shape.length + c * itK.shape.length +  itK.index];

                    v += xv * kv;
                }
            }
            Otmp[itO.index] = v;
        }

        if (fusedActivation != Layers.FusableActivation.None)
        {
            ApplyFusedActivation(Otmp, O, fusedActivation);
            ReleaseTensorFloat(Otmp);
        }
    }

    void ConvTransposeND(TensorFloat X, TensorFloat W, TensorFloat B, TensorFloat O, Span<int> strides, Span<int> pads, Span<int> outputPadding, Layers.FusableActivation fusedActivation)
    {
        var Otmp = (fusedActivation != Layers.FusableActivation.None) ? AllocTensorFloat(O.shape) : O;

        BurstTensorData.Pin(X);
        BurstTensorData.Pin(W);
        if (B != null)
            BurstTensorData.Pin(B);
        BurstTensorData.Pin(O);

        var inputChannels = X.shape[1];

        var itK = new TensorNDIterator(W.shape);
        var itX = new TensorNDIterator(X.shape);
        itX = itX.RemoveDim(0);
        itX = itX.RemoveDim(0);

        for (var itO = new TensorNDIterator(O.shape); itO.HasNext(); itO.MoveNext())
        {
            var n = itO[0];
            var k = itO[1];
            var v = B == null ? 0 : B[k];
            itK[1] = k;

            for (var c = 0; c < inputChannels; ++c)
            {
                itK[0] = c;

                itX.Reset();
                for (; itX.HasNext(); itX.MoveNext())
                {
                    var outOfBounds = false;

                    for (var i = 0; i < strides.Length; i++)
                    {
                        var ox = itX[i];
                        var dx = itO[2 + i] + pads[i] - ox * strides[i];

                        if ((dx < 0) || (dx >= W.shape[2 + i]))
                        {
                            outOfBounds = true;
                            break;
                        }

                        itK[2 + i] = dx;
                    }

                    if (outOfBounds)
                        continue;

                    var xv = X[n * X.shape[1] * itX.shape.length + c * itX.shape.length + itX.index];
                    var kv = W[itK.index];

                    v += xv * kv;
                }
            }
            O[itO.index] = v;
        }

        if (fusedActivation != Layers.FusableActivation.None)
        {
            ApplyFusedActivation(Otmp, O, fusedActivation);
            ReleaseTensorFloat(Otmp);
        }
    }

    void ResizeND(TensorFloat X, TensorFloat O, ReadOnlySpan<float> scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode = Layers.NearestMode.RoundPreferFloor, Layers.CoordTransformMode coordTransformMode = Layers.CoordTransformMode.HalfPixel)
    {
        bool firstAlloc = false;
        for (var i = 0; i < scale.Length; i++)
        {
            var Otmp = i == scale.Length - 1 ? O : AllocTensorFloat(ShapeInference.Resize(X.shape, i, scale[i]));
            Resize1D(X, Otmp, i, scale[i], interpolationMode, nearestMode, coordTransformMode);
            if (firstAlloc)
                ReleaseTensorFloat(X);
            X = Otmp;
            firstAlloc = true;
        }
    }

    void Resize1D(TensorFloat X, TensorFloat O, int axis, float scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode, Layers.CoordTransformMode coordTransformMode)
    {
        BurstTensorData.Pin(X);
        BurstTensorData.Pin(O);

        var itX = new TensorNDIterator(X.shape);

        for (var itO = new TensorNDIterator(O.shape); itO.HasNext(); itO.MoveNext())
        {
            itX.CopyNDIndex(itO);

            OpsUtils.GetScaleAndBias(X.shape[axis], O.shape[axis], scale, coordTransformMode, interpolationMode, nearestMode, out float outputScale, out float outputBias);

            float inputCoord = Math.Max(0.0f, itO[axis] * outputScale + outputBias);

            if (interpolationMode == Layers.InterpolationMode.Linear)
            {
                int indexValue = (int)inputCoord;
                float x_c0 = inputCoord - Mathf.Floor(inputCoord);
                float x_c1 = 1.0f - x_c0;

                itX[axis] = Mathf.Clamp(indexValue, 0, X.shape[axis] - 1);
                float x0 = X[itX.index];
                itX[axis] = Mathf.Clamp(indexValue + 1, 0, X.shape[axis] - 1);
                float x1 = X[itX.index];

                O[itO.index] = x_c0 * x1 + x_c1 * x0;
            }
            else
            {
                int indexValue = 0;
                switch (nearestMode)
                {
                    case Layers.NearestMode.RoundPreferFloor:
                    case Layers.NearestMode.Ceil:
                        indexValue = (int)Mathf.Ceil(inputCoord);
                        break;
                    case Layers.NearestMode.RoundPreferCeil:
                    case Layers.NearestMode.Floor:
                        indexValue = (int)Mathf.Floor(inputCoord);
                        break;
                }

                itX[axis] = Mathf.Clamp(indexValue, 0, X.shape[axis] - 1);
                O[itO.index] = X[itX.index];
            }
        }
    }

    void ApplyLocalPoolingOperator(TensorFloat X, TensorFloat O, int[] pool, int[] stride, int[] pad, Func<float> initOp, Func<float, float, float> accumulateOp, Func<float, int, float> normalizeOp)
    {
        BurstTensorData.Pin(X);
        BurstTensorData.Pin(O);

        var itX = new TensorNDIterator(X.shape);
        var itP = new TensorNDIterator(new TensorShape(pool));
        for (var itO = new TensorNDIterator(O.shape); itO.HasNext(); itO.MoveNext())
        {
            itX[0] = itO[0];
            itX[1] = itO[1];

            float acc = initOp();
            int elementCount = 0;

            itP.Reset();
            for (; itP.HasNext(); itP.MoveNext())
            {
                bool outOfBounds = false;
                for (int i = 0; i < pool.Length; i++)
                {
                    int ox = itO[2 + i] * stride[i] + itP[i] - pad[i];

                    if ((ox < 0) || (ox >= X.shape[2 + i]))
                    {
                        outOfBounds = true;
                        break;
                    }

                    itX[2 + i] = ox;
                }

                if (!outOfBounds)
                {
                    acc = accumulateOp(acc, X[itX.index]);
                    elementCount++;
                }
            }

            O[itO.index] = normalizeOp(acc, elementCount);
        }
    }

    void MaxPoolND(TensorFloat X, TensorFloat O, int[] pool, int[] stride, int[] pad)
    {
        Func<float> initOp = () => float.MinValue;
        Func<float, float, float> accumulateOp = (acc, v) => Mathf.Max(acc, v);
        Func<float, int, float> normalizeOp = (acc, elementCount) => acc;
        ApplyLocalPoolingOperator(X, O, pool, stride, pad, initOp, accumulateOp, normalizeOp);
    }

    void AveragePoolND(TensorFloat X, TensorFloat O, int[] pool, int[] stride, int[] pad)
    {
        Func<float> initOp = () => 0.0f;
        Func<float, float, float> accumulateOp = (acc, v) => acc + v;
        Func<float, int, float> normalizeOp = (acc, elementCount) => acc / elementCount;
        ApplyLocalPoolingOperator(X, O, pool, stride, pad, initOp, accumulateOp, normalizeOp);
    }

    void ScatterElementsReduce(TensorInt X, TensorInt indices, TensorInt updates, TensorInt O, int axis, Layers.ScatterReductionMode reduction)
    {
        MemCopy(X, O);

        BurstTensorData.Pin(X);
        BurstTensorData.Pin(indices);
        BurstTensorData.Pin(updates);
        BurstTensorData.Pin(O);

        //TODO: verify this.
        var itO = new TensorNDIterator(O.shape);
        for (var itIndices = new TensorNDIterator(indices.shape); itIndices.HasNext(); itIndices.MoveNext())
        {
            itO = itIndices;

            var index = indices[itIndices.index];
            index = index < 0 ? X.shape[axis] + index : index;

            itO[axis] = index;

            if (reduction == Layers.ScatterReductionMode.None)
                O[itO.index] = updates[itIndices.index];
            else if (reduction == Layers.ScatterReductionMode.Add)
                O[itO.index] += updates[itIndices.index];
            else if (reduction == Layers.ScatterReductionMode.Mul)
                O[itO.index] *= updates[itIndices.index];
        }
    }
}
} // namespace Unity.Sentis
