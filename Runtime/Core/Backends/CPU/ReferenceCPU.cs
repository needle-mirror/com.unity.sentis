using System;
using Unity.Sentis;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace Unity.Sentis {

/// <summary>
/// Represents a CPU backend ops.
/// </summary>
public partial class CPUOps : IOps
{
    bool m_OwnAllocator;
    internal ITensorAllocator m_Allocator;

    /// <inheritdoc/>
    public virtual DeviceType deviceType => DeviceType.CPU;

    /// <summary>
    /// Initializes and returns an instance of `CPUOps`.
    /// </summary>
    /// <param name="allocator">The allocator to use when allocating tensors.</param>
    public CPUOps(ITensorAllocator allocator = null)
    {
        if (allocator == null)
        {
            m_OwnAllocator = true;
            m_Allocator = new TensorCachingAllocator();
        }
        else
        {
            m_OwnAllocator = false;
            m_Allocator = allocator;
        }
    }

    /// <inheritdoc/>
    public virtual void PostLayerCleanup()
    {
        m_Allocator.PostLayerCleanup();
    }

    /// <inheritdoc/>
    public virtual Tensor NewTensor(TensorShape shape, DataType dataType, AllocScope scope)
    {
        return m_Allocator.Alloc(shape, dataType, DeviceType.CPU, scope);
    }

    /// <summary>
    /// Allocate a new `TensorFloat` of a given shape.
    /// </summary>
    protected TensorFloat NewTensorFloat(TensorShape shape, AllocScope scope)
    {
        return NewTensor(shape, DataType.Float, scope) as TensorFloat;
    }

    /// <summary>
    /// Allocate a new `TensorInt` of a given shape.
    /// </summary>
    protected TensorInt NewTensorInt(TensorShape shape, AllocScope scope)
    {
        return NewTensor(shape, DataType.Int, scope) as TensorInt;
    }

    /// <summary>
    /// Allocate a new `TensorFloat` of a given shape using the `AllocScope.LayerOutput` scope.
    /// </summary>
    protected TensorFloat NewOutputTensorFloat(TensorShape shape)
    {
        return NewTensorFloat(shape, AllocScope.LayerOutput);
    }

    /// <summary>
    /// Allocate a new `TensorInt` of a given shape using the `AllocScope.LayerOutput` scope.
    /// </summary>
    protected TensorInt NewOutputTensorInt(TensorShape shape)
    {
        return NewTensorInt(shape, AllocScope.LayerOutput);
    }

    /// <summary>
    /// Allocate a new `Tensor` of a given shape and data type using the `AllocScope.LayerOutput` scope.
    /// </summary>
    protected Tensor NewOutputTensor(TensorShape shape, DataType dataType)
    {
        return NewTensor(shape, dataType, AllocScope.LayerOutput);
    }

    /// <summary>
    /// Allocate a new `TensorFloat` of a given shape using the `AllocScope.InternalToLayer` scope.
    /// </summary>
    protected TensorFloat NewTempTensorFloat(TensorShape shape)
    {
        return NewTensorFloat(shape, AllocScope.InternalToLayer);
    }

    /// <summary>
    /// Allocate a new `TensorInt` of a given shape using the `AllocScope.InternalToLayer` scope.
    /// </summary>
    protected TensorInt NewTempTensorInt(TensorShape shape)
    {
        return NewTensorInt(shape, AllocScope.InternalToLayer);
    }

    /// <inheritdoc/>
    public virtual void ResetAllocator(bool keepCachedMemory = true)
    {
        m_Allocator.Reset(keepCachedMemory);
    }

    /// <summary>
    /// Disposes of the ops and any associated memory.
    /// </summary>
    public virtual void Dispose()
    {
        if (m_OwnAllocator)
            ResetAllocator(keepCachedMemory: false);
    }

    TensorFloat ConvND(TensorFloat X, TensorFloat K, TensorFloat B, int groups, int[] stride, int[] pad, int[] dilation, Layers.FusableActivation fusedActivation)
    {
        var Oshape = ShapeInference.Conv(X.shape, K.shape, B.shape, groups, stride, pad, dilation);
        var O = (fusedActivation != Layers.FusableActivation.None) ? NewTempTensorFloat(Oshape) : NewOutputTensorFloat(Oshape);
        if (O.shape.HasZeroDims())
            return O;

        int inputGroupedChannels = X.shape[1] / groups;
        int outputGroupedChannels = O.shape[1] / groups;

        var itK = new TensorNDIterator(K.shape);
        itK = itK.RemoveDim(0);
        itK = itK.RemoveDim(0);

        var itX = new TensorNDIterator(X.shape);
        for (var itO = new TensorNDIterator(O.shape); itO.HasNext(); itO.MoveNext())
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
                    for(int i = 0; i < stride.Length; i++)
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
            O[itO.index] = v;
        }

        if (fusedActivation != Layers.FusableActivation.None)
            O = ApplyFusedActivation(O, fusedActivation);

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Compress(Tensor X, TensorInt condition, int axis)
    {
        var numCondition = condition.shape.length;

        var indices = NewTempTensorInt(condition.shape);
        var numIndices = 0;

        for (var i = 0; i < numCondition; i++)
        {
            if (condition[i] == 0)
                continue;
            indices[numIndices] = i;
            numIndices++;
        }

        return CompressWithIndices(X, indices, numIndices, axis);
    }

    /// <inheritdoc/>
    public virtual TensorInt NonMaxSuppression(TensorFloat boxes, TensorFloat scores, int maxOutputBoxesPerClass = 0, float iouThreshold = 0, float scoreThreshold = 0, Layers.CenterPointBox centerPointBox = Layers.CenterPointBox.Corners)
    {
        // Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
        // Bounding boxes with score less than scoreThreshold are removed.
        // maxOutputBoxesPerClass represents the maximum number of boxes to be selected per batch per class.
        // This algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system result in the same boxes being selected by the algorithm.
        // Bounding box format is indicated by attribute centerPointBox. Corners - diagonal y,x pairs (coords or normalized values). Center - center coords + width and height.
        // iouThreshold represents the threshold for deciding whether boxes overlap too much with respect to IOU. 0-1 range. 0 - no filtering.
        // The output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes sorted in descending order and grouped by batch and class.

        ShapeInference.NonMaxSuppression(boxes.shape, scores.shape, iouThreshold);
        if (boxes.shape.HasZeroDims() || scores.shape.HasZeroDims() || maxOutputBoxesPerClass <= 0)
            return NewOutputTensorInt(new TensorShape(0, 3));

        // allocate the maximum possible output size tensor
        var outputData = new int[scores.shape[0] * scores.shape[1] * maxOutputBoxesPerClass * 3];
        // array of the current selected output indexes
        var selectedIndexes = new int[maxOutputBoxesPerClass];
        // array of the current selected output scores
        var selectedScores = new float[maxOutputBoxesPerClass];
        // keep a track of total output boxes
        int numberOfBoxes = 0;

        // find boxes to keep and then combine them into the single output tensor grouped by current batch and class
        for (int batch = 0; batch < scores.shape[0]; batch++)
        {
            for (int classID = 0; classID < scores.shape[1]; classID++)
            {
                //keep a track of selected boxes per batch and class
                int selectedBoxes = 0;
                Array.Clear(selectedIndexes, 0, maxOutputBoxesPerClass);
                Array.Clear(selectedScores, 0, maxOutputBoxesPerClass);
                // iterate over input boxes for the current batch and class
                for (int i = 0; i < scores.shape[2]; i++)
                {
                    // check if the score is lower that the scoreThreshold
                    if (scores[batch, classID, i] < scoreThreshold)
                        continue;

                    // initialize insert index to last position
                    int insertIndex = selectedBoxes;
                    bool isIgnoreBox = false;

                    // compare input boxes to the already selected boxes
                    for (int j = 0; j < selectedBoxes; j++)
                    {
                        // if insert index is still default, i.e. box score is lower than previous sorted boxes, compare to see if this is the correct insert index
                        if ((insertIndex == selectedBoxes) && scores[batch, classID, i] > scores[batch, classID, selectedIndexes[j]])
                            insertIndex = j;

                        // if not excessive overlap with this box consider next box
                        if (NotIntersectOverUnion(
                                boxes[batch, i, 0],
                                boxes[batch, i, 1],
                                boxes[batch, i, 2],
                                boxes[batch, i, 3],
                                boxes[batch, selectedIndexes[j], 0],
                                boxes[batch, selectedIndexes[j], 1],
                                boxes[batch, selectedIndexes[j], 2],
                                boxes[batch, selectedIndexes[j], 3],
                                centerPointBox, iouThreshold))
                            continue;

                        // new box has lower score than overlap box so do not output new box
                        if (insertIndex >= selectedBoxes)
                        {
                            isIgnoreBox = true;
                            break;
                        }

                        // new box has higher score than overlap box so remove overlap box from list, no need to shift memory if it is in final position
                        if (j < (maxOutputBoxesPerClass - 1))
                        {
                            // remove the overlaping box index and score values from the current selected box array by shifting the memory
                            // selectedIndexes/selectedScores = [x x x j y y y]
                            // <- shift y y y by one
                            // [x x x y y y]
                            unsafe
                            {
                                fixed (int* dst = &selectedIndexes[j])
                                    UnsafeUtility.MemMove(dst, dst + 1, (maxOutputBoxesPerClass - (j + 1)) * sizeof(int));
                                fixed (float* dst = &selectedScores[j])
                                    UnsafeUtility.MemMove(dst, dst + 1, (maxOutputBoxesPerClass - (j + 1)) * sizeof(int));
                            }
                        }
                        selectedBoxes--;
                        j--;
                    }

                    // either new box has lower score than an overlap box or there are already maxOutputBoxesPerClass with a better score, do not output new box
                    if (isIgnoreBox || insertIndex >= maxOutputBoxesPerClass)
                        continue;

                    // shift subsequent boxes forward by one in sorted array to make space for new box, no need if new box is after all boxes or or at end of array
                    if (insertIndex < selectedBoxes && insertIndex < (maxOutputBoxesPerClass - 1))
                    {
                        // shift memory to free a slot for a new box index and score values
                        // selectedIndexes/selectedScores = [x x x y y y]
                        // -> shift y y y by one
                        // [x x x insertIndex y y y]
                        unsafe
                        {
                            fixed (int* dst = &selectedIndexes[insertIndex])
                                UnsafeUtility.MemMove(dst + 1, dst, (maxOutputBoxesPerClass - (insertIndex + 1)) * sizeof(int));
                            fixed (float* dst = &selectedScores[insertIndex])
                                UnsafeUtility.MemMove(dst + 1, dst, (maxOutputBoxesPerClass - (insertIndex + 1)) * sizeof(int));
                        }
                    }

                    // record the score and index values of the selected box
                    // [x x x insertIndex y y y]
                    // insert box
                    // [x x x i y y y]
                    // [x x x score y y y]
                    selectedIndexes[insertIndex] = i;
                    selectedScores[insertIndex] = scores[batch, classID, i];
                    selectedBoxes++;
                }

                // gather outputs
                for (int i = 0; i < selectedBoxes; i++)
                {
                    // box is identified by its batch, class and index
                    outputData[numberOfBoxes * 3 + 0] = batch;
                    outputData[numberOfBoxes * 3 + 1] = classID;
                    outputData[numberOfBoxes * 3 + 2] = selectedIndexes[i];
                    numberOfBoxes++;
                }
            }
        }

        // create output tensor of correct length by trimming outputData
        var O = NewOutputTensorInt(new TensorShape(numberOfBoxes, 3));
        var pinO = ArrayTensorData.Pin(O, false);
        NativeTensorArray.Copy(outputData, pinO.array, numberOfBoxes * 3);
        return O;
    }

    bool NotIntersectOverUnion(float x1, float y1, float w1, float h1, float x2, float y2, float w2, float h2, Layers.CenterPointBox centerPointBox, float iouThreshold)
    {
        //inputs:
        //center_point_box:
        // 0 - diagonal y,x pairs (coords or normalized values)
        // 1 - center coords + width and height
        // Can be optimised by calculating each box area and PyTorch corner data outside IOU

        float b1x1;
        float b1x2;
        float b1y1;
        float b1y2;
        float b2x1;
        float b2x2;
        float b2y1;
        float b2y2;

        //convert inputs to: top left and bottom right corners of two rectangles
        //PyTorch
        if (centerPointBox == Layers.CenterPointBox.Center)
        {
            b1x1 = x1 - 0.5f * w1;
            b1x2 = x1 + 0.5f * w1;
            b1y1 = y1 - 0.5f * h1;
            b1y2 = y1 + 0.5f * h1;
            b2x1 = x2 - 0.5f * w2;
            b2x2 = x2 + 0.5f * w2;
            b2y1 = y2 - 0.5f * h2;
            b2y2 = y2 + 0.5f * h2;
        }
        //TensorFlow
        else //CenterPointBox.Corners
        {
            if (y1 < h1)
            {
                b1x1 = y1;
                b1x2 = h1;
            }
            else
            {
                b1x1 = h1;
                b1x2 = y1;
            }

            if (x1 < w1)
            {
                b1y1 = x1;
                b1y2 = w1;
            }
            else
            {
                b1y1 = w1;
                b1y2 = x1;
            }

            if (y2 < h2)
            {
                b2x1 = y2;
                b2x2 = h2;
            }
            else
            {
                b2x1 = h2;
                b2x2 = y2;
            }

            if (x2 < w2)
            {
                b2y1 = x2;
                b2y2 = w2;
            }
            else
            {
                b2y1 = w2;
                b2y2 = x2;
            }
        }

        //intersection rectangle
        float xMax = Math.Max(b1x1, b2x1);
        float yMax = Math.Max(b1y1, b2y1);
        float xMin = Math.Min(b1x2, b2x2);
        float yMin = Math.Min(b1y2, b2y2);

        //check if intersection rectangle exist
        if (xMin <= xMax || yMin <= yMax)
        {
            return true;
        }

        float intersectionArea = (xMin - xMax) * (yMin - yMax);
        float b1area = Math.Abs((b1x2 - b1x1) * (b1y2 - b1y1));
        float b2area = Math.Abs((b2x2 - b2x1) * (b2y2 - b2y1));
        return intersectionArea / (b1area + b2area - intersectionArea) <= iouThreshold;
    }

    TensorFloat ResizeND(TensorFloat X, float[] scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode = Layers.NearestMode.RoundPreferFloor, Layers.CoordTransformMode coordTransformMode = Layers.CoordTransformMode.HalfPixel)
    {
        var O = Resize1D(X, 0, scale[0], interpolationMode, nearestMode, coordTransformMode);
        for (int i = 1; i < scale.Length; i++)
        {
            O = Resize1D(O, i, scale[i], interpolationMode, nearestMode, coordTransformMode);
        }

        return O;
    }

    TensorFloat Resize1D(TensorFloat X, int axis, float scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode, Layers.CoordTransformMode coordTransformMode)
    {
        var O = NewOutputTensorFloat(ShapeInference.Resize(X.shape, axis, scale));

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

        return O;
    }

    TensorFloat ApplyLocalPoolingOperator(TensorFloat X, int[] pool, int[] stride, int[] pad, Func<float> initOp, Func<float, float, float> accumulateOp, Func<float, int, float> normalizeOp)
    {
        var O = NewOutputTensorFloat(ShapeInference.ApplyPool(X.shape, pool, stride, pad));

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

        return O;
    }

    TensorFloat MaxPoolND(TensorFloat X, int[] pool, int[] stride, int[] pad)
    {
        Func<float> initOp = () => float.MinValue;
        Func<float, float, float> accumulateOp = (acc, v) => Mathf.Max(acc, v);
        Func<float, int, float> normalizeOp = (acc, elementCount) => acc;
        return ApplyLocalPoolingOperator(X, pool, stride, pad, initOp, accumulateOp, normalizeOp);
    }

    TensorFloat AveragePoolND(TensorFloat X, int[] pool, int[] stride, int[] pad)
    {
        Func<float> initOp = () => 0.0f;
        Func<float, float, float> accumulateOp = (acc, v) => acc + v;
        Func<float, int, float> normalizeOp = (acc, elementCount) => acc / elementCount;
        return ApplyLocalPoolingOperator(X, pool, stride, pad, initOp, accumulateOp, normalizeOp);
    }

    /// <inheritdoc/>
    public virtual TensorFloat LRN(TensorFloat X, float alpha, float beta, float bias, int size)
    {
        // https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        // However divide the sum by size to follow onnx and pytorch implementation
        // ONNX https://github.com/onnx/onnx/blob/master/docs/Operators.md#LRN
        // PYTORCH https://github.com/pytorch/pytorch/blob/1465970a343e61f2f2b104859ca7f5d7e03f5d02/torch/nn/functional.py#L2069
        // Tensorflow don't and follow the paper to the letter https://github.com/tensorflow/tensorflow/blob/e6faa845c51bb69465146d93646947fd2ba53efa/tensorflow/python/kernel_tests/lrn_op_test.py#L53
        // However they bake the division to alpha when exporting to ONNX https://github.com/onnx/tensorflow-onnx/blob/7c37ccb97e0fd478ce093910c4a1411b18e44fd7/tf2onnx/onnx_opset/math.py
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;
        float sizef = size;

        var itRemap = new TensorNDIterator(O.shape);
        for (var it = new TensorNDIterator(O.shape); it.HasNext(); it.MoveNext())
        {
            int c = it[1];
            float regionCenter = (sizef - 1.0f) / 2.0f;
            int regionStart = Math.Max(0, c - (int)Mathf.Floor(regionCenter));
            int regionEnd = Math.Min(X.shape[1], c + (int)Mathf.Ceil(regionCenter)+1);
            float sumOfSquared = 0.0f;
            for (int ci = regionStart; ci < regionEnd; ++ci)
            {
                itRemap.CopyNDIndex(it);
                itRemap[1] = ci;
                float regionValue = X[itRemap.index];
                sumOfSquared += regionValue * regionValue;
            }

            O[it.index] = X[it.index] / Mathf.Pow(bias + alpha * sumOfSquared / sizef, beta);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Multinomial(TensorFloat X, int count, float? seed)
    {
        var O = NewOutputTensorInt(ShapeInference.Multinomial(X.shape, count));

        uint finalSeed = Random.GetOpSeed(seed);
        finalSeed = finalSeed == 0 ? 1 : finalSeed;
        var random = new Mathematics.Random(finalSeed);

        // Tensorflow Multinomial for reference
        // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/multinomial_op.cc
        for (int n = 0; n < X.shape[0]; ++n)
        {
            var maxLogP = Mathf.NegativeInfinity;
            for (int i = 0; i < X.shape[1]; ++i)
                maxLogP = Mathf.Max(X[n, i], maxLogP);

            float sumOfProbabilities = 0f;
            for (int i = 0; i < X.shape[1]; ++i)
                sumOfProbabilities += Mathf.Exp(X[n, i] - maxLogP); // NOTE: X contains log-probabilities

            for (int sample = 0; sample < count; ++sample)
            {
                float p = random.NextFloat() * sumOfProbabilities;

                int i = 0;
                float cumulativeP = 0f;
                while (i < X.shape[1] && p > cumulativeP)
                {
                    cumulativeP += Mathf.Exp(X[n, i] - maxLogP);
                    i++;
                }
                Logger.AssertIsTrue(i > 0, "Multinomial.ValueError: need at least one cumulative sample {0}", i);
                O[n, sample] = i - 1;
            }
        }

        return O;
    }

    /// <inheritdoc/>
    public TensorInt NonZero(TensorFloat X)
    {
        int nbNonZeroIndices = 0;
        var end = X.shape.length;
        for (int i = 0; i < end; ++i)
        {
            if (X[i] != 0.0f)
                nbNonZeroIndices += 1;
        }

        var O = NewOutputTensorInt(new TensorShape(X.shape.rank, nbNonZeroIndices));
        if (O.shape.HasZeroDims())
            return O;

        int nonZeroIndicesIdx = 0;
        for (var it = new TensorNDIterator(X.shape); it.HasNext(); it.MoveNext())
        {
            if (X[it.index] != 0.0f)
            {
                for (int i = 0; i < X.shape.rank; i++)
                    O[i * nbNonZeroIndices + nonZeroIndicesIdx] = it[i];
                nonZeroIndicesIdx++;
            }
        }

        return O;
    }

    /// <inheritdoc/>
    public TensorInt NonZero(TensorInt X)
    {
        int nbNonZeroIndices = 0;
        var end = X.shape.length;
        for (int i = 0; i < end; ++i)
        {
            if (X[i] != 0)
                nbNonZeroIndices += 1;
        }

        var O = NewOutputTensorInt(new TensorShape(X.shape.rank, nbNonZeroIndices));
        if (O.shape.HasZeroDims())
            return O;

        int nonZeroIndicesIdx = 0;
        for (var it = new TensorNDIterator(X.shape); it.HasNext(); it.MoveNext())
        {
            if (X[it.index] != 0)
            {
                for (int i = 0; i < X.shape.rank; i++)
                    O[i * nbNonZeroIndices + nonZeroIndicesIdx] = it[i];
                nonZeroIndicesIdx++;
            }
        }

        return O;
    }

    Tensor ScatterElementsReduce(TensorInt X, TensorInt indices, TensorInt updates, int axis, Layers.ScatterReductionMode reduction)
    {
        var O = Copy(X) as TensorInt;
        if (O.shape.HasZeroDims())
            return O;

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

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt ScatterND(TensorInt X, TensorInt indices, TensorInt updates, Layers.ScatterReductionMode reduction)
    {
        var O = Copy(X) as TensorInt;
        if (O.shape.HasZeroDims())
            return O;

        var indexRemapDim = indices.shape[-1];
        var indicesLength = indices.shape.Length(0, -1);
        var updatesLength = updates.shape.length / indicesLength;

        var itX = new TensorNDIterator(X.shape);
        for (var i = 0; i < indicesLength; i++)
        {
            var indexO = 0;
            var trailing = 1;
            for (var j = (indexRemapDim -1); j >= 0; j--)
            {
                indexO += trailing * indices[i * indexRemapDim + j];
                trailing *= X.shape[j];
            }

            for (var k = 0; k < updatesLength; k++)
            {
                var vw = updates[i * updatesLength + k];

                if (reduction == Layers.ScatterReductionMode.None)
                    O[indexO * updatesLength + k] = vw;
                else if (reduction == Layers.ScatterReductionMode.Add)
                    O[indexO * updatesLength + k] += vw;
                else if (reduction == Layers.ScatterReductionMode.Mul)
                    O[indexO * updatesLength + k] *= vw;
            }
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Shape(Tensor X, int start = 0, int end = TensorShape.maxRank)
    {
        start = start < 0 ? start + X.shape.rank : start;
        end = end < 0 ? end + X.shape.rank : end;
        start = Mathf.Clamp(start, 0, X.shape.rank);
        end = Mathf.Clamp(end, 0, X.shape.rank);

        Logger.AssertIsTrue(end >= start, "Shape.InputError: start value cannot be greater than end value for shape slicing");
        var O = NewOutputTensorInt(new TensorShape(end - start));
        var arrayO = ArrayTensorData.Pin(O, uploadCache: false).array;

        for (var i = start; i < end; i++)
            arrayO.Set<int>(i - start, X.shape[i]);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Size(TensorShape shape)
    {
        var O = NewOutputTensorInt(new TensorShape());
        var arrayO = ArrayTensorData.Pin(O, uploadCache: false).array;

        arrayO.Set<int>(0, shape.length);

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Copy(Tensor X)
    {
        // make shallow copy and patch the shape, if already managed by allocator
        if (X.allocator != null)
            return X.ShallowCopy();

        var O = NewOutputTensor(X.shape, X.dataType);
        if (O.shape.HasZeroDims())
            return O;
        MemCopy(X, O);
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Reshape(Tensor X, TensorShape newShape)
    {
        Logger.AssertAreEqual(X.shape.length, newShape.length, "Reshape.LengthError: in/out tensorshape must have the same # of elements : ({0}, {1})", X.shape.length, newShape.length);
        // if already managed by allocator, can do a shallow copy
        if (X.allocator != null)
            return X.ShallowReshape(newShape);

        var O = NewOutputTensor(newShape, X.dataType);
        if (O.shape.HasZeroDims())
            return O;
        MemCopy(X, O);
        return O;
    }

    /// <summary>
    /// Sets final output tensor for W, R, initialH and initialC from provided input tensors
    /// if no input is provided the tensor is cleared to 0 as a default
    /// otherwise if the input tensor can be used directly in the calculation this will early out
    /// </summary>
    void SetRnnInput(TensorFloat X, TensorFloat O, int index, int count, int length, int strideX)
    {
        if (X == O)
            return;
        if (X == null)
            MemSet(O, 0);
        else
            MemCopyStride(X, O, strideX, length, length, count, index * length, 0);
    }

    /// <summary>
    /// Sets intermediate input tensors for Y_h and Y_c from intermediate output tensor
    /// if the calculation is single direction and sequenceFirst layout then the output
    /// tensor will be used directly and this command early outs
    /// </summary>
    void SetRnnOutput(TensorFloat X, TensorFloat O, int index, int count, int length, int strideO)
    {
        if (X == O)
            return;
        MemCopyStride(X, O, length, strideO, length, count, 0, index * length);
    }

    /// <inheritdoc/>
    public virtual TensorFloat[] LSTM(TensorFloat X, TensorFloat W, TensorFloat R, TensorFloat B, TensorInt sequenceLens, TensorFloat initialH, TensorFloat initialC, TensorFloat P, Layers.RnnDirection direction, Layers.RnnActivation[] activations, float[] activationAlpha, float[] activationBeta, bool inputForget, float clip = float.MaxValue, Layers.RnnLayout layout = Layers.RnnLayout.SequenceFirst)
    {
        ShapeInference.LSTM(X.shape, W.shape, R.shape, layout, out var shapeY, out var shapeY_h, out var shapeY_c);
        var O = NewOutputTensorFloat(shapeY);
        var Y_h = NewOutputTensorFloat(shapeY_h);
        var Y_c = NewOutputTensorFloat(shapeY_c);
        if (O.shape.HasZeroDims())
            return new[] { O, Y_h, Y_c };

        var seqLength = X.shape[layout == Layers.RnnLayout.SequenceFirst ? 0 : 1];
        var batchSize = X.shape[layout == Layers.RnnLayout.SequenceFirst ? 1 : 0];
        var inputSize = X.shape[2];
        var hiddenSize = R.shape[2];
        var numDirections = W.shape[0];

        var W1 = numDirections == 2 ? NewTempTensorFloat(new TensorShape(1, 4 * hiddenSize, inputSize)) : W;
        var R1 = numDirections == 2 ? NewTempTensorFloat(new TensorShape(1, 4 * hiddenSize, hiddenSize)) : R;

        var Bi = B;
        if (Bi == null)
        {
            Bi = NewTempTensorFloat(new TensorShape(numDirections, 8 * hiddenSize));
            MemSet(Bi, 0);
        }
        var sequenceLensi = sequenceLens;
        if (sequenceLensi == null)
        {
            sequenceLensi = NewTempTensorInt(new TensorShape(batchSize));
            MemSet(sequenceLensi, math.asint(seqLength));
        }
        var Pi = P;
        if (Pi == null)
        {
            Pi = NewTempTensorFloat(new TensorShape(numDirections, 3 * hiddenSize));
            MemSet(Pi, 0);
        }

        var Y_h1 = layout == Layers.RnnLayout.SequenceFirst ? (numDirections == 2 ? NewTempTensorFloat(new TensorShape(1, batchSize, hiddenSize)) : Y_h) : NewTempTensorFloat(new TensorShape(batchSize, 1, hiddenSize));
        var Y_c1 = layout == Layers.RnnLayout.SequenceFirst ? (numDirections == 2 ? NewTempTensorFloat(new TensorShape(1, batchSize, hiddenSize)) : Y_c) : NewTempTensorFloat(new TensorShape(batchSize, 1, hiddenSize));

        var Y_hcLower = layout == Layers.RnnLayout.SequenceFirst ? batchSize * hiddenSize : hiddenSize;
        var Y_hcUpper = layout == Layers.RnnLayout.SequenceFirst ? 1 : batchSize;

        for (var i = 0; i < numDirections; i++)
        {
            SetRnnInput(W, W1, i, 1, 4 * hiddenSize * inputSize, 0);
            SetRnnInput(R, R1, i, 1, 4 * hiddenSize * hiddenSize, 0);
            SetRnnInput(initialH, Y_h1, i, Y_hcUpper, Y_hcLower, numDirections * Y_hcLower);
            SetRnnInput(initialC, Y_c1, i, Y_hcUpper, Y_hcLower, numDirections * Y_hcLower);
            var isReverse = direction == Layers.RnnDirection.Reverse || (direction == Layers.RnnDirection.Bidirectional && i == 1);
            SinglePassLSTM(X, W1, R1, Bi, sequenceLensi, Pi, O, Y_h1, Y_c1, activations, activationAlpha, activationBeta, inputForget, clip, isReverse, i, layout);
            SetRnnOutput(Y_h1, Y_h, i, Y_hcUpper, Y_hcLower, numDirections * Y_hcLower);
            SetRnnOutput(Y_c1, Y_c, i, Y_hcUpper, Y_hcLower, numDirections * Y_hcLower);
        }

        return new[] { O, Y_h, Y_c };
    }
}
} // namespace Unity.Sentis
