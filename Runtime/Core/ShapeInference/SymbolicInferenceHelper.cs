using System;
using System.ComponentModel;
using Unity.Sentis.Layers;
using UnityEngine;

namespace Unity.Sentis
{
    static class SymbolicInference
    {
        public static SymbolicTensorShape MatMul2D(SymbolicTensorShape shapeX, bool transposeX, SymbolicTensorShape shapeY, bool transposeY)
        {
            shapeX.DeclareRank(2);
            shapeY.DeclareRank(2);

            var mulXDim = transposeX ? shapeX[0] : shapeX[1];
            var mulYDim = transposeY ? shapeY[1] : shapeY[0];
            Logger.AssertIsTrue(!mulXDim.isValue || !mulYDim.isValue || mulXDim == mulYDim, "MatMul2D.ValueError: failed, dims not equal");

            return new SymbolicTensorShape(transposeX ? shapeX[1] : shapeX[0], transposeY ? shapeY[0] : shapeY[1]);
        }

        public static SymbolicTensorShape MatMul(SymbolicTensorShape shapeX, SymbolicTensorShape shapeY)
        {
            return shapeX.MatMul(shapeY);
        }

        public static SymbolicTensorShape Reduce(SymbolicTensorShape shapeX, int axis, bool keepDim = true)
        {
            if (!shapeX.hasRank)
                return SymbolicTensorShape.UnknownShape;

            var reducedShape = new SymbolicTensorShape(shapeX);

            // reducing on a zero axis will result in a zero rather than a one
            if (shapeX[axis].isValue)
                reducedShape[axis] = shapeX[axis].value == 0 ? SymbolicTensorDim.Zero : SymbolicTensorDim.One;
            else
                reducedShape[axis] = SymbolicTensorDim.Unknown;

            return !keepDim ? reducedShape.Squeeze(axis) : reducedShape;
        }

        public static SymbolicTensorShape Reduce(SymbolicTensorShape shapeX, SymbolicTensorShape shapeAxes, PartialTensor axes, bool keepDim, bool noopWithEmptyAxis)
        {
            if (axes.isPartiallyKnown)
            {
                if (axes.shape.length == 0)
                    return Reduce(shapeX, keepDim, noopWithEmptyAxis);

                var reducedShape = new SymbolicTensorShape(shapeX);
                if (!axes.IsFullyKnown() && reducedShape.hasRank)
                {
                    // replace any non 0 or 1 dims with unknown (0 and 1 stay the same whether reduced or not)
                    for (var i = 0; i < reducedShape.rank; i++)
                    {
                        if (reducedShape[i].isValue && (reducedShape[i].value == 0 || reducedShape[i].value == 1))
                            continue;
                        reducedShape[i] = SymbolicTensorDim.Unknown;
                    }
                }
                for (var i = 0; i < axes.shape.length; i++)
                {
                    if (!axes[i].isValue)
                        continue;
                    var axis = axes[i].value;
                    // reducing on a zero axis will result in a zero rather than a one
                    if (shapeX[axis].isValue)
                        reducedShape[axis] = shapeX[axis].value == 0 ? SymbolicTensorDim.Zero : SymbolicTensorDim.One;
                    else
                        reducedShape[axis] = SymbolicTensorDim.Unknown;
                }

                return !keepDim ? Squeeze(reducedShape, axes) : reducedShape;
            }

            if (shapeAxes.IsFullyKnown())
            {
                if (shapeAxes[0].value == 0)
                    return Reduce(shapeX, keepDim, noopWithEmptyAxis);

                return keepDim ? SymbolicTensorShape.UnknownOfRankLike(shapeX) : SymbolicTensorShape.UnknownShape;
            }

            return keepDim && !noopWithEmptyAxis ? SymbolicTensorShape.UnknownOfRankLike(shapeX) : SymbolicTensorShape.UnknownShape;
        }

        public static SymbolicTensorShape Reduce(SymbolicTensorShape shapeX, bool keepDim, bool noopWithEmptyAxis)
        {
            if (noopWithEmptyAxis)
                return shapeX;

            return keepDim ? SymbolicTensorShape.OnesLike(shapeX) : new SymbolicTensorShape();
        }

        public static SymbolicTensorShape Squeeze(SymbolicTensorShape shapeX, PartialTensor axes)
        {
            if (!axes.isPartiallyKnown)
                return SymbolicTensorShape.UnknownShape;
            if (!axes.IsFullyKnown())
                return SymbolicTensorShape.UnknownOfRank(shapeX.rank - axes.shape.length);
            return shapeX.Squeeze(axes.ToIntArray());
        }

        public static SymbolicTensorShape Unsqueeze(SymbolicTensorShape shapeX, PartialTensor axes)
        {
            if (!axes.isPartiallyKnown)
                return SymbolicTensorShape.UnknownShape;
            if (!axes.IsFullyKnown())
                return SymbolicTensorShape.UnknownOfRank(shapeX.rank + axes.shape.length);
            return shapeX.Unsqueeze(axes.ToIntArray());
        }

        public static SymbolicTensorShape DepthToSpace(SymbolicTensorShape shapeX, int blocksize)
        {
            shapeX.DeclareRank(4);
            return new SymbolicTensorShape(shapeX[0], shapeX[1] / (blocksize * blocksize), shapeX[2] * blocksize, shapeX[3] * blocksize);
        }

        public static SymbolicTensorShape SpaceToDepth(SymbolicTensorShape shapeX, int blockSize)
        {
            shapeX.DeclareRank(4);
            return new SymbolicTensorShape(shapeX[0], shapeX[1] * (blockSize * blockSize), shapeX[2] / blockSize, shapeX[3] / blockSize);
        }

        public static SymbolicTensorShape Multinomial(SymbolicTensorShape shapeX, int count)
        {
            shapeX.DeclareRank(2);
            return new SymbolicTensorShape(shapeX[0], new SymbolicTensorDim(count));
        }

        public static SymbolicTensorShape OneHot(SymbolicTensorShape shapeX, int axis, PartialTensor depth)
        {
            if (!shapeX.hasRank)
                return SymbolicTensorShape.UnknownShape;

            var shapeOut = shapeX.Unsqueeze(axis);
            shapeOut[axis] = depth[0].ToSymbolicTensorDim();

            return shapeOut;
        }

        public static SymbolicTensorShape RoiAlign(SymbolicTensorShape shapeX, SymbolicTensorShape shapeRois, SymbolicTensorShape shapeIndices, int h, int w)
        {
            var shapeOut = SymbolicTensorShape.UnknownOfRank(4);

            shapeRois.DeclareRank(2);
            if (shapeRois[1].isValue)
                Logger.AssertAreEqual(4, shapeRois[1].value, "RoiAlign.ValueError: incorrect number of num_rois, expecting 4 got, {0}", shapeRois[1].value);
            shapeOut[0] = shapeRois[0];

            shapeX.DeclareRank(4);
            shapeOut[1] = shapeX[1];

            shapeIndices.DeclareRank(1);
            shapeOut[0] = SymbolicTensorDim.MaxDefinedDim(shapeOut[0], shapeIndices[0]);

            shapeOut[2] = new SymbolicTensorDim(h);
            shapeOut[3] = new SymbolicTensorDim(w);

            return shapeOut;
        }

        public static SymbolicTensorShape Flatten(SymbolicTensorShape shapeX, int axis)
        {
            if (!shapeX.hasRank)
            {
                if (axis == 0)
                    return new SymbolicTensorShape(SymbolicTensorDim.One, SymbolicTensorDim.Unknown);
                return SymbolicTensorShape.UnknownOfRank(2);
            }

            axis = shapeX.Axis(axis);

            var shapeOut = SymbolicTensorShape.Ones(2);
            for (var i = 0; i < axis; i++)
            {
                shapeOut[0] *= shapeX[i];
            }
            for (var i = axis; i < shapeX.rank; i++)
            {
                shapeOut[1] *= shapeX[i];
            }

            return shapeOut;
        }

        public static SymbolicTensorShape Tile(SymbolicTensorShape shapeX, int[] repeats)
        {
            shapeX.DeclareRank(repeats.Length);

            var shapeOut = SymbolicTensorShape.Ones(shapeX.rank);

            for (var i = 0; i < shapeOut.rank; i++)
            {
                shapeOut[i] = shapeX[i] * repeats[i];
            }

            return shapeOut;
        }

        public static SymbolicTensorShape Pad(SymbolicTensorShape shapeX, SymbolicTensorShape shapePads, PartialTensor pads)
        {
            if (shapePads.hasRank)
            {
                Logger.AssertIsTrue(shapePads.rank == 1, "Pad.ValueError: pads must be rank 1");
                if (shapePads[0].isValue)
                {
                    Logger.AssertIsTrue(shapePads[0].value % 2 == 0, "Pad.ValueError: length of pads must divide by 2");
                    shapeX.DeclareRank(shapePads[0].value / 2);
                }
            }

            if (!shapeX.hasRank)
                return SymbolicTensorShape.UnknownShape;

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank);

            for (var i = 0; i < shapeOut.rank; i++)
            {
                var dimPad = pads[i] + pads[i + shapeOut.rank];
                shapeOut[i] = shapeX[i] + dimPad.ToSymbolicTensorDim();
            }

            return shapeOut;
        }

        public static SymbolicTensorShape Reshape(SymbolicTensorShape shapeX, SymbolicTensorShape shapeShape, ref PartialTensor tensorShape, bool allowZero)
        {
            shapeShape.DeclareRank(1);

            if (!tensorShape.isPartiallyKnown)
            {
                if (shapeShape[0].isValue)
                    return SymbolicTensorShape.UnknownOfRank(shapeShape[0].value);

                return SymbolicTensorShape.UnknownShape;
            }

            tensorShape.AssertRank(1);

            if (tensorShape.IsFullyKnown())
                return Reshape(shapeX, tensorShape.ToIntArray(), allowZero);

            var shapeOut = SymbolicTensorShape.UnknownOfRank(tensorShape.shape.length);

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (!tensorShape[i].isValue || tensorShape[i].value < 0)
                    continue;
                if (tensorShape[i].value == 0)
                    shapeOut[i] = allowZero ? SymbolicTensorDim.Zero : shapeX[i];
                else
                    shapeOut[i] = new SymbolicTensorDim(tensorShape[i].value);
            }

            // input: (6, 'A', 'B')
            // shape: (1, 'A', 3, 2) -> (1, 0, 3, 2) if allowZero (1, -1, 3, 2) else
            // shape: (3, 'C') -> (3, -1)
            int minusCount = 0;
            var newSize = new int[shapeOut.rank];
            for (var i = 0; i < shapeOut.rank; i++)
            {
                var dim = tensorShape[i];
                if (dim.isUnknown)
                {
                    newSize[i] = -1;
                    minusCount++;
                }
                else if (dim.isValue)
                {
                    newSize[i] = dim.value;
                    if (dim.value == -1)
                        minusCount++;
                }
                else
                {
                    if (!allowZero && i < shapeX.rank && shapeX[i].isParam && shapeX[i].param == dim.param)
                    {
                        newSize[i] = 0;
                    }
                    else
                    {
                        newSize[i] = -1;
                        minusCount++;
                    }
                }
            }

            if (minusCount <= 1)
            {
                for (int i = 0; i < shapeOut.rank; i++)
                    tensorShape[i] = new PartialTensorElement(newSize[i]);
                shapeOut = Reshape(shapeX, newSize, allowZero);
            }

            return shapeOut;
        }

        static SymbolicTensorShape Reshape(SymbolicTensorShape shapeX, int[] shapeOutArray, bool allowZero)
        {
            var inputSize = SymbolicTensorDim.One;

            if (shapeX.hasRank)
            {
                for (var i = 0; i < shapeX.rank; i++)
                {
                    inputSize *= shapeX[i];
                }
            }

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeOutArray.Length);

            var negativeOutputIndex = -1;
            var outputSize = SymbolicTensorDim.One;

            for (var i = 0; i < shapeOutArray.Length; i++)
            {
                if (shapeOutArray[i] == -1)
                {
                    Logger.AssertIsTrue(negativeOutputIndex < 0, "Reshape.ValueError: multiple axes with dimension -1 given");
                    negativeOutputIndex = i;
                }
                else if (shapeOutArray[i] == 0 && !allowZero)
                {
                    if (shapeX.hasRank)
                    {
                        Logger.AssertIsTrue(i < shapeX.rank, "Reshape.ValueError: 0 dimension has no corresponding axis in input");
                        shapeOut[i] = shapeX[i];
                        outputSize *= shapeX[i];
                    }
                    else
                    {
                        shapeOut[i] = SymbolicTensorDim.Unknown;
                        outputSize = SymbolicTensorDim.Unknown;
                    }
                }
                else
                {
                    shapeOut[i] = new SymbolicTensorDim(shapeOutArray[i]);
                    outputSize *= shapeOutArray[i];
                }
            }

            if (negativeOutputIndex < 0 && shapeX.hasRank && inputSize.isValue && outputSize.isValue)
                Logger.AssertIsTrue(inputSize == outputSize, "Reshape.SizeError: can't reshape between different sizes");

            if (negativeOutputIndex >= 0 && shapeX.hasRank)
                shapeOut[negativeOutputIndex] = inputSize / outputSize;

            return shapeOut;
        }

        public static SymbolicTensorShape Transpose(SymbolicTensorShape shapeX, int[] perm)
        {
            if (perm != null)
                shapeX.DeclareRank(perm.Length);

            if (!shapeX.hasRank)
                return SymbolicTensorShape.UnknownShape;

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank);

            if (perm == null || perm.Length == 0)
            {
                // reverse axes
                for (var i = 0; i < shapeX.rank; i++)
                {
                    shapeOut[i] = shapeX[shapeX.rank - 1 - i];
                }
            }
            else
            {
                uint axesBitMask = 0;
                for (var i = 0; i < perm.Length; i++)
                {
                    var axis = shapeX.Axis(perm[i]);
                    Logger.AssertIsTrue(((axesBitMask >> axis) & 1U) == 0,"Transpose.ValueError: permutation must be a permutation of the axis (0, rank-1)");
                    axesBitMask |= 1U << axis;
                    shapeOut[i] = shapeX[axis];
                }
            }

            return shapeOut;
        }

        public static SymbolicTensorShape Compress(SymbolicTensorShape shapeX, SymbolicTensorShape shapeCondition)
        {
            var isZero = shapeX.Length() * shapeCondition.Length() == SymbolicTensorDim.Zero;
            return new SymbolicTensorShape(isZero ? SymbolicTensorDim.Zero : SymbolicTensorDim.Unknown);
        }

        public static SymbolicTensorShape Compress(SymbolicTensorShape shapeX, SymbolicTensorShape shapeCondition, int axis)
        {
            var isZero = shapeX.Length() * shapeCondition.Length() == SymbolicTensorDim.Zero;
            var outputShape = new SymbolicTensorShape(shapeX);
            outputShape[axis] = isZero ? SymbolicTensorDim.Zero : SymbolicTensorDim.Unknown;
            return outputShape;
        }

        public static SymbolicTensorShape Gather(SymbolicTensorShape shapeX, SymbolicTensorShape shapeIndices, int axis)
        {
            if (!shapeX.hasRank)
                return SymbolicTensorShape.UnknownShape;

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            if (!shapeIndices.hasRank)
                return SymbolicTensorShape.UnknownShape;

            axis = shapeX.Axis(axis);

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank - 1 + shapeIndices.rank);

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (i < axis)
                    shapeOut[i] = shapeX[i];
                else if (i < axis + shapeIndices.rank)
                    shapeOut[i] = shapeIndices[i - axis];
                else
                    shapeOut[i] = shapeX[i - shapeOut.rank];
            }

            return shapeOut;
        }

        public static SymbolicTensorShape GatherElements(SymbolicTensorShape shapeX, SymbolicTensorShape shapeIndices, int axis)
        {
            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            if (shapeX.hasRank)
            {
                shapeX.Axis(axis);
                shapeIndices.DeclareRank(shapeX.rank);
            }

            return shapeIndices;
        }

        public static SymbolicTensorShape GatherND(SymbolicTensorShape shapeX, SymbolicTensorShape shapeIndices, int batchDims)
        {
            // from https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= batchDims : true, "RankError: incorrect rank, expecting at least {0}, got {1}", batchDims, shapeX.rank);
            Logger.AssertIsTrue(shapeIndices.hasRank ? shapeIndices.rank >= batchDims : true, "RankError: incorrect rank, expecting at least {0}, got {1}", batchDims, shapeIndices.rank);

            if (!shapeX.hasRank || !shapeIndices.hasRank)
                return SymbolicTensorShape.UnknownShape;

            if (!shapeIndices[-1].isValue)
                return SymbolicTensorShape.UnknownShape;

            Logger.AssertIsTrue(batchDims + shapeIndices[-1].value <= shapeX.rank, "GatherND.InputError: last indices dim too large");
            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank + shapeIndices.rank - shapeIndices[-1].value - 1 - batchDims);
            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (i < batchDims)
                    shapeOut[i] = SymbolicTensorDim.MaxDefinedDim(shapeX[i], shapeIndices[i]);
                else if (i < shapeIndices.rank - 1)
                    shapeOut[i] = shapeIndices[i];
                else
                    shapeOut[i] = shapeX[i - shapeOut.rank];
            }

            return shapeOut;
        }

        public static SymbolicTensorShape Concat(SymbolicTensorShape[] inputShapes, int axis)
        {
            Logger.AssertIsTrue(inputShapes.Length > 0, "Concat.InputError: can't broadcast shapes array of size 0");

            var rank = SymbolicTensorDim.Unknown;
            foreach (var inputShape in inputShapes)
            {
                if (inputShape.hasRank)
                    rank = SymbolicTensorDim.MaxDefinedDim(rank, new SymbolicTensorDim(inputShape.rank));
            }

            if (rank.isUnknown)
                return SymbolicTensorShape.UnknownShape;

            for (var i = 0; i < inputShapes.Length; i++)
            {
                inputShapes[i].DeclareRank(rank.value);
            }

            var shapeOut = SymbolicTensorShape.UnknownOfRank(rank.value);
            axis = shapeOut.Axis(axis);

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (i == axis)
                {
                    shapeOut[i] = SymbolicTensorDim.Zero;
                    foreach (var shape in inputShapes)
                    {
                        shapeOut[i] += shape[i];
                    }
                }
                else
                {
                    shapeOut[i] = SymbolicTensorDim.Unknown;
                    foreach (var shape in inputShapes)
                    {
                        shapeOut[i] = SymbolicTensorDim.MaxDefinedDim(shapeOut[i], shape[i]);
                    }
                }
            }

            return shapeOut;
        }

        public static SymbolicTensorShape Broadcast(SymbolicTensorShape[] inputShapes)
        {
            Logger.AssertIsTrue(inputShapes.Length > 0, "Broadcast.InputError: can't broadcast shapes array of size 0");

            if (inputShapes.Length == 1)
                return inputShapes[0];

            if (inputShapes.Length == 2)
                return inputShapes[0].Broadcast(inputShapes[1]);

            var outRank = 0;
            foreach (var shape in inputShapes)
            {
                if (!shape.hasRank)
                    return SymbolicTensorShape.UnknownShape;

                outRank = Mathf.Max(outRank, shape.rank);
            }

            var shapeOut = SymbolicTensorShape.Ones(outRank);

            foreach (var inputShape in inputShapes)
            {
                for (var j = 0; j < inputShape.rank; j++)
                {
                    shapeOut[shapeOut.rank - inputShape.rank + j] = SymbolicTensorDim.Broadcast(shapeOut[shapeOut.rank - inputShape.rank + j], inputShape[j]);
                }
            }

            return shapeOut;
        }

        public static SymbolicTensorShape Slice(SymbolicTensorShape[] inputShapes, PartialTensor starts, PartialTensor ends, PartialTensor? axesOptional = null, PartialTensor? stepsOptional = null)
        {
            var dataShape = inputShapes[0];

            if (!dataShape.hasRank)
                return SymbolicTensorShape.UnknownShape;

            var axes = axesOptional ?? PartialTensor.Range(0, dataShape.rank);

            if (!axes.isPartiallyKnown)
                return SymbolicTensorShape.UnknownOfRank(dataShape.rank);

            var steps = stepsOptional ?? PartialTensor.ConstantOfShape(axes.shape, 1);

            var shapeOut = new SymbolicTensorShape(dataShape);

            uint axesBitMask = 0;
            for (var i = 0; i < axes.shape.length; i++)
            {
                var axisElement = axes[i];
                if (!axisElement.isValue)
                    return SymbolicTensorShape.UnknownOfRank(dataShape.rank);
                var axis = shapeOut.Axis(axisElement.value);
                Logger.AssertIsTrue(((axesBitMask >> axis) & 1U) == 0, "Slice.InputError: Axis cannot repeat in axes list");
                axesBitMask |= 1U << axis;
                shapeOut[axis] = SliceDim(dataShape[axis], starts[i], ends[i], steps[i]);
            }

            return shapeOut;
        }

        public static SymbolicTensorShape ResizeSizes(SymbolicTensorShape shapeX, SymbolicTensorShape shapeSizes, PartialTensor tensorSizes)
        {
            shapeSizes.DeclareRank(1);

            if (shapeSizes[0].isValue)
                shapeX.DeclareRank(shapeSizes[0].value);

            if (tensorSizes.isPartiallyKnown)
                return tensorSizes.ToSymbolicTensorShape();

            return SymbolicTensorShape.UnknownOfRankLike(shapeX);
        }

        public static SymbolicTensorShape ResizeScales(SymbolicTensorShape shapeX, SymbolicTensorShape shapeScales, float[] scales)
        {
            shapeScales.DeclareRank(1);

            if (shapeScales[0].isValue)
                shapeX.DeclareRank(shapeScales[0].value);

            if (scales == null)
                return SymbolicTensorShape.UnknownOfRankLike(shapeX);

            shapeX.DeclareRank(scales.Length);

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank);

            for (var i = 0; i < shapeOut.rank; i++)
            {
                shapeOut[i] = ScaleDim(shapeX[i], scales[i]);
            }

            return shapeOut;
        }

        public static SymbolicTensorShape Pool(SymbolicTensorShape shapeX, int[] kernelShape, int[] strides, int[] padding, AutoPad autoPad, bool ceilMode)
        {
            shapeX.DeclareRank(2 + kernelShape.Length);

            Logger.AssertIsTrue(strides == null || shapeX.rank - 2 == strides.Length, "Pool.InputError: strides must have same number of values as spatial dimensions or be null");
            Logger.AssertIsTrue(padding == null || (shapeX.rank - 2) * 2 == padding.Length, "Pool.InputError: padding must have twice the number of values as spatial dimensions or be null");

            var shapeOut = new SymbolicTensorShape(shapeX);

            for (var i = 2; i < shapeOut.rank; i++)
            {
                var s = strides == null ? 1 : strides[i - 2];
                var p = (padding == null || autoPad != AutoPad.NotSet) ? 0 : (padding[i - 2] + padding[i - 2 + (shapeX.rank - 2)]);
                shapeOut[i] = PoolDim(shapeX[i], kernelShape[i - 2], s, p, 1, ceilMode, autoPad);
            }

            return shapeOut;
        }

        public static SymbolicTensorShape Conv(SymbolicTensorShape shapeX, SymbolicTensorShape shapeKernel, SymbolicTensorShape shapeBias, int[] strides, int[] padding, int[] dilations, AutoPad autoPad)
        {
            if (!shapeX.hasRank)
                return SymbolicTensorShape.UnknownShape;

            shapeKernel.DeclareRank(shapeX.rank);
            shapeBias.DeclareRank(1);

            Logger.AssertIsTrue(strides == null || shapeX.rank - 2 == strides.Length, "Conv.InputError: strides must be the same length as the spatial dimensions or be null");
            Logger.AssertIsTrue(padding == null || 2 * (shapeX.rank - 2) == padding.Length, "Conv.InputError: padding must have two values per spatial dimension or be null");
            Logger.AssertIsTrue(dilations == null || shapeX.rank - 2 == dilations.Length, "Conv.InputError: dilations must be the same length as the spatial dimensions or be null");

            var shapeOut = SymbolicTensorShape.UnknownOfRank(shapeX.rank);

            shapeOut[0] = shapeX[0];
            shapeOut[1] = SymbolicTensorDim.MaxDefinedDim(shapeKernel[0], shapeBias[0]);

            for (var i = 2; i < shapeOut.rank; i++)
            {
                var stride = strides == null ? 1 : strides[i - 2];
                var pad = padding == null || autoPad != AutoPad.NotSet ? 0 : padding[i - 2] + padding[i - 2 + (shapeX.rank - 2)];
                var dilation = dilations == null ? 1 : dilations[i - 2];
                shapeOut[i] = ConvDim(shapeX[i], shapeKernel[i], stride, pad, dilation, autoPad);
            }

            return shapeOut;
        }

        public static SymbolicTensorShape ConvTranspose(SymbolicTensorShape shapeX, SymbolicTensorShape shapeKernel, SymbolicTensorShape shapeBias, int[] strides, int[] padding, AutoPad autoPad, int[] outputPadding)
        {
            if (!shapeX.hasRank)
                return SymbolicTensorShape.UnknownShape;

            shapeKernel.DeclareRank(shapeX.rank);

            Logger.AssertIsTrue(strides == null || shapeX.rank - 2 == strides.Length, "ConvTranspose.InputError: strides must have two less values than the rank of input or be null");
            Logger.AssertIsTrue(padding == null || 2 * (shapeX.rank - 2) == padding.Length, "ConvTranspose.InputError: padding must have two values per spatial dimension or be null");
            Logger.AssertIsTrue(outputPadding == null || shapeX.rank - 2 == outputPadding.Length, "ConvTranspose.InputError: outputPadding must have two values per spatial dimension or be null");

            var shapeOut = SymbolicTensorShape.Ones(shapeX.rank);

            shapeOut[0] = shapeX[0];
            shapeOut[1] = SymbolicTensorDim.MaxDefinedDim(shapeKernel[1], shapeBias[0]);

            for (var i = 2; i < shapeOut.rank; i++)
            {
                var stride = strides == null ? 1 : strides[i - 2];
                var pad = padding == null || autoPad != AutoPad.NotSet ? 0 : padding[i - 2] + padding[i - 2 + (shapeX.rank - 2)];
                var dilation = 1;
                var outputPad = outputPadding == null ? 0 : outputPadding[i - 2];
                shapeOut[i] = ConvTransposeDim(shapeX[i], shapeKernel[i], stride, pad, dilation, outputPad, autoPad);
            }

            return shapeOut;
        }

        public static SymbolicTensorShape GlobalPool(SymbolicTensorShape shapeX)
        {
            if (!shapeX.hasRank)
                return SymbolicTensorShape.UnknownShape;

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 3 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 3, shapeX.rank);

            var shapeOut = new SymbolicTensorShape(shapeX);

            for (var i = 2; i < shapeOut.rank; i++)
            {
                shapeOut[i] = SymbolicTensorDim.One;
            }

            return shapeOut;
        }

        public static SymbolicTensorShape LSTM(SymbolicTensorShape[] shapes, int hiddenSize, int numDirections, RnnLayout layout, out SymbolicTensorShape shapeY, out SymbolicTensorShape shapeYH, out SymbolicTensorShape shapeYC)
        {
            var shapeX = shapes[0];
            var shapeW = shapes[1];
            var shapeR = shapes[2];

            var seqLength = SymbolicTensorDim.Unknown;
            var batchSize = SymbolicTensorDim.Unknown;

            shapeX.DeclareRank(3);
            shapeW.DeclareRank(3);
            shapeR.DeclareRank(3);

            seqLength = SymbolicTensorDim.MaxDefinedDim(seqLength, layout == RnnLayout.SequenceFirst ? shapeX[0] : shapeX[1]);
            batchSize = SymbolicTensorDim.MaxDefinedDim(batchSize, layout == RnnLayout.SequenceFirst ? shapeX[1] : shapeX[0]);

            if (shapes.Length > 3 && shapes[3] is var shapeB)
                shapeB.DeclareRank(2);

            if (shapes.Length > 4 && shapes[4] is var shapeSequenceLens)
            {
                shapeSequenceLens.DeclareRank(1);
                batchSize = SymbolicTensorDim.MaxDefinedDim(batchSize, shapeSequenceLens[0]);
            }

            if (shapes.Length > 5 && shapes[5] is var shapeInitialH)
            {
                shapeInitialH.DeclareRank(3);
                batchSize = SymbolicTensorDim.MaxDefinedDim(batchSize, layout == RnnLayout.SequenceFirst ? shapeInitialH[1] : shapeInitialH[0]);
            }

            if (shapes.Length > 6 && shapes[6] is var shapeInitialC)
            {
                shapeInitialC.DeclareRank(3);
                batchSize = SymbolicTensorDim.MaxDefinedDim(batchSize, layout == RnnLayout.SequenceFirst ? shapeInitialC[1] : shapeInitialC[0]);
            }

            if (shapes.Length > 7 && shapes[7] is var shapeP)
                shapeP.DeclareRank(2);

            if (layout == RnnLayout.SequenceFirst)
            {
                shapeY = new SymbolicTensorShape(seqLength, new SymbolicTensorDim(numDirections), batchSize, new SymbolicTensorDim(hiddenSize));
                shapeYH = new SymbolicTensorShape(new SymbolicTensorDim(numDirections), batchSize, new SymbolicTensorDim(hiddenSize));
                shapeYC = new SymbolicTensorShape(new SymbolicTensorDim(numDirections), batchSize, new SymbolicTensorDim(hiddenSize));
            }
            else
            {
                shapeY = new SymbolicTensorShape(batchSize, seqLength, new SymbolicTensorDim(numDirections), new SymbolicTensorDim(hiddenSize));
                shapeYH = new SymbolicTensorShape(batchSize, new SymbolicTensorDim(numDirections), new SymbolicTensorDim(hiddenSize));
                shapeYC = new SymbolicTensorShape(batchSize, new SymbolicTensorDim(numDirections), new SymbolicTensorDim(hiddenSize));
            }

            return shapeY;
        }

        public static SymbolicTensorShape TopK(SymbolicTensorShape shapeX, SymbolicTensorShape shapeK, int axis, out SymbolicTensorShape shapeValues, out SymbolicTensorShape shapeIndices)
        {
            if (!shapeX.hasRank)
            {
                shapeValues = SymbolicTensorShape.UnknownShape;
                shapeIndices = SymbolicTensorShape.UnknownShape;
                return shapeValues;
            }

            shapeK.DeclareRank(1);
            if (shapeK[0].isValue)
                Logger.AssertIsTrue(shapeK[0].value == 1, "TopK.InputError: k must be a single value");

            shapeValues = new SymbolicTensorShape(shapeX);
            shapeIndices = new SymbolicTensorShape(shapeX);

            axis = shapeX.Axis(axis);

            shapeValues[axis] = SymbolicTensorDim.Unknown;
            shapeIndices[axis] = SymbolicTensorDim.Unknown;

            return shapeValues;
        }

        public static SymbolicTensorShape PRelu(SymbolicTensorShape shapeX, SymbolicTensorShape shapeSlope)
        {
            if (!shapeX.hasRank)
                return SymbolicTensorShape.UnknownShape;

            if (!shapeSlope.hasRank)
                return shapeX;

            Logger.AssertIsTrue(shapeSlope.rank <= shapeX.rank, "PRelu.InputError: slope shape must be unidirectional broadcastable to input");
            var numInitialDims = shapeX.rank - shapeSlope.rank;
            var shapeOut = new SymbolicTensorShape(shapeX);

            for (var i = 0; i < shapeSlope.rank; i++)
            {
                if (shapeSlope[i].isValue && shapeSlope[i].value == 1)
                    continue;
                shapeOut[numInitialDims + i] = SymbolicTensorDim.MaxDefinedDim(shapeOut[numInitialDims + i], shapeSlope[i]);
            }

            return shapeOut;
        }

        public static SymbolicTensorShape FromShape(SymbolicTensorShape shapeX)
        {
            shapeX.DeclareRank(1);

            if (shapeX[0].isValue)
                return SymbolicTensorShape.UnknownOfRank(shapeX[0].value);

            return SymbolicTensorShape.UnknownShape;
        }

        public static SymbolicTensorShape Expand(SymbolicTensorShape shapeX, SymbolicTensorShape shapeShape)
        {
            if (!shapeX.hasRank)
                return SymbolicTensorShape.UnknownShape;

            shapeShape.DeclareRank(1);

            if (shapeShape[0].isValue)
                return SymbolicTensorShape.UnknownOfRank(Mathf.Max(shapeX.rank, shapeShape[0].value));

            return SymbolicTensorShape.UnknownShape;
        }

        public static SymbolicTensorShape Expand(SymbolicTensorShape shapeX, PartialTensor shapeTensor)
        {
            return shapeTensor.ToSymbolicTensorShape().Broadcast(shapeX);
        }

        public static SymbolicTensorShape Shape(SymbolicTensorShape shapeShape, int start, int end)
        {
            if (start == end)
                return new SymbolicTensorShape(SymbolicTensorDim.Zero);

            if (!shapeShape.hasRank)
                return SymbolicTensorShape.UnknownOfRank(1);

            start = start < 0 ? start + shapeShape.rank : start;
            end = end < 0 ? end + shapeShape.rank : end;
            start = Mathf.Clamp(start, 0, shapeShape.rank);
            end = Mathf.Clamp(end, 0, shapeShape.rank);

            Logger.AssertIsTrue(end >= start, "Shape.InputError: start value cannot be greater than end value for shape slicing");

            return new SymbolicTensorShape(new SymbolicTensorDim(end - start));
        }

        public static SymbolicTensorShape ScaleBias(SymbolicTensorShape shapeX, SymbolicTensorShape shapeScale, SymbolicTensorShape shapeBias)
        {
            var c = SymbolicTensorDim.Unknown;
            shapeScale.DeclareRank(1);
            c = SymbolicTensorDim.MaxDefinedDim(c, shapeScale[0]);
            shapeBias.DeclareRank(1);
            c = SymbolicTensorDim.MaxDefinedDim(c, shapeBias[0]);
            if (!shapeX.hasRank)
                return SymbolicTensorShape.UnknownShape;

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 2 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 2, shapeX.rank);

            var shapeOut = new SymbolicTensorShape(shapeX);
            shapeOut[1] = SymbolicTensorDim.MaxDefinedDim(shapeOut[1], c);
            return shapeOut;
        }

        public static SymbolicTensorShape ScatterElements(SymbolicTensorShape shapeX, SymbolicTensorShape shapeIndices, SymbolicTensorShape shapeUpdates, int axis, ScatterReductionMode reduction)
        {
            if (!shapeX.hasRank && !shapeIndices.hasRank && !shapeUpdates.hasRank)
                return SymbolicTensorShape.UnknownShape;

            if (!shapeX.hasRank && shapeIndices.hasRank)
                shapeX = SymbolicTensorShape.UnknownOfRank(shapeIndices.rank);

            if (!shapeX.hasRank && shapeUpdates.hasRank)
                shapeX = SymbolicTensorShape.UnknownOfRank(shapeUpdates.rank);

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            shapeIndices.DeclareRank(shapeX.rank);
            shapeUpdates.DeclareRank(shapeX.rank);

            // throw error if axis incorrect
            shapeX.Axis(axis);

            // throw error if indices and updates don't match
            for (var i = 0; i < shapeIndices.rank; i++)
            {
                SymbolicTensorDim.MaxDefinedDim(shapeIndices[i], shapeUpdates[i]);
            }

            return shapeX;
        }

        public static SymbolicTensorShape ScatterND(SymbolicTensorShape shapeX, SymbolicTensorShape shapeIndices, SymbolicTensorShape shapeUpdates, ScatterReductionMode reduction)
        {
            Logger.AssertIsTrue(shapeIndices.hasRank ? shapeIndices.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", shapeIndices.rank, 1);

            if (shapeIndices.hasRank && shapeUpdates.hasRank && shapeIndices[-1].isValue)
                shapeX.DeclareRank(shapeUpdates.rank - (shapeIndices.rank - shapeIndices[-1].value - 1));

            if (!shapeX.hasRank)
                return SymbolicTensorShape.UnknownShape;

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            return shapeX;
        }

        public static SymbolicTensorShape Normalization(SymbolicTensorShape shapeX, SymbolicTensorShape shapeScale, SymbolicTensorShape shapeBias)
        {
            var c = SymbolicTensorDim.Unknown;
            shapeScale.DeclareRank(1);
            c = SymbolicTensorDim.MaxDefinedDim(c, shapeScale[0]);
            shapeBias.DeclareRank(1);
            c = SymbolicTensorDim.MaxDefinedDim(c, shapeBias[0]);
            if (!shapeX.hasRank)
                return SymbolicTensorShape.UnknownShape;

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 2 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 2, shapeX.rank);
            shapeScale.DeclareRank(1);

            var shapeOut = new SymbolicTensorShape(shapeX);
            shapeOut[1] = SymbolicTensorDim.MaxDefinedDim(shapeOut[1], c);
            return shapeOut;
        }

        public static SymbolicTensorShape LRN(SymbolicTensorShape shapeX)
        {
            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 2 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 2, shapeX.rank);

            return shapeX;
        }

        public static SymbolicTensorShape Tile(SymbolicTensorShape shapeX, SymbolicTensorShape shapeRepeats, PartialTensor tensorRepeats)
        {
            shapeRepeats.DeclareRank(1);

            if (!tensorRepeats.isPartiallyKnown)
            {
                if (shapeRepeats[0].isValue && !shapeX.hasRank)
                    shapeX = SymbolicTensorShape.UnknownOfRank(shapeRepeats[0].value);
                Logger.AssertIsTrue(!shapeRepeats[0].isValue || shapeRepeats[0].value == shapeX.rank, "Tile.InputError: repeats value must be equal to input rank");
                return !shapeX.hasRank ? SymbolicTensorShape.UnknownShape : SymbolicTensorShape.UnknownOfRank(shapeX.rank);
            }

            tensorRepeats.AssertRank(1);
            shapeX.DeclareRank(tensorRepeats.shape.length);

            var shapeOut = new SymbolicTensorShape(shapeX);
            for (var i = 0; i < shapeOut.rank; i++)
            {
                shapeOut[i] *= tensorRepeats[i].ToSymbolicTensorDim();
            }
            return shapeOut;
        }

        public static SymbolicTensorDim SliceDim(SymbolicTensorDim dimX, PartialTensorElement start, PartialTensorElement end, PartialTensorElement step)
        {
            Logger.AssertIsTrue(step != 0, "Slice.InputError: Step cannot be 0");

            if (dimX.isValue && start.isValue && end.isValue && step.isValue)
                return new SymbolicTensorDim(ShapeInference.SliceDim(dimX.value, start.value, end.value, step.value));

            if (start.isUnknown || end.isUnknown)
                return SymbolicTensorDim.Unknown;

            if (start == end)
                return SymbolicTensorDim.Zero;

            var dimXElement = PartialTensorElement.FromSymbolicTensorDim(dimX);
            if (step > 0)
            {
                if (start == dimXElement || start == int.MaxValue || start >= dimX)
                    return SymbolicTensorDim.Zero;
                if (end == 0 || end == int.MinValue || dimX >= -end)
                    return SymbolicTensorDim.Zero;
                if (step == 1 && (start == 0 || start == int.MinValue) && (end == dimXElement || end == int.MaxValue))
                    return dimX;
            }
            else if (step < 0)
            {
                if (end == dimXElement || end == int.MaxValue || end >= dimX)
                    return SymbolicTensorDim.Zero;
                if (start == 0 || start == int.MinValue || dimX >= -start)
                    return SymbolicTensorDim.Zero;
                if (step == -1 && (end == -1 || end == int.MinValue) && (start == dimX || start == int.MaxValue))
                    return dimX;
            }

            return SymbolicTensorDim.Unknown;
        }

        static SymbolicTensorDim ScaleDim(SymbolicTensorDim dimX, float scale)
        {
            // ReSharper disable once CompareOfFloatsByEqualityOperator
            if (scale == 1f)
                return dimX;

            if (dimX.isValue)
                return new SymbolicTensorDim(Mathf.RoundToInt(dimX.value * scale));

            return SymbolicTensorDim.Unknown;
        }

        static SymbolicTensorDim ConvDim(SymbolicTensorDim dimX, SymbolicTensorDim dimKernel, int stride, int padding, int dilation, AutoPad autoPad)
        {
            if (dimKernel.isValue)
                return PoolDim(dimX, dimKernel.value, stride, padding, dilation, false, autoPad);

            if (dimKernel.isParam && (autoPad is AutoPad.SameLower || autoPad is AutoPad.SameUpper))
                return PoolDim(dimX, 0, stride, padding, dilation, false, autoPad);

            return SymbolicTensorDim.Unknown;
        }

        static SymbolicTensorDim ConvTransposeDim(SymbolicTensorDim dimX, SymbolicTensorDim dimKernel, int stride, int padding, int dilation, int outputPadding, AutoPad autoPad)
        {
            if (autoPad == AutoPad.NotSet)
                return stride * (dimX - 1) + outputPadding + (dimKernel - 1) * dilation + 1 - padding;

            return dimX * stride;
        }

        static SymbolicTensorDim PoolDim(SymbolicTensorDim dimX, int kernel, int stride, int padding, int dilation, bool ceilMode, AutoPad autoPad)
        {
            switch (autoPad)
            {
                case AutoPad.Valid:
                    return (dimX - ((kernel - 1) * dilation + 1) + 1).DivideWithRounding(stride, 1);
                case AutoPad.SameLower:
                case AutoPad.SameUpper:
                    return dimX.DivideWithRounding(stride, 1);
                case AutoPad.NotSet:
                    return (dimX + padding - ((kernel - 1) * dilation + 1)).DivideWithRounding(stride, ceilMode ? 1 : -1) + 1;
                default:
                    throw new InvalidEnumArgumentException();
            }
        }

        public static SymbolicTensorShape Dense(SymbolicTensorShape shapeX, SymbolicTensorShape shapeW, SymbolicTensorShape shapeB)
        {
            var shapeOut = shapeX.MatMul(shapeW);
            if (shapeOut.hasRank)
                shapeOut[-1] = SymbolicTensorDim.MaxDefinedDim(shapeB[0], shapeOut[-1]);
            return shapeOut;
        }

        /// <summary>
        /// Split a symbolic dim as equally as possible into numOutputs
        /// this is used for the Split layer
        /// return as a PartialTensor to easily work with case where splits are
        /// given as PartialTensor
        /// </summary>
        public static PartialTensor SplitDim(SymbolicTensorDim dim, int numOutputs)
        {
            // initialize as unknown partial tensor of length numOutputs
            var symbolicSplit = new PartialTensor(new TensorShape(numOutputs));

            if (dim.isParam && numOutputs == 1)
                symbolicSplit[0] = new PartialTensorElement(dim.param);

            if (!dim.isValue)
                return symbolicSplit;

            var splitLength = Mathf.CeilToInt(dim.value / (float)numOutputs);
            for (var i = 0; i < numOutputs - 1; i++)
            {
                symbolicSplit[i] = new PartialTensorElement(splitLength);
            }

            // final split length is the (possible smaller) remainder along the axis
            var lastSplitLength = dim.value - (splitLength * (numOutputs - 1));
            Logger.AssertIsTrue(lastSplitLength >= 0, "Split.InputError: split axis too small for numOutputs");
            symbolicSplit[numOutputs - 1] = new PartialTensorElement(lastSplitLength);

            return symbolicSplit;
        }
    }
}
