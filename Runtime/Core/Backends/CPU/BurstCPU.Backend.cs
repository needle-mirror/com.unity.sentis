using UnityEngine;
using UnityEngine.Assertions;
using System;
using Unity.Collections;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Sentis.BurstTensorData;

namespace Unity.Sentis {

// BurstCPU.Core.cs -- definition of class CPUBackend, Pin(), BurstTensorData
// BurstCPU.Backend.cs  -- impl. IBackend, job schedulers
// BurstCPU.Jobs.cs -- impl. jobs

public partial class CPUBackend
{
    internal static FencedMemoryAlloc s_tmpMemBlock0 = new FencedMemoryAlloc();
    internal static FencedMemoryAlloc s_tmpMemBlock1 = new FencedMemoryAlloc();

    static BLASPlugin s_BLAS = BLASPluginFactory.CreateNativeBLASPlugin();

    unsafe void ScheduleSGEMM(
        IDependableMemoryResource pinX, int XM, int XN,
        IDependableMemoryResource pinY, int YM, int YN,
        IDependableMemoryResource pinO, int OM, int ON,
        bool transposeA = false, bool transposeB = false, bool accumulateC = false, int offsetY = 0)
    {
        var data = new BatchMatrixMultiplyHelper();
        data.A = (float*)pinX.rawPtr;
        data.B = (float*)pinY.rawPtr + offsetY;
        data.C = (float*)pinO.rawPtr;
        data.M = transposeA ? XN : XM;
        data.N = transposeB ? YM : YN;
        data.K = transposeA ? XM : XN;
        data.lda = XN;
        data.ldb = YN;
        data.ldc = ON;
        data.transposeA = transposeA;
        data.transposeB = transposeB;
        data.accumulateC = accumulateC;
        data.batchCount = 1;

        JobHandle dependsOn = JobHandle.CombineDependencies(pinO.reuse, pinX.fence, pinY.fence);
        JobHandle jobFence;

        if (s_BLAS != null)
        {
            var job = new BatchMatrixMultiplyWithPluginJob();
            job.data = data;
            jobFence = job.Schedule(dependsOn);
        }
        else
        {
            var job = new BatchMatrixMultiplyJob();
            job.data = data;
            jobFence = job.Schedule(dependsOn);
        }

        pinO.fence = pinX.reuse = pinY.reuse = jobFence;
    }

    /// <inheritdoc/>
    public virtual TensorFloat MatMul2D(TensorFloat X, bool xTranspose, TensorFloat Y, bool yTranspose)
    {
        var Oshape = ShapeInference.Gemm(X.shape, Y.shape, xTranspose, yTranspose);
        if (X.shape.HasZeroDims() || Y.shape.HasZeroDims())
            return ConstantOfShape(Oshape, 0.0f);
        var O = NewOutputTensorFloat(Oshape);

        var pinX = Pin(X);
        var pinY = Pin(Y);
        var pinO = Pin(O, clearOnInit: false);

        ScheduleSGEMM(
            pinX, X.shape[0], X.shape[1],
            pinY, Y.shape[0], Y.shape[1],
            pinO, O.shape[0], O.shape[1],
            transposeA: xTranspose, transposeB: yTranspose);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat MatMul(TensorFloat X, TensorFloat Y)
    {
        var Oshape = X.shape.MatMul(Y.shape);
        if (X.shape.HasZeroDims() || Y.shape.HasZeroDims())
            return ConstantOfShape(Oshape, 0.0f);
        var O = NewOutputTensorFloat(Oshape);

        var pinX = Pin(X);
        var pinY = Pin(Y);
        var pinO = Pin(O, clearOnInit: false);

        var xShape = X.shape.rank == 1 ? new TensorShape(1, X.shape[0]) : X.shape;
        var yShape = Y.shape.rank == 1 ? new TensorShape(Y.shape[0], 1) : Y.shape;

        var data = new BatchMatrixMultiplyHelper();
        unsafe
        {
            data.A = (float*)pinX.rawPtr;
            data.B = (float*)pinY.rawPtr;
            data.C = (float*)pinO.rawPtr;
            data.M = xShape[-2];
            data.N = yShape[-1];
            data.K = xShape[-1];
            data.lda = data.K;
            data.ldb = data.N;
            data.ldc = data.N;
            data.Prepare(xShape, yShape);
        }

        JobHandle dependsOn = JobHandle.CombineDependencies(pinO.reuse, pinX.fence, pinY.fence);
        JobHandle jobFence;

        if (s_BLAS != null)
        {
            var job = new BatchMatrixMultiplyWithPluginJob();
            job.data = data;
            jobFence = job.Schedule(dependsOn);
        }
        else
        {
            var job = new BatchMatrixMultiplyJob();
            job.data = data;
            jobFence = job.Schedule(dependsOn);
        }

        pinO.fence = pinX.reuse = pinY.reuse = jobFence;

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Dense(TensorFloat X, TensorFloat W, TensorFloat B, Layers.FusableActivation fusedActivation)
    {
        var Oshape = X.shape.MatMul(W.shape);
        var O = (fusedActivation != Layers.FusableActivation.None) ? NewTempTensorFloat(Oshape) : NewOutputTensorFloat(Oshape);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = Pin(X);
        var pinW = Pin(W);
        var pinB = Pin(B);
        var pinO = Pin(O, clearOnInit: false);

        { // O = broadcast(B)
            // @TODO: move broadcast B directly into MatrixMultiplyJob
            var job = new VectorBroadcast1DJob();
            job.length = B.shape[0];
            job.repeat = O.shape.length / B.shape[0];
            job.ScheduleXO(pinB, pinO);
        }

        ScheduleSGEMM(
            pinX, X.shape.Length(0, -1), X.shape[-1],
            pinW, W.shape[0], W.shape[1],
            pinO, O.shape.Length(0, -1), O.shape[-1], accumulateC: true);

        if (fusedActivation != Layers.FusableActivation.None)
            O = ApplyFusedActivation(O, fusedActivation);

        return O;
    }

    TensorFloat ApplyFusedActivation(TensorFloat X, Layers.FusableActivation fusedActivation)
    {
        switch (fusedActivation)
        {
            case Layers.FusableActivation.None:
                return X;
            case Layers.FusableActivation.Relu:
                return Relu(X);
            default:
                throw new NotImplementedException();
        }
    }

    /// <inheritdoc/>
    public virtual TensorFloat Conv(TensorFloat X, TensorFloat K, TensorFloat B, int groups, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
    {
        if (X.shape.rank > 5)
            return ConvND(X, K, B, groups, strides, pads, dilations, fusedActivation);

        var Oshape = ShapeInference.Conv(X.shape, K.shape, groups, strides, pads, dilations);
        var O = NewOutputTensorFloat(Oshape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new ConvJob();
        int arrayLength = job.Prepare(X.shape, K.shape, O.shape, groups, strides, pads, dilations, fusedActivation);
        if (B != null)
        {
            job.useBias = true;
            job.ScheduleXSBO(Pin(X), Pin(K), Pin(B), Pin(O, clearOnInit: false), arrayLength, 1);
        }
        else
        {
            job.useBias = false;
            var pinX = Pin(X);
            var pinW = Pin(K);
            var pinO = Pin(O, clearOnInit: false);
            var fenceBeforeJobStart = JobHandle.CombineDependencies(pinX.fence, pinW.fence, pinO.reuse);
            unsafe
            {
                job.X = new ReadOnlyMemResource { ptr = pinX.rawPtr };
                job.S = new ReadOnlyMemResource { ptr = pinW.rawPtr };
                job.O = new ReadWriteMemResource { ptr = pinO.rawPtr };
            }
            var jobFence = job.Schedule(arrayLength, 1, fenceBeforeJobStart);
            pinX.reuse = jobFence;
            pinW.reuse = jobFence;
            pinO.fence = jobFence;
        }

        O = ApplyFusedActivation(O, fusedActivation);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ConvTranspose(TensorFloat X, TensorFloat W, TensorFloat B, Span<int> strides, Span<int> pads, Span<int> outputPadding, Layers.FusableActivation fusedActivation)
    {
        if (X.shape.rank >= 5)
            return ConvTransposeND(X, W, B, strides, pads, outputPadding, fusedActivation);

        var Oshape = ShapeInference.ConvTranspose(X.shape, W.shape, strides, pads, outputPadding);
        var O = (fusedActivation != Layers.FusableActivation.None) ? NewTempTensorFloat(Oshape) : NewOutputTensorFloat(Oshape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new ConvTransposeJob();

        job.Prepare(X.shape, W.shape, O.shape, strides, pads);

        var batchCount = X.shape[0];
        var inputChannels = X.shape[1];
        var outputChannels = O.shape[1];

        var columnBuffer = NewTempTensorFloat(new TensorShape(outputChannels * job.kernelSize, job.inputSize));
        var pinCB = Pin(columnBuffer, clearOnInit: false);

        for (var batchIndex = 0; batchIndex < batchCount; batchIndex++)
        {
            ScheduleSGEMM(
                Pin(W), inputChannels, outputChannels * job.kernelSize,
                Pin(X), inputChannels, job.inputSize,
                pinCB, outputChannels * job.kernelSize, job.inputSize,
                transposeA: true, offsetY: batchIndex * inputChannels * job.inputSize);

            job.offsetO = batchIndex * outputChannels * job.outputSize;
            if (B != null)
            {
                job.useBias = true;
                job.ScheduleXBO(pinCB, Pin(B), Pin(O, clearOnInit: false), outputChannels, 1);
            }
            else
            {
                job.useBias = false;
                var pinO = Pin(O, clearOnInit: false);
                var fenceBeforeJobStart = JobHandle.CombineDependencies(pinCB.fence, pinO.reuse);
                unsafe
                {
                    job.X = new ReadOnlyMemResource { ptr = pinCB.rawPtr };
                    job.O = new ReadWriteMemResource { ptr = pinO.rawPtr };
                }
                var jobFence = job.Schedule(outputChannels, 1, fenceBeforeJobStart);
                pinCB.reuse = jobFence;
                pinO.fence = jobFence;
            }
        }

        if (fusedActivation != Layers.FusableActivation.None)
            O = ApplyFusedActivation(O, fusedActivation);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Resize(TensorFloat X, ReadOnlySpan<float> scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode = Layers.NearestMode.RoundPreferFloor, Layers.CoordTransformMode coordTransformMode = Layers.CoordTransformMode.HalfPixel)
    {
        int rankX = X.shape.rank;

        // Handle only the common cases of NCHW or NCDHW resizes where NC is not scaled and delegate
        // the uncommon cases to the reference implementation.
        if ((rankX != 4 && rankX != 5) || scale[0] != 1.0f || scale[1] != 1.0f)
            return ResizeND(X, scale, interpolationMode, nearestMode, coordTransformMode);

        var O = NewOutputTensorFloat(ShapeInference.Resize(X.shape, scale));
        if (O.shape.HasZeroDims())
            return O;

        var pinX = Pin(X);
        var pinO = Pin(O, clearOnInit: false);

        // Fast path for 2D nearest mode resize where the scales are integers.
        if (rankX == 4 && interpolationMode == Layers.InterpolationMode.Nearest &&
            scale[2] == math.floor(scale[2]) && scale[3] == math.floor(scale[3]) &&
            (nearestMode == Layers.NearestMode.Floor || nearestMode == Layers.NearestMode.RoundPreferFloor) &&
            coordTransformMode == Layers.CoordTransformMode.Asymmetric)
        {
            var job = new UpsampleNearest2DJob();
            job.inputHeight = X.shape[2];
            job.inputWidth = X.shape[3];
            job.scaleHeight = (int)scale[2];
            job.scaleWidth = (int)scale[3];

            job.ScheduleBatchXO(pinX, pinO, X.shape.length, 1024);

            return O;
        }

        FencedMemoryAlloc tables = s_tmpMemBlock0;
        int spatialDims = rankX - 2;

        // Compute the number of output dimension indices for the helper tables constructed below.
        int tableElementCount = 0;
        for (int i = 2; i < rankX; i++)
            tableElementCount += O.shape[i];

        if (interpolationMode == Layers.InterpolationMode.Linear)
        {
            // For each output dimension index, cache the lower and upper input indices and the
            // fractional distribution for the linear interpolation.
            tables.Allocate(tableElementCount * 3, JobsUtility.CacheLineSize, Allocator.TempJob);

            var initTablesJob = new ResizeLinearInitTablesJob();
            var resizeJob = new ResizeLinearJob();

            unsafe
            {
                initTablesJob.spatialDims = spatialDims;
                resizeJob.spatialDims = spatialDims;
                resizeJob.inputSize = 1;

                for (int i = 0; i < spatialDims; i++)
                {
                    initTablesJob.inputShape[i] = X.shape[2 + i];
                    initTablesJob.outputShape[i] = O.shape[2 + i];
                    resizeJob.inputSize *= X.shape[2 + i];
                    resizeJob.outputShape[i] = O.shape[2 + i];

                    OpsUtils.GetScaleAndBias(X.shape[2 + i], O.shape[2 + i], scale[2 + i], coordTransformMode, interpolationMode, nearestMode, out float outputScale, out float outputBias);
                    initTablesJob.scales[i] = outputScale;
                    initTablesJob.biases[i] = outputBias;
                }
            }

            initTablesJob.ScheduleO(tables);
            resizeJob.ScheduleBatchXBO(pinX, tables, pinO, O.shape.length, 1024);
        }
        else
        {
            // For each output dimension index, cache the nearest input index.
            tables.Allocate(tableElementCount * 1, JobsUtility.CacheLineSize, Allocator.TempJob);

            var initTablesJob = new ResizeNearestInitTablesJob();
            var resizeJob = new ResizeNearestJob();

            unsafe
            {
                initTablesJob.spatialDims = spatialDims;
                if (nearestMode == Layers.NearestMode.RoundPreferCeil)
                    initTablesJob.nearestMode = Layers.NearestMode.Floor;
                else if (nearestMode == Layers.NearestMode.RoundPreferFloor)
                    initTablesJob.nearestMode = Layers.NearestMode.Ceil;
                else
                    initTablesJob.nearestMode = nearestMode;
                resizeJob.spatialDims = spatialDims;
                resizeJob.inputSize = 1;

                for (int i = 0; i < spatialDims; i++)
                {
                    initTablesJob.inputShape[i] = X.shape[2 + i];
                    initTablesJob.outputShape[i] = O.shape[2 + i];
                    resizeJob.inputSize *= X.shape[2 + i];
                    resizeJob.outputShape[i] = O.shape[2 + i];

                    OpsUtils.GetScaleAndBias(X.shape[2 + i], O.shape[2 + i], scale[2 + i], coordTransformMode, interpolationMode, nearestMode, out float outputScale, out float outputBias);
                    initTablesJob.scales[i] = outputScale;
                    initTablesJob.biases[i] = outputBias;
                }
            }

            initTablesJob.ScheduleO(tables);
            resizeJob.ScheduleBatchXBO(pinX, tables, pinO, O.shape.length, 1024);
        }

        unsafe
        {
            var job = new MemFreeJob();
            job.allocator = Allocator.TempJob;
            job.buffer0 = tables.rawPtr;
            job.Schedule(pinO.fence);
        }

        tables.ClearState();

        return O;
    }

    static readonly int[] permutationsDepthToSpaceDCR = new int[] { 0, 3, 4, 1, 5, 2 };
    static readonly int[] permutationsDepthToSpaceCRD = new int[] { 0, 1, 4, 2, 5, 3 };
    static readonly int[] permutationsSpaceToDepth = new int[] { 0, 3, 5, 1, 2, 4 };

    /// <inheritdoc/>
    public virtual TensorFloat DepthToSpace(TensorFloat X, int blocksize, Layers.DepthToSpaceMode mode)
    {
        var O = NewOutputTensorFloat(ShapeInference.DepthToSpace(X.shape, blocksize));
        if (O.shape.HasZeroDims())
            return O;

        int[] permutations;
        int dim1, dim3;

        if (mode == Layers.DepthToSpaceMode.DepthColumnRow)
        {
            permutations = permutationsDepthToSpaceDCR;
            dim1 = blocksize;
            dim3 = O.shape[1];
        }
        else
        {
            permutations = permutationsDepthToSpaceCRD;
            dim1 = O.shape[1];
            dim3 = blocksize;
        }

        var reshape = new TensorShape(X.shape[0], dim1, blocksize, dim3, X.shape[2], X.shape[3]);

        var job = new TransposeJob();
        job.iteratorX.Prepare(reshape, permutations);
        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat SpaceToDepth(TensorFloat X, int blocksize)
    {
        var O = NewOutputTensorFloat(ShapeInference.SpaceToDepth(X.shape, blocksize));
        if (O.shape.HasZeroDims())
            return O;

        var reshape = new TensorShape(X.shape[0], X.shape[1], X.shape[2] / blocksize, blocksize, X.shape[3] / blocksize, blocksize);

        var job = new TransposeJob();
        job.iteratorX.Prepare(reshape, permutationsSpaceToDepth);
        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat MaxPool(TensorFloat X, int[] pool, int[] stride, int[] pad)
    {
        if (X.shape.rank > 4)
            return MaxPoolND(X, pool, stride, pad);

        var O = NewOutputTensorFloat(ShapeInference.ApplyPool(X.shape, pool, stride, pad));
        if (O.shape.HasZeroDims())
            return O;

        var job = new MaxPool2DJob();
        if (X.shape.rank == 4)
        {
            job.inputHeight = X.shape[2];
            job.inputWidth = X.shape[3];
            job.outputHeight = O.shape[2];
            job.outputWidth = O.shape[3];
            job.poolHeight = pool[0];
            job.poolWidth = pool[1];
            job.strideHeight = stride[0];
            job.strideWidth = stride[1];
            job.padHeight = pad[0];
            job.padWidth = pad[1];
        }
        else
        {
            job.inputHeight = 1;
            job.inputWidth = X.shape[2];
            job.outputHeight = 1;
            job.outputWidth = O.shape[2];
            job.poolHeight = 1;
            job.poolWidth = pool[0];
            job.strideHeight = 1;
            job.strideWidth = stride[0];
            job.padHeight = 0;
            job.padWidth = pad[0];
        }
        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat AveragePool(TensorFloat X, int[] pool, int[] stride, int[] pad)
    {
        if (X.shape.rank > 4)
            return AveragePoolND(X, pool, stride, pad);

        var O = NewOutputTensorFloat(ShapeInference.ApplyPool(X.shape, pool, stride, pad));
        if (O.shape.HasZeroDims())
            return O;

        var job = new AveragePool2DJob();
        if (X.shape.rank == 4)
        {
            job.inputHeight = X.shape[2];
            job.inputWidth = X.shape[3];
            job.outputHeight = O.shape[2];
            job.outputWidth = O.shape[3];
            job.poolHeight = pool[0];
            job.poolWidth = pool[1];
            job.strideHeight = stride[0];
            job.strideWidth = stride[1];
            job.padHeight = pad[0];
            job.padWidth = pad[1];
        }
        else
        {
            job.inputHeight = 1;
            job.inputWidth = X.shape[2];
            job.outputHeight = 1;
            job.outputWidth = O.shape[2];
            job.poolHeight = 1;
            job.poolWidth = pool[0];
            job.strideHeight = 1;
            job.strideWidth = stride[0];
            job.padHeight = 0;
            job.padWidth = pad[0];
        }
        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat GlobalMaxPool(TensorFloat X)
    {
        var O = NewOutputTensorFloat(ShapeInference.GlobalPool(X.shape));
        if (O.shape.HasZeroDims())
            return O;

        var job = new ReduceMaxFloatJob();
        job.innerLength = 1;
        job.reduceLength = X.shape.Strides(1);
        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat GlobalAveragePool(TensorFloat X)
    {
        var O = NewOutputTensorFloat(ShapeInference.GlobalPool(X.shape));
        if (O.shape.HasZeroDims())
            return O;

        int strideX = X.shape.Strides(1);

        var job = new ReduceMeanFloatJob();
        job.innerLength = 1;
        job.reduceLength = strideX;
        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual void GlobalAverageVariancePool(TensorFloat O, TensorFloat X, int axis)
    {
        if (O.shape.HasZeroDims())
            return;

        var job = new GlobalAverageVariancePoolJob();
        job.spatialDims = X.shape.Length(axis);

        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length / 2, 1024);
    }

    /// <inheritdoc/>
    public virtual TensorFloat InstanceNormalization(TensorFloat X, TensorFloat S, TensorFloat B, float epsilon)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = Pin(X);
        var pinS = Pin(S);
        var pinB = Pin(B);
        var pinO = Pin(O, clearOnInit: false);

        // Allocate memory
        var reduceOpShape = ShapeInference.GlobalAverageVariancePool(X.shape);
        var meanVariance = NewTempTensorFloat(reduceOpShape);

        GlobalAverageVariancePool(meanVariance, X, 2);

        var job = new InstanceNormalizationTailJob();
        job.channels = X.shape[1];
        job.spatialDims = X.shape.length / (X.shape[0] * X.shape[1]);
        job.epsilon = epsilon;

        job.ScheduleXSBWO(pinX, pinS, pinB, Pin(meanVariance), pinO, O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat AxisNormalization(TensorFloat X, TensorFloat S, TensorFloat B, float epsilon)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        int axis = X.shape.Axis(-1);

        var reducedShape = X.shape.Reduce(axis);
        reducedShape[axis] = 2;

        int axisDim = X.shape[axis];
        int outerLength = X.shape.Length(0, -1);

        var meanVariance = NewTempTensorFloat(reducedShape);
        GlobalAverageVariancePool(meanVariance, X, -1);

        var job = new AxisNormalizationTailJob();
        job.axisDim = axisDim;
        job.outerLength = outerLength;
        job.epsilon = epsilon;

        job.ScheduleXSBWO(Pin(X), Pin(S), Pin(B), Pin(meanVariance), Pin(O, clearOnInit: false), outerLength, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ScaleBias(TensorFloat X, TensorFloat S, TensorFloat B)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new ScaleBiasJob();
        job.channels = X.shape[1];
        job.spatialLength = X.shape.Length(2);

        job.ScheduleBatchXSBO(Pin(X), Pin(S), Pin(B), Pin(O, clearOnInit: true), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat BatchNormalization(TensorFloat X, TensorFloat S, TensorFloat B, TensorFloat mean, TensorFloat variance, float epsilon)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var pinX = Pin(X);
        var pinS = Pin(S);
        var pinB = Pin(B);
        var pinM = Pin(mean);
        var pinV = Pin(variance);
        var pinO = Pin(O, clearOnInit: false);

        var job = new BatchNormalizationJob();
        job.channels = X.shape[1];
        job.spatialLength = X.shape.Length(2);
        job.epsilon = epsilon;
        unsafe
        {
            job.Xptr = (float*)pinX.rawPtr;
            job.Sptr = (float*)pinS.rawPtr;
            job.Bptr = (float*)pinB.rawPtr;
            job.Mptr = (float*)pinM.rawPtr;
            job.Vptr = (float*)pinV.rawPtr;
            job.Optr = (float*)pinO.rawPtr;
        }

        pinO.fence = pinX.reuse = pinS.reuse = pinB.reuse = pinM.reuse = pinV.reuse =
                    job.ScheduleBatch(O.shape.length, 1024, JobHandle.CombineDependencies(pinO.reuse, pinX.fence, JobHandle.CombineDependencies(pinS.fence, pinB.fence, JobHandle.CombineDependencies(pinM.fence, pinV.fence))));

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Cast(Tensor X, DataType toType)
    {
        if (X.dataType == toType)
            return Copy(X);

        var O = NewOutputTensor(X.shape, toType);
        if (O.shape.HasZeroDims())
            return O;

        if (toType == DataType.Float)
        {
            var job = new CastToFloatJob();
            job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }
        else
        {
            var job = new CastToIntJob();
            job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt IsInf(TensorFloat X, bool detectNegative, bool detectPositive)
    {
        var O = NewOutputTensorInt(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new IsInfJob();
        job.detectNegative = detectNegative;
        job.detectPositive = detectPositive;
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat PRelu(TensorFloat X, TensorFloat S)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new PReluJob();
        var outputLength = job.broadcast.Prepare(X.shape, S.shape);
        job.ScheduleBatchXBO(Pin(X), Pin(S), Pin(O, clearOnInit: false), outputLength, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat LeakyRelu(TensorFloat X, float alpha)
    {
        Logger.AssertIsTrue(alpha <= 1, "LeakyRelu.ValueError: alpha is supposed to be <= 1, got {0}", alpha);
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new LeakyReluJob();
        job.alpha = alpha;
        job.f1 = 0.5f * (1f + alpha);
        job.f2 = 0.5f * (1f - alpha);
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat HardSigmoid(TensorFloat X, float alpha, float beta)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new HardSigmoidJob();
        job.alpha = alpha;
        job.beta = beta;
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Elu(TensorFloat X, float alpha)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new EluJob();
        job.alpha = alpha;
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Gelu(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new GeluJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Selu(TensorFloat X, float alpha, float gamma)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new SeluJob();
        job.alpha = alpha;
        job.gamma = gamma;
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Clip(TensorFloat X, float min, float max)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new ClipJob();
        job.minV = min;
        job.maxV = max;
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Celu(TensorFloat X, float alpha)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new CeluJob();
        job.alpha = alpha;
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Shrink(TensorFloat X, float bias, float lambd)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new ShrinkJob();
        job.bias = bias;
        job.lambd = lambd;
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ThresholdedRelu(TensorFloat X, float alpha)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new ThresholdedReluJob();
        job.alpha = alpha;
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    static TensorIndex[] s_operandIndicesOne = new TensorIndex[1];
    static TensorShape[] s_operandShapesOne = new TensorShape[1];
    static TensorIndex[] s_operandIndicesTwo = new TensorIndex[2];
    static TensorShape[] s_operandShapesTwo = new TensorShape[2];

    /// <inheritdoc/>
    public virtual TensorFloat Einsum(string equation, params TensorFloat[] operands)
    {
        switch (operands.Length)
        {
            case 1:
            {
                s_operandShapesOne[0] = operands[0].shape;
                EinsumHelper.ParseEquationString(equation, s_operandShapesOne, ref s_operandIndicesOne, out var outputIndices, out var outputShape, out var sumIndices, out var sumShape, out var numIndices);

                var job = new EinsumOneJob();

                unsafe
                {
                    EinsumHelper.PinOperandStrides(operands[0].shape, s_operandIndicesOne[0], outputIndices, sumIndices, job.outStridesA, job.sumStridesA);
                    OpsUtils.PinTensorShapeStrides(outputShape, job.outLengths, job.outStrides);
                    OpsUtils.PinTensorShapeStrides(sumShape, job.sumLengths, job.sumStrides);
                }

                job.sumSize = sumShape.length;
                job.sumRank = sumShape.rank;
                job.outRank = outputShape.rank;

                var O = NewOutputTensorFloat(outputShape);

                job.ScheduleXO(Pin(operands[0]), Pin(O, clearOnInit: false), outputShape.length, 1024);
                return O;
            }
            case 2:
            {
                s_operandShapesTwo[0] = operands[0].shape;
                s_operandShapesTwo[1] = operands[1].shape;
                EinsumHelper.ParseEquationString(equation, s_operandShapesTwo, ref s_operandIndicesTwo, out var outputIndices, out var outputShape, out var sumIndices, out var sumShape, out var numIndices);

                var job = new EinsumTwoJob();

                unsafe
                {
                    EinsumHelper.PinOperandStrides(operands[0].shape, s_operandIndicesTwo[0], outputIndices, sumIndices, job.outStridesA, job.sumStridesA);
                    EinsumHelper.PinOperandStrides(operands[1].shape, s_operandIndicesTwo[1], outputIndices, sumIndices, job.outStridesB, job.sumStridesB);
                    OpsUtils.PinTensorShapeStrides(outputShape, job.outLengths, job.outStrides);
                    OpsUtils.PinTensorShapeStrides(sumShape, job.sumLengths, job.sumStrides);
                }

                job.sumSize = sumShape.length;
                job.sumRank = sumShape.rank;
                job.outRank = outputShape.rank;

                var O = NewOutputTensorFloat(outputShape);

                job.ScheduleXBO(Pin(operands[0]), Pin(operands[1]), Pin(O, clearOnInit: false), outputShape.length, 1024);
                return O;
            }
            default:
                return EinsumND(equation, operands);
        }
    }

    /// <inheritdoc/>
    public virtual Tensor Concat(Tensor[] tensors, int axis)
    {
        var O = NewOutputTensor(TensorShapeHelper.ConcatShape(tensors, axis), tensors[0].dataType);
        if (O.shape.HasZeroDims())
            return O;

        unsafe
        {
            // copy tensor data interleaved into O
            var pinO = Pin(O, clearOnInit: false);
            int offsetO = 0;

            var job = new CopyStrideJob();
            job.strideO = O.shape.Length(axis);
            var count = O.shape.Length(0, axis);

            var outputFences = stackalloc JobHandle[tensors.Length];
            int outputFencesCount = 0;

            for (int i = 0; i < tensors.Length; ++i)
            {
                if (tensors[i].shape.HasZeroDims())
                    continue;

                var pinX = Pin(tensors[i]);
                int lengthX = tensors[i].shape.Length(axis);

                job.strideX = lengthX;
                job.length = lengthX;
                job.offsetO = offsetO;

                var jobFence = job.ScheduleXO(pinX, pinO, count, 1024);
                outputFences[outputFencesCount++] = jobFence;

                offsetO += lengthX;
            }

            pinO.fence = JobHandleUnsafeUtility.CombineDependencies(outputFences, outputFencesCount);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Slice(Tensor X, ReadOnlySpan<int> starts, ReadOnlySpan<int> ends, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps)
    {
        var O = NewOutputTensor(X.shape.Slice(starts, ends, axes, steps), X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var job = new SliceJob();
        job.sliceParams.Prepare(X.shape, O.shape, starts, axes, steps);

        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Split(Tensor X, int axis, int start, int end)
    {
        var O = NewOutputTensor(X.shape.Split(axis, start, end), X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var job = new SliceJob();
        job.sliceParams.PrepareSplit(X.shape, O.shape, axis, start);

        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Pad(TensorFloat X, ReadOnlySpan<int> pad, Layers.PadMode padMode, float constant)
    {
        if (padMode != Layers.PadMode.Constant)
            Assert.IsFalse(X.shape.HasZeroDims(), "ValueError: zero dimensions input for Pad operator is not supported");

        var Oshape = X.shape.Pad(pad);
        if (X.shape.HasZeroDims())
            return ConstantOfShape(Oshape, constant);
        var O = NewOutputTensorFloat(Oshape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new PadJob();
        job.padMode = padMode;
        job.constant = constant;
        job.padParams.Prepare(X.shape, pad);
        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Transpose(Tensor X)
    {
        var O = NewOutputTensor(X.shape.Transpose(), X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var job = new TransposeJob();
        job.iteratorX.Prepare(X.shape);
        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Transpose(Tensor X, int[] permutations)
    {
        var O = NewOutputTensor(X.shape.Transpose(permutations), X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var job = new TransposeJob();
        job.iteratorX.Prepare(X.shape, permutations);
        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Pow(TensorFloat A, TensorInt B)
    {
        var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new PowFloatIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Where(TensorInt C, Tensor A, Tensor B)
    {
        var O = NewOutputTensor(A.shape.Broadcast(B.shape.Broadcast(C.shape)), A.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var job = new WhereJob();
        unsafe
        {
            OpsUtils.PinTensorShapeStrides(O.shape, job.shapeO, job.stridesO);
            OpsUtils.PinTensorShapeStrides(C.shape, job.shapeC, job.stridesC);
            OpsUtils.PinTensorShapeStrides(A.shape, job.shapeA, job.stridesA);
            OpsUtils.PinTensorShapeStrides(B.shape, job.shapeB, job.stridesB);
        }
        job.rank = O.shape.rank;

        job.ScheduleXSBO(Pin(C), Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Tile(Tensor X, ReadOnlySpan<int> repeats)
    {
        var O = NewOutputTensor(X.shape.Tile(repeats), X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var job = new TileJob();
        unsafe
        {
            OpsUtils.PinTensorShapeStrides(O.shape, job.shapeO, job.stridesO);
            OpsUtils.PinTensorShapeStrides(X.shape, job.shapeX, job.stridesX);
        }
        job.rank = O.shape.rank;

        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ConstantOfShape(TensorShape X, float value)
    {
        var O = NewOutputTensorFloat(X);
        if (O.shape.HasZeroDims())
            return O;
        MemSet(O, math.asint(value));
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt ConstantOfShape(TensorShape X, int value)
    {
        var O = NewOutputTensorInt(X);
        if (O.shape.HasZeroDims())
            return O;
        MemSet(O, value);
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Expand(Tensor X, TensorShape newShape)
    {
        var O = NewOutputTensor(X.shape.Broadcast(newShape), X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var job = new ExpandJob();
        unsafe
        {
            OpsUtils.PinTensorShapeStrides(O.shape, job.shapeO, job.stridesO);
            OpsUtils.PinTensorShapeStrides(X.shape, job.shapeX, job.stridesX);
        }
        job.rank = O.shape.rank;

        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    protected virtual Tensor CompressWithIndices(Tensor X, TensorInt indices, int numIndices, int axis)
    {
        var O = NewOutputTensor(ShapeInference.Compress(X.shape, numIndices, axis), X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var job = new GatherJob();
        job.innerLength = X.shape.Strides(axis);
        job.indicesLength = numIndices;
        job.axisDim = X.shape[axis];

        job.ScheduleBatchXBO(Pin(X), Pin(indices), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Gather(Tensor X, TensorInt indices, int axis)
    {
        var O = NewOutputTensor(ShapeInference.Gather(X.shape, indices.shape, axis), X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var job = new GatherJob();
        job.innerLength = X.shape.Strides(axis);
        job.indicesLength = indices.shape.length;
        job.axisDim = X.shape[axis];

        job.ScheduleBatchXBO(Pin(X), Pin(indices), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor GatherElements(Tensor X, TensorInt indices, int axis)
    {
        var O = NewOutputTensor(indices.shape, X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var job = new GatherElementsJob();
        job.endLength = X.shape.Strides(axis);
        job.startLength = X.shape.Length(0, axis);
        job.axisDim = X.shape[axis];

        job.ScheduleXBO(Pin(X), Pin(indices), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor GatherND(Tensor X, TensorInt indices, int batchDims)
    {
        var O = NewOutputTensor(ShapeInference.GatherND(X.shape, indices.shape, batchDims), X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var job = new GatherNDJob();
        job.rankX = X.shape.rank;
        job.rankO = O.shape.rank;
        job.rankIndices = indices.shape.rank;
        job.indexSize = indices.shape[-1];
        job.batchDims = batchDims;

        unsafe
        {
            OpsUtils.PinTensorShapeStrides(O.shape, job.shapeO, job.stridesO);
            OpsUtils.PinTensorShapeStrides(X.shape, job.shapeX, job.stridesX);
            OpsUtils.PinTensorShapeStrides(indices.shape, job.shapeIndices, job.stridesIndices);
        }

        job.ScheduleXBO(Pin(X), Pin(indices), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor ScatterElements(Tensor X, TensorInt indices, Tensor updates, int axis, Layers.ScatterReductionMode reduction)
    {
        if (X.dataType == DataType.Int && reduction != Layers.ScatterReductionMode.None)
            return ScatterElementsReduce(X as TensorInt, indices, updates as TensorInt, axis, reduction);

        var O = NewOutputTensor(X.shape, X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        MemCopy(X, O);

        var job = new ScatterElementsJob();
        job.endLength = X.shape.Strides(axis);
        job.axisDim = X.shape[axis];
        job.axisDimIndices = indices.shape[axis];
        job.reduction = (int)reduction;

        // When reduction != ScatterReductionMode.None, the reduction is allowed to have duplicate output
        // indices. To avoid race conditions updating the output tensor, force these reduction modes to run
        // on a single worker by setting the inner loop length to int.MaxValue.

        job.ScheduleXBO(Pin(updates), Pin(indices), Pin(O, clearOnInit: true), indices.shape.length,
            (reduction == Layers.ScatterReductionMode.None) ? 1024 : int.MaxValue);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ScatterND(TensorFloat X, TensorInt indices, TensorFloat updates, Layers.ScatterReductionMode reduction)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        MemCopy(X, O);

        int indexRemapDim = indices.shape[-1];
        int indicesLength = indices.shape.Length(0, -1);
        int updatesLength = updates.shape.length / indicesLength;

        var job = new ScatterNDJob();
        job.updatesLength = updatesLength;
        job.indexRemapDim = indexRemapDim;
        job.reduction = reduction;
        unsafe
        {
            int trailing = 1;
            for (int j = (indexRemapDim-1); j >= 0; j--)
            {
                job.trailing[j] = trailing;
                trailing *= X.shape[j];
            }
        }
        job.ScheduleXSBO(Pin(X), Pin(indices), Pin(updates), Pin(O, clearOnInit: true), updatesLength * indicesLength, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt OneHot(TensorInt X, int axis, int depth, int offValue, int onValue)
    {
        var O = NewOutputTensorInt(ShapeInference.OneHot(X.shape, axis, depth));
        if (O.shape.HasZeroDims())
            return O;

        var job = new OneHotJob();
        job.depth = depth;
        job.offValue = offValue;
        job.onValue = onValue;
        job.axis = O.shape.Axis(axis);
        unsafe
        {
            OpsUtils.PinTensorShapeStrides(O.shape, job.shapeO, job.stridesO);
            OpsUtils.PinTensorShapeStrides(X.shape, job.shapeX, job.stridesX);
        }
        job.rankO = O.shape.rank;

        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor[] TopK(TensorFloat X, int k, int axis, bool largest, bool sorted)
    {
        var outputShape = new TensorShape(X.shape);
        outputShape[axis] = k;

        var values = NewOutputTensorFloat(outputShape);
        var indices = NewOutputTensorInt(outputShape);
        if (outputShape.HasZeroDims())
            return new Tensor[] { values, indices };

        int reduceLength = X.shape[axis];
        int innerLength = X.shape.Strides(axis);
        int outerLength = X.shape.length / (reduceLength * innerLength);

        var pinX = Pin(X);
        var pinV = Pin(values, clearOnInit: false);
        var pinI = Pin(indices, clearOnInit: false);
        var job = new TopKJob();
        job.innerLength = innerLength;
        job.reduceLength = reduceLength;
        job.maxK = k;
        job.largest = largest;
        unsafe
        {
            job.Xptr = (float*)pinX.rawPtr;
            job.Valuesptr = (float*)pinV.rawPtr;
            job.Indicesptr = (int*)pinI.rawPtr;
        }
        pinX.reuse = pinV.fence = pinI.fence = job.Schedule(outerLength * innerLength, 32, JobHandle.CombineDependencies(pinX.fence, pinV.reuse, pinI.reuse));

        return new Tensor[] { values, indices };
    }

    /// <inheritdoc/>
    public virtual TensorFloat RoiAlign(TensorFloat X, TensorFloat Rois, TensorInt Indices, Layers.RoiPoolingMode mode, int outputHeight, int outputWidth, int samplingRatio, float spatialScale)
    {
        var O = NewOutputTensorFloat(ShapeInference.RoiAlign(X.shape, Rois.shape, Indices.shape, outputHeight, outputWidth));
        if (O.shape.HasZeroDims())
            return O;

        var job = new RoiAlignJob();
        job.numRois = Rois.shape[0];
        job.inputChannels = X.shape[1];
        job.inputHeight = X.shape[2];
        job.inputWidth  = X.shape[3];
        job.inputSpatialSize = X.shape[2] * X.shape[3];
        job.inputBatchOffset = X.shape[1] * X.shape[2] * X.shape[3];
        job.outputHeight = outputHeight;
        job.outputWidth = outputWidth;
        job.normalizeOHeight = 1.0f / outputHeight;
        job.normalizeOWidth = 1.0f / outputWidth;
        job.samplingRatio = samplingRatio;
        job.spatialScale = spatialScale;
        job.mode = mode;
        job.ScheduleBatchXSBO(Pin(X), Pin(Rois), Pin(Indices), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat RandomNormal(TensorShape s, float mean, float scale, float? seed)
    {
        var O = NewOutputTensorFloat(s);
        if (O.shape.HasZeroDims())
            return O;

        var job = new RandomNormalJob();
        job.seed = Random.GetOpSeed(seed);
        job.mean = mean;
        job.scale = scale;
        job.ScheduleO(Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat RandomUniform(TensorShape s, float low, float high, float? seed)
    {
        var O = NewOutputTensorFloat(s);
        if (O.shape.HasZeroDims())
            return O;

        var job = new RandomUniformJob();
        job.seed = Random.GetOpSeed(seed);
        job.low = low;
        job.high = high;
        job.ScheduleO(Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Bernoulli(TensorFloat X, DataType dataType, float? seed)
    {
        var O = NewOutputTensor(X.shape, dataType);
        if (O.shape.HasZeroDims())
            return O;

        var job = new BernoulliJob();
        job.seed = Random.GetOpSeed(seed);
        job.dataType = dataType;
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);

        return O;
    }

    /// <summary>
    /// Copy blocks of values from X to O, we copy 'count' blocks each of length 'length' values with initial offsets
    /// given by 'offsetX', 'offsetO' and with strides given by 'strideX', 'strideO'
    /// </summary>
    protected virtual void MemCopy(Tensor X, Tensor O, int length = -1, int offsetX = 0, int offsetO = 0)
    {
        length = length < 0 ? O.shape.length - offsetO : length;
        if (length == 0)
            return;
        Logger.AssertIsTrue(length > 0, "MemCopy.InputError: copy length must be greater than 0");
        Logger.AssertIsTrue(offsetX >= 0, "MemCopy.BoundsError: copy out of bounds for tensor X");
        Logger.AssertIsTrue(offsetX + length <= X.shape.length, "MemCopy.BoundsError: copy out of bounds for tensor X");
        Logger.AssertIsTrue(offsetO >= 0, "MemCopy.BoundsError: copy out of bounds for tensor O");
        Logger.AssertIsTrue(offsetO + length <= O.shape.length, "MemCopy.BoundsError: copy out of bounds for tensor O");
        var job = new CopyJob();
        job.offsetX = offsetX;
        job.offsetO = offsetO;
        job.length = length;
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false));
    }

    /// <summary>
    /// Copy blocks of values from X to O, we copy 'count' blocks each of length 'length' values with initial offsets
    /// given by 'offsetX', 'offsetO' and with strides given by 'strideX', 'strideO'
    /// </summary>
    protected virtual void MemCopyStride(Tensor X, Tensor O, int strideX, int strideO, int length, int count, int offsetX = 0, int offsetO = 0)
    {
        if (length == 0 || count == 0)
            return;
        Logger.AssertIsTrue(length > 0, "MemCopy.InputError: copy stride length must be greater than 0");
        Logger.AssertIsTrue(count > 0, "MemCopy.InputError: copy stride count must be greater than 0");
        Logger.AssertIsTrue(offsetX >= 0, "MemCopy.BoundsError: copy stride out of bounds for tensor X");
        Logger.AssertIsTrue(offsetX + (count - 1) * strideX + length <= X.shape.length, "MemCopy.BoundsError: copy stride out of bounds for tensor X");
        Logger.AssertIsTrue(offsetO >= 0, "MemCopy.BoundsError: copy stride out of bounds for tensor O");
        Logger.AssertIsTrue(offsetO + (count - 1) * strideO + length <= O.shape.length, "MemCopy.BoundsError: copy stride out of bounds for tensor O");
        var job = new CopyStrideJob();
        job.offsetX = offsetX;
        job.offsetO = offsetO;
        job.strideX = strideX;
        job.strideO = strideO;
        job.length = length;
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), count, 1024);
    }

    /// <summary>
    /// Set values of O to value
    /// </summary>
    protected virtual void MemSet(Tensor O, int value)
    {
        if (value == 0)
        {
            var job = new ClearJob();
            job.length = O.shape.length;
            job.ScheduleO(Pin(O, clearOnInit: false));
        }
        else
        {
            var job = new SetJob();
            job.memValue = value;
            job.ScheduleO(Pin(O, clearOnInit: false), O.shape.length, 1024);
        }
    }

    /// <summary>
    /// Computes a single pass LSTM either forward or reverse
    /// dirIndex and layout are used to calculate where to index the various
    /// tensors in bidirectional and batch first layout passes
    /// X has given layout
    /// W, R are cropped to single direction
    /// P, B are full number of directions
    /// Y has given layout and full number of directions (matches output of Layer)
    /// Y_h, Y_c are SequenceFirst layout and cropped to single direction
    /// HtxRT and XsixWT are temp vectors of the correct dimension for the intermediate results of the matmuls
    /// activations, activationAlpha and activationBeta have full number of dimensions
    /// </summary>
    protected virtual void SinglePassLSTM(TensorFloat X, TensorFloat W, TensorFloat R, TensorFloat B, TensorInt sequenceLens, TensorFloat P, TensorFloat Y, TensorFloat Y_h, TensorFloat Y_c, Layers.RnnActivation[] activations, float[] activationAlpha, float[] activationBeta, bool inputForget, float clip, bool isReverse, int dirIndex, Layers.RnnLayout layout)
    {
        var pinY = Pin(Y, clearOnInit: false);

        var pinX = Pin(X);
        var pinW = Pin(W);
        var pinB = Pin(B);
        var pinR = Pin(R);
        var pinP = Pin(P);
        var pinY_h = Pin(Y_h);
        var pinY_c = Pin(Y_c);
        var pinSequenceLens = Pin(sequenceLens);

        var numDirections = B.shape[0];
        var inputSize = X.shape[2];
        var hiddenSize = R.shape[2];

        var seqLength = X.shape[0];
        var batchSize = X.shape[1];

        var xStrideSeq = batchSize * 4 * hiddenSize;
        var xStrideBatch = 4 * hiddenSize;

        var yStrideDir = batchSize * hiddenSize;
        var yStrideSeq = numDirections * batchSize * hiddenSize;
        var yStrideBatch = hiddenSize;

        if (layout == Layers.RnnLayout.BatchFirst)
        {
            seqLength = X.shape[1];
            batchSize = X.shape[0];

            xStrideSeq = 4 * hiddenSize;
            xStrideBatch = seqLength * 4 * hiddenSize;

            yStrideDir = hiddenSize;
            yStrideSeq = numDirections * hiddenSize;
            yStrideBatch = seqLength * numDirections * hiddenSize;
        }

        var HtxRT = NewTempTensorFloat(new TensorShape(batchSize * 4 * hiddenSize));
        var XsixWT = NewTempTensorFloat(new TensorShape(seqLength * batchSize * 4 * hiddenSize));

        var pinHtxRT = Pin(HtxRT, clearOnInit: false);
        var pinXsixWT = Pin(XsixWT, clearOnInit: false);

        ScheduleSGEMM(pinX, seqLength * batchSize, inputSize, pinW, 4 * hiddenSize, inputSize, pinXsixWT, seqLength * batchSize, 4 * hiddenSize, transposeB: true);

        var endJob = new LSTMEndJob();
        endJob.fActivation = activations[3 * dirIndex + 0];
        endJob.fAlpha = activationAlpha[3 * dirIndex + 0];
        endJob.fBeta = activationBeta[3 * dirIndex + 0];
        endJob.gActivation = activations[3 * dirIndex + 1];
        endJob.gAlpha = activationAlpha[3 * dirIndex + 1];
        endJob.gBeta = activationBeta[3 * dirIndex + 1];
        endJob.hActivation = activations[3 * dirIndex + 2];
        endJob.hAlpha = activationAlpha[3 * dirIndex + 2];
        endJob.hBeta = activationBeta[3 * dirIndex + 2];
        endJob.hiddenSize = hiddenSize;
        endJob.xStride = xStrideBatch;
        endJob.yStride = yStrideBatch;
        endJob.inputForget = inputForget;
        endJob.clip = clip;
        unsafe
        {
            endJob.YHptr = (float*)pinY_h.rawPtr;
            endJob.YCptr = (float*)pinY_c.rawPtr;
            endJob.Bptr = (float*)pinB.rawPtr + dirIndex * 8 * hiddenSize;
            endJob.Pptr = (float*)pinP.rawPtr + dirIndex * 3 * hiddenSize;
            endJob.HtxRTptr = (float*)pinHtxRT.rawPtr;
            endJob.SequenceLensptr = (int*)pinSequenceLens.rawPtr;
        }

        for (var i = 0; i < seqLength; i++)
        {
            var seqIndex = isReverse ? seqLength - 1 - i : i;

            ScheduleSGEMM(pinY_h, batchSize, hiddenSize, pinR, 4 * hiddenSize, hiddenSize, pinHtxRT, batchSize, 4 * hiddenSize, transposeB: true);

            unsafe
            {
                endJob.seqIndex = seqIndex;
                endJob.Yptr = (float*)pinY.rawPtr + dirIndex * yStrideDir + seqIndex * yStrideSeq;
                endJob.XsixWTptr = (float*)pinXsixWT.rawPtr + seqIndex * xStrideSeq;

                pinY_h.fence = pinY_c.fence = pinY.fence = pinP.reuse = pinB.reuse = pinXsixWT.reuse = pinHtxRT.reuse = pinSequenceLens.reuse =
                    endJob.Schedule(batchSize, 1, JobHandle.CombineDependencies(pinY.reuse, pinY_h.reuse, JobHandle.CombineDependencies(pinY_c.reuse, pinP.fence, JobHandle.CombineDependencies(pinB.fence, pinXsixWT.fence, JobHandle.CombineDependencies(pinHtxRT.fence, pinSequenceLens.fence)))));
            }
        }
    }

    /// <summary>
    /// Prepares `Tensor` for use with CPU backend.
    /// </summary>
    /// <param name="X">`Tensor` to prepare for CPU backend.</param>
    /// <param name="clearOnInit">Whether to copy tensor data to CPU backend.</param>
    /// <returns>`Tensor` once prepared for CPU backend.</returns>
    public virtual Tensor PinToDevice(Tensor X, bool clearOnInit = true)
    {
        Pin(X, clearOnInit);
        return X;
    }
}
} // namespace Unity.Sentis
