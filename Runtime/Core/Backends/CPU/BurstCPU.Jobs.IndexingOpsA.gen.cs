// This is auto-generated -- do not modify directly

using UnityEngine;
using System;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using static Unity.Burst.Intrinsics.X86.Avx;
using static Unity.Burst.Intrinsics.X86.Fma;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Unity.Sentis
{
partial class CPUBackend
{


[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
internal unsafe struct TileJob : IJobParallelFor, IJobResourceDeclarationXO
{

    public int rank;
    public fixed int shapeO[8];
    public fixed int stridesO[8];
    public fixed int shapeX[8];
    public fixed int stridesX[8];
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        int indexX = 0;
        for (int axis = 0; axis < rank; axis++)
        {
            indexX += (((threadIdx / stridesO[(TensorShape.maxRank-1) - axis]) % shapeO[(TensorShape.maxRank-1) - axis]) % shapeX[(TensorShape.maxRank-1) - axis]) * stridesX[(TensorShape.maxRank-1) - axis];
        }

        Optr[threadIdx] = Xptr[indexX];
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
internal unsafe struct GatherElementsFastJob : IJobParallelFor, IJobResourceDeclarationXBO
{

    public int inputAxisSize;
    public int inputAxisElementStride;
    public int indicesAxisElementStride;
    public int indicesAxisMinusOneElementStride;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {

        // Xptr                             is the input tensor
        // Bptr                             is the indices tensor
        // Optr                             is the output tensor (shaped as the indices)
        // inputAxisSize                    is X.shape[axis], the size of the axis dimension for the input tensor (gather source)
        // indicesAxisElementStride         is the element-to-element stride for elements on the axis dimension in the indices and output tensor.
        // indicesAxisMinusOneElementStride is indicesAxisElementStride * indicesAxisSize, ie the stride for elements on axis-1 in the indices and output tensor.

        int trailingOffset = threadIdx % indicesAxisElementStride;
        int outerMostElementNum = threadIdx / (indicesAxisMinusOneElementStride);

        int index = (int)Bptr[threadIdx];
        index = index < 0 ? inputAxisSize + index : index;

        Optr[threadIdx] = Xptr[outerMostElementNum * inputAxisSize * inputAxisElementStride + index * inputAxisElementStride + trailingOffset];
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
internal unsafe struct GatherElementsJob : IJobParallelFor, IJobResourceDeclarationXBO
{

    public int inputAxisSize;
    public int posAxis;
    public int rank;
    public fixed int stridesO[8];
    public fixed int stridesX[8];
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        // Note that the strides arrays here are compacted at the head: "rank" values starting at element [0] !

        // Xptr                             is the input tensor (source of gather)
        // Bptr                             is the indices tensor
        // Optr                             is the output tensor (shaped as indices)
        // stridesX                         are the strides of the input tensor X
        // stridesO                         are the strides of the indices and output tensor O
        // posAxis                          is the positive gather axis
        // rank                             is the rank of the tensors
        int index = (int)Bptr[threadIdx];
        uint posIndex = (uint)(index < 0 ? inputAxisSize + index : index);

        uint remainder;
        uint idxOnCurDim;
        uint inputLinearIdx;
        int axis = (int)(uint)posAxis;
        int curDim;

        remainder = (uint)threadIdx;
        inputLinearIdx = 0;

        for (curDim = 0; curDim < axis; curDim++)
        {
            idxOnCurDim = remainder / (uint)stridesO[curDim];
            remainder = remainder % (uint)stridesO[curDim];
            inputLinearIdx += idxOnCurDim * (uint)stridesX[curDim];
        }

        // process axis dimension and scatter re-indexing
        remainder = remainder % (uint)stridesO[curDim];
        inputLinearIdx += posIndex * (uint)stridesX[curDim];
        curDim++;

        // We assume that the tensors are compact, no strides on the innermost dimensions, so we dont do the loop at curDim == rank - 1
        for (; curDim < rank - 1; curDim++)
        {
            idxOnCurDim = remainder / (uint)stridesO[curDim];
            remainder = remainder % (uint)stridesO[curDim];
            inputLinearIdx += idxOnCurDim * (uint)stridesX[curDim];
        }

        // curDim == rank - 1 == innermost (assume stride 1; also, obviously X.shape[rank-1] >= O and indices.shape[rank-1].
        // Also, even if curDim == rank (ie axis was rank-1 and we already processed last/innermost dim), we can still safely
        // do the last step in any case because the remainder of anything % 1 (stride[rank-1] is always 1) will be 0:
        inputLinearIdx += remainder;

        Optr[threadIdx] = Xptr[(int)inputLinearIdx];
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
internal unsafe struct ScatterElementsFastJob : IJobParallelFor, IJobResourceDeclarationXBO
{

    public int outAxisSize;
    public int indicesAxisElementStride;
    public int outAxisElementStride;
    public int indicesAxisMinusOneElementStride;
    public int reductionType;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        //int outAxisSize = axisDim;
        //int indicesAxisSize = axisDimIndices;
        //int indicesAxisElementStride = endLength;
        //int outAxisElementStride = endLengthX;
        //int reductionType = reduction;

        // Xptr                             is the updates tensor
        // Bptr                             is the indices tensor
        // Optr                             is the output tensor
        // outAxisSize                      is Optr.shape[axis], the size of the axis dimension for the output (and input) tensor
        // indicesAxisSize                  is Bptr.shape[axis]
        // indicesAxisElementStride         is the element-to-element stride for elements on the axis dimension in the indices and updates tensor.
        // outAxisElementStride             is the element-to-element stride for elements on the axis dimension in the indices and updates tensor.
        // indicesAxisMinusOneElementStride is indicesAxisElementStride * indicesAxisSize, ie the stride for elements on axis-1 in the indices and updates tensor.

        // Imagine casting the indices tensor data (on which the linear - aka flat - threadIdx and the dispatch is based/overlaid on)
        // into a new tensor having 3 dimensions: all the original innermost dimensions innermore to the axis are flattened into one
        // of size indicesAxisElementStride (which incidentally is the element stride for the axis dimension), the middle dimension is the axis
        // and the outermost dimension folds all the rest of the tensor data into 2D slices of indicesAxisSize X indicesAxisElementStride.
        //
        // trailingOffset will be the (also linear/flat since the innermost dim) element number on the innermost dim,
        // outerMostElementNum will be the 2D slice number on that folded-on, outermost, "axis - 1" dimension:
        //
        int trailingOffset = threadIdx % indicesAxisElementStride;
        //int outerMostElementNum = threadIdx / (indicesAxisElementStride * indicesAxisSize);
        int outerMostElementNum = threadIdx / (indicesAxisMinusOneElementStride);

        // Also note that the fast path here assumes that for innermost dimensions:
        //
        //      output.shape[trailing_i] == indices.shape[trailing_i] where "trailing_i" is axis+1, ..., rank-1 (ie all innermost dims innermore to the axis)
        //      OR
        //      indices.shape[trailing_outer_to_first_inner_non_matching_dim_i] == 1
        //            where "trailing_outer_to_first_inner_non_matching_dim_i" is axis+1, ..., first_non_matching_dim-1
        //            where "first_non_matching_dim" is the first dimension, when scanning from the innermost (rank-1) to outer, that
        //            doesn't match (all those innermore to that one are assumed to match).
        //
        // In that way, we have a continuous block of memory for tailing dimensions of the tensor that match, and any other don't actually have more than a single
        // element (index 0) in the indices tensor before we reach the axis dimencion.
        // In that way, any trailing offset calculated from the indices tensor linear (ie flat)  index "threadIdx", trailing_offset := threadIdx % indicesAxisElementStride
        // can't straddle or cross an element in an outermore non matching-size dimension (straddle meaning be at element != 0 on that outermore dimension)
        // before it actually crosses an axis element (which is properly handled).
        //
        // Also, for the outermost dimensions we have all of these conditions:
        //
        //      a) output.shape[axis] >= indices.shape[axis]
        //      b) output.shape[axis-1] >= indices.shape[axis-1]
        //
        //          and let the comparison on this axis-1 dim be noted "axis-1size is equal" or "axis-1size is NOT equal"
        //
        //          ie the axis dimension and the one "one step outermore" to the axis can also be different but with a caveat:
        //
        //      c)
        //
        //      ("axis-1size is equal" AND
        //            (  output.shape[outermost_i] == indices.shape[outermost_i]
        //            OR indices.shape[outermost_i] == 1                         ), where outermost_i is axis-2, ..., 0)
        //      OR
        //      ("axis-1size is NOT equal" AND indices.shape[outermost_i] == 1, where outermost_i is axis-2, ..., 0)
        //
        //      The later condition (c) can also thus be written as
        //
        //      (indices.shape[outermost_i] == 1, where outermost_i is axis-2, ..., 0)
        //      OR
        //      (output.shape[outermost_i] == indices.shape[outermost_i] AND output.shape[axis-1] == indices.shape[axis-1],
        //          where outermost_i is axis-2, ..., 0)
        //
        //      => ie for all dimensions outermore than axis, either they are all equal or
        //      the indices/updates tensor dimensions outermore than **axis-1** are of size one.
        //
        //      This later condition ensures that even if element numbers are not the same on axis-1, in the compacted/flattened tensor
        //      representation, outerMostElementNum in the indices of the compacted tensor can never straddle across a different element
        //      in one of the outermost_i dimensions in the (original - non compacted view of the) output tensor.
        //      This is obviously because the corresponding multi-index in the original tensor will have 0 subindices along those
        //      dimensions, and so the conversion to the output multi-index doesn't implicate any strides along those dimensions
        //      (since ... + 0*stride = + 0).
        //
        int index = (int)Bptr[threadIdx];
        index = index < 0 ? outAxisSize + index : index;

        if (reductionType == 0)
            Optr[outerMostElementNum * outAxisSize * outAxisElementStride + index * outAxisElementStride + trailingOffset] = Xptr[threadIdx];
        else if (reductionType == 1)
            Optr[outerMostElementNum * outAxisSize * outAxisElementStride + index * outAxisElementStride + trailingOffset] += Xptr[threadIdx];
        else if (reductionType == 2)
            Optr[outerMostElementNum * outAxisSize * outAxisElementStride + index * outAxisElementStride + trailingOffset] *= Xptr[threadIdx];
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
internal unsafe struct ScatterElementsJob : IJobParallelFor, IJobResourceDeclarationXBO
{

    public int outAxisSize;
    public int posAxis;
    public int rank;
    public int reductionType;
    public fixed int stridesO[8];
    public fixed int stridesX[8];
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        // Note that the strides arrays here are compacted at the head: "rank" values starting at element [0] !

        // Xptr                             is the updates tensor
        // Bptr                             is the indices tensor
        // Optr                             is the output tensor
        // stridesX                         are the strides of the updates tensor X and thus of the indices tensor B
        // stridesO                         are the strides of the output tensor O
        // posAxis                          is the positive scatter axis
        // rank                             is the rank of the tensors
        int index = (int)Bptr[threadIdx];
        uint posIndex = (uint) (index < 0 ? outAxisSize + index : index);

        uint remainder;
        uint idxOnCurDim;
        uint outLinearIdx;
        int axis = (int)(uint)posAxis;
        int curDim;

        remainder = (uint)threadIdx;
        outLinearIdx = 0;
        for (curDim = 0; curDim < axis; curDim++)
        {
            idxOnCurDim = remainder / (uint)stridesX[curDim];
            remainder = remainder % (uint)stridesX[curDim];
            outLinearIdx += idxOnCurDim * (uint)stridesO[curDim];
        }

        // process axis dimension and scatter re-indexing
        remainder = remainder % (uint)stridesX[curDim];
        outLinearIdx += posIndex * (uint)stridesO[curDim];
        curDim++;

        // We assume that the tensors are compact, no strides on the innermost dimensions, so we dont do the loop at curDim == rank - 1
        for (; curDim < rank - 1; curDim++)
        {
            idxOnCurDim = remainder / (uint)stridesX[curDim];
            remainder = remainder % (uint)stridesX[curDim];
            outLinearIdx += idxOnCurDim * (uint)stridesO[curDim];
        }
        // curDim == rank - 1 == innermost (assume stride 1; also, obviously O.shape[rank-1] >= indices or updates.shape[rank-1].
        // Also, even if curDim == rank (ie axis was rank-1 and we already processed last/innermost dim), we can still safely
        // do the last step in any case because the remainder of anything % 1 (stride[rank-1] is always 1) will be 0:
        outLinearIdx += remainder;

        if (reductionType == 0)
            Optr[(int)outLinearIdx] = Xptr[threadIdx];
        else if (reductionType == 1)
            Optr[(int)outLinearIdx] += Xptr[threadIdx];
        else if (reductionType == 2)
            Optr[(int)outLinearIdx] *= Xptr[threadIdx];
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
internal unsafe struct ExpandJob : IJobParallelFor, IJobResourceDeclarationXO
{

    public int rank;
    public fixed int shapeO[8];
    public fixed int stridesO[8];
    public fixed int shapeX[8];
    public fixed int stridesX[8];
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        int indexX = 0;
        for (int axis = 0; axis < rank; axis++)
        {
            indexX += (((threadIdx / stridesO[(TensorShape.maxRank - 1) - axis]) % shapeO[(TensorShape.maxRank - 1) - axis]) % shapeX[(TensorShape.maxRank - 1) - axis]) * stridesX[(TensorShape.maxRank - 1) - axis];
        }

        Optr[threadIdx] = Xptr[indexX];
    }
}

}
}
