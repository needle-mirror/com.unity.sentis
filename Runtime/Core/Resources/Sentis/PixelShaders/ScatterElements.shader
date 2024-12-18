Shader "Hidden/Sentis/ScatterElements"
{
    Properties
    {
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma multi_compile_local _ ScatterInt
            #pragma multi_compile_local ReduceNone ReduceAdd ReduceMul
            #pragma multi_compile_local _ UseDiv4Mask
            #pragma multi_compile_local _ NoFastPath
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"
            #include "HLSLSupport.cginc"
            #include "../ComputeShaders/Tensor.cginc"

            #ifdef ScatterInt
            #define DTYPE4 int4
            #define DTYPE int
            DECLARE_TENSOR(X, int);
            DECLARE_TENSOR(W, int);
            #else
            #define DTYPE4 float4
            #define DTYPE float
            DECLARE_TENSOR(X, float);
            DECLARE_TENSOR(W, float);
            #endif
            DECLARE_TENSOR(B, int);
            DECLARE_TENSOR_BLOCK_STRIDE(X, DTYPE)
            DECLARE_TENSOR_BLOCK_STRIDE(B, int)
            DECLARE_TENSOR_BLOCK_STRIDE(W, DTYPE)
            DECLARE_TENSOR_BLOCK_STRIDE_O;

            // Important: all tensors X, B, W and O are assumed to be blocked (4-element-chunked) along the same dimension,
            // (confusingly called axis in CommonPixelShader.cginc, not to be confused with the scatter axis here!)
            // and the blocked dimension is NOT the scatter axis (as when accessing the indices tensor B we wont have to
            // handle potential blocked conversion from the indices specified) otherwise the code below will not work.
            //
            // Also, all quantities here are from the *blocked* shape of the tensors
            // but outAxisSize and NumIndices (which is the indices axis dimension size) are equal to the non blocked shape
            // since like just noted, we don't block/chunk along the scatter axis.
            //
            uint NumIndices;                  // ie indicesAxisSize

            uint outAxisSize;
            #ifndef NoFastPath
            uint outAxisElementStride;        // could include a blocked component if tensor is blocked along innermore dim than axis (ie blockedAxis > scatterAxis)
            uint indicesAxisElementStride;    // could include a blocked component if tensor is blocked along innermore dim than axis (ie blockedAxis > scatterAxis)
            #endif
            uint indicesLinearSize;           // full tensor linear size, from blocked shape too
            //uint indicesAxisSliceLinearSize;  // from blocked shape too: eg if indices.shape == (5,2,3,4) and axis is 1, a slice along axis 1 is of size 5*3*4

            #ifdef UseDiv4Mask
            uint indicesDiv4RemainderMask[4]; // see below
            #endif

            #ifdef NoFastPath
            uint StridesO[8];                 // these are all from the blocked (4-sized chunks on one dimension) tensor shape
            uint ShapeB[8];                   // these are all from the blocked (4-sized chunks on one dimension) tensor shape
            uint StridesB[8];                 // and are also all compacted at the head of the arrays:
                                              // eg ShapeB[0] really corresponds to dimension 0 shape (the outermost).

            uint posAxis;                     // positive axis
            uint RankX;                       // rank of all tensors in fact
            #endif

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                DTYPE4 v = SampleBlockX(blockIndexO);

                uint indicesPotentialTailLinearIdx;
                uint outAxisElement;

                #ifndef NoFastPath
                // FAST PATH: Simple linear to pseudo-multi-index (in blocked indices tensor) conversion
                //
                // (note the convention of "shape" and Unravel return order are different in CommonPixelShader.cginc than everywhere else in our code,
                // TODO cleanup!)

                // multiIdx_lowerScatterAxisUpperO here is a multi index (reverse of our and numpy usual convention here) of trailing linear offset,
                // element num on axis and element num on outermost folded/compacted dimension.
                // Important also: multiIdx_lowerScatterAxisUpperO is still overall blocked somewhere but not for the axis index,
                // ie not the multiIdx_lowerScatterAxisUpperO.y component.
                uint3 multiIdx_lowerScatterAxisUpperO = Unravel(uint2(outAxisElementStride, outAxisSize), blockIndexO);

                // Alias to make the fastpath / generic path code cleaner: denote this "potential tail linear offset" for indices calculated
                // from output linear index. Note that this is not the right tail when fast path conditions aren't true even if < indicesAxisElementStride:
                indicesPotentialTailLinearIdx = multiIdx_lowerScatterAxisUpperO.x;
                outAxisElement = multiIdx_lowerScatterAxisUpperO.y;

                #else // NoFastPath follows:

                // NO FAST PATH: Full linear to pseudo-multi-index (in blocked indices tensor) conversion in 3 parts.

                // Part 1 and 2 of linear to pseudo-multi-index:

                // In the generic robust path, still calculate multiIdx_lowerScatterAxisUpperO, but in steps using the full strides,
                // except unlike compute, to early out, we reverse the calculation by going from innermost dim to outermost,
                // calculating first the tail linear offset in the indices tensor (tail innermore to the axis).
                // We don't consider scattering further if we are beyond the indices tensor.
                // (Remember we do scatter as gather and the dispatch is done from the input/output tensor shape which is larger or equal to indices)
                //
                // The axis is not necessarily the optimal split point to check for an early out (could use rank/2) but we need
                // to special case at that split point anyway because we must force the axis component of the indices multi-index to 0.
                // (again because of "scatter as a gather search" we don't re-index that axis component
                // we actually don't sum it to indicesLinearIdx since we need to search the whole axis later for the gather part).
                //
                // If we pass below that indicesAxisElementStride threshold, we proceed with the outermost part component of the
                // indices blocked multi-index (called Upper in multiIdx_lowerScatterAxisUpperO) and again will break if that
                // indicesLinearIdxAtAxisElement0 would exceed the whole indicesLinearSize.

                uint remainder;
                uint idxOnCurDim;
                uint indicesLinearIdx;
                uint axis = posAxis;
                int curDim;

                bool idxOnCurDimValidForIndicesTensor; // TODO

                uint rank = min(RankX, SHAPE_MAXRANK);
                axis = min(axis, rank - 1);

                // Note: contrary to when going from outermost to innermost, we need shapeOfSrc[] instead of stridesOfSrc[]
                // and the modulo (div remainder) gives the multi-index part while the dividend is further used to continue
                // the conversion
                // ie instead of
                //  idxOnCurDim = remainder / stridesO[curDim];
                //  remainder = remainder % stridesO[curDim];
                // we do
                //  idxOnCurDim = remainder % shapeO[curDim];
                //  remainder = remainder / shapeO[curDim];
                remainder = blockIndexO;
                indicesLinearIdx = 0;
                idxOnCurDimValidForIndicesTensor = true;

                UNITY_UNROLL
                for (curDim = 0; (curDim < axis) && idxOnCurDimValidForIndicesTensor; curDim++)
                {
                    idxOnCurDim = remainder / StridesO[curDim];
                    remainder = remainder % StridesO[curDim];
                    indicesLinearIdx += idxOnCurDim * StridesB[curDim];
                    idxOnCurDimValidForIndicesTensor = (idxOnCurDim < ShapeB[curDim]);
                }

                // note: we don't early return v; because shader compilers can produce faulty code with early/multiple returns!
                if (idxOnCurDimValidForIndicesTensor)
                {
                    // Process axis dimension.
                    // Normally we don't care about the scatter axis index component (ie idxOnCurDim)
                    // since we do scatter re-indexing (the whole purpose of the indices tensor)
                    // but here we don't do scatter re-indexing in scatter as gather and want element 0
                    // (since we gather-search at the start of the scatter axis).
                    // We still need to store the outAxisElement to find a match in the search:
                    outAxisElement = remainder / StridesO[curDim];
                    remainder = remainder % StridesO[curDim];
                    //indicesLinearIdx += (instead of posIndex we have 0 here) * StridesB[curDim];
                    curDim++;

                    // For the axis, we don't need to check (idxOnCurDim < ShapeB[curDim]) since we will scan from 0 to NumIndices,
                    // so never overflowing

                    // To join common non fast path/fastpath code:
                    uint indicesAxisElementStride = StridesB[axis];

                #endif // NoFastPath


                    // ! Fast path / non-fast path common code,
                    //   at this point assume these are valid:
                    //
                    //      outAxisElement
                    //      indicesAxisElementStride
                    //      outAxisSize

                    #ifndef NoFastPath
                    // In fastpath indicesPotentialTailLinearIdx = multiIdx_lowerScatterAxisUpperO.x will be blockIndexO % outAxisElementStride.
                    // See compute shader version of ScatterElementsFast: this is not necessarily matching the trailing offset
                    // of the indices tensor ie != blockIndexO % indicesAxisElementStride.
                    if (indicesPotentialTailLinearIdx < indicesAxisElementStride)
                    #endif
                    {
                        #ifdef NoFastPath
                        // NO FAST PATH:
                        // Part 3 of linear to pseudo-multi-index:

                        idxOnCurDimValidForIndicesTensor = true;
                        // Note that the starting linear index blockIndexO could still overrun the tensor's end because the dispatch might be
                        // larger and in any case the last dimension could also be larger so we still check for the above except if it's the axis!
                        // We reset this to true here since we just processed the axis and if the axis is the innermost dim,
                        // we don't enter the loop below, and don't need to test idxOnCurDimValidForIndicesTensor either!
                        UNITY_UNROLL
                        for (; (curDim < rank-1) && idxOnCurDimValidForIndicesTensor; curDim++)
                        {
                            idxOnCurDim = remainder / StridesO[curDim];
                            remainder = remainder % StridesO[curDim];
                            indicesLinearIdx += idxOnCurDim * StridesB[curDim];
                            idxOnCurDimValidForIndicesTensor = (idxOnCurDim < ShapeB[curDim]);
                        }
                        // Remainder is 0 here if we already processed rank-1 and curDim == rank, so we systematically add
                        // the last remainder (with strides[rank-1] = 1):
                        indicesLinearIdx += remainder;
                        uint indicesLinearIdxAtAxisElement0 = indicesLinearIdx;

                        // note: we don't early return v; because shader compilers can produce faulty code with early/multiple returns!
                        if (idxOnCurDimValidForIndicesTensor)
                        {
                        #else
                            // FAST PATH: Simple linear to pseudo-multi-index (in blocked indices tensor) conversion

                            uint indicesLinearIdxAtAxisElement0 = multiIdx_lowerScatterAxisUpperO.x + indicesAxisElementStride * NumIndices * multiIdx_lowerScatterAxisUpperO.z;
                            // multiIdx_lowerScatterAxisUpperO.z is outerMostElementNum in compacted/folded tensor view, see compute code.

                            // Indices shape can be smaller than the input/output.
                            //
                            // (For fast path, in particular from dimension 0 to the axis (inclusive) dimension,
                            // can be 1 or, for the scatter axis-1 and scatter axis dimension, can be anything smaller than the corresponding shape
                            // in the input/output tensors).
                            //
                            // Make sure we don't clobber the output tensor with zeroes by overflowing indices/updates: without this test,
                            // texture access would obviously not fail but return 0 for indices and 0 for updates payload and thus would match
                            // all elements on the starting axis slice (slice element #0 of the axis) for all higher elements k on axis and outermore
                            // dims where k doesn't exist in indices and updates.
                            if (indicesLinearIdxAtAxisElement0 < indicesLinearSize)
                        #endif // above this, fastpath
                            {
                                // Another trickyness here: even if we blocked all tensors along the same dim, we could be at an uneven div4 last
                                // element along the blocked dimension, and would need to mask out some garbage from the sampled indices block
                                // by overriding the calculated mask.
                                // (But note this is out of the for loop as we don't block along the scatter axis)
                                //
                                // Remember:
                                //
                                // When we load a block from SampleBlock*(linearPixel), we get 4 values that are NOT necessarily contiguous
                                // in the flattened tensor, but that have a stride == to the product of the shape[i] for i == dims innermore than
                                // the axis on which we blocked the tensor
                                // (eg for O texture using DECLARE_TENSOR_BLOCK_STRIDE_O, this is StrideAxisO in CommonPixelShader.cginc).
                                // There could be padding garbage in the last blocked element, it depends if the original size of the blocked axis
                                // was evenly divisible by 4.
                                // To know if we need to handle that special case, we need to know the blockedAxisIndex corresponding to the
                                // linear index we're using to get the sample.
                                // That special case happens for when the linear index touches the last element (imagining a multi-index corresponding
                                // to the linear one) on the blockedAxis and this is identified by blockedAxisIndex being == to (blockedAxisSize-1).
                                //
                                #ifdef UseDiv4Mask
                                uint blockedAxisElementStrideB = StrideAxisB; // !Important: for StrideAxisB here, the axis is blockedAxis for B,
                                uint blockedAxisSizeB = DimBlockedB;          // see CommonPixelShader.cginc and eg UnravelO (cleanup the mess in there)

                                uint3 multiIdx_lowerBlockedAxisUpperB = Unravel(uint2(blockedAxisElementStrideB, blockedAxisSizeB), indicesLinearIdxAtAxisElement0);
                                uint blockedAxisIndexB = multiIdx_lowerBlockedAxisUpperB[1];

                                uint4 div4RemainderGarbageMask = uint4(indicesDiv4RemainderMask[0], indicesDiv4RemainderMask[1], indicesDiv4RemainderMask[2], indicesDiv4RemainderMask[3]);
                                div4RemainderGarbageMask = (blockedAxisIndexB == (blockedAxisSizeB - 1)) ? div4RemainderGarbageMask : int4(1,1,1,1);
                                #endif

                                // Note that indicesLinearIdxAtAxisElement0 += indicesAxisElementStride; in the loop below
                                // won't change blockedAxisIndexB as we are sure the scatter axis is NOT the blocked axis!
                                uint indicesLinearIdxAtAxisElementJ = indicesLinearIdxAtAxisElement0;
                                for (uint j = 0; j < NumIndices; j++)
                                {
                                    int4 indicesJ = SampleBlockB(indicesLinearIdxAtAxisElementJ);
                                    int4 outAxisElement4 = int4(outAxisElement, outAxisElement, outAxisElement, outAxisElement);
                                    int4 mask = (indicesJ == outAxisElement4 || (indicesJ + int4(outAxisSize, outAxisSize, outAxisSize, outAxisSize)) == outAxisElement4) ? 1 : 0;
                                    //int4 mask = ((indicesJ + saturate(-sign(indicesJ))*outAxisSize) == outAxisElement) ? 1 : 0;
                                    DTYPE4 updates = SampleBlockW(indicesLinearIdxAtAxisElementJ);

                                    #ifdef UseDiv4Mask
                                    mask *= div4RemainderGarbageMask;
                                    #endif

                                    #ifdef ReduceNone
                                    v = v * (1 - mask) + updates * mask;
                                    #elif ReduceAdd
                                    v = v + updates * mask;
                                    #elif ReduceMul
                                    v = v * ((1 - mask) + updates * mask);
                                    #endif
                                    indicesLinearIdxAtAxisElementJ += indicesAxisElementStride;
                                }
                            } // fastpath only: if (indicesLinearIdxAtAxisElement0 < indicesLinearSize)

                        #ifdef NoFastPath
                        } // multi-index overflow, early out
                        #endif
                    } // fastpath only: if (indicesPotentialTailLinearIdx < indicesAxisElementStride)

                #ifdef NoFastPath
                } // multi-index overflow, early out
                #endif

                return v;
            }
            ENDCG
        }
    }
}
