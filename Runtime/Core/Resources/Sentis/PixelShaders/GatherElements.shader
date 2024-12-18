Shader "Hidden/Sentis/GatherElements"
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
            #pragma multi_compile_local _ GatherInt
            #pragma multi_compile_local _ NoFastPath
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"
            #include "HLSLSupport.cginc"
            #include "../ComputeShaders/Tensor.cginc"

            #ifdef GatherInt
            #define DTYPE int
            #define DTYPE4 int4
            DECLARE_TENSOR(X, int);
            #else
            #define DTYPE float
            #define DTYPE4 float4
            DECLARE_TENSOR(X, float);
            #endif
            DECLARE_TENSOR(B, int);
            DECLARE_TENSOR_BLOCK_STRIDE(X, DTYPE)
            DECLARE_TENSOR_BLOCK_STRIDE(B, int)
            DECLARE_TENSOR_BLOCK_STRIDE_O;

            uint inputAxisSize;
            #ifndef NoFastPath
            uint inputAxisElementStride;           // These are all NON BLOCKED (ie original tensor shape): see below, this is because we use GetIndexO vs GetBlockIndexO
            uint indicesAxisElementStride;         // (cf ScatterElements.shader)
            uint indicesAxisMinusOneElementStride; // indicesAxisElementStride * indicesAxisSize, ie the stride for elements on axis-1 in the indices tensor.
            #endif
            #ifdef NoFastPath
            uint StridesO[8];                 // these are all from the NON blocked tensor shapes (here O shape match indices, X is inputs)
            uint StridesX[8];                 // and are all compacted at the head of the arrays:
                                              // eg StridesO[0] really corresponds to dimension 0 shape (the outermost).

            uint posAxis;                     // positive axis
            uint RankX;                       // rank of all tensors in fact
            #endif

            DTYPE4 frag(v2f j, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint4 inputLinearIdx4;

                uint4 indexO4 = GetIndexO(screenPos); // this will get a non blocked index because all values passed on GPU above (strides etc) are non blocked here but dispatch and tensor data are blocked!

                int4 index4 = SampleElementsB(indexO4);
                index4 = index4 < 0 ? inputAxisSize + index4 : index4;

                #ifndef NoFastPath

                int4 trailingOffset4 = indexO4 % indicesAxisElementStride;
                int4 outerMostElementNum4 = indexO4 / (indicesAxisMinusOneElementStride);

                inputLinearIdx4 = outerMostElementNum4 * inputAxisElementStride * inputAxisSize + index4 * inputAxisElementStride + trailingOffset4;

                #else

                uint4 remainder4;
                uint4 idxOnCurDim4;
                uint axis = posAxis;
                uint curDim;

                uint rank = min(RankX, SHAPE_MAXRANK);
                axis = min(axis, rank - 1);

                remainder4 = indexO4;
                inputLinearIdx4 = 0;

                UNITY_UNROLL
                for (curDim = 0; curDim < axis; curDim++)
                {
                    idxOnCurDim4 = remainder4 / StridesO[curDim];
                    remainder4 = remainder4 % StridesO[curDim];
                    inputLinearIdx4 += idxOnCurDim4 * StridesX[curDim];
                }

                // process axis dimension and scatter re-indexing
                //We get index4 for instead of doing idxOnCurDim4 = remainder4 / StridesO[curDim]
                remainder4 = remainder4 % StridesO[curDim];
                inputLinearIdx4 += index4 * StridesX[curDim];
                curDim++;

                // We assume that the tensors are compact, no strides on the innermost dimensions, so we dont do the loop at curDim == rank - 1
                UNITY_UNROLL
                for (; curDim < rank - 1; curDim++)
                {
                    idxOnCurDim4 = remainder4 / StridesO[curDim];
                    remainder4 = remainder4 % StridesO[curDim];
                    inputLinearIdx4 += idxOnCurDim4 * StridesX[curDim];
                }
                // curDim == rank - 1 == innermost (assume stride 1; also, obviously X.shape[rank-1] >= O and indices.shape[rank-1]
                // so we can safely do the last step as:
                inputLinearIdx4 += remainder4;

                #endif // NoFastPath

                return SampleElementsX(inputLinearIdx4);
            }
            ENDCG
        }
    }
}
