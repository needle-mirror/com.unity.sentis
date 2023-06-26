Shader "Hidden/Sentis/Concat"
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
            #pragma multi_compile _ BLOCKWISE
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR_BLOCK_STRIDE_O;
            DECLARE_TENSOR(A);
            DECLARE_TENSOR(B);

            uint StrideAxis, DimAxisA, DimAxisB;
            uint ConcatLengthA;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                float4 v;
                uint blockIndexO = GetBlockIndexO(screenPos);
                #ifdef BLOCKWISE
                uint3 lowerAxisUpper = Unravel(uint2(StrideAxis, DimAxisO), blockIndexO);
                if (lowerAxisUpper[1] < ConcatLengthA)
                {
                    uint blockIndexA = Ravel(uint2(StrideAxis, DimAxisA), lowerAxisUpper);
                    v = SampleBlockA(blockIndexA);
                } else
                {
                    lowerAxisUpper[1] -= ConcatLengthA;
                    uint blockIndexB = Ravel(uint2(StrideAxis, DimAxisB), lowerAxisUpper);
                    v = SampleBlockB(blockIndexB);
                }
                #else
                uint3 lowerAxisDiv4Upper = UnravelO(blockIndexO);
                uint4 axis = UnblockAxis(lowerAxisDiv4Upper[1]);
                float4 maskB = axis < ConcatLengthA ? 0.0f : 1.0f;

                uint blockIndexA = Ravel(uint2(StrideAxis, DimAxisA), lowerAxisDiv4Upper);
                float4 va = SampleBlockA(blockIndexA);
                uint4 axisB = axis - ConcatLengthA;
                uint4 axisBDiv4 = axisB >> 2;
                uint4 axisBMod4 = axisB & 3;
                uint4 blockIndexB4 = lowerAxisDiv4Upper[0] + StrideAxis * (axisBDiv4 + DimAxisB * lowerAxisDiv4Upper[2]);
                float4 vb = SampleElementsB(blockIndexB4, axisBMod4);
                v = lerp(va, vb, maskB);
                #endif

                return v;
            }
            ENDCG
        }
    }
}
