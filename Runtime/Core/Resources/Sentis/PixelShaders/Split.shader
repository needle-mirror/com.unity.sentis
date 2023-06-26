Shader "Hidden/Sentis/Split"
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

            DECLARE_TENSOR(X);

            uint StrideAxis, DimAxisX;
            uint SplitStart, SplitLength;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint3 lowerAxisUpper = Unravel(uint2(StrideAxis, SplitLength), blockIndexO);

                float4 v = 0;
                #ifdef BLOCKWISE
                lowerAxisUpper[1] += SplitStart;
                uint blockIndexX = Ravel(uint2(StrideAxis, DimAxisX), lowerAxisUpper);
                v = SampleBlockX(blockIndexX);
                #else
                uint4 axisX4 = UnblockAxis(lowerAxisUpper[1]) + SplitStart;
                uint4 blockIndexX4 = lowerAxisUpper[0] + StrideAxis * ((axisX4 >> 2) + DimAxisX * lowerAxisUpper[2]);
                v = SampleElementsX(blockIndexX4, axisX4 & 3);
                #endif

                return v;
            }
            ENDCG
        }
    }
}
