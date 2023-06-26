Shader "Hidden/Sentis/Expand"
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
            #pragma multi_compile _ BLOCKEDDIM_RANK1
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X);

            uint DimO[8];
            uint StridesX[8];

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint blockIndexX = 0;
                uint n = blockIndexO;
                [unroll]
                for (uint j = 0; j < 8; j++)
                {
                    blockIndexX += (n % DimO[j]) * StridesX[j];
                    n /= DimO[j];
                }

                float4 v = SampleBlockX(blockIndexX);

                #ifdef BLOCKEDDIM_RANK1
                v = v.x;
                #endif


                return v;
            }
            ENDCG
        }
    }
}
