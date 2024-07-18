Shader "Hidden/Sentis/TopP"
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

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, float);
            DECLARE_TENSOR(B, float);

            uint StrideAxisX, DimAxisX;

            int4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint2 lowerUpper = Unravel(uint1(StrideAxisX), blockIndexO);
                uint blockIndexX = lowerUpper[0] * StrideAxisX * DimAxisX;

                float4 acc4 = 0;
                int4 accIdx4 = 0;
                float4 prob = SampleBlockB(blockIndexO);
                for (int j = 0; j < (int)DimAxisX; j++)
                {
                    bool4 c4 = prob >= acc4;
                    accIdx4 = c4 ? j : accIdx4;
                    float4 p4 = SampleBlockX(blockIndexX + j);
                    acc4 += p4;
                }

                return accIdx4;
            }
            ENDCG
        }
    }
}
