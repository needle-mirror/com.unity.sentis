Shader "Hidden/Sentis/RMSNormalizationTail"
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
            DECLARE_TENSOR(S, float);
            DECLARE_TENSOR(K, float);

            uint reduceLength;
            float epsilon;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                float4 v = SampleBlockX(blockIndexO);

                uint blockIndexA = blockIndexO / reduceLength;
                uint blockIndexS = blockIndexO % reduceLength;
                float4 meanSqr = SampleBlockK(blockIndexA);
                float scale = SampleBlockS(blockIndexS).x;
                return scale * v / sqrt(meanSqr + epsilon);
            }
            ENDCG
        }
    }
}
