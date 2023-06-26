Shader "Hidden/Sentis/Reshape"
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

            DECLARE_TENSOR(X);
            DECLARE_TENSOR_BLOCK_STRIDE(X);

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                float4 v = 0;
                #ifdef BLOCKWISE
                uint blockIndexO = GetBlockIndexO(screenPos);
                v = SampleBlockX(blockIndexO);
                #else
                uint4 index4 = GetIndexO(screenPos);
                v = SampleBlockX(index4);
                #endif

                return v;
            }
            ENDCG
        }
    }
}
