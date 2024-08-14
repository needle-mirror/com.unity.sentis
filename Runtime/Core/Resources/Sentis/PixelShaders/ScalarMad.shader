Shader "Hidden/Sentis/ScalarMad"
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
            #pragma multi_compile_local _ INT

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"


            #if defined(INT)
            #define DTYPE4 int4
            int sInt;
            int bInt;
            DECLARE_TENSOR(X, int);
            #else
            #define DTYPE4 float4
            float s;
            float b;
            DECLARE_TENSOR(X, float);
            #endif

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                DTYPE4 vx = SampleBlockX(blockIndexO);

                #if defined(INT)
                int4 v = sInt * vx + bInt;
                #else
                float4 v = s * vx + b;
                #endif

                return v;
            }
            ENDCG
        }
    }
}
