Shader "Hidden/Sentis/TextureConversion/ComputeBufferToTexture"
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
            #pragma multi_compile_local EXACT LINEAR

            #pragma vertex vert
            #pragma fragment frag

            #include "../PixelShaders/CommonVertexShader.cginc"
            #include "TensorToTextureUtils.cginc"

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                #if defined(SHADER_API_MOBILE)
                return float4(0, 0, 0, 0);
                #endif
                uint2 O_pos = (uint2)(screenPos.xy - 0.5f);
                return ComputeColor(O_pos);
            }
            ENDCG
        }
    }
}
