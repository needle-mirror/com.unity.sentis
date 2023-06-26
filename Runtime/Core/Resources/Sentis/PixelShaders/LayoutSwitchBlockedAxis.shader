Shader "Hidden/Sentis/LayoutSwitchBlockedAxis"
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

            DECLARE_TENSOR_BLOCK_STRIDE_O;

            DECLARE_TENSOR(X);
            DECLARE_TENSOR_BLOCK_STRIDE(X);

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint4 index = GetIndexO(screenPos);
                return SampleBlockX(index);
            }
            ENDCG
        }
    }
}
