Shader "Hidden/Sentis/OneHot"
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

            DECLARE_TENSOR(X, int);

            uint StrideAxis, DimAxisO;
            int onValue, offValue;

            int4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint3 lowerAxisUpper = Unravel(uint2(StrideAxis, DimAxisO), blockIndexO);
                int4 indices = SampleBlockX(Ravel(uint1(StrideAxis), lowerAxisUpper.xz));
                bool4 mask4 = (indices == (int4)lowerAxisUpper.y) || ((indices + (int)DimAxisO) == (int4)lowerAxisUpper.y);
                return mask4 ? onValue : offValue;
            }
            ENDCG
        }
    }
}
