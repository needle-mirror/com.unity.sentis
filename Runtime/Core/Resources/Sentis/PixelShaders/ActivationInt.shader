Shader "Hidden/Sentis/ActivationInt"
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
            #pragma multi_compile_local Sign Not Abs Neg Clip Square

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR_BLOCK_STRIDE_O;

            int Alpha;
            int Beta;

            DECLARE_TENSOR(X, int);

            int4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                int4 v = SampleBlockX(blockIndexO);
                #if defined(Abs)
                    v = abs(v);
                #endif
                #if defined(Neg)
                    v = -v;
                #endif
                #if defined(Sign)
                    v = sign(v);
                #endif
                #ifdef Not
                    v = v == 0 ? 1 : 0;
                #endif
                #if defined(Clip)
                    v = min(Beta, max(v, Alpha));
                #endif
                #if defined(Square)
                    v = v * v;
                #endif
                return v;
            }
            ENDCG
        }
    }
}
