Shader "Hidden/Sentis/LocalPool"
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
            #pragma multi_compile MAXPOOL AVGPOOL

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            #define FLT_MIN -3.402823466e+38F

            DECLARE_TENSOR(X);

            uint O_width, O_height, O_channelsDiv4;
            uint X_width, X_height, X_channelsDiv4;

            int StrideY, StrideX, PadY, PadX, PoolY, PoolX;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint4 w_h_cDiv4_n = Unravel(uint3(O_width, O_height, O_channelsDiv4), blockIndexO);
                uint w = w_h_cDiv4_n[0];
                uint h = w_h_cDiv4_n[1];
                uint cDiv4 = w_h_cDiv4_n[2];
                uint n = w_h_cDiv4_n[3];

                uint offsetX = X_width * X_height * (cDiv4 + X_channelsDiv4 * n);

                float counter = 0.0f;
                float4 accVal = 0.0f;
                #ifdef MAXPOOL
                accVal = FLT_MIN;
                #endif
                for (int dy = 0; dy < PoolY; ++dy)
                for (int dx = 0; dx < PoolX; ++dx)
                {
                    uint oy = (h * StrideY + dy) - PadY;
                    uint ox = (w * StrideX + dx) - PadX;

                    if (oy >= X_height) continue;
                    if (ox >= X_width) continue;
                    uint blockIndexX = offsetX + Ravel(uint1(X_width), uint2(ox, oy));
                    float4 v = SampleBlockX(blockIndexX);
                    #ifdef MAXPOOL
                    accVal = max(accVal, v);
                    #endif
                    #ifdef AVGPOOL
                    accVal += v;
                    #endif
                    counter += 1.0f;
                }
                #ifdef AVGPOOL
                accVal /= counter;
                #endif

                return accVal;
            }
            ENDCG
        }
    }
}
