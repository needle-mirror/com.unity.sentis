Shader "Hidden/Sentis/DepthwiseConv2D"
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
            #pragma multi_compile NONE RELU

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(B);
            DECLARE_TENSOR(K);
            DECLARE_TENSOR(X);

            uint O_width, O_height, O_channelsDiv4;
            uint K_width, K_height;
            uint X_width, X_height, X_channelsDiv4;

            uint StrideY, StrideX;
            uint PadY, PadX;
            uint DilationY, DilationX;

            float4 ApplyFusedActivation(float4 v)
            {
                #ifdef RELU
                return max(v, 0);
                #endif
                return v;
            }

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint4 w_h_kDiv4_n = Unravel(uint3(O_width, O_height, O_channelsDiv4), blockIndexO);
                uint w = w_h_kDiv4_n[0];
                uint h = w_h_kDiv4_n[1];
                uint kDiv4 = w_h_kDiv4_n[2];
                uint n = w_h_kDiv4_n[3];

                float4 acc4 = SampleBlockB(kDiv4);

                for (uint dy = 0; dy < K_height; ++dy)
                {
                    for (uint dx = 0; dx < K_width; ++dx)
                    {
                        uint oy = h * StrideY + DilationY * dy - PadY;
                        uint ox = w * StrideX + DilationX * dx - PadX;

                        if (oy >= X_height) continue;
                        if (ox >= X_width) continue;

                        uint blockIndexX = ox + X_width * (oy + X_height * (kDiv4 + X_channelsDiv4 * n));
                        uint blockIndexK = dx + K_width * (dy + K_height * kDiv4);
                        float4 vx = SampleBlockX(blockIndexX);
                        float4 vk = SampleBlockK(blockIndexK);
                        acc4 += vx * vk;
                    }
                }

                return ApplyFusedActivation(acc4);
            }
            ENDCG
        }
    }
}
