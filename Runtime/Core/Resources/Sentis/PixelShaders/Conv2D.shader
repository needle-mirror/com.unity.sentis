Shader "Hidden/Sentis/Conv2D"
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
            uint X_width, X_height, X_channels, X_channelsDiv4;

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
                uint4 k4Offset = (K_width * K_height * X_channelsDiv4) * UnblockAxis(kDiv4);
                uint batchXOffset = X_width * X_height * X_channelsDiv4 * n;

                float4 acc4 = SampleBlockB(kDiv4);

                float4 k0 = 0;
                float4 k1 = 0;
                float4 k2 = 0;
                float4 k3 = 0;

                for (uint dy = 0; dy < K_height; ++dy)
                {
                    for (uint dx = 0; dx < K_width; ++dx)
                    {
                        uint oy = (h * StrideY + DilationY * dy) - PadY;
                        uint ox = (w * StrideX + DilationX * dx) - PadX;

                        bool maskX = (oy < X_height) && (ox < X_width);
                        for (uint cDiv4 = 0; cDiv4 < X_channelsDiv4; ++cDiv4)
                        {
                            float4 v = SampleBlockX(batchXOffset + Ravel(uint2(X_width, X_height), uint3(ox, oy, cDiv4)));
                            v *= maskX * (UnblockAxis(cDiv4) < X_channels ? 1.0f : 0.0f);

                            uint4 kIndex4 = k4Offset + Ravel(uint2(K_width, K_height), uint3(dx, dy, cDiv4));
                            k0 = SampleBlockK(kIndex4.x);
                            k1 = SampleBlockK(kIndex4.y);
                            k2 = SampleBlockK(kIndex4.z);
                            k3 = SampleBlockK(kIndex4.w);

                            acc4 += mul(float4x4(k0, k1, k2, k3), v);
                        }
                    }
                }

                return ApplyFusedActivation(acc4);
            }
            ENDCG
        }
    }
}
