Shader "Hidden/Sentis/Conv2DTrans"
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
            uint K_width, K_height, K_mDivGroup;
            uint X_width, X_height, X_channelsDiv4;

            int StrideY, StrideX;
            int PadY, PadX;

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
                int w = w_h_kDiv4_n[0];
                int h = w_h_kDiv4_n[1];
                uint kDiv4 = w_h_kDiv4_n[2];
                uint n = w_h_kDiv4_n[3];
                const uint4 k4Offset = (K_width * K_height) * UnblockAxis(kDiv4);
                const uint batchXOffset = X_width * X_height * X_channelsDiv4 * n;

                const uint xDelta = X_width * X_height;
                const uint kDelta = K_width * K_height * K_mDivGroup;

                const uint oyMin = h - PadY + StrideY - 1 < 0 ? 0 : (h - PadY + StrideY - 1) / (uint)StrideY;
                const uint oyMax = min(X_height, ceil((K_height + h - PadY + StrideY - 1) / StrideY));
                const uint oxMin = w - PadX + StrideX - 1 < 0 ? 0 : (w - PadX + StrideX - 1) / (uint)StrideX;
                const uint oxMax = min(X_width, ceil((K_width + w - PadX + StrideX - 1) / StrideX));

                float4 acc4 = SampleBlockB(kDiv4);

                float4 k0 = 0;
                float4 k1 = 0;
                float4 k2 = 0;
                float4 k3 = 0;

                for (uint oy = oyMin, dy = K_height - 1 - (oyMin * StrideY - h + PadY); oy < oyMax; oy++, dy -= StrideY)
                {
                    for (uint ox = oxMin, dx = K_width - 1 - (oxMin * StrideX - w + PadX); ox < oxMax; ox++, dx -= StrideX)
                    {
                        uint xIndex = batchXOffset + Ravel(uint1(X_width), uint2(ox, oy));
                        uint4 kIndex4 = k4Offset + Ravel(uint1(K_width), uint2(dx, dy));

                        for (uint cDiv4 = 0; cDiv4 < X_channelsDiv4; ++cDiv4)
                        {
                            float4 v = SampleBlockX(xIndex);

                            k0 = SampleBlockK(kIndex4.x);
                            k1 = SampleBlockK(kIndex4.y);
                            k2 = SampleBlockK(kIndex4.z);
                            k3 = SampleBlockK(kIndex4.w);

                            acc4 += mul(float4x4(k0, k1, k2, k3), v);

                            xIndex += xDelta;
                            kIndex4 += kDelta;
                        }
                    }
                }
                return ApplyFusedActivation(acc4);
            }
            ENDCG
        }
    }
}
