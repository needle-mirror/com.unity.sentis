Shader "Hidden/Sentis/GroupedConv2D"
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

            uint O_width, O_height, O_channels, O_channelsDiv4;
            uint K_width, K_height, K_channelsDivGroupDiv4;
            uint X_width, X_height, X_channels, X_channelsDiv4;

            uint StrideY, StrideX;
            uint PadY, PadX;
            uint DilationY, DilationX;
            uint Groups;

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
                uint4 k4 = UnblockAxis(kDiv4);

                float4 acc4 = SampleBlockB(kDiv4);;

                uint inputGroupedChannels = X_channels / Groups;
                uint outputGroupedChannels = O_channels / Groups;

                for (uint dy = 0; dy < K_height; ++dy)
                {
                    for (uint dx = 0; dx < K_width; ++dx)
                    {
                        uint oy = h * StrideY + DilationY * dy - PadY;
                        uint ox = w * StrideX + DilationX * dx - PadX;

                        if (oy >= X_height) continue;
                        if (ox >= X_width) continue;

                        for (uint c = 0; c < inputGroupedChannels; ++c)
                        {
                            uint4 xc4 = (k4 / outputGroupedChannels) * inputGroupedChannels + c;
                            uint4 blockIndexX4 = ox + X_width * (oy + X_height * ((xc4 >> 2) + X_channelsDiv4 * n));
                            uint4 blockIndexK4 = dx + K_width * (dy + K_height * ((c >> 2) + K_channelsDivGroupDiv4 * k4));
                            float4 x4 = SampleElementsX(blockIndexX4, xc4 & 3);
                            float4 k4 = SampleElementsK(blockIndexK4, c & 3);
                            acc4 += x4 * k4;
                        }
                    }
                }

                return ApplyFusedActivation(acc4);
            }
            ENDCG
        }
    }
}
