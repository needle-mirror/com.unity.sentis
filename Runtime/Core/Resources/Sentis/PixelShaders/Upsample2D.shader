Shader "Hidden/Sentis/Upsample2D"
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
            #pragma multi_compile LINEAR NEAREST_FLOOR NEAREST_CEIL

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X);

            uint O_width, O_height, O_channelsDiv4;
            uint X_width, X_height, X_channelsDiv4;

            float4 Scale;
            float4 Bias;

            float4 BilinearInterpolation(float fracSrcPosX, float fracSrcPosY, float4 p00, float4 p01, float4 p10, float4 p11)
            {
                float4 v = p00 * (1 - fracSrcPosX) * (1 - fracSrcPosY) +
                           p01 * (1 - fracSrcPosX) * fracSrcPosY +
                           p10 * fracSrcPosX       * (1 - fracSrcPosY) +
                           p11 * fracSrcPosX       * fracSrcPosY;
                return v;
            }

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint4 w_h_cDiv4_n = Unravel(uint3(O_width, O_height, O_channelsDiv4), blockIndexO);
                uint w = w_h_cDiv4_n[0];
                uint h = w_h_cDiv4_n[1];
                uint cDiv4 = w_h_cDiv4_n[2];
                uint n = w_h_cDiv4_n[3];
                int offset = X_width * X_height * (cDiv4 + X_channelsDiv4 * n);

                float srcPosY = h * Scale[0] + Bias[0];
                float srcPosX = w * Scale[1] + Bias[1];

                float4 v = 0;
                #if defined(LINEAR)
                    float floorSrcPosX = floor(srcPosX);
                    float floorSrcPosY = floor(srcPosY);
                    float fracSrcPosX = srcPosX - floorSrcPosX;
                    float fracSrcPosY = srcPosY - floorSrcPosY;

                    int xLower = clamp((int)floorSrcPosX + 0, 0, (int)X_width - 1);
                    int xUpper = clamp((int)floorSrcPosX + 1, 0, (int)X_width - 1);
                    int yLower = clamp((int)floorSrcPosY + 0, 0, (int)X_height - 1);
                    int yUpper = clamp((int)floorSrcPosY + 1, 0, (int)X_height - 1);

                    //from https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/trilinear-interpolation
                    float4 p00 = SampleBlockX(xLower + X_width * yLower + offset);
                    float4 p01 = SampleBlockX(xLower + X_width * yUpper + offset);
                    float4 p10 = SampleBlockX(xUpper + X_width * yLower + offset);
                    float4 p11 = SampleBlockX(xUpper + X_width * yUpper + offset);
                    v = BilinearInterpolation(fracSrcPosX, fracSrcPosY, p00, p01, p10, p11);
                #else
                    int oy;
                    int ox;
                    #if defined(NEAREST_FLOOR)
                        oy = clamp((int)floor(srcPosY), 0, (int)X_height - 1);
                        ox = clamp((int)floor(srcPosX), 0, (int)X_width - 1);
                    #elif defined(NEAREST_CEIL)
                        oy = clamp((int)ceil(srcPosY), 0, (int)X_height - 1);
                        ox = clamp((int)ceil(srcPosX), 0, (int)X_width - 1);
                    #endif
                    v = SampleBlockX(ox + X_width * oy + offset);
                #endif

                return v;
            }
            ENDCG
        }
    }
}
