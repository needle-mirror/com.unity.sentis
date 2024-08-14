Shader "Hidden/Sentis/GridSample"
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
            #pragma multi_compile_local GRIDSAMPLE1D GRIDSAMPLE2D GRIDSAMPLE3D
            #pragma multi_compile_local LINEAR NEAREST
            #pragma multi_compile_local ZEROS BORDER REFLECTION
            #pragma multi_compile_local _ ALIGN_CORNERS

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, float);
            DECLARE_TENSOR(S, float);

            uint O_width, O_height, O_depth, O_channelsDiv4;
            uint X_width, X_height, X_depth, X_channelsDiv4;

            float4 BilinearInterpolation(float fracSrcPosX, float fracSrcPosY, float4 p00, float4 p01, float4 p10, float4 p11)
            {
                float4 v = p00 * (1 - fracSrcPosX) * (1 - fracSrcPosY) +
                           p01 * (1 - fracSrcPosX) * fracSrcPosY +
                           p10 * fracSrcPosX       * (1 - fracSrcPosY) +
                           p11 * fracSrcPosX       * fracSrcPosY;
                return v;
            }

            float4 SampleXWithOOB(int4 pos, uint offset)
            {
                #ifdef ZEROS
                #ifdef GRIDSAMPLE1D
                if (pos.x < 0 || pos.x >= X_width)
                #endif
                #ifdef GRIDSAMPLE2D
                if (pos.x < 0 || pos.x >= X_width || pos.y < 0 || pos.y >= X_height)
                #endif
                #ifdef GRIDSAMPLE3D
                if (pos.x < 0 || pos.x >= X_width || pos.y < 0 || pos.y >= X_height || pos.z < 0 || pos.z >= X_depth)
                #endif
                return 0;
                #endif

                #ifdef GRIDSAMPLE1D
                return SampleBlockX(offset + pos.x);
                #endif
                #ifdef GRIDSAMPLE2D
                return SampleBlockX(offset + pos.y * X_width + pos.x);
                #endif
                #ifdef GRIDSAMPLE3D
                return SampleBlockX(offset + pos.z * X_width * X_height + pos.y * X_width + pos.x);
                #endif
            }

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint n = blockIndexO;
                uint w = n % O_width;
                n /= O_width;
                #if defined(GRIDSAMPLE2D) | defined(GRIDSAMPLE3D)
                uint h = n % O_height;
                n /= O_height;
                #ifdef GRIDSAMPLE3D
                uint d = n % O_depth;
                n /= O_depth;
                #endif
                #endif
                uint cDiv4 = n % O_channelsDiv4;
                n /= O_channelsDiv4;

                #ifdef GRIDSAMPLE1D
                float4 srcPos = SampleBlockS(n * O_width + w);
                int offset = X_width * (cDiv4 + X_channelsDiv4 * n);
                #endif
                #ifdef GRIDSAMPLE2D
                float4 srcPos = SampleBlockS((n * O_height + h) * O_width + w);
                int offset = X_width * X_height * (cDiv4 + X_channelsDiv4 * n);
                #endif
                #ifdef GRIDSAMPLE3D
                float4 srcPos = SampleBlockS(((n * O_depth + d) * O_height + h) * O_width + w);
                int offset = X_width * X_height * X_depth * (cDiv4 + X_channelsDiv4 * n);
                #endif

                #ifdef REFLECTION
                srcPos = abs(((srcPos - 1.0) % 4.0 + 4.0) % 4.0 - 2.0) - 1.0;
                #endif
                #ifdef BORDER
                srcPos = clamp(srcPos, -1.0, 1.0);
                #endif

                int4 Xshape = int4(X_width, X_height, X_depth, 0);

                #ifdef ALIGN_CORNERS
                srcPos = (0.5f * (srcPos + 1.0)) * (Xshape - 1.0);
                #else
                srcPos = (0.5f * (srcPos + 1.0)) * Xshape - 0.5;
                #endif

                float4 v = 0;

                #if defined(NEAREST)
                    int4 pos_i = round(srcPos);

                    #if defined(BORDER) || defined(REFLECTION)
                    pos_i = clamp(pos_i, 0, Xshape - int4(1, 1, 1, 1));
                    #endif

                    v = SampleXWithOOB(pos_i, offset);
                #endif
                #if defined(LINEAR)
                    int4 pos_i_0 = floor(srcPos);
                    int4 pos_i_1 = pos_i_0 + 1;
                    float4 pos_r = srcPos - pos_i_0;

                    #if defined(BORDER) || defined(REFLECTION)
                    pos_i_0 = max(pos_i_0, 0);
                    pos_i_1 = min(pos_i_1, Xshape - int4(1, 1, 1, 1));
                    #endif

                    #ifdef GRIDSAMPLE1D
                    float4 v0 = SampleXWithOOB(pos_i_0, offset);
                    float4 v1 = SampleXWithOOB(pos_i_1, offset);
                    v = (1 - pos_r.x) * v0 + pos_r.x * v1;
                    #endif
                    #ifdef GRIDSAMPLE2D
                    float4 v00 = SampleXWithOOB(pos_i_0, offset);
                    float4 v01 = SampleXWithOOB(int4(pos_i_0.x, pos_i_1.y, 0, 0), offset);
                    float4 v10 = SampleXWithOOB(int4(pos_i_1.x, pos_i_0.y, 0, 0), offset);
                    float4 v11 = SampleXWithOOB(pos_i_1, offset);
                    v = BilinearInterpolation(pos_r.x, pos_r.y, v00, v01, v10, v11);
                    #endif
                    #ifdef GRIDSAMPLE3D
                    float4 v000 = SampleXWithOOB(pos_i_0, offset);
                    float4 v001 = SampleXWithOOB(int4(pos_i_0.x, pos_i_0.y, pos_i_1.z, 0), offset);
                    float4 v010 = SampleXWithOOB(int4(pos_i_0.x, pos_i_1.y, pos_i_0.z, 0), offset);
                    float4 v011 = SampleXWithOOB(int4(pos_i_0.x, pos_i_1.y, pos_i_1.z, 0), offset);
                    float4 v100 = SampleXWithOOB(int4(pos_i_1.x, pos_i_0.y, pos_i_0.z, 0), offset);
                    float4 v101 = SampleXWithOOB(int4(pos_i_1.x, pos_i_0.y, pos_i_1.z, 0), offset);
                    float4 v110 = SampleXWithOOB(int4(pos_i_1.x, pos_i_1.y, pos_i_0.z, 0), offset);
                    float4 v111 = SampleXWithOOB(pos_i_1, offset);
                    float4 v0 = BilinearInterpolation(pos_r.x, pos_r.y, v000, v010, v100, v110);
                    float4 v1 = BilinearInterpolation(pos_r.x, pos_r.y, v001, v011, v101, v111);
                    v = v0 * (1 - pos_r.z) + v1 * pos_r.z;
                    #endif
                #endif

                return v;
            }
            ENDCG
        }
    }
}
