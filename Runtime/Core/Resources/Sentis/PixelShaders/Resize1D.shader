Shader "Hidden/Sentis/Resize1D"
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
            #pragma multi_compile_local LINEAR NEAREST_FLOOR NEAREST_CEIL

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, float);

            uint innerLength, outAxisSize, inputAxisSize;

            float Scale;
            float Bias;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint n = blockIndexO;
                uint oInner = n % innerLength;
                n /= innerLength;
                uint oAxis = n % outAxisSize;
                n /= outAxisSize;
                uint oOuter = n;

                float srcPosX = oAxis * Scale + Bias;

                float4 v = 0;
                #if defined(LINEAR)
                    float floorSrcPosX = floor(srcPosX);
                    float fracSrcPosX = srcPosX - floorSrcPosX;
                    int xLower = clamp((int)floorSrcPosX + 0, 0, (int)inputAxisSize - 1);
                    int xUpper = clamp((int)floorSrcPosX + 1, 0, (int)inputAxisSize - 1);

                    //from https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/trilinear-interpolation
                    float4 p0 = SampleBlockX((oOuter * inputAxisSize + xLower) * innerLength + oInner);
                    float4 p1 = SampleBlockX((oOuter * inputAxisSize + xUpper) * innerLength + oInner);
                    v = p0 * (1 - fracSrcPosX) + p1 * fracSrcPosX;
                #else
                    #if defined(NEAREST_FLOOR)
                        int ox = clamp((int)floor(srcPosX), 0, (int)inputAxisSize - 1);
                    #else // defined(NEAREST_CEIL)
                        int ox = clamp((int)ceil(srcPosX), 0, (int)inputAxisSize - 1);
                    #endif
                    v = SampleBlockX((oOuter * inputAxisSize + ox) * innerLength + oInner);
                #endif

                return v;
            }
            ENDCG
        }
    }
}
