Shader "Hidden/Sentis/SliceSet"
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
            #pragma multi_compile_local _ BLOCKWISE
            #pragma multi_compile_local _ INT
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR_BLOCK_STRIDE_O;

            #ifdef INT
            #define DTYPE4 int4
            #define DTYPE int
            DECLARE_TENSOR(X, int);
            DECLARE_TENSOR(V, int);
            #else
            #define DTYPE4 float4
            #define DTYPE float
            DECLARE_TENSOR(X, float);
            DECLARE_TENSOR(V, float);
            #endif
            DECLARE_TENSOR_BLOCK_STRIDE(X, DTYPE);
            DECLARE_TENSOR_BLOCK_STRIDE(V, DTYPE);

            uint StridesV[8];
            uint Starts[8];
            uint Steps[8];
            uint ShapeO[8];
            uint ShapeV[8];

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                DTYPE4 v = 0;
                #ifdef BLOCKWISE
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint n = blockIndexO;
                uint blockIndexV = 0;
                uint mask = 1;
                for (uint j = 0; j < 8; j++)
                {
                    uint m = n % ShapeO[j];
                    int d = clamp(((int)m - (int)Starts[j]) / (int)Steps[j], 0, (int)ShapeV[j] - 1);
                    blockIndexV += d * (int)StridesV[j];
                    mask *= (m == (int)Starts[j] + d * (int)Steps[j]);
                    n /= ShapeO[j];
                }
                v = mask * SampleBlockV(blockIndexV) + (1 - mask) * SampleElementsX(blockIndexO);
                #else
                uint4 indexO4 = GetIndexO(screenPos);
                uint4 n4 = indexO4;
                uint4 indexV4 = 0;
                uint4 mask4 = 1;
                for (uint j = 0; j < 8; j++)
                {
                    uint4 m4 = n4 % ShapeO[j];
                    int4 d4 = clamp(((int4)m4 - (int4)Starts[j]) / (int4)Steps[j], 0, (int4)ShapeV[j] - 1);
                    indexV4 += d4 * (int4)StridesV[j];
                    mask4 *= (m4 == (int4)Starts[j] + d4 * (int4)Steps[j]);
                    n4 /= ShapeO[j];
                }
                v = mask4 * SampleElementsV(indexV4) + (1 - mask4) * SampleElementsX(indexO4);
                #endif

                return v;
            }
            ENDCG
        }
    }
}
