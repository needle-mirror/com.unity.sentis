Shader "Hidden/Sentis/Reduce"
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
            #pragma multi_compile REDUCEMIN REDUCEMAX REDUCESUM REDUCESUMSQUARE REDUCEMEAN REDUCEPROD REDUCEL1 REDUCEL2 REDUCESQRT REDUCELOGSUM REDUCELOGSUMEXP

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X);

            uint StrideAxisX, DimAxisX;

            float Normalization;

            #if defined(REDUCELOGSUMEXP)
            float4 maxVal;
            #endif

            #define FLT_MAX 3.402823466e+38F
            #define FLT_MIN -3.402823466e+38F

            inline float4 Default4()
            {
                #if defined(REDUCEMIN)
                return FLT_MAX;
                #elif defined(REDUCEMAX)
                return FLT_MIN;
                #elif defined(REDUCEPROD)
                return 1.0f;
                #else
                return 0.0f;
                #endif
            }

            inline float4 Initialize4(float4 v)
            {
                #if defined(REDUCESUMSQUARE) | defined(REDUCEL2)
                return v * v;
                #elif defined(REDUCEL1)
                return abs(v);
                #elif defined(REDUCELOGSUMEXP)
                return exp(v - maxVal);
                #else
                return v;
                #endif
            }

            inline float4 Reduce4(float4 acc, float4 v)
            {
                #if defined(REDUCEMIN)
                return min(acc, v);
                #elif defined(REDUCEMAX)
                return max(acc, v);
                #elif defined(REDUCEPROD)
                return acc * v;
                #else
                return acc + v;
                #endif
            }

            inline float4 Finalize4(float4 acc)
            {
                #if defined(REDUCEMEAN)
                return Normalization * acc;
                #elif defined(REDUCESQRT) | defined(REDUCEL2)
                return sqrt(acc);
                #elif defined(REDUCELOGSUM)
                float4 u = log(acc);
                bool4 accNaN = acc <= 0.0f;
                u.x = accNaN.x ? 0.0f : u.x;
                u.y = accNaN.y ? 0.0f : u.y;
                u.z = accNaN.z ? 0.0f : u.z;
                u.w = accNaN.w ? 0.0f : u.w;
                return u;
                #elif defined(REDUCELOGSUMEXP)
                return log(acc) + maxVal;
                #else
                return acc;
                #endif
            }

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint2 lowerUpper = Unravel(uint1(StrideAxisX), blockIndexO);

                float4 acc4 = Default4();
                uint blockIndexXMin = Ravel(uint1(StrideAxisX * DimAxisX), lowerUpper);
                uint blockIndexXMax = blockIndexXMin + StrideAxisX * DimAxisX;
                uint blockIndexX;
                #if defined(REDUCELOGSUMEXP)
                maxVal = FLT_MIN;
                for (blockIndexX = blockIndexXMin; blockIndexX < blockIndexXMax; blockIndexX += StrideAxisX)
                {
                    float4 v = SampleBlockX(blockIndexX);
                    maxVal = max(maxVal, v);
                }
                #endif
                for (blockIndexX = blockIndexXMin; blockIndexX < blockIndexXMax; blockIndexX += StrideAxisX)
                {
                    float4 v = SampleBlockX(blockIndexX);
                    v = Initialize4(v);
                    acc4 = Reduce4(acc4, v);
                }
                acc4 = Finalize4(acc4);

                return acc4;
            }
            ENDCG
        }
    }
}
