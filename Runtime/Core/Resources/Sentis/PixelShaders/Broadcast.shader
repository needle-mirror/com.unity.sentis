Shader "Hidden/Sentis/Broadcast"
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
            #pragma multi_compile _ BLOCKEDDIM_RANK1_A BLOCKEDDIM_RANK1_B
            #pragma multi_compile Add Sub Mul Div Pow Min Max FMod Mean

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(A);
            DECLARE_TENSOR(B);

            uint DimO[8];
            uint StridesA[8];
            uint StridesB[8];

            #ifdef Mean
            float alpha, beta;
            #endif

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint blockIndexA = 0;
                uint blockIndexB = 0;
                uint n = blockIndexO;
                [unroll]
                for (uint j = 0; j < 8; j++)
                {
                    uint k = (n % DimO[j]);
                    n /= DimO[j];
                    blockIndexA += k * StridesA[j];
                    blockIndexB += k * StridesB[j];
                }

                float4 v = 0.0f;

                float4 va = SampleBlockA(blockIndexA);
                float4 vb = SampleBlockB(blockIndexB);

                #ifdef BLOCKEDDIM_RANK1_A
                va = va.x;
                #endif
                #ifdef BLOCKEDDIM_RANK1_B
                vb = vb.x;
                #endif

                #ifdef Add
                    v = va + vb;
                #endif
                #ifdef Sub
                    v = va - vb;
                #endif
                #ifdef Mul
                    v = va * vb;
                #endif
                #ifdef Pow
                    float4 u = pow(va, vb);
                    bool4 vNaN = (va < 0.0f && floor(vb) == vb) || (va == 0.0f && vb == 0.0f);
                    v.x = vNaN.x ? 0.0f : u.x;
                    v.y = vNaN.y ? 0.0f : u.y;
                    v.z = vNaN.z ? 0.0f : u.z;
                    v.w = vNaN.w ? 0.0f : u.w;
                #endif
                #ifdef Min
                    v = min(va, vb);
                #endif
                #ifdef Max
                    v = max(va, vb);
                #endif
                #ifdef Mean
                    v = alpha * va + beta * vb;
                #endif
                #ifdef FMod
                    float4 u = fmod(va, vb);
                    bool4 vNaN = vb == 0.0f;
                    v.x = vNaN.x ? 0.0f : u.x;
                    v.y = vNaN.y ? 0.0f : u.y;
                    v.z = vNaN.z ? 0.0f : u.z;
                    v.w = vNaN.w ? 0.0f : u.w;
                #endif
                #ifdef Div
                    float4 u = va / vb;
                    bool4 vNaN = vb == 0.0f;
                    v.x = vNaN.x ? 0.0f : u.x;
                    v.y = vNaN.y ? 0.0f : u.y;
                    v.z = vNaN.z ? 0.0f : u.z;
                    v.w = vNaN.w ? 0.0f : u.w;
                #endif

                return v;
            }
            ENDCG
        }
    }
}
