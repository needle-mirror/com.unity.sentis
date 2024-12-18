Shader "Hidden/Sentis/GatherND"
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
            #pragma vertex vert
            #pragma fragment frag

            uint StridesB[8]; // Leave this declaration precisely here, this avoids triggered a bug on xbox series, see below.

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, float);
            DECLARE_TENSOR(B, int);
            DECLARE_TENSOR_BLOCK_STRIDE(X, float)
            DECLARE_TENSOR_BLOCK_STRIDE(B, int)
            DECLARE_TENSOR_BLOCK_STRIDE_O;

            //uint StridesB[8] placing this here still trigger the issue described above
            uint ShapeO[8];
            uint ShapeX[8];
            uint ShapeB[8];
            uint StridesO[8];
            uint StridesX[8];
            //uint StridesB[8] placing this here still trigger the issue described above
            uint RankX, RankO, RankB;

            uint iStart, iEndIndices, iEndX, iStartB, iEndB;

            float4 frag(v2f j, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint4 indexO4 = GetIndexO(screenPos);
                uint4 itIndices = 0;
                uint4 itX = 0;
                uint i;

                // iterate up to point where i == iEndX
                for (i = iStart; i < iEndX; i++)
                {
                    uint4 itO = (indexO4 / StridesO[i]) % ShapeO[i];
                    itIndices += itO * StridesB[(RankO - RankB) + i];
                    itX += itO * StridesX[(RankO - RankX) + i];
                }

                // finish indices
                for (i = iEndX; i < iEndIndices; i++)
                {
                    itIndices += ((indexO4 / StridesO[i]) % ShapeO[i]) * StridesB[(RankO - RankB) + i];
                }

                itIndices -= iStartB;

                for (i = iStartB; i < iEndB; i++)
                {
                    int4 index4 = SampleElementsB(itIndices + i);
                    index4 = index4 < 0 ? ShapeX[i] + index4 : index4;
                    itX += index4 * StridesX[i];
                }

                for (; i < 8; i++)
                {
                    itX += ((indexO4 / StridesO[i]) % ShapeO[i]) * StridesX[i];
                }

                return SampleElementsX(itX);
            }
            ENDCG
        }
    }
}
