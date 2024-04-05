//
// This was generated in SRP / HDRP, but since integrated and modified for Sentis for now
// (until we re-architect the framework, add the new parallel primitives, etc.)
//

#ifndef GPUPREFIXSUM_DATA_CS_HLSL
#define GPUPREFIXSUM_DATA_CS_HLSL
//
// UnityEngine.Rendering.GPUPrefixSum+ShaderDefs:  static fields
//
#define GATHER_SCALE_BIAS_CLAMPA_ITEMS_PER_THREAD (4)
#define GROUP_SIZE (128)
#define LOG2_GROUP_SIZE (7)
#define ARGS_BUFFER_STRIDE (16)
#define ARGS_BUFFER_UPPER (0)
#define ARGS_BUFFER_LOWER (8)

// Generated from UnityEngine.Rendering.GPUPrefixSum+LevelOffsets
// PackingRules = Exact
struct LevelOffsets
{
    uint count;
    uint offset;
    uint parentOffset;
    uint parentAlignedUpCount;
};

//
// Accessors for UnityEngine.Rendering.GPUPrefixSum+LevelOffsets
//
uint GetCount(LevelOffsets value)
{
    return value.count;
}
uint GetOffset(LevelOffsets value)
{
    return value.offset;
}
uint GetParentOffset(LevelOffsets value)
{
    return value.parentOffset;
}
uint GetParentAlignedUpCount(LevelOffsets value)
{
    return value.parentAlignedUpCount;
}

#endif
