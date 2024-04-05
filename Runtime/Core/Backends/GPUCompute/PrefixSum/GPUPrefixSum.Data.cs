using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Unity.Sentis
{
    public partial struct GPUPrefixSum
    {
        // GenerateHLSL
        // Keep in sync with /GPUCompute/PrefixSum/GPUPrefixSum.Data.cs.hlsl
        internal static class ShaderDefs
        {
            public const int GroupSize = 128;
            public const int GatherScaleBiasClampaItemsPerThread = 4;

            // Stride of the indirect arguement buffer in uints, the buffer is split into two sections dispatch options ( a lower or upper arguement set )
            public const int ArgsBufferStride = 16;
            public const int ArgsBufferUpper  = 0;
            public const int ArgsBufferLower  = 8;

            // How many thread groups the "value" can be divided in, rounded up.
            public static int DivUpGroup(int value)
            {
                return (value + GroupSize - 1) / GroupSize;
            }

            // Smallest multiple of GroupSize >= value
            public static int AlignUpGroup(int value)
            {
                return DivUpGroup(value) * GroupSize;
            }

            // For a number of max elements as inputs, output total size required by whole pyramid of buffers and the number of levels in it.
            public static void CalculateTotalPyramidBufferSize(int maxElementCount, out int totalSize, out int levelCounts)
            {
                int alignedSupportMaxCount = AlignUpGroup(maxElementCount);
                totalSize = alignedSupportMaxCount;
                levelCounts = 1;
                while (alignedSupportMaxCount > GroupSize)
                {
                    alignedSupportMaxCount = AlignUpGroup(DivUpGroup(alignedSupportMaxCount));
                    totalSize += alignedSupportMaxCount;
                    ++levelCounts;
                }
            }
        }

        /// <summary>
        /// Structure defining level offsets.
        /// </summary>
        // GenerateHLSL
        // Keep in sync with /GPUCompute/PrefixSum/GPUPrefixSum.Data.cs.hlsl
        public struct LevelOffsets
        {
            /// <summary> Number of elements at this level per concurrent independent sum 
            /// For level 0, also the number per concurrent independent sums but exactly the number in the original input,
            /// ie not aligned to the next multiple of group size. Other levels are aligned.
            /// </summary>
            public uint count;

            /// <summary> Level offset. </summary>
            public uint offset;

            /// <summary> Parent level offset. </summary>
            public uint parentOffset;

            /// <summary> Parent level count but aligned up to group size even if parent is level 0 </summary>
            public uint parentAlignedUpCount;
        }

        /// <summary>
        /// Data structure containing the runtime resources that are bound by the command buffer.
        /// </summary>
        public struct SupportResources
        {
            internal bool ownsResources;

            internal bool useRawIOBuffers;        // else Structured

            internal bool groupEachIOPyramidLevels;     // Kernel behavior when processing multiple concurrent prefix sums:
            internal bool groupEachIOSingleLevelLists;  // 1) Should each pyramid be totally independent and separated by a fixed stride, one pyramid (per sum) after the other.
                                                        //
                                                        //   -pyramid level-to-level (for any level) stride does not depend on actual (ie IndirectDirectArgs or DirectArgs) data input
                                                        //    like actual number of concurrent sums (listCount) or perListElementCount.
                                                        //    The stride will thus always be dependent on the max capacity of the ressources, and == maxFullPyramidElementCount
                                                        //
                                                        //   -Otherwise, each levels for *all* concurrent prefix sums can be packed together at the cost of a little extra
                                                        //    bookeeping, such that we have in prefixBuffer0 
                                                        //    [level 0 for concurrent prefix sum #0][level 0 for concurrent prefix sum #1]...[level 0 for concurrent prefix sum #listCount-1]
                                                        //    [level 1 for concurrent prefix sum #0]...[level 1 for concurrent prefix sum #listCount-1]
                                                        //    ...
                                                        //    [level needed based on input - 1, for concurrent prefix sum #0]...[level needed based on input - 1 for concurrent prefix sum #listCount-1]
                                                        //
                                                        //    The actual stride depend on the actual arguments of the run itself, not the ressources.
                                                        //    If the dispatch is driven by indirect args (see inputCountsBuffer) this is all on the GPU, even though
                                                        //    we can predictably recover those strides anywhere from the known indirect args, group_size and AlignUpGroup().
                                                        //    If a more complex dependency existed, it could just be output by the kernel generating indirect-dependent dispatch arguments,
                                                        //    like it already outputs in totalLevelCountBuffer.
                                                        //
                                                        // 2) Similar to 1) but for prefixBuffer1 (and 2), that each store a single level for each concurrent prefix sums
                                                        //    during each level of the algorithm.
                                                        //
                                                        //   -When not packed (or grouped), single levels of each concurrent sum are separated by a fixed stride determined by the ressources,
                                                        //   and since those buffers carry a maximum of maxLevel1PrefixInputCount (level 1 is the largest ever stored in these, each level needs
                                                        //   less and less space), this is also the stride between each concurrent sum.
                                                        //
                                                        //   -When packed, each prefix sum is packed according to the size (aligned up to group_size) of each level, so each level
                                                        //   has a different concurrent sum to next concurrent sum data stride.
                                                        //   These ping pong buffers are not accessed after the algorithm for now so this is not an issue but that stride
                                                        //   can also be predictably recovered if needed.

            internal int  maxConcurrentSums; // multiple prefix sums can be run in parallel and independently: indicates how many to reserve storage appropriately

            // For each parallel sums (identical parameters for each only):

            internal int  maxAlignedElementCount;     // number of elements (max) at level 0, ie for a whole independent prefix sum input, aligned up to the groupsize
            internal int  maxLevel1PrefixInputCount;  // number of elements (max) at level 1, (and for prefixBuffer1 and 2 if needed) aligned up to the groupsize
            internal int  maxFullPyramidElementCount; // pyramid total capacity (for each prefix sum if more than one supported)
            internal int  maxLevelCount;              // maximum pyramid levels given the pyramid allocated (level 0 to max-1)

            // prefixBuffer0 is a whole pyramid with first the level 0 per-group prefixes (until down-adding pass, where it ends up being the full prefix)
            // than the prefix of the per-group totals of the previous level, so the pyramid size is divided (rounding up to groupsize) by groupsize
            // at each level (groupsize is thread group size in threads).

            // prefixBuffer1 is sized maxAlignedElementCount

            internal ComputeBuffer prefixBuffer0;                // IO, full pyramid
            internal ComputeBuffer prefixBuffer1;                // IO, element list(s) for input (one level only)
            internal ComputeBuffer prefixBuffer2;                // IO, element list(s) for input (one level only, same as Buffer1, for ping-pong)

            internal GraphicsBuffer totalLevelCountBuffer;       // raw, content generated by the kernel generating indirect-dependent dispatch arguments and aux info, here only total levels really needed
            internal GraphicsBuffer levelOffsetsBuffer;          // structured<LevelOffsets> content generated by the kernel generating indirect-dependent dispatch arguments and aux info such as this per level list
            internal GraphicsBuffer indirectDispatchArgsBuffer;  // raw, inputs to the kernel generating indirect-dependent dispatch arguments

            /// <summary>The prefix sum result.</summary>
            public ComputeBuffer output => prefixBuffer0;
            public int GetOutputConcurrentSumsStride(int numOfElementsPerSumUsed)
            {
                if (!groupEachIOPyramidLevels)
                    return maxFullPyramidElementCount;

                // if groupEachIOPyramidLevels:
                int res = ShaderDefs.AlignUpGroup(numOfElementsPerSumUsed);
                return res;
            }

            /// <summary>
            /// Allocate support resources to accomodate a max count.
            /// </summary>
            /// <param name="maxElementCount">The max element count.</param>
            /// <param name="maxConcurrentSums">The max num of parallel sums that can be run</param>
            /// <returns>The created support resources.</returns>
            public static SupportResources Create(int maxElementCount, int maxConcurrentSums = 1, bool usePingPongInsteadOfGather = true, bool useRawIOBuffers = true,
                                                  bool groupEachIOPyramidLevels = true, bool groupEachIOSingleLevelLists = true)
            {
                var resources = new SupportResources() 
                {
                    maxAlignedElementCount = 0, ownsResources = true, useRawIOBuffers = useRawIOBuffers,
                    groupEachIOPyramidLevels = groupEachIOPyramidLevels, groupEachIOSingleLevelLists = groupEachIOSingleLevelLists,
                };
                resources.EnsureConfigOrResize(maxElementCount, maxConcurrentSums, usePingPongInsteadOfGather, useRawIOBuffers);
                return resources;
            }

            internal void EnsureConfigOrResize(int newMaxElementCount, int newMaxConcurrentSums = 1, bool usePingPongInsteadOfGather = true, bool newUseRawIOBuffers = false,
                                 bool newGroupEachIOPyramidLevels = true, bool newGroupEachIOSingleLevelLists = true)
            {
                if (!ownsResources)
                    throw new Exception("Cannot resize resources unless they are owned. Use GpuPrefixSumSupportResources.Create() for this.");

                if (newMaxConcurrentSums > 65535) // D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION
                    throw new Exception("Number of parallel sums requested not supported.");

                newMaxElementCount = Math.Max(newMaxElementCount, 1); //at bare minimum support a single group.
                newMaxConcurrentSums = Math.Max(newMaxConcurrentSums, 1);

                // Commented but gives same result, not necessary to test though
                //int newMaxLevel1PrefixInputCount = ShaderDefs.AlignUpGroup(ShaderDefs.DivUpGroup(newMaxElementCount));

                // Note these don't change resource config or allocations, but the returned inferred stride
                // between each independent concurrent sums results and the kernel configurations for packing
                // working sets during kernel runs.
                groupEachIOPyramidLevels = newGroupEachIOPyramidLevels;
                groupEachIOSingleLevelLists = newGroupEachIOSingleLevelLists;

                if ( useRawIOBuffers == newUseRawIOBuffers
                    && maxAlignedElementCount >= newMaxElementCount
                    && maxConcurrentSums >= newMaxConcurrentSums
                    //&& maxLevel1PrefixInputCount >= newMaxLevel1PrefixInputCount
                    && (!usePingPongInsteadOfGather || (prefixBuffer2 != null)))
                    return;

                Dispose();
                ShaderDefs.CalculateTotalPyramidBufferSize(newMaxElementCount, out int totalSize, out int levelCounts);

                maxConcurrentSums          = newMaxConcurrentSums;
                maxAlignedElementCount     = ShaderDefs.AlignUpGroup(newMaxElementCount);
                maxLevel1PrefixInputCount  = ShaderDefs.AlignUpGroup(ShaderDefs.DivUpGroup(maxAlignedElementCount));//newMaxElementCount instead of maxAlignedElementCount would work too;
                maxFullPyramidElementCount = totalSize;
                maxLevelCount              = levelCounts;

#if USE_GRAPHICSBUFFER
                GraphicsBuffer.Target targetType = useRawIOBuffers ? GraphicsBuffer.Target.Raw : GraphicsBuffer.Target.Structured;

                prefixBuffer0              = new GraphicsBuffer(targetType, maxFullPyramidElementCount * maxConcurrentSums, 4);
                // Bug in SRP core: no resize will happen if maxAlignedElementCount is enough for a new value of param "newMaxElementCount",
                // and this buffer wouldn't be re-allocated to fit it!
                // bug: prefixBuffer1              = new GraphicsBuffer(GraphicsBuffer.Target.Raw, newMaxElementCount, 4);
                // Just use maxAlignedElementCount even if technically newMaxElementCount would be enough, actually even less is needed:
                // prefixBuffer1 is used as input after the first level (level 0, which uses the main user input buffer), and at that point,
                // the required number of items has already dropped by 1/GROUP_SIZE, ie becomes AlignUpGroup(DivUpGroup(newMaxElementCount))
                //prefixBuffer1              = new GraphicsBuffer(GraphicsBuffer.Target.Raw, maxAlignedElementCount * maxConcurrentSums, 4);
                
                prefixBuffer1              = new GraphicsBuffer(targetType, maxLevel1PrefixInputCount * maxConcurrentSums, 4);
                prefixBuffer2              = usePingPongInsteadOfGather ? new GraphicsBuffer(targetType, maxLevel1PrefixInputCount * maxConcurrentSums, 4) : null;
#else
                ComputeBufferType bufferType = useRawIOBuffers ? ComputeBufferType.Raw : ComputeBufferType.Structured;

                prefixBuffer0              = new ComputeBuffer(maxFullPyramidElementCount * maxConcurrentSums, 4, bufferType);
                prefixBuffer1              = new ComputeBuffer(maxLevel1PrefixInputCount * maxConcurrentSums, 4, bufferType);
                prefixBuffer2              = usePingPongInsteadOfGather ? new ComputeBuffer(maxLevel1PrefixInputCount * maxConcurrentSums, 4, bufferType) : null;
#endif
                totalLevelCountBuffer      = new GraphicsBuffer(GraphicsBuffer.Target.Raw, 1, 4);
                levelOffsetsBuffer         = new GraphicsBuffer(GraphicsBuffer.Target.Structured, levelCounts, System.Runtime.InteropServices.Marshal.SizeOf<LevelOffsets>());
                indirectDispatchArgsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, ShaderDefs.ArgsBufferStride * levelCounts, sizeof(uint));//3 arguments for upp dispatch, 3 arguments for lower dispatch
            }

            /// <summary>
            /// Dispose the supporting resources.
            /// </summary>
            public void Dispose()
            {
                if (maxAlignedElementCount == 0 || !ownsResources)
                    return;

                maxAlignedElementCount = 0;

                void TryFreeBuffer(ComputeBuffer resource)
                {
                    if (resource != null)
                    {
                        resource.Dispose();
                        resource = null;
                    }
                }
                void TryFreeGfxBuffer(GraphicsBuffer resource)
                {
                    if (resource != null)
                    {
                        resource.Dispose();
                        resource = null;
                    }
                }

                TryFreeBuffer(prefixBuffer0);
                TryFreeBuffer(prefixBuffer1);
                TryFreeBuffer(prefixBuffer2);
                TryFreeGfxBuffer(levelOffsetsBuffer);
                TryFreeGfxBuffer(indirectDispatchArgsBuffer);
                TryFreeGfxBuffer(totalLevelCountBuffer);
            }
        }

        /// <summary>
        /// Arguments for a direct prefix sum.
        /// </summary>
        public struct DirectArgs
        {
            /// <summary>An inclusive or exclusive prefix sum.</summary>
            public bool             exclusive;
            /// <summary>Do prefix sum by casting each input element as their bit count.</summary>
            public bool             castInputAsBitcounts;

            /// <summary>The size of the (or each) input list.</summary>
            public int              perListElementCount;
            /// <summary>The number of input lists if executing multiple prefix sums in parallel.</summary>
            public int              listCount;
            //public int              inputsBufferListToListStride;
            // assume stride is perListElementCount, ie all lists are packed in the main input
            /// <summary>The input list(s) (if more than one list, they are all packed one after the other).</summary>
            public ComputeBuffer   inputs;
            /// <summary>Required runtime resources.</summary>
            public SupportResources supportResources;
        }

        /// <summary>
        /// Arguments for an indirect prefix sum.
        /// </summary>
        public struct IndirectDirectArgs
        {
            /// <summary>An inclusive or exclusive prefix sum.</summary>
            public bool             exclusive;
            /// <summary>Do prefix sum by casting each input element as their bit count.</summary>
            public bool             castInputAsBitcounts;

            /// <summary>Byte offset of the input count inside the input count buffer.</summary>
            public int              perListElementCountBufferByteOffset;
            /// <summary>Byte offset of the number of lists inside the input count buffer.</summary>
            public int              listCountBufferByteOffset;
            //public int              inputsBufferListToListStride;
            // assume stride is perListElementCount, ie all lists are packed in the main input
            /// <summary>GPU buffer defining the size of the (or of each) input list and their number.</summary>
            public GraphicsBuffer   inputCountsBuffer;
            /// <summary>The input list.</summary>
            public ComputeBuffer    inputs;
            /// <summary>Required runtime resources.</summary>
            public SupportResources supportResources;
        }

        /// <summary>
        /// Structure defining any required assets used by the GPU sort.
        /// </summary>
        public struct SystemResources
        {
            /// <summary>
            /// The compute asset that defines all of the kernels for the GPU prefix sum.
            /// </summary>
            public ComputeShader computeAsset;

            internal enum KeywordIds
            {
                //UsingRawIOBuffers,
                PackAllSumsSingleLevelsIO,
                PackAllSumsPyramidLevels,
                //PrefixSumPassFillsNextInput,
            }

            internal LocalKeyword[] Keywords;

            internal int kernelCalculateLevelDispatchArgsFromConst;
            internal int kernelCalculateLevelDispatchArgsFromBuffer;
            internal int kernelPrefixSumOnGroup;
            internal int kernelPrefixSumOnGroupExclusive;
            internal int kernelPrefixSumNextInput;
            internal int kernelPrefixSumResolveParent;
            internal int kernelPrefixSumResolveParentExclusive;

            internal int kernelPrefixSumOnGroupOrigInputAsBitCnt;
            internal int kernelPrefixSumOnGroupExclusiveOrigInputAsBitCnt;

            internal int kernelPrefixSumOnGroupFillNext;
            internal int kernelPrefixSumOnGroupExclusiveFillNext;

            internal int kernelMainPrefixSumOnGroupOrigInputAsBitCntFillNext;
            internal int kernelMainPrefixSumOnGroupExclusiveOrigInputAsBitCntFillNext;

            internal int kernelMainPrefixSumResolveParentOrigInputAsBitCnt;
            internal int kernelMainPrefixSumResolveParentExclusiveOrigInputAsBitCnt;

            internal int kernelMainGatherScaleBiasClampAbove;

            internal void InsertAllSetKeywords(CommandBuffer cmdBuffer, in SupportResources supportResources, bool usePingPongInsteadOfGather)
            {
                if (computeAsset == null)
                    return;

                if (supportResources.prefixBuffer2 == null)
                {
                    // TODO: message
                    usePingPongInsteadOfGather = false;
                }
                //cmdBuffer.SetKeyword(computeAsset, Keywords[(int)KeywordIds.UsingRawIOBuffers], supportResources.useRawIOBuffers);
                cmdBuffer.SetKeyword(computeAsset, Keywords[(int)KeywordIds.PackAllSumsSingleLevelsIO], supportResources.groupEachIOSingleLevelLists);
                cmdBuffer.SetKeyword(computeAsset, Keywords[(int)KeywordIds.PackAllSumsPyramidLevels], supportResources.groupEachIOPyramidLevels);
                //cmdBuffer.SetKeyword(computeAsset, Keywords[KeywordIds.PrefixSumPassFillsNextInput], usePingPongInsteadOfGather);
            }

            internal void LoadKeywords()
            {
                if (computeAsset == null)
                    return;

                Keywords = new LocalKeyword[2]
                {
                    //new(computeAsset, "PREFIX_SUM_USES_RAW_IO"),
                    new(computeAsset, "PACK_CONCURRENT_SUM_SINGLE_LEVELS"),
                    new(computeAsset, "PACK_CONCURRENT_SUM_PYRAMID_LEVELS"),
                    //new(computeAsset, "PREFIX_SUM_FILLS_NEXT_INPUT")
                };
            }

            internal void LoadKernels()
            {
                if (computeAsset == null)
                    return;

                kernelCalculateLevelDispatchArgsFromConst  = computeAsset.FindKernel("MainCalculateLevelDispatchArgsFromConst");
                kernelCalculateLevelDispatchArgsFromBuffer = computeAsset.FindKernel("MainCalculateLevelDispatchArgsFromBuffer");

                kernelPrefixSumNextInput = computeAsset.FindKernel("MainPrefixSumNextInput");

                // prefix
                kernelPrefixSumOnGroup                     = computeAsset.FindKernel("MainPrefixSumOnGroup");
                kernelPrefixSumOnGroupExclusive            = computeAsset.FindKernel("MainPrefixSumOnGroupExclusive");

                // prefix with initial input cast as bitcnts
                kernelPrefixSumOnGroupOrigInputAsBitCnt                = computeAsset.FindKernel("MainPrefixSumOnGroupOrigInputAsBitCnt");
                kernelPrefixSumOnGroupExclusiveOrigInputAsBitCnt       = computeAsset.FindKernel("MainPrefixSumOnGroupExclusiveOrigInputAsBitCnt");

                // prefix with direct output for next pass (ping pong buffers)
                kernelPrefixSumOnGroupFillNext                         = computeAsset.FindKernel("MainPrefixSumOnGroupFillNext");
                kernelPrefixSumOnGroupExclusiveFillNext                = computeAsset.FindKernel("MainPrefixSumOnGroupExclusiveFillNext");

                // prefix with both of the above
                kernelMainPrefixSumOnGroupOrigInputAsBitCntFillNext          = computeAsset.FindKernel("MainPrefixSumOnGroupOrigInputAsBitCntFillNext");
                kernelMainPrefixSumOnGroupExclusiveOrigInputAsBitCntFillNext = computeAsset.FindKernel("MainPrefixSumOnGroupExclusiveOrigInputAsBitCntFillNext");

                // prefix downpass of partial prefix resolve on parents, but with initial input cast as bitcnts
                kernelMainPrefixSumResolveParentOrigInputAsBitCnt          = computeAsset.FindKernel("MainPrefixSumResolveParentOrigInputAsBitCnt");
                kernelMainPrefixSumResolveParentExclusiveOrigInputAsBitCnt = computeAsset.FindKernel("MainPrefixSumResolveParentExclusiveOrigInputAsBitCnt");

                kernelPrefixSumResolveParent          = computeAsset.FindKernel("MainPrefixSumResolveParent");
                kernelPrefixSumResolveParentExclusive = computeAsset.FindKernel("MainPrefixSumResolveParentExclusive");

                kernelMainGatherScaleBiasClampAbove = computeAsset.FindKernel("MainGatherScaleBiasClampAbove");
            }

            internal (int, int) GetPrefixSumAndResolveKernelVariants(bool isExclusive, bool castInputAsBitcounts, bool usePingPongInsteadOfGather)
            {
                switch ((castInputAsBitcounts, usePingPongInsteadOfGather, isExclusive))
                {
                    case (false, false, false):
                        return (kernelPrefixSumOnGroup, kernelPrefixSumResolveParent);
                        //break;
                    case (false, false, true):
                        return (kernelPrefixSumOnGroupExclusive, kernelPrefixSumResolveParentExclusive);
                        //break;
                    case (false, true, false):
                        return (kernelPrefixSumOnGroupFillNext, kernelPrefixSumResolveParent);
                        //break;
                    case (false, true, true):
                        return (kernelPrefixSumOnGroupExclusiveFillNext, kernelPrefixSumResolveParentExclusive);
                        //break;
                    case (true, false, false):
                        return (kernelPrefixSumOnGroupOrigInputAsBitCnt, kernelMainPrefixSumResolveParentOrigInputAsBitCnt);
                        //break;
                    case (true, false, true):
                        return (kernelPrefixSumOnGroupExclusiveOrigInputAsBitCnt, kernelMainPrefixSumResolveParentExclusiveOrigInputAsBitCnt);
                        //break;
                    case (true, true, false):
                        return (kernelMainPrefixSumOnGroupOrigInputAsBitCntFillNext, kernelMainPrefixSumResolveParentOrigInputAsBitCnt);
                        //break;
                    case (true, true, true):
                        return (kernelMainPrefixSumOnGroupExclusiveOrigInputAsBitCntFillNext, kernelMainPrefixSumResolveParentExclusiveOrigInputAsBitCnt);
                        //break;
                }
            }
        }
    }
}
