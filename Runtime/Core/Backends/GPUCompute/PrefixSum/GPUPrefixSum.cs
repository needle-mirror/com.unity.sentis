using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Unity.Sentis
{
    /// <summary>
    /// Utility class for computing inclusive or exclusive prefix sums, directly or indirectly dispatched on the GPU.
    /// </summary>
    public partial struct GPUPrefixSum
    {
        private SystemResources resources;

        /// <summary>
        /// Initializes a re-usable GPU prefix sum instance.
        /// </summary>
        /// <param name="resources">The required system resources.</param>
        public GPUPrefixSum(SystemResources resources)
        {
            this.resources = resources;
            this.resources.LoadKernels();
            this.resources.LoadKeywords();
        }

        int[] PackPrefixSumArgs(int a, int b, int c, int d)
        {
            return new int[] { a, b, c, d };
        }

        internal void ExecuteCommonIndirect(CommandBuffer cmdBuffer, ComputeBuffer inputsBuffer, in SupportResources supportResources, bool isExclusive, bool castInputAsBitcounts,
                                            bool usePingPongBuffers)
        {
            bool usePingPong = usePingPongBuffers;

            (int sumOnGroupKernel, int sumResolveParentKernel) k = resources.GetPrefixSumAndResolveKernelVariants(isExclusive, castInputAsBitcounts, usePingPong);

            //this should have been done before the indirect builder kernel:
            //resources.InsertAllSetKeywords(cmdBuffer, supportResources, usePingPong);

            // _PrefixSumIntArgs2 doesn't depend on loops below, set it once here:
            // .y = stride in pyramid buffer for each results pyramid of concurrent prefix sums if more than one:
            var packedArgs2 = PackPrefixSumArgs(0, supportResources.maxFullPyramidElementCount, 0, supportResources.maxLevel1PrefixInputCount);
            cmdBuffer.SetComputeIntParams(resources.computeAsset, ShaderIDs._PrefixSumIntArgs2, packedArgs2);

            //up the hierarchy: at each level, schedule prefix sums of (per group) partial prefix sums of previous level
            bool nextInputBufferIs1 = false;
            for (int levelId = 0; levelId < supportResources.maxLevelCount; ++levelId)
            {
                var packedArgs = PackPrefixSumArgs(0, supportResources.maxLevelCount, 0, levelId);
                cmdBuffer.SetComputeIntParams(resources.computeAsset, ShaderIDs._PrefixSumIntArgs, packedArgs);

                if (levelId == 0)
                {
                    cmdBuffer.SetComputeBufferParam(resources.computeAsset, k.sumOnGroupKernel, ShaderIDs._InputBuffer, inputsBuffer);
                }
                else
                {
                    cmdBuffer.SetComputeBufferParam(resources.computeAsset, k.sumOnGroupKernel, ShaderIDs._InputBuffer, supportResources.prefixBuffer1);
                }

                if (usePingPong)
                {
                    nextInputBufferIs1 = !nextInputBufferIs1;
                    var nextInputBuffer = nextInputBufferIs1 ? supportResources.prefixBuffer1 : supportResources.prefixBuffer2;
                    cmdBuffer.SetComputeBufferParam(resources.computeAsset, k.sumOnGroupKernel, ShaderIDs._NextInputBuffer, nextInputBuffer);
                }

                cmdBuffer.SetComputeBufferParam(resources.computeAsset, k.sumOnGroupKernel, ShaderIDs._TotalLevelsBuffer, supportResources.totalLevelCountBuffer);
                cmdBuffer.SetComputeBufferParam(resources.computeAsset, k.sumOnGroupKernel, ShaderIDs._LevelsOffsetsBuffer, supportResources.levelOffsetsBuffer);
                cmdBuffer.SetComputeBufferParam(resources.computeAsset, k.sumOnGroupKernel, ShaderIDs._OutputBuffer, supportResources.prefixBuffer0);
                cmdBuffer.DispatchCompute(resources.computeAsset, k.sumOnGroupKernel, supportResources.indirectDispatchArgsBuffer, (uint)(levelId * ShaderDefs.ArgsBufferStride * 4));

                if (levelId == supportResources.maxLevelCount - 1)
                    continue;

                if (!usePingPong)
                {
                    cmdBuffer.SetComputeBufferParam(resources.computeAsset, resources.kernelPrefixSumNextInput, ShaderIDs._InputBuffer, supportResources.prefixBuffer0);
                    cmdBuffer.SetComputeBufferParam(resources.computeAsset, resources.kernelPrefixSumNextInput, ShaderIDs._LevelsOffsetsBuffer, supportResources.levelOffsetsBuffer);
                    cmdBuffer.SetComputeBufferParam(resources.computeAsset, resources.kernelPrefixSumNextInput, ShaderIDs._OutputBuffer, supportResources.prefixBuffer1);
                    cmdBuffer.DispatchCompute(resources.computeAsset, resources.kernelPrefixSumNextInput, supportResources.indirectDispatchArgsBuffer, (uint)((levelId + 1) * ShaderDefs.ArgsBufferStride * 4));
                }
            }

            //down the hierarchy: add back "recursively" the full prefix offsets (from a given level fully resolved prefix sum) that corresponds to
            // the true prefix missing for each group partial prefix of the parent level:
            for (int levelId = supportResources.maxLevelCount - 1; levelId >= 1; --levelId)
            {
                var packedArgs = PackPrefixSumArgs(0, supportResources.maxLevelCount, 0, levelId);
                cmdBuffer.SetComputeIntParams(resources.computeAsset, ShaderIDs._PrefixSumIntArgs, packedArgs);
                cmdBuffer.SetComputeBufferParam(resources.computeAsset, k.sumResolveParentKernel, ShaderIDs._InputBuffer, inputsBuffer);
                cmdBuffer.SetComputeBufferParam(resources.computeAsset, k.sumResolveParentKernel, ShaderIDs._OutputBuffer, supportResources.prefixBuffer0);
                cmdBuffer.SetComputeBufferParam(resources.computeAsset, k.sumResolveParentKernel, ShaderIDs._LevelsOffsetsBuffer, supportResources.levelOffsetsBuffer);
                cmdBuffer.DispatchCompute(resources.computeAsset, k.sumResolveParentKernel, supportResources.indirectDispatchArgsBuffer, (uint)(((levelId - 1) * ShaderDefs.ArgsBufferStride + ShaderDefs.ArgsBufferLower) * 4));
            }
        }

        /// <summary>
        /// Prefix sum a list of data from a CPU-defined count.
        /// </summary>
        /// <param name="cmdBuffer">Command Buffer for recording the prefix sum commands.</param>
        /// <param name="arguments">Runtime arguments for the prefix sum.</param>
        /// <exception cref="Exception">When the input data is invalid.</exception>
        public void DispatchDirect(CommandBuffer cmdBuffer, in DirectArgs arguments, bool debugJustDispatchIndirectBuilder = false)
        {
            if (arguments.supportResources.prefixBuffer0 == null || arguments.supportResources.prefixBuffer1 == null)
                throw new Exception("Support resources are not valid.");

            if (arguments.inputs == null)
                throw new Exception("Input source buffer cannot be null.");

            if (arguments.perListElementCount > arguments.supportResources.maxAlignedElementCount)
                throw new Exception("Input count exceeds maximum count of support resources. Ensure to create support resources with enough space.");

            bool usePingPong = true && arguments.supportResources.prefixBuffer2 != null; // Instead of gather

            // Set keywords to select kernels
            resources.InsertAllSetKeywords(cmdBuffer, arguments.supportResources, usePingPong);

            //Generate level offsets first, from const value.
            var packedArgs = PackPrefixSumArgs(arguments.perListElementCount, arguments.supportResources.maxLevelCount, 0, 0);
            // num of parallel sums to run (listCount) and .y = stride in pyramid buffer for each results pyramid of concurrent prefix sums if more than one:
            var packedArgs2 = PackPrefixSumArgs(arguments.listCount, arguments.supportResources.maxFullPyramidElementCount, 0, arguments.supportResources.maxLevel1PrefixInputCount);
            cmdBuffer.SetComputeIntParams(resources.computeAsset, ShaderIDs._PrefixSumIntArgs, packedArgs);
            cmdBuffer.SetComputeIntParams(resources.computeAsset, ShaderIDs._PrefixSumIntArgs2, packedArgs2);
            cmdBuffer.SetComputeBufferParam(resources.computeAsset, resources.kernelCalculateLevelDispatchArgsFromConst, ShaderIDs._OutputLevelsOffsetsBuffer, arguments.supportResources.levelOffsetsBuffer);
            cmdBuffer.SetComputeBufferParam(resources.computeAsset, resources.kernelCalculateLevelDispatchArgsFromConst, ShaderIDs._OutputDispatchLevelArgsBuffer, arguments.supportResources.indirectDispatchArgsBuffer);
            cmdBuffer.SetComputeBufferParam(resources.computeAsset, resources.kernelCalculateLevelDispatchArgsFromConst, ShaderIDs._OutputTotalLevelsBuffer, arguments.supportResources.totalLevelCountBuffer);
            cmdBuffer.DispatchCompute(resources.computeAsset, resources.kernelCalculateLevelDispatchArgsFromConst, 1, 1, 1);

            if (!debugJustDispatchIndirectBuilder)
                ExecuteCommonIndirect(cmdBuffer, arguments.inputs, arguments.supportResources, arguments.exclusive, arguments.castInputAsBitcounts, usePingPongBuffers:usePingPong);
        }

        public void DispatchGatherScaleBiasClampAbove(CommandBuffer cmdBuffer, ComputeBuffer input, ComputeBuffer output, int elementCount, int elementStride, int scale, int bias, int clampToMaxValue)
        {
            int dispatchSizeInThreads = ShaderDefs.AlignUpGroup((elementCount + ShaderDefs.GatherScaleBiasClampaItemsPerThread - 1) / ShaderDefs.GatherScaleBiasClampaItemsPerThread);
            int groupCount = ShaderDefs.DivUpGroup(dispatchSizeInThreads);

            var packedArgs = PackPrefixSumArgs(elementStride, scale, bias, clampToMaxValue);
            var packedArgs2 = PackPrefixSumArgs(elementCount, dispatchSizeInThreads, 0, 0);
            cmdBuffer.SetComputeIntParams(resources.computeAsset, ShaderIDs._PrefixSumIntArgs, packedArgs);
            cmdBuffer.SetComputeIntParams(resources.computeAsset, ShaderIDs._PrefixSumIntArgs2, packedArgs2);
            cmdBuffer.SetComputeBufferParam(resources.computeAsset, resources.kernelMainGatherScaleBiasClampAbove, ShaderIDs._InputBuffer, input);
            cmdBuffer.SetComputeBufferParam(resources.computeAsset, resources.kernelMainGatherScaleBiasClampAbove, ShaderIDs._OutputBuffer, output);
            cmdBuffer.DispatchCompute(resources.computeAsset, resources.kernelMainGatherScaleBiasClampAbove, groupCount, 1, 1);
        }


        /// <summary>
        /// Prefix sum a list of data from a GPU-defined count.
        /// </summary>
        /// <param name="cmdBuffer">Command Buffer for recording the prefix sum commands.</param>
        /// <param name="arguments">Runtime arguments for the prefix sum.</param>
        /// <exception cref="Exception">When the input data is invalid.</exception>

        // WARNING: NOT TESTED!
        public void DispatchIndirect(CommandBuffer cmdBuffer, in IndirectDirectArgs arguments, bool debugJustDispatchIndirectBuilder = false)
        {
            if (arguments.supportResources.prefixBuffer0 == null || arguments.supportResources.prefixBuffer1 == null)
                throw new Exception("Support resources are not valid.");

            if (arguments.inputs == null || arguments.inputCountsBuffer == null)
                throw new Exception("Input source buffer and inputCountsBuffer cannot be null.");

            bool usePingPong = true && arguments.supportResources.prefixBuffer2 != null; // Instead of gather

            // Set keywords to select kernels
            resources.InsertAllSetKeywords(cmdBuffer, arguments.supportResources, usePingPong);

            //Generate level offsets first, from const value.
            var packedArgs = PackPrefixSumArgs(0, arguments.supportResources.maxLevelCount, arguments.perListElementCountBufferByteOffset, 0);
            // num of parallel sums to run (listCount) and .y = stride in pyramid buffer for each results pyramid of concurrent prefix sums if more than one:
            var packedArgs2 = PackPrefixSumArgs(0, arguments.supportResources.maxFullPyramidElementCount, arguments.listCountBufferByteOffset, arguments.supportResources.maxLevel1PrefixInputCount);
            cmdBuffer.SetComputeIntParams(resources.computeAsset, ShaderIDs._PrefixSumIntArgs, packedArgs);
            cmdBuffer.SetComputeIntParams(resources.computeAsset, ShaderIDs._PrefixSumIntArgs2, packedArgs2);
            cmdBuffer.SetComputeBufferParam(resources.computeAsset, resources.kernelCalculateLevelDispatchArgsFromBuffer, ShaderIDs._InputCountsBuffer, arguments.inputCountsBuffer);
            cmdBuffer.SetComputeBufferParam(resources.computeAsset, resources.kernelCalculateLevelDispatchArgsFromBuffer, ShaderIDs._OutputLevelsOffsetsBuffer, arguments.supportResources.levelOffsetsBuffer);
            cmdBuffer.SetComputeBufferParam(resources.computeAsset, resources.kernelCalculateLevelDispatchArgsFromBuffer, ShaderIDs._OutputDispatchLevelArgsBuffer, arguments.supportResources.indirectDispatchArgsBuffer);
            cmdBuffer.SetComputeBufferParam(resources.computeAsset, resources.kernelCalculateLevelDispatchArgsFromBuffer, ShaderIDs._OutputTotalLevelsBuffer, arguments.supportResources.totalLevelCountBuffer);
            cmdBuffer.DispatchCompute(resources.computeAsset, resources.kernelCalculateLevelDispatchArgsFromBuffer, 1, 1, 1);

            if (!debugJustDispatchIndirectBuilder)
                ExecuteCommonIndirect(cmdBuffer, arguments.inputs, arguments.supportResources, arguments.exclusive, arguments.castInputAsBitcounts, usePingPongBuffers:usePingPong);
        }
    }
}
