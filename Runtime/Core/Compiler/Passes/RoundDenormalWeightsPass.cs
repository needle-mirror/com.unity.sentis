using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Burst;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class RoundDenormalWeightsPass : IModelPass
    {
        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard, CompileSynchronously = true)]
        internal unsafe struct RoundDenormalJob : IJobParallelFor
        {
            [NoAlias] [NativeDisableUnsafePtrRestriction] public uint* ptr;

            public void Execute(int index)
            {
                if (float.IsSubnormal(ptr[index]))
                    ptr[index] = 0;
            }
        }

        public void Run(ref Model model)
        {
            foreach (var constant in model.constants)
            {
                if (constant.weights == null || constant.dataType != DataType.Float)
                    continue;

                unsafe
                {
                    var job = new RoundDenormalJob
                    {
                        ptr = (uint*)constant.weights.RawPtr
                    };
                    var jobHandle = job.Schedule(constant.shape.length, 32);
                    jobHandle.Complete();
                }
            }
        }
    }
}
