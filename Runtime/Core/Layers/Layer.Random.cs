using System;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents the abstract base class for layers which generate random values in the output tensor.
    /// </summary>
    abstract class RandomLayer : Layer
    {
        public bool hasSeed;
        public int seed;
        [NonSerialized]
        Random m_Random;

        protected int NextSeed => m_Random.NextSeed();

        public void ResetSeed()
        {
            m_Random = hasSeed ? new Random(seed) : new Random();
        }

        protected RandomLayer(int[] outputs, int[] inputs, int? seed)
            : base(outputs, inputs)
        {
            hasSeed = seed.HasValue;
            this.seed = seed ?? 0;
            ResetSeed();
        }
    }

    /// <summary>
    /// Represents a `RandomNormal` random layer. This generates an output tensor of a given shape with random values in a normal distribution with given `mean` and `scale`, and an optional `seed` value.
    /// </summary>
    class RandomNormal : RandomLayer
    {
        public float mean;
        public float scale;
        public int[] shape;

        public RandomNormal(int output, int[] shape, float mean, float scale, int? seed)
            : base(new[] { output }, Array.Empty<int>(), seed)
        {
            this.mean = mean;
            this.scale = scale;
            this.shape = shape;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, new SymbolicTensorShape(new TensorShape(shape))));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(shape), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.RandomNormal(O, mean, scale, NextSeed);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, mean: {mean}, scale: {scale}, hasSeed: {hasSeed}, seed: {seed}";
        }

        internal override string profilerTag => "RandomNormal";
    }

    /// <summary>
    /// Represents a `RandomNormalLike` random layer. This generates an output tensor with the same shape as the input tensor with random values in a normal distribution, with given `mean` and `scale`, and an optional `seed` value.
    /// </summary>
    class RandomNormalLike : RandomLayer
    {
        public float mean;
        public float scale;

        public RandomNormalLike(int output, int input, float mean, float scale, int? seed)
            : base(new[] { output }, new[] { input }, seed)
        {
            this.mean = mean;
            this.scale = scale;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, ctx.GetPartialTensor(inputs[0]).shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var shapeX = ctx.storage.GetTensorShape(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shapeX, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.RandomNormal(O, mean, scale, NextSeed);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, mean: {mean}, scale: {scale}, hasSeed: {hasSeed}, seed: {seed}";
        }

        internal override string profilerTag => "RandomNormalLike";
    }

    /// <summary>
    /// Represents a `RandomUniform` random layer. This generates an output tensor of a given shape with random values in a uniform distribution between a given `low` and `high`, from an optional `seed` value.
    /// </summary>
    class RandomUniform : RandomLayer
    {
        public float low;
        public float high;
        public int[] shape;

        public RandomUniform(int output, int[] shape, float low, float high, int? seed)
            : base(new[] { output }, Array.Empty<int>(), seed)
        {
            this.low = low;
            this.high = high;
            this.shape = shape;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, new SymbolicTensorShape(new TensorShape(shape))));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(shape), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.RandomUniform(O, low, high, NextSeed);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, low: {low}, high: {high}, hasSeed: {hasSeed}, seed: {seed}";
        }

        internal override string profilerTag => "RandomUniform";
    }

    /// <summary>
    /// Represents a `RandomUniformLike` random layer. This generates an output tensor with the same shape as the input tensor random values in a uniform distribution between a given `low` and `high`, from an optional `seed` value.
    /// </summary>
    class RandomUniformLike : RandomLayer
    {
        public float low;
        public float high;

        public RandomUniformLike(int output, int input, float low, float high, int? seed)
            : base(new[] { output }, new[] { input }, seed)
        {
            this.low = low;
            this.high = high;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Float, ctx.GetPartialTensor(inputs[0]).shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var shapeX = ctx.storage.GetTensorShape(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shapeX, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.RandomUniform(O, low, high, NextSeed);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, mean: {low}, scale: {high}, hasSeed: {hasSeed}, seed: {seed}";
        }

        internal override string profilerTag => "RandomUniformLike";
    }

    /// <summary>
    /// Represents a `Bernoulli` random layer. This generates an output tensor with values 0 or 1 from a Bernoulli distribution. The input tensor contains the probabilities used for generating the output values.
    /// </summary>
    class Bernoulli : RandomLayer
    {
        public DataType dataType;

        public Bernoulli(int output, int input, DataType dataType, int? seed)
            : base(new[] { output }, new[] { input }, seed)
        {
            this.dataType = dataType;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(outputs[0], new PartialTensor(dataType, ctx.GetPartialTensor(inputs[0]).shape));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Bernoulli(X, O, NextSeed);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, dataType: {dataType}, hasSeed: {hasSeed}, seed: {seed}";
        }

        internal override string profilerTag => "Bernoulli";
    }

    /// <summary>
    /// Represents a `Multinomial` random layer. This generates an output tensor with values from a multinomial distribution according to the probabilities given by the input tensor.
    /// </summary>
    class Multinomial : RandomLayer
    {
        public int count;

        public Multinomial(int output, int input, int count, int? seed)
            : base(new[] { output }, new[] { input }, seed)
        {
            this.count = count;
        }

        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;
            ctx.AddPartialTensor(outputs[0], new PartialTensor(DataType.Int, new SymbolicTensorShape(shapeX[0], SymbolicTensorDim.Int(count))));
        }

        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(X.shape[0], count), DataType.Int, ctx.backend.backendType) as TensorInt;

            var Xtmp = ctx.storage.AllocateTensor(X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            var random = ctx.storage.AllocateTensor(new TensorShape(X.shape[0], count), DataType.Float, ctx.backend.backendType) as TensorFloat;

            ctx.backend.RandomUniform(random, 0, 1, NextSeed);
            ctx.backend.Softmax(X, Xtmp, -1);
            ctx.backend.TopP(Xtmp, random, O);

            ctx.storage.Dispose(Xtmp);
            ctx.storage.Dispose(random);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, count: {count}, hasSeed: {hasSeed}, seed: {seed}";
        }

        internal override string profilerTag => "Multinomial";
    }
}
