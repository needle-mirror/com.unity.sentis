using System;
using System.Runtime.Serialization;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents the abstract base class for layers which generate random values in the output tensor.
    /// </summary>
    [Serializable]
    [NonDeterministicOutput]
    abstract class RandomLayer : Layer
    {
        /// <summary>
        /// Whether the layer has a provided seed value for the random number generator.
        /// </summary>
        public bool hasSeed;
        /// <summary>
        /// The seed value for the random number generator. The layer does not use this value if `hasSeed` is `false`.
        /// </summary>
        public int seed;
        [NonSerialized]
        Random m_Random;

        /// <summary>
        /// Gets the next seed value for execution.
        /// </summary>
        protected int NextSeed => m_Random.NextSeed();

        /// <summary>
        /// Resets the state of the random number generator to its initial state.
        ///
        /// If `hasSeed` is `false` then the random number generator is seeded by System.Random().
        /// </summary>
        public void ResetSeed()
        {
            m_Random = hasSeed ? new Random(seed) : new Random();
        }

        /// <summary>
        /// Initializes and returns an instance of `RandomLayer`.
        /// </summary>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the layer generates a seed using `System.Random()`.</param>
        protected RandomLayer(int? seed)
        {
            hasSeed = seed.HasValue;
            this.seed = seed ?? 0;
            ResetSeed();
        }
    }

    /// <summary>
    /// Represents a `RandomNormal` random layer. This generates an output tensor of a given shape with random values in a normal distribution with given `mean` and `scale`, and an optional `seed` value.
    /// </summary>
    [Serializable]
    class RandomNormal : RandomLayer
    {
        /// <summary>
        /// The mean of the normal distribution used to generate the output.
        /// </summary>
        public float mean;
        /// <summary>
        /// The standard deviation of the normal distribution used to generate the output.
        /// </summary>
        public float scale;
        /// <summary>
        /// The shape for the output tensor as an int array.
        /// </summary>
        public int[] shape;

        /// <summary>
        /// Initializes and returns an instance of `RandomNormal` random layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="shape">The shape for the output tensor as an array of ints.</param>
        /// <param name="mean">The mean of the normal distribution used to generate the output.</param>
        /// <param name="scale">The standard deviation of the normal distribution used to generate the output.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the layer generates a seed using `System.Random()`.</param>
        public RandomNormal(string name, int[] shape, float mean, float scale, int? seed)
            : base(seed)
        {
            this.index = name;
            this.inputs = Array.Empty<string>();
            this.mean = mean;
            this.scale = scale;
            this.shape = shape;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(index, new PartialTensor(DataType.Float, new SymbolicTensorShape(new TensorShape(shape))));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var O = ctx.vars.AllocateTensorAndStore(index, new TensorShape(shape), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.RandomNormal(O, mean, scale, NextSeed);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, mean: {mean}, scale: {scale}, hasSeed: {hasSeed}, seed: {seed}";
        }

        internal override string profilerTag => "RandomNormal";
    }

    /// <summary>
    /// Represents a `RandomNormalLike` random layer. This generates an output tensor with the same shape as the input tensor with random values in a normal distribution, with given `mean` and `scale`, and an optional `seed` value.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.NoDataDependencyInputs(0)]
    class RandomNormalLike : RandomLayer
    {
        /// <summary>
        /// The mean of the normal distribution used to generate the output.
        /// </summary>
        public float mean;
        /// <summary>
        /// The standard deviation of the normal distribution used to generate the output.
        /// </summary>
        public float scale;

        /// <summary>
        /// Initializes and returns an instance of `RandomNormalLike` random layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer. The layer does not use the values of this tensor in the computation.</param>
        /// <param name="mean">The mean of the normal distribution used to generate the output.</param>
        /// <param name="scale">The standard deviation of the normal distribution used to generate the output.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the layer generates a seed using `System.Random()`.</param>
        public RandomNormalLike(string name, string input, float mean, float scale, int? seed)
            : base(seed)
        {
            this.index = name;
            this.inputs = new[] { input };
            this.mean = mean;
            this.scale = scale;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(index, new PartialTensor(DataType.Float, ctx.GetPartialTensor(inputs[0]).shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.RandomNormal(O, mean, scale, NextSeed);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, mean: {mean}, scale: {scale}, hasSeed: {hasSeed}, seed: {seed}";
        }

        internal override string profilerTag => "RandomNormalLike";
    }

    /// <summary>
    /// Represents a `RandomUniform` random layer. This generates an output tensor of a given shape with random values in a uniform distribution between a given `low` and `high`, from an optional `seed` value.
    /// </summary>
    [Serializable]
    class RandomUniform : RandomLayer
    {
        /// <summary>
        /// The lower end of the interval of the uniform distribution used to generate the output.
        /// </summary>
        public float low;
        /// <summary>
        /// The upper end of the interval of the uniform distribution used to generate the output.
        /// </summary>
        public float high;
        /// <summary>
        /// The shape for the output tensor as an int array.
        /// </summary>
        public int[] shape;

        /// <summary>
        /// Initializes and returns an instance of `RandomUniform` random layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="shape">The shape for the output tensor as an array of ints.</param>
        /// <param name="low">The lower end of the interval of the uniform distribution used to generate the output.</param>
        /// <param name="high">The upper end of the interval of the uniform distribution used to generate the output.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the layer generates a seed using `System.Random()`.</param>
        public RandomUniform(string name, int[] shape, float low, float high, int? seed)
            : base(seed)
        {
            this.index = name;
            this.inputs = Array.Empty<string>();
            this.low = low;
            this.high = high;
            this.shape = shape;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(index, new PartialTensor(DataType.Float, new SymbolicTensorShape(new TensorShape(shape))));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var O = ctx.vars.AllocateTensorAndStore(index, new TensorShape(shape), DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.RandomUniform(O, low, high, NextSeed);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, low: {low}, high: {high}, hasSeed: {hasSeed}, seed: {seed}";
        }

        internal override string profilerTag => "RandomUniform";
    }

    /// <summary>
    /// Represents a `RandomUniformLike` random layer. This generates an output tensor with the same shape as the input tensor random values in a uniform distribution between a given `low` and `high`, from an optional `seed` value.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.NoDataDependencyInputs(0)]
    class RandomUniformLike : RandomLayer
    {
        /// <summary>
        /// The lower end of the interval of the uniform distribution used to generate the output.
        /// </summary>
        public float low;
        /// <summary>
        /// The upper end of the interval of the uniform distribution used to generate the output.
        /// </summary>
        public float high;

        /// <summary>
        /// Initializes and returns an instance of `RandomUniformLike` random layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer. The layer does not use the values of this tensor in the computation.</param>
        /// <param name="low">The lower end of the interval of the uniform distribution used to generate the output.</param>
        /// <param name="high">The upper end of the interval of the uniform distribution used to generate the output.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the layer generates a seed using `System.Random()`.</param>
        public RandomUniformLike(string name, string input, float low, float high, int? seed)
            : base(seed)
        {
            this.index = name;
            this.inputs = new[] { input };
            this.low = low;
            this.high = high;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(index, new PartialTensor(DataType.Float, ctx.GetPartialTensor(inputs[0]).shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]);
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.RandomUniform(O, low, high, NextSeed);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, mean: {low}, scale: {high}, hasSeed: {hasSeed}, seed: {seed}";
        }

        internal override string profilerTag => "RandomUniformLike";
    }

    /// <summary>
    /// Represents a `Bernoulli` random layer. This generates an output tensor with values 0 or 1 from a Bernoulli distribution. The input tensor contains the probabilities used for generating the output values.
    /// </summary>
    [Serializable]
    class Bernoulli : RandomLayer
    {
        /// <summary>
        /// The data type of the output as a `DataType`.
        /// </summary>
        public DataType dataType;

        /// <summary>
        /// Initializes and returns an instance of `Bernoulli` random layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the probabilities tensor of the layer.</param>
        /// <param name="dataType">The data type of the output as a `DataType`.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the layer generates a seed using `System.Random()`.</param>
        public Bernoulli(string name, string input, DataType dataType, int? seed)
            : base(seed)
        {
            this.index = name;
            this.inputs = new[] { input };
            this.dataType = dataType;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            ctx.AddPartialTensor(index, new PartialTensor(dataType, ctx.GetPartialTensor(inputs[0]).shape));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.vars.AllocateTensorAndStore(index, X.shape, dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Bernoulli(X, O, NextSeed);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, dataType: {dataType}, hasSeed: {hasSeed}, seed: {seed}";
        }

        internal override string profilerTag => "Bernoulli";
    }

    /// <summary>
    /// Represents a `Multinomial` random layer. This generates an output tensor with values from a multinomial distribution according to the probabilities given by the input tensor.
    /// </summary>
    [Serializable]
    class Multinomial : RandomLayer
    {
        /// <summary>
        /// The number of times to sample the input.
        /// </summary>
        public int count;

        /// <summary>
        /// Initializes and returns an instance of `Multinomial` random layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the probabilities tensor of the layer.</param>
        /// <param name="count">The number of times to sample the input.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the layer generates a seed using `System.Random()`.</param>
        public Multinomial(string name, string input, int count, int? seed)
            : base(seed)
        {
            this.index = name;
            this.inputs = new[] { input };
            this.count = count;
        }

        /// <inheritdoc/>
        internal override void InferPartial(PartialInferenceContext ctx)
        {
            var shapeX = ctx.GetPartialTensor(inputs[0]).shape;
            ctx.AddPartialTensor(index, new PartialTensor(DataType.Int, new SymbolicTensorShape(shapeX[0], SymbolicTensorDim.Int(count))));
        }

        /// <inheritdoc/>
        public override void Execute(ExecutionContext ctx)
        {
            var X = ctx.vars.GetTensor(inputs[0]) as TensorFloat;
            var O = ctx.vars.AllocateTensorAndStore(index, new TensorShape(X.shape[0], count), DataType.Int, ctx.backend.backendType) as TensorInt;

            var Xtmp = ctx.vars.AllocateTensor(X.shape, DataType.Float, ctx.backend.backendType) as TensorFloat;
            var random = ctx.vars.AllocateTensor(new TensorShape(count), DataType.Float, ctx.backend.backendType) as TensorFloat;

            ctx.backend.RandomUniform(random, 0, 1, seed);
            ctx.backend.Softmax(X, Xtmp, -1);

            ctx.backend.TopP(Xtmp, random, O);


            ctx.vars.Dispose(Xtmp);
            ctx.vars.Dispose(random);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, count: {count}, hasSeed: {hasSeed}, seed: {seed}";
        }

        internal override string profilerTag => "Multinomial";
    }
}
