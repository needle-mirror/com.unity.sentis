using System;
using System.Runtime.Serialization;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents the abstract base class for layers which generate random values in the output tensor.
    /// </summary>
    [Serializable]
    [NonDeterministicOutput]
    public abstract class RandomLayer : Layer
    {
        /// <summary>
        /// Whether the layer has a provided seed value for the random number generator.
        /// </summary>
        public bool hasSeed;
        /// <summary>
        /// The seed value for the random number generator. The layer does not use this value if `hasSeed` is `false`.
        /// </summary>
        public float seed;
        [NonSerialized]
        Random m_Random;

        protected float NextSeed => m_Random.NextFloatSeed();

        [OnDeserialized]
        internal void OnDeserializedMethod(StreamingContext context)
        {
            ResetSeed();
        }

        /// <summary>
        /// Resets the state of the random number generator to its initial state.
        ///
        /// If `hasSeed` is `false` then the random number generator is seeded by System.Random().
        /// </summary>
        public void ResetSeed()
        {
            m_Random = hasSeed ? new Random(seed) : new Random();
        }

        protected RandomLayer(float? seed)
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
    public class RandomNormal : RandomLayer
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
        /// The shape for the output tensor as a `TensorShape`.
        /// </summary>
        public TensorShape shape;

        /// <summary>
        /// Initializes and returns an instance of `RandomNormal` random layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="shape">The shape for the output tensor as an array of ints.</param>
        /// <param name="mean">The mean of the normal distribution used to generate the output.</param>
        /// <param name="scale">The standard deviation of the normal distribution used to generate the output.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the layer generates a seed using `System.Random()`.</param>
        public RandomNormal(string name, int[] shape, float mean, float scale, float? seed)
            : base(seed)
        {
            this.name = name;
            this.inputs = Array.Empty<string>();
            this.mean = mean;
            this.scale = scale;
            this.shape = new TensorShape(shape);
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return new SymbolicTensorShape(shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.RandomNormal(shape, mean, scale, NextSeed);
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
    public class RandomNormalLike : RandomLayer
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
        public RandomNormalLike(string name, string input, float mean, float scale, float? seed)
            : base(seed)
        {
            this.name = name;
            this.inputs = new[] { input };
            this.mean = mean;
            this.scale = scale;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return inputShapes[0];
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.RandomNormal(inputTensors[0].shape, mean, scale, NextSeed);
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
    public class RandomUniform : RandomLayer
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
        /// The shape for the output tensor as a `TensorShape`.
        /// </summary>
        public TensorShape shape;

        /// <summary>
        /// Initializes and returns an instance of `RandomUniform` random layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="shape">The shape for the output tensor as an array of ints.</param>
        /// <param name="low">The lower end of the interval of the uniform distribution used to generate the output.</param>
        /// <param name="high">The upper end of the interval of the uniform distribution used to generate the output.</param>
        /// <param name="seed">The optional seed to use for the random number generation. If this is `null` the layer generates a seed using `System.Random()`.</param>
        public RandomUniform(string name, int[] shape, float low, float high, float? seed)
            : base(seed)
        {
            this.name = name;
            this.inputs = Array.Empty<string>();
            this.low = low;
            this.high = high;
            this.shape = new TensorShape(shape);
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return new SymbolicTensorShape(shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.RandomUniform(shape, low, high, NextSeed);
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
    public class RandomUniformLike : RandomLayer
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
        public RandomUniformLike(string name, string input, float low, float high, float? seed)
            : base(seed)
        {
            this.name = name;
            this.inputs = new[] { input };
            this.low = low;
            this.high = high;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return inputShapes[0];
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.RandomUniform(inputTensors[0].shape, low, high, NextSeed);
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
    public class Bernoulli : RandomLayer
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
        public Bernoulli(string name, string input, DataType dataType, float? seed)
            : base(seed)
        {
            this.name = name;
            this.inputs = new[] { input };
            this.dataType = dataType;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return inputShapes[0];
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.Bernoulli(inputTensors[0] as TensorFloat, dataType, NextSeed);
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
    public class Multinomial : RandomLayer
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
        public Multinomial(string name, string input, int count, float? seed)
            : base(seed)
        {
            this.name = name;
            this.inputs = new[] { input };
            this.count = count;
        }

        internal override SymbolicTensorShape InferOutputShape(SymbolicTensorShape[] inputShapes, ShapeInferenceContext ctx)
        {
            return SymbolicInference.Multinomial(inputShapes[0], count);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            return ctx.ops.Multinomial(inputTensors[0] as TensorFloat, count, NextSeed);
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, count: {count}, hasSeed: {hasSeed}, seed: {seed}";
        }

        internal override string profilerTag => "Multinomial";
    }
}
