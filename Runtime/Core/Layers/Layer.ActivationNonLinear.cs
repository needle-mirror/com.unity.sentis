using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a `LogSoftmax` activation layer along an axis: f(x, axis) = log(Softmax(x, axis)).
    /// </summary>
    class LogSoftmax : Activation
    {
        public int axis;

        public LogSoftmax(int output, int input, int axis = -1)
            : base(output, input)
        {
            this.axis = axis;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.LogSoftmax(X, O, axis);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        public override string opName => "LogSoftmax";
    }

    /// <summary>
    /// Represents a `Softmax` activation layer along an axis: f(x, axis) = exp(X) / ReduceSum(exp(X), axis).
    /// </summary>
    class Softmax : Activation
    {
        public int axis;

        public Softmax(int output, int input, int axis = -1)
            : base(output, input)
        {
            this.axis = axis;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Softmax(X, O, axis);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        public override string opName => "Softmax";
    }
}
