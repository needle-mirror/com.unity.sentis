// This is auto-generated -- do not modify directly
using System;
using UnityEngine.Assertions;
using Unity.Sentis;
using static Unity.Sentis.CPUTensorData;

namespace Unity.Sentis {

partial class CPUBackend
{
    /// <inheritdoc/>
    public void Add(Tensor<float> A, Tensor<float> B, Tensor<float> O)
    {
        var job = new AddFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Sub(Tensor<float> A, Tensor<float> B, Tensor<float> O)
    {
        var job = new SubFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Mul(Tensor<float> A, Tensor<float> B, Tensor<float> O)
    {
        var job = new MulFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Div(Tensor<float> A, Tensor<float> B, Tensor<float> O)
    {
        var job = new DivFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Add(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new AddIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Sub(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new SubIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Mul(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new MulIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Div(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new DivIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Pow(Tensor<float> A, Tensor<float> B, Tensor<float> O)
    {
        var job = new PowFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Greater(Tensor<float> A, Tensor<float> B, Tensor<int> O)
    {
        var job = new GreaterFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Greater(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new GreaterIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void GreaterOrEqual(Tensor<float> A, Tensor<float> B, Tensor<int> O)
    {
        var job = new GreaterOrEqualFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void GreaterOrEqual(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new GreaterOrEqualIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Less(Tensor<float> A, Tensor<float> B, Tensor<int> O)
    {
        var job = new LessFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void LessOrEqual(Tensor<float> A, Tensor<float> B, Tensor<int> O)
    {
        var job = new LessOrEqualFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Equal(Tensor<float> A, Tensor<float> B, Tensor<int> O)
    {
        var job = new EqualFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Less(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new LessIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void LessOrEqual(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new LessOrEqualIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Equal(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new EqualIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Or(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new OrJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void And(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new AndJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Xor(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new XorJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Mod(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new ModIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Mod(Tensor<float> A, Tensor<float> B, Tensor<float> O)
    {
        var job = new ModFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void FMod(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new FModIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void FMod(Tensor<float> A, Tensor<float> B, Tensor<float> O)
    {
        var job = new FModFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Min(Tensor<float> A, Tensor<float> B, Tensor<float> O)
    {
        var job = new MinFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Max(Tensor<float> A, Tensor<float> B, Tensor<float> O)
    {
        var job = new MaxFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Min(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new MinIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }

    /// <inheritdoc/>
    public void Max(Tensor<int> A, Tensor<int> B, Tensor<int> O)
    {
        var job = new MaxIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 32);
    }


    /// <inheritdoc/>
    public void Abs(Tensor<float> X, Tensor<float> O)
    {
        var job = new AbsFloatJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Abs(Tensor<int> X, Tensor<int> O)
    {
        var job = new AbsIntJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Neg(Tensor<float> X, Tensor<float> O)
    {
        var job = new NegFloatJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Neg(Tensor<int> X, Tensor<int> O)
    {
        var job = new NegIntJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Square(Tensor<float> X, Tensor<float> O)
    {
        var job = new SquareFloatJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Square(Tensor<int> X, Tensor<int> O)
    {
        var job = new SquareIntJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Sign(Tensor<float> X, Tensor<float> O)
    {
        var job = new SignFloatJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Sign(Tensor<int> X, Tensor<int> O)
    {
        var job = new SignIntJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void IsNaN(Tensor<float> X, Tensor<int> O)
    {
        var job = new IsNaNJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Not(Tensor<int> X, Tensor<int> O)
    {
        var job = new NotJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Ceil(Tensor<float> X, Tensor<float> O)
    {
        var job = new CeilJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Floor(Tensor<float> X, Tensor<float> O)
    {
        var job = new FloorJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Round(Tensor<float> X, Tensor<float> O)
    {
        var job = new RoundJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Reciprocal(Tensor<float> X, Tensor<float> O)
    {
        var job = new ReciprocalJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Sqrt(Tensor<float> X, Tensor<float> O)
    {
        var job = new SqrtJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Exp(Tensor<float> X, Tensor<float> O)
    {
        var job = new ExpJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Log(Tensor<float> X, Tensor<float> O)
    {
        var job = new LogJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Acos(Tensor<float> X, Tensor<float> O)
    {
        var job = new AcosJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Acosh(Tensor<float> X, Tensor<float> O)
    {
        var job = new AcoshJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Asin(Tensor<float> X, Tensor<float> O)
    {
        var job = new AsinJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Asinh(Tensor<float> X, Tensor<float> O)
    {
        var job = new AsinhJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Atan(Tensor<float> X, Tensor<float> O)
    {
        var job = new AtanJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Atanh(Tensor<float> X, Tensor<float> O)
    {
        var job = new AtanhJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Cos(Tensor<float> X, Tensor<float> O)
    {
        var job = new CosJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Cosh(Tensor<float> X, Tensor<float> O)
    {
        var job = new CoshJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Sin(Tensor<float> X, Tensor<float> O)
    {
        var job = new SinJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Sinh(Tensor<float> X, Tensor<float> O)
    {
        var job = new SinhJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Tan(Tensor<float> X, Tensor<float> O)
    {
        var job = new TanJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Tanh(Tensor<float> X, Tensor<float> O)
    {
        var job = new TanhJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Relu(Tensor<float> X, Tensor<float> O)
    {
        var job = new ReluJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Relu6(Tensor<float> X, Tensor<float> O)
    {
        var job = new Relu6Job();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Softplus(Tensor<float> X, Tensor<float> O)
    {
        var job = new SoftplusJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Swish(Tensor<float> X, Tensor<float> O)
    {
        var job = new SwishJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Sigmoid(Tensor<float> X, Tensor<float> O)
    {
        var job = new SigmoidJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Erf(Tensor<float> X, Tensor<float> O)
    {
        var job = new ErfJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void Softsign(Tensor<float> X, Tensor<float> O)
    {
        var job = new SoftsignJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void HardSwish(Tensor<float> X, Tensor<float> O)
    {
        var job = new HardSwishJob();
        job.length = O.shape.length;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
    }

    /// <inheritdoc/>
    public void ReduceMin(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMinFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceMinFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceMinFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceMin(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMinIntJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else
            {
                var Otmp = AllocTensorInt(shapeXreduced);
                var job = new ReduceMinIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorInt(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceMinIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorInt(X);
    }

    /// <inheritdoc/>
    public void ReduceMax(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMaxFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceMaxFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceMaxFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceMax(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMaxIntJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else
            {
                var Otmp = AllocTensorInt(shapeXreduced);
                var job = new ReduceMaxIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorInt(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceMaxIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorInt(X);
    }

    /// <inheritdoc/>
    public void ReduceSum(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceSumFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceSumFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceSumFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceSum(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceSumIntJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else
            {
                var Otmp = AllocTensorInt(shapeXreduced);
                var job = new ReduceSumIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorInt(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceSumIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorInt(X);
    }

    /// <inheritdoc/>
    public void ReduceSumSquare(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceSumSquareFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isInitial = true;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else if (isInitial)
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceSumSquareFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isInitial = false;
                isXTempAlloc = true;
            }
            else
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceSumFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        if (isInitial)
        {
            var job = new ReduceSumSquareFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        else
        {
            var job = new ReduceSumFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceSumSquare(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceSumSquareIntJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isInitial = true;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else if (isInitial)
            {
                var Otmp = AllocTensorInt(shapeXreduced);
                var job = new ReduceSumSquareIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorInt(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isInitial = false;
                isXTempAlloc = true;
            }
            else
            {
                var Otmp = AllocTensorInt(shapeXreduced);
                var job = new ReduceSumIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorInt(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        if (isInitial)
        {
            var job = new ReduceSumSquareIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        else
        {
            var job = new ReduceSumIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorInt(X);
    }

    /// <inheritdoc/>
    public void ReduceMeanSquare(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMeanSquareFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isInitial = true;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else if (isInitial)
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceMeanSquareFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 1024);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isInitial = false;
                isXTempAlloc = true;
            }
            else
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceSumFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 1024);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        if (isInitial)
        {
            var job = new ReduceMeanSquareFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        else
        {
            var job = new ReduceSumFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceMean(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMeanFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceMeanFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceMeanFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceProd(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceProdFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceProdFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceProdFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceProd(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceProdIntJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else
            {
                var Otmp = AllocTensorInt(shapeXreduced);
                var job = new ReduceProdIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorInt(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceProdIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorInt(X);
    }

    /// <inheritdoc/>
    public void ReduceL1(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceL1FloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isInitial = true;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else if (isInitial)
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceL1FloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isInitial = false;
                isXTempAlloc = true;
            }
            else
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceSumFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        if (isInitial)
        {
            var job = new ReduceL1FloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        else
        {
            var job = new ReduceSumFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceL1(Tensor<int> X, Tensor<int> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceL1IntJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isInitial = true;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else if (isInitial)
            {
                var Otmp = AllocTensorInt(shapeXreduced);
                var job = new ReduceL1IntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorInt(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isInitial = false;
                isXTempAlloc = true;
            }
            else
            {
                var Otmp = AllocTensorInt(shapeXreduced);
                var job = new ReduceSumIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorInt(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        if (isInitial)
        {
            var job = new ReduceL1IntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        else
        {
            var job = new ReduceSumIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorInt(X);
    }

    /// <inheritdoc/>
    public void ReduceL2(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceL2FloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isInitial = true;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else if (isInitial)
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceSumSquareFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isInitial = false;
                isXTempAlloc = true;
            }
            else
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceSumFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        if (isInitial)
        {
            var job = new ReduceL2FloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        else
        {
            var job = new ReduceSqrtFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceLogSum(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceLogSumFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceSumFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceLogSumFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceLogSumExp(Tensor<float> X, Tensor<float> O, ReadOnlySpan<int> axes)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceLogSumExpFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
            return;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape;
        shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isXTempAlloc = false;

        for (int i = 1; i < axes.Length; i++)
        {
            axis = X.shape.Axis(axes[i]);
            dimX = X.shape[axis];
            Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

            if ((axis == (prevAxis + 1)))
            {
                innerLength /= dimX;
                reduceLength *= dimX;
            }
            else
            {
                var Otmp = AllocTensorFloat(shapeXreduced);
                var job = new ReduceLogSumExpFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 32);

                if (isXTempAlloc)
                    ReleaseTensorFloat(X);
                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isXTempAlloc = true;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceLogSumExpFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ArgMax(Tensor<float> X, Tensor<int> O, int axis, bool selectLastIndex)
    {
        Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

        if (selectLastIndex)
        {
            var job = new ArgMaxFloatLastJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        else
        {
            var job = new ArgMaxFloatFirstJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
    }

    /// <inheritdoc/>
    public void ArgMax(Tensor<int> X, Tensor<int> O, int axis, bool selectLastIndex)
    {
        Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

        if (selectLastIndex)
        {
            var job = new ArgMaxIntLastJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        else
        {
            var job = new ArgMaxIntFirstJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
    }

    /// <inheritdoc/>
    public void ArgMin(Tensor<float> X, Tensor<int> O, int axis, bool selectLastIndex)
    {
        Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

        if (selectLastIndex)
        {
            var job = new ArgMinFloatLastJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        else
        {
            var job = new ArgMinFloatFirstJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
    }

    /// <inheritdoc/>
    public void ArgMin(Tensor<int> X, Tensor<int> O, int axis, bool selectLastIndex)
    {
        Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

        if (selectLastIndex)
        {
            var job = new ArgMinIntLastJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
        else
        {
            var job = new ArgMinIntFirstJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 32);
        }
    }

    /// <inheritdoc/>
    public void Softmax(Tensor<float> X, Tensor<float> O, int axis)
    {
        var job = new SoftmaxJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length / job.reduceLength, 32);
    }

    /// <inheritdoc/>
    public void LogSoftmax(Tensor<float> X, Tensor<float> O, int axis)
    {
        var job = new LogSoftmaxJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length / job.reduceLength, 32);
    }

    /// <inheritdoc/>
    public void Hardmax(Tensor<float> X, Tensor<float> O, int axis)
    {
        var job = new HardmaxJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length / job.reduceLength, 32);
    }

    /// <inheritdoc/>
    public void CumSum(Tensor<float> X, Tensor<float> O, int axis, bool reverse, bool exclusive)
    {
        var job = new CumSumFloatJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.reverse = reverse;
        job.exclusive = exclusive;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length / job.reduceLength, 32);
    }

    /// <inheritdoc/>
    public void CumSum(Tensor<int> X, Tensor<int> O, int axis, bool reverse, bool exclusive)
    {
        var job = new CumSumIntJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.reverse = reverse;
        job.exclusive = exclusive;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length / job.reduceLength, 32);
    }

    /// <inheritdoc/>
    public void Tril(Tensor X, Tensor O, int k)
    {
        var job = new TrilJob();
        job.widthX = X.shape[-1];
        job.heightX = X.shape[-2];
        job.diagonalK = k;
        job.ScheduleXO(Pin(X), Pin(O), X.shape.Length(0, -1), 32);
    }

    /// <inheritdoc/>
    public void Triu(Tensor X, Tensor O, int k)
    {
        var job = new TriuJob();
        job.widthX = X.shape[-1];
        job.heightX = X.shape[-2];
        job.diagonalK = k;
        job.ScheduleXO(Pin(X), Pin(O), X.shape.Length(0, -1), 32);
    }

    /// <inheritdoc/>
    public void Range(Tensor<float> O, float start, float delta)
    {
        var job = new RangeFloatJob();
        job.alpha = start;
        job.beta = delta;
        job.length = O.shape.length;
        job.ScheduleBatchO(Pin(O), O.shape.length, 32);
    }
    /// <inheritdoc/>
    public void Range(Tensor<int> O, int start, int delta)
    {
        var job = new RangeIntJob();
        job.alphai = start;
        job.betai = delta;
        job.length = O.shape.length;
        job.ScheduleBatchO(Pin(O), O.shape.length, 32);
    }
}

}
