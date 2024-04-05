// This is auto-generated -- do not modify directly
using System;
using UnityEngine.Assertions;
using Unity.Sentis;
using static Unity.Sentis.BurstTensorData;

namespace Unity.Sentis {

public partial class CPUBackend
{
    /// <inheritdoc/>
    public void Add(TensorFloat A, TensorFloat B, TensorFloat O)
    {
        var job = new AddFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Sub(TensorFloat A, TensorFloat B, TensorFloat O)
    {
        var job = new SubFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Mul(TensorFloat A, TensorFloat B, TensorFloat O)
    {
        var job = new MulFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Div(TensorFloat A, TensorFloat B, TensorFloat O)
    {
        var job = new DivFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Add(TensorInt A, TensorInt B, TensorInt O)
    {
        var job = new AddIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Sub(TensorInt A, TensorInt B, TensorInt O)
    {
        var job = new SubIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Mul(TensorInt A, TensorInt B, TensorInt O)
    {
        var job = new MulIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Div(TensorInt A, TensorInt B, TensorInt O)
    {
        var job = new DivIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Pow(TensorFloat A, TensorFloat B, TensorFloat O)
    {
        var job = new PowFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Greater(TensorFloat A, TensorFloat B, TensorInt O)
    {
        var job = new GreaterFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Greater(TensorInt A, TensorInt B, TensorInt O)
    {
        var job = new GreaterIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void GreaterOrEqual(TensorFloat A, TensorFloat B, TensorInt O)
    {
        var job = new GreaterOrEqualFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void GreaterOrEqual(TensorInt A, TensorInt B, TensorInt O)
    {
        var job = new GreaterOrEqualIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Less(TensorFloat A, TensorFloat B, TensorInt O)
    {
        var job = new LessFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void LessOrEqual(TensorFloat A, TensorFloat B, TensorInt O)
    {
        var job = new LessOrEqualFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Equal(TensorFloat A, TensorFloat B, TensorInt O)
    {
        var job = new EqualFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Less(TensorInt A, TensorInt B, TensorInt O)
    {
        var job = new LessIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void LessOrEqual(TensorInt A, TensorInt B, TensorInt O)
    {
        var job = new LessOrEqualIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Equal(TensorInt A, TensorInt B, TensorInt O)
    {
        var job = new EqualIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Or(TensorInt A, TensorInt B, TensorInt O)
    {
        var job = new OrJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void And(TensorInt A, TensorInt B, TensorInt O)
    {
        var job = new AndJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Xor(TensorInt A, TensorInt B, TensorInt O)
    {
        var job = new XorJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Mod(TensorInt A, TensorInt B, TensorInt O)
    {
        var job = new ModIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void Mod(TensorFloat A, TensorFloat B, TensorFloat O)
    {
        var job = new ModFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void FMod(TensorInt A, TensorInt B, TensorInt O)
    {
        var job = new FModIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }

    /// <inheritdoc/>
    public void FMod(TensorFloat A, TensorFloat B, TensorFloat O)
    {
        var job = new FModFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O), outputLength, 1024);
    }


    /// <inheritdoc/>
    public void Min(TensorFloat[] tensors, TensorFloat O)
    {
        var Otmp = (tensors.Length > 2) ? AllocTensorFloat(O.shape) : null;

        var A = tensors[0];
        var shapeA = A.shape;
        var curO = tensors.Length % 2 == 0 ? O : Otmp;
        for (int t = 1; t < tensors.Length; t++)
        {
            var job = new MinFloatJob();
            var B = tensors[t];

            var outputLength = job.broadcast.Prepare(shapeA, B.shape);
            job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(curO), outputLength, 1024);

            A = curO;
            shapeA = shapeA.Broadcast(B.shape);
            curO = curO == O ? Otmp : O;
        }

        ReleaseTensorFloat(Otmp);
        Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");
    }

    /// <inheritdoc/>
    public void Max(TensorFloat[] tensors, TensorFloat O)
    {
        var Otmp = (tensors.Length > 2) ? AllocTensorFloat(O.shape) : null;

        var A = tensors[0];
        var shapeA = A.shape;
        var curO = tensors.Length % 2 == 0 ? O : Otmp;
        for (int t = 1; t < tensors.Length; t++)
        {
            var job = new MaxFloatJob();
            var B = tensors[t];

            var outputLength = job.broadcast.Prepare(shapeA, B.shape);
            job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(curO), outputLength, 1024);

            A = curO;
            shapeA = shapeA.Broadcast(B.shape);
            curO = curO == O ? Otmp : O;
        }

        ReleaseTensorFloat(Otmp);
        Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");
    }

    /// <inheritdoc/>
    public void Mean(TensorFloat[] tensors, TensorFloat O)
    {
        var Otmp = (tensors.Length > 2) ? AllocTensorFloat(O.shape) : null;

        var A = tensors[0];
        var shapeA = A.shape;
        var curO = tensors.Length % 2 == 0 ? O : Otmp;
        for (int t = 1; t < tensors.Length; t++)
        {
            var job = new MeanFloatJob();
            job.beta = 1.0f / tensors.Length;
            job.alpha = t == 1 ? job.beta : 1.0f;
            var B = tensors[t];

            var outputLength = job.broadcast.Prepare(shapeA, B.shape);
            job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(curO), outputLength, 1024);

            A = curO;
            shapeA = shapeA.Broadcast(B.shape);
            curO = curO == O ? Otmp : O;
        }

        ReleaseTensorFloat(Otmp);
        Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");
    }

    /// <inheritdoc/>
    public void Sum(TensorFloat[] tensors, TensorFloat O)
    {
        var Otmp = (tensors.Length > 2) ? AllocTensorFloat(O.shape) : null;

        var A = tensors[0];
        var shapeA = A.shape;
        var curO = tensors.Length % 2 == 0 ? O : Otmp;
        for (int t = 1; t < tensors.Length; t++)
        {
            var job = new AddFloatJob();
            var B = tensors[t];

            var outputLength = job.broadcast.Prepare(shapeA, B.shape);
            job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(curO), outputLength, 1024);

            A = curO;
            shapeA = shapeA.Broadcast(B.shape);
            curO = curO == O ? Otmp : O;
        }

        ReleaseTensorFloat(Otmp);
        Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");
    }

    /// <inheritdoc/>
    public void Min(TensorInt[] tensors, TensorInt O)
    {
        var Otmp = (tensors.Length > 2) ? AllocTensorInt(O.shape) : null;

        var A = tensors[0];
        var shapeA = A.shape;
        var curO = tensors.Length % 2 == 0 ? O : Otmp;
        for (int t = 1; t < tensors.Length; t++)
        {
            var job = new MinIntJob();
            var B = tensors[t];

            var outputLength = job.broadcast.Prepare(shapeA, B.shape);
            job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(curO), outputLength, 1024);

            A = curO;
            shapeA = shapeA.Broadcast(B.shape);
            curO = curO == O ? Otmp : O;
        }

        ReleaseTensorInt(Otmp);
        Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");
    }

    /// <inheritdoc/>
    public void Max(TensorInt[] tensors, TensorInt O)
    {
        var Otmp = (tensors.Length > 2) ? AllocTensorInt(O.shape) : null;

        var A = tensors[0];
        var shapeA = A.shape;
        var curO = tensors.Length % 2 == 0 ? O : Otmp;
        for (int t = 1; t < tensors.Length; t++)
        {
            var job = new MaxIntJob();
            var B = tensors[t];

            var outputLength = job.broadcast.Prepare(shapeA, B.shape);
            job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(curO), outputLength, 1024);

            A = curO;
            shapeA = shapeA.Broadcast(B.shape);
            curO = curO == O ? Otmp : O;
        }

        ReleaseTensorInt(Otmp);
        Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");
    }

    /// <inheritdoc/>
    public void Abs(TensorFloat X, TensorFloat O)
    {
        var job = new AbsFloatJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Abs(TensorInt X, TensorInt O)
    {
        var job = new AbsIntJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Neg(TensorFloat X, TensorFloat O)
    {
        var job = new NegFloatJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Neg(TensorInt X, TensorInt O)
    {
        var job = new NegIntJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Square(TensorFloat X, TensorFloat O)
    {
        var job = new SquareFloatJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Square(TensorInt X, TensorInt O)
    {
        var job = new SquareIntJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Sign(TensorFloat X, TensorFloat O)
    {
        var job = new SignFloatJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Sign(TensorInt X, TensorInt O)
    {
        var job = new SignIntJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void IsNaN(TensorFloat X, TensorInt O)
    {
        var job = new IsNaNJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Not(TensorInt X, TensorInt O)
    {
        var job = new NotJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Ceil(TensorFloat X, TensorFloat O)
    {
        var job = new CeilJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Floor(TensorFloat X, TensorFloat O)
    {
        var job = new FloorJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Round(TensorFloat X, TensorFloat O)
    {
        var job = new RoundJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Reciprocal(TensorFloat X, TensorFloat O)
    {
        var job = new ReciprocalJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Sqrt(TensorFloat X, TensorFloat O)
    {
        var job = new SqrtJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Exp(TensorFloat X, TensorFloat O)
    {
        var job = new ExpJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Log(TensorFloat X, TensorFloat O)
    {
        var job = new LogJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Acos(TensorFloat X, TensorFloat O)
    {
        var job = new AcosJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Acosh(TensorFloat X, TensorFloat O)
    {
        var job = new AcoshJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Asin(TensorFloat X, TensorFloat O)
    {
        var job = new AsinJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Asinh(TensorFloat X, TensorFloat O)
    {
        var job = new AsinhJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Atan(TensorFloat X, TensorFloat O)
    {
        var job = new AtanJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Atanh(TensorFloat X, TensorFloat O)
    {
        var job = new AtanhJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Cos(TensorFloat X, TensorFloat O)
    {
        var job = new CosJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Cosh(TensorFloat X, TensorFloat O)
    {
        var job = new CoshJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Sin(TensorFloat X, TensorFloat O)
    {
        var job = new SinJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Sinh(TensorFloat X, TensorFloat O)
    {
        var job = new SinhJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Tan(TensorFloat X, TensorFloat O)
    {
        var job = new TanJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Tanh(TensorFloat X, TensorFloat O)
    {
        var job = new TanhJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Relu(TensorFloat X, TensorFloat O)
    {
        var job = new ReluJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Relu6(TensorFloat X, TensorFloat O)
    {
        var job = new Relu6Job();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Softplus(TensorFloat X, TensorFloat O)
    {
        var job = new SoftplusJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Swish(TensorFloat X, TensorFloat O)
    {
        var job = new SwishJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Sigmoid(TensorFloat X, TensorFloat O)
    {
        var job = new SigmoidJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Erf(TensorFloat X, TensorFloat O)
    {
        var job = new ErfJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Softsign(TensorFloat X, TensorFloat O)
    {
        var job = new SoftsignJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void HardSwish(TensorFloat X, TensorFloat O)
    {
        var job = new HardSwishJob();
        job.ScheduleXO(Pin(X), Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void ReduceMin(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMinFloatJob();
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

        {
            var job = new ReduceMinFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceMin(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMinIntJob();
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
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 1024);

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
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        if (isXTempAlloc)
            ReleaseTensorInt(X);
    }

    /// <inheritdoc/>
    public void ReduceMax(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMaxFloatJob();
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

        {
            var job = new ReduceMaxFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceMax(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMaxIntJob();
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
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 1024);

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
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        if (isXTempAlloc)
            ReleaseTensorInt(X);
    }

    /// <inheritdoc/>
    public void ReduceSum(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceSumFloatJob();
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
    public void ReduceSum(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceSumIntJob();
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
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 1024);

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
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        if (isXTempAlloc)
            ReleaseTensorInt(X);
    }

    /// <inheritdoc/>
    public void ReduceSumSquare(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceSumSquareFloatJob();
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
                var job = new ReduceSumSquareFloatJob();
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
            var job = new ReduceSumSquareFloatJob();
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
    public void ReduceSumSquare(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceSumSquareIntJob();
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
                var Otmp = AllocTensorInt(shapeXreduced);
                var job = new ReduceSumSquareIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 1024);

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
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 1024);

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
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        else
        {
            var job = new ReduceSumIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        if (isXTempAlloc)
            ReleaseTensorInt(X);
    }

    /// <inheritdoc/>
    public void ReduceMean(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMeanFloatJob();
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

        {
            var job = new ReduceMeanFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceProd(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceProdFloatJob();
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

        {
            var job = new ReduceProdFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceProd(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceProdIntJob();
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
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 1024);

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
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        if (isXTempAlloc)
            ReleaseTensorInt(X);
    }

    /// <inheritdoc/>
    public void ReduceL1(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceL1FloatJob();
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
                var job = new ReduceL1FloatJob();
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
            var job = new ReduceL1FloatJob();
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
    public void ReduceL1(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceL1IntJob();
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
                var Otmp = AllocTensorInt(shapeXreduced);
                var job = new ReduceL1IntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 1024);

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
                job.ScheduleBatchXO(Pin(X), Pin(Otmp), Otmp.shape.length, 1024);

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
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        else
        {
            var job = new ReduceSumIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        if (isXTempAlloc)
            ReleaseTensorInt(X);
    }

    /// <inheritdoc/>
    public void ReduceL2(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceL2FloatJob();
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
                var job = new ReduceSumSquareFloatJob();
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
            var job = new ReduceL2FloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        else
        {
            var job = new ReduceSqrtFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceLogSum(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceLogSumFloatJob();
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

        {
            var job = new ReduceLogSumFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ReduceLogSumExp(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
    {
        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceLogSumExpFloatJob();
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

        {
            var job = new ReduceLogSumExpFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        if (isXTempAlloc)
            ReleaseTensorFloat(X);
    }

    /// <inheritdoc/>
    public void ArgMax(TensorFloat X, TensorInt O, int axis, bool keepdim, bool selectLastIndex)
    {
        Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

        if (selectLastIndex)
        {
            var job = new ArgMaxFloatLastJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        else
        {
            var job = new ArgMaxFloatFirstJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
    }

    /// <inheritdoc/>
    public void ArgMax(TensorInt X, TensorInt O, int axis, bool keepdim, bool selectLastIndex)
    {
        Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

        if (selectLastIndex)
        {
            var job = new ArgMaxIntLastJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        else
        {
            var job = new ArgMaxIntFirstJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
    }

    /// <inheritdoc/>
    public void ArgMin(TensorFloat X, TensorInt O, int axis, bool keepdim, bool selectLastIndex)
    {
        Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

        if (selectLastIndex)
        {
            var job = new ArgMinFloatLastJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        else
        {
            var job = new ArgMinFloatFirstJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
    }

    /// <inheritdoc/>
    public void ArgMin(TensorInt X, TensorInt O, int axis, bool keepdim, bool selectLastIndex)
    {
        Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

        if (selectLastIndex)
        {
            var job = new ArgMinIntLastJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
        else
        {
            var job = new ArgMinIntFirstJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length, 1024);
        }
    }

    /// <inheritdoc/>
    public void Softmax(TensorFloat X, TensorFloat O, int axis)
    {
        var job = new SoftmaxJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length / job.reduceLength, 1024);
    }

    /// <inheritdoc/>
    public void LogSoftmax(TensorFloat X, TensorFloat O, int axis)
    {
        var job = new LogSoftmaxJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length / job.reduceLength, 1024);
    }

    /// <inheritdoc/>
    public void Hardmax(TensorFloat X, TensorFloat O, int axis)
    {
        var job = new HardmaxJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length / job.reduceLength, 1024);
    }

    /// <inheritdoc/>
    public void CumSum(TensorFloat X, TensorFloat O, int axis, bool reverse, bool exclusive)
    {
        var job = new CumSumFloatJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.reverse = reverse;
        job.exclusive = exclusive;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length / job.reduceLength, 1024);
    }

    /// <inheritdoc/>
    public void CumSum(TensorInt X, TensorInt O, int axis, bool reverse, bool exclusive)
    {
        var job = new CumSumIntJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.reverse = reverse;
        job.exclusive = exclusive;
        job.ScheduleBatchXO(Pin(X), Pin(O), O.shape.length / job.reduceLength, 1024);
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
    public void Range(TensorFloat O, float start, float delta)
    {
        var job = new RangeFloatJob();
        job.start = start;
        job.delta = delta;
        job.ScheduleO(Pin(O), O.shape.length, 1024);
    }

    /// <inheritdoc/>
    public void Range(TensorInt O, int start, int delta)
    {
        var job = new RangeIntJob();
        job.start = start;
        job.delta = delta;
        job.ScheduleO(Pin(O), O.shape.length, 1024);
    }

}

} // namespace Unity.Sentis
