// This is auto-generated -- do not modify directly
using System;
using UnityEngine.Assertions;
using Unity.Sentis;
using static Unity.Sentis.BurstTensorData;

namespace Unity.Sentis {

public partial class CPUBackend
{
    /// <inheritdoc/>
    public virtual TensorFloat Add(TensorFloat A, TensorFloat B)
    {
        var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new AddFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Sub(TensorFloat A, TensorFloat B)
    {
        var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new SubFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Mul(TensorFloat A, TensorFloat B)
    {
        var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new MulFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Div(TensorFloat A, TensorFloat B)
    {
        var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new DivFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Add(TensorInt A, TensorInt B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new AddIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Sub(TensorInt A, TensorInt B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new SubIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Mul(TensorInt A, TensorInt B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new MulIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Div(TensorInt A, TensorInt B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new DivIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Pow(TensorFloat A, TensorFloat B)
    {
        var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new PowFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Greater(TensorFloat A, TensorFloat B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new GreaterFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Greater(TensorInt A, TensorInt B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new GreaterIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt GreaterOrEqual(TensorFloat A, TensorFloat B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new GreaterOrEqualFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt GreaterOrEqual(TensorInt A, TensorInt B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new GreaterOrEqualIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Less(TensorFloat A, TensorFloat B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new LessFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt LessOrEqual(TensorFloat A, TensorFloat B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new LessOrEqualFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Equal(TensorFloat A, TensorFloat B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new EqualFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Less(TensorInt A, TensorInt B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new LessIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt LessOrEqual(TensorInt A, TensorInt B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new LessOrEqualIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Equal(TensorInt A, TensorInt B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new EqualIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Or(TensorInt A, TensorInt B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new OrJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt And(TensorInt A, TensorInt B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new AndJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Xor(TensorInt A, TensorInt B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new XorJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Mod(TensorInt A, TensorInt B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new ModIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt FMod(TensorInt A, TensorInt B)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new FModIntJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat FMod(TensorFloat A, TensorFloat B)
    {
        var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(A, B));
        if (O.shape.HasZeroDims())
            return O;

        var job = new FModFloatJob();
        var outputLength = job.broadcast.Prepare(A.shape, B.shape);
        job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(O, clearOnInit: false), outputLength, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Min(TensorFloat[] tensors)
    {
        var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(tensors));
        if (O.shape.HasZeroDims())
            return O;

        var Otmp = (tensors.Length > 2) ? NewTempTensorFloat(O.shape) : null;

        var A = tensors[0];
        var shapeA = A.shape;
        var curO = tensors.Length % 2 == 0 ? O : Otmp;
        for (int t = 1; t < tensors.Length; t++)
        {
            var job = new MinFloatJob();
            var B = tensors[t];

            var outputLength = job.broadcast.Prepare(shapeA, B.shape);
            job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(curO, clearOnInit: false), outputLength, 1024);

            A = curO;
            shapeA = shapeA.Broadcast(B.shape);
            curO = curO == O ? Otmp : O;
        }

        Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Max(TensorFloat[] tensors)
    {
        var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(tensors));
        if (O.shape.HasZeroDims())
            return O;

        var Otmp = (tensors.Length > 2) ? NewTempTensorFloat(O.shape) : null;

        var A = tensors[0];
        var shapeA = A.shape;
        var curO = tensors.Length % 2 == 0 ? O : Otmp;
        for (int t = 1; t < tensors.Length; t++)
        {
            var job = new MaxFloatJob();
            var B = tensors[t];

            var outputLength = job.broadcast.Prepare(shapeA, B.shape);
            job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(curO, clearOnInit: false), outputLength, 1024);

            A = curO;
            shapeA = shapeA.Broadcast(B.shape);
            curO = curO == O ? Otmp : O;
        }

        Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Mean(TensorFloat[] tensors)
    {
        var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(tensors));
        if (O.shape.HasZeroDims())
            return O;

        var Otmp = (tensors.Length > 2) ? NewTempTensorFloat(O.shape) : null;

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
            job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(curO, clearOnInit: false), outputLength, 1024);

            A = curO;
            shapeA = shapeA.Broadcast(B.shape);
            curO = curO == O ? Otmp : O;
        }

        Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Sum(TensorFloat[] tensors)
    {
        var O = NewOutputTensorFloat(TensorShapeHelper.BroadcastShape(tensors));
        if (O.shape.HasZeroDims())
            return O;

        var Otmp = (tensors.Length > 2) ? NewTempTensorFloat(O.shape) : null;

        var A = tensors[0];
        var shapeA = A.shape;
        var curO = tensors.Length % 2 == 0 ? O : Otmp;
        for (int t = 1; t < tensors.Length; t++)
        {
            var job = new AddFloatJob();
            var B = tensors[t];

            var outputLength = job.broadcast.Prepare(shapeA, B.shape);
            job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(curO, clearOnInit: false), outputLength, 1024);

            A = curO;
            shapeA = shapeA.Broadcast(B.shape);
            curO = curO == O ? Otmp : O;
        }

        Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Min(TensorInt[] tensors)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(tensors));
        if (O.shape.HasZeroDims())
            return O;

        var Otmp = (tensors.Length > 2) ? NewTempTensorInt(O.shape) : null;

        var A = tensors[0];
        var shapeA = A.shape;
        var curO = tensors.Length % 2 == 0 ? O : Otmp;
        for (int t = 1; t < tensors.Length; t++)
        {
            var job = new MinIntJob();
            var B = tensors[t];

            var outputLength = job.broadcast.Prepare(shapeA, B.shape);
            job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(curO, clearOnInit: false), outputLength, 1024);

            A = curO;
            shapeA = shapeA.Broadcast(B.shape);
            curO = curO == O ? Otmp : O;
        }

        Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Max(TensorInt[] tensors)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(tensors));
        if (O.shape.HasZeroDims())
            return O;

        var Otmp = (tensors.Length > 2) ? NewTempTensorInt(O.shape) : null;

        var A = tensors[0];
        var shapeA = A.shape;
        var curO = tensors.Length % 2 == 0 ? O : Otmp;
        for (int t = 1; t < tensors.Length; t++)
        {
            var job = new MaxIntJob();
            var B = tensors[t];

            var outputLength = job.broadcast.Prepare(shapeA, B.shape);
            job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(curO, clearOnInit: false), outputLength, 1024);

            A = curO;
            shapeA = shapeA.Broadcast(B.shape);
            curO = curO == O ? Otmp : O;
        }

        Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Sum(TensorInt[] tensors)
    {
        var O = NewOutputTensorInt(TensorShapeHelper.BroadcastShape(tensors));
        if (O.shape.HasZeroDims())
            return O;

        var Otmp = (tensors.Length > 2) ? NewTempTensorInt(O.shape) : null;

        var A = tensors[0];
        var shapeA = A.shape;
        var curO = tensors.Length % 2 == 0 ? O : Otmp;
        for (int t = 1; t < tensors.Length; t++)
        {
            var job = new AddIntJob();
            var B = tensors[t];

            var outputLength = job.broadcast.Prepare(shapeA, B.shape);
            job.ScheduleBatchXBO(Pin(A), Pin(B), Pin(curO, clearOnInit: false), outputLength, 1024);

            A = curO;
            shapeA = shapeA.Broadcast(B.shape);
            curO = curO == O ? Otmp : O;
        }

        Logger.AssertIsTrue(curO != O, "Output tensor should have been the persistent one.");

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Abs(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new AbsFloatJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Abs(TensorInt X)
    {
        var O = NewOutputTensorInt(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new AbsIntJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Neg(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new NegFloatJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Neg(TensorInt X)
    {
        var O = NewOutputTensorInt(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new NegIntJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Sign(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new SignFloatJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Sign(TensorInt X)
    {
        var O = NewOutputTensorInt(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new SignIntJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt IsNaN(TensorFloat X)
    {
        var O = NewOutputTensorInt(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new IsNaNJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Cast(TensorInt X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new CastToFloatJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Cast(TensorFloat X)
    {
        var O = NewOutputTensorInt(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new CastToIntJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Not(TensorInt X)
    {
        var O = NewOutputTensorInt(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new NotJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Ceil(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new CeilJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Floor(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new FloorJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Round(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new RoundJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Reciprocal(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new ReciprocalJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Sqrt(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new SqrtJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Square(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new SquareJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Exp(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new ExpJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Log(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new LogJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Acos(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new AcosJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Acosh(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new AcoshJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Asin(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new AsinJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Asinh(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new AsinhJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Atan(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new AtanJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Atanh(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new AtanhJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Cos(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new CosJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Cosh(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new CoshJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Sin(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new SinJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Sinh(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new SinhJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Tan(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new TanJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Tanh(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new TanhJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Relu(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new ReluJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Relu6(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new Relu6Job();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Softplus(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new SoftplusJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Swish(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new SwishJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Sigmoid(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new SigmoidJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Erf(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new ErfJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Softsign(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new SoftsignJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat HardSwish(TensorFloat X)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new HardSwishJob();
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ReduceMin(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorFloat(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMinFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;

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
                var Otmp = NewTempTensorFloat(shapeXreduced);
                var job = new ReduceMinFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceMinFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt ReduceMin(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorInt(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMinIntJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;

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
                var Otmp = NewTempTensorInt(shapeXreduced);
                var job = new ReduceMinIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceMinIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ReduceMax(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorFloat(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMaxFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;

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
                var Otmp = NewTempTensorFloat(shapeXreduced);
                var job = new ReduceMaxFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceMaxFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt ReduceMax(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorInt(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMaxIntJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;

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
                var Otmp = NewTempTensorInt(shapeXreduced);
                var job = new ReduceMaxIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceMaxIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ReduceSum(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorFloat(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceSumFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;

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
                var Otmp = NewTempTensorFloat(shapeXreduced);
                var job = new ReduceSumFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceSumFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt ReduceSum(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorInt(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceSumIntJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;

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
                var Otmp = NewTempTensorInt(shapeXreduced);
                var job = new ReduceSumIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceSumIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ReduceSumSquare(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorFloat(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceSumSquareFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isInitial = true;

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
                var Otmp = NewTempTensorFloat(shapeXreduced);
                var job = new ReduceSumSquareFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isInitial = false;
            }
            else
            {
                var Otmp = NewTempTensorFloat(shapeXreduced);
                var job = new ReduceSumFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        if (isInitial)
        {
            var job = new ReduceSumSquareFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }
        else
        {
            var job = new ReduceSumFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt ReduceSumSquare(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorInt(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceSumSquareIntJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isInitial = true;

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
                var Otmp = NewTempTensorInt(shapeXreduced);
                var job = new ReduceSumSquareIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isInitial = false;
            }
            else
            {
                var Otmp = NewTempTensorInt(shapeXreduced);
                var job = new ReduceSumIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        if (isInitial)
        {
            var job = new ReduceSumSquareIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }
        else
        {
            var job = new ReduceSumIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ReduceMean(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorFloat(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceMeanFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;

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
                var Otmp = NewTempTensorFloat(shapeXreduced);
                var job = new ReduceMeanFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceMeanFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ReduceProd(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorFloat(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceProdFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;

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
                var Otmp = NewTempTensorFloat(shapeXreduced);
                var job = new ReduceProdFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceProdFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt ReduceProd(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorInt(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceProdIntJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;

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
                var Otmp = NewTempTensorInt(shapeXreduced);
                var job = new ReduceProdIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceProdIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ReduceL1(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorFloat(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceL1FloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isInitial = true;

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
                var Otmp = NewTempTensorFloat(shapeXreduced);
                var job = new ReduceL1FloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isInitial = false;
            }
            else
            {
                var Otmp = NewTempTensorFloat(shapeXreduced);
                var job = new ReduceSumFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        if (isInitial)
        {
            var job = new ReduceL1FloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }
        else
        {
            var job = new ReduceSumFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt ReduceL1(TensorInt X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorInt(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceL1IntJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isInitial = true;

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
                var Otmp = NewTempTensorInt(shapeXreduced);
                var job = new ReduceL1IntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isInitial = false;
            }
            else
            {
                var Otmp = NewTempTensorInt(shapeXreduced);
                var job = new ReduceSumIntJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        if (isInitial)
        {
            var job = new ReduceL1IntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }
        else
        {
            var job = new ReduceSumIntJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ReduceL2(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorFloat(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceL2FloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;
        bool isInitial = true;

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
                var Otmp = NewTempTensorFloat(shapeXreduced);
                var job = new ReduceSumSquareFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
                isInitial = false;
            }
            else
            {
                var Otmp = NewTempTensorFloat(shapeXreduced);
                var job = new ReduceSumFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        if (isInitial)
        {
            var job = new ReduceL2FloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }
        else
        {
            var job = new ReduceSqrtFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ReduceLogSum(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorFloat(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceLogSumFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;

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
                var Otmp = NewTempTensorFloat(shapeXreduced);
                var job = new ReduceSumFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceLogSumFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat ReduceLogSumExp(TensorFloat X, ReadOnlySpan<int> axes, bool keepdim)
    {
        TensorShape Oshape = X.shape.Reduce(axes, keepdim);
        var O = NewOutputTensorFloat(Oshape);
        if (Oshape.HasZeroDims())
            return O;

        if (axes == null || axes.Length == 0)
        {
            var job = new ReduceLogSumExpFloatJob();
            job.innerLength = 1;
            job.reduceLength = X.shape.length;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
            return O;
        }

        // Accumulate reduce axis until non contiguity
        // X: (2 3 4 5 6), reduce 0,1,4
        // reduce 0 + 1 will result in a fused reduce on axis 2*3
        // 4 breaks contiguity, thus we perform the previous reduce and start procedure over

        int axis = X.shape.Axis(axes[0]);
        int innerLength = X.shape.Strides(axis);
        int dimX = X.shape[axis];
        int reduceLength = dimX;
        TensorShape shapeXreduced = X.shape; shapeXreduced[axis] = 1;
        int prevAxis = axis;

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
                var Otmp = NewTempTensorFloat(shapeXreduced);
                var job = new ReduceLogSumExpFloatJob();
                job.innerLength = innerLength;
                job.reduceLength = reduceLength;
                job.ScheduleBatchXO(Pin(X), Pin(Otmp, clearOnInit: false), Otmp.shape.length, 1024);

                X = Otmp;
                innerLength = X.shape.Strides(axis);
                reduceLength = dimX;
            }

            shapeXreduced[axis] = 1;
            prevAxis = axis;
        }

        {
            var job = new ReduceLogSumExpFloatJob();
            job.innerLength = innerLength;
            job.reduceLength = reduceLength;
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt ArgMax(TensorFloat X, int axis, bool keepdim, bool selectLastIndex = false)
    {
        var O = NewOutputTensorInt(X.shape.Reduce(axis, keepdim));
        if (O.shape.HasZeroDims())
            return O;
        Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

        if (selectLastIndex)
        {
            var job = new ArgMaxFloatLastJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }
        else
        {
            var job = new ArgMaxFloatFirstJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt ArgMax(TensorInt X, int axis, bool keepdim, bool selectLastIndex = false)
    {
        var O = NewOutputTensorInt(X.shape.Reduce(axis, keepdim));
        if (O.shape.HasZeroDims())
            return O;
        Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

        if (selectLastIndex)
        {
            var job = new ArgMaxIntLastJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }
        else
        {
            var job = new ArgMaxIntFirstJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt ArgMin(TensorFloat X, int axis, bool keepdim, bool selectLastIndex = false)
    {
        var O = NewOutputTensorInt(X.shape.Reduce(axis, keepdim));
        if (O.shape.HasZeroDims())
            return O;
        Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

        if (selectLastIndex)
        {
            var job = new ArgMinFloatLastJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }
        else
        {
            var job = new ArgMinFloatFirstJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt ArgMin(TensorInt X, int axis, bool keepdim, bool selectLastIndex = false)
    {
        var O = NewOutputTensorInt(X.shape.Reduce(axis, keepdim));
        if (O.shape.HasZeroDims())
            return O;
        Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

        if (selectLastIndex)
        {
            var job = new ArgMinIntLastJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }
        else
        {
            var job = new ArgMinIntFirstJob();
            job.innerLength = X.shape.Strides(axis);
            job.reduceLength = X.shape[axis];
            job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length, 1024);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Softmax(TensorFloat X, int axis)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new SoftmaxJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length / job.reduceLength, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat LogSoftmax(TensorFloat X, int axis)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new LogSoftmaxJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length / job.reduceLength, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Hardmax(TensorFloat X, int axis)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new HardmaxJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length / job.reduceLength, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat CumSum(TensorFloat X, int axis, bool reverse, bool exclusive)
    {
        var O = NewOutputTensorFloat(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new CumSumFloatJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.reverse = reverse;
        job.exclusive = exclusive;
        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length / job.reduceLength, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt CumSum(TensorInt X, int axis, bool reverse, bool exclusive)
    {
        var O = NewOutputTensorInt(X.shape);
        if (O.shape.HasZeroDims())
            return O;

        var job = new CumSumIntJob();
        job.innerLength = X.shape.Strides(axis);
        job.reduceLength = X.shape[axis];
        job.reverse = reverse;
        job.exclusive = exclusive;
        job.ScheduleBatchXO(Pin(X), Pin(O, clearOnInit: false), O.shape.length / job.reduceLength, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Tril(Tensor X, int k)
    {
        var O = NewOutputTensor(X.shape, X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var job = new TrilJob();
        job.widthX  = X.shape[-1];
        job.heightX = X.shape[-2];
        job.diagonalK = k;
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), X.shape.Length(0, -1), 32);

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Triu(Tensor X, int k)
    {
        var O = NewOutputTensor(X.shape, X.dataType);
        if (O.shape.HasZeroDims())
            return O;

        var job = new TriuJob();
        job.widthX  = X.shape[-1];
        job.heightX = X.shape[-2];
        job.diagonalK = k;
        job.ScheduleXO(Pin(X), Pin(O, clearOnInit: false), X.shape.Length(0, -1), 32);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorFloat Range(float start, float limit, float delta)
    {
        var O = NewOutputTensorFloat(ShapeInference.Range(start, limit, delta));
        if (O.shape.HasZeroDims())
            return O;

        var job = new RangeFloatJob();
        job.start = start;
        job.delta = delta;
        job.ScheduleO(Pin(O), O.shape.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public virtual TensorInt Range(int start, int limit, int delta)
    {
        var O = NewOutputTensorInt(ShapeInference.Range(start, limit, delta));
        if (O.shape.HasZeroDims())
            return O;

        var job = new RangeIntJob();
        job.start = start;
        job.delta = delta;
        job.ScheduleO(Pin(O), O.shape.length, 1024);

        return O;
    }

}

} // namespace Unity.Sentis
