using System;
using System.Linq;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents an input definition of a model at build time. This holds a data type and shape.
    /// </summary>
    public readonly struct InputDef
    {
        /// <summary>
        /// The data type of the input.
        /// </summary>
        public DataType DataType { get; }
        /// <summary>
        /// The shape of the input.
        /// </summary>
        public SymbolicTensorShape Shape { get; }

        /// <summary>
        /// Initializes and returns an instance of `InputDef` with data type and tensor shape.
        /// </summary>
        /// <param name="dataType">The data type of the input.</param>
        /// <param name="shape">The shape of the input.</param>
        public InputDef(DataType dataType, TensorShape shape)
        {
            DataType = dataType;
            Shape = new SymbolicTensorShape(shape);
        }

        /// <summary>
        /// Initializes and returns an instance of `InputDef` with data type and symbolic tensor shape.
        /// </summary>
        /// <param name="dataType">The data type of the input.</param>
        /// <param name="shape">The shape of the input.</param>
        public InputDef(DataType dataType, SymbolicTensorShape shape)
        {
            DataType = dataType;
            Shape = shape;
        }

        /// <summary>
        /// Initializes and returns an instance of `InputDef` implicitly with data type and symbolic tensor shape from model input.
        /// </summary>
        /// <param name="input">The model input to use for data type and shape.</param>
        /// <returns>The input def.</returns>
        public static implicit operator InputDef(Model.Input input) => FromModelInput(input);

        /// <summary>
        /// Initializes and returns an array of `InputDef` with data types and tensor shapes taken from model inputs.
        /// </summary>
        /// <param name="model">The model to use for input data types and shapes.</param>
        /// <returns>The input def array.</returns>
        public static InputDef[] FromModel(Model model) => model.inputs.Select(i => (InputDef)i).ToArray();

        /// <summary>
        /// Initializes and returns an instance of `InputDef` with data type and tensor shape taken from model input.
        /// </summary>
        /// <param name="input">The model input to use for data type and shape.</param>
        /// <returns>The input def.</returns>
        public static InputDef FromModelInput(Model.Input input) => new(input.dataType, input.shape);

        /// <summary>
        /// Initializes and returns an instance of `InputDef` with data type and tensor shape taken from tensor.
        /// </summary>
        /// <param name="tensor">The tensor to use for data type and shape.</param>
        /// <returns>The input def.</returns>
        public static InputDef FromTensor(Tensor tensor) => new(tensor.dataType, tensor.shape);

        /// <summary>
        /// Initializes and returns an an array of `InputDef` with data types and tensor shapes taken from tensors.
        /// </summary>
        /// <param name="tensors">The tensors to use for data type and shape.</param>
        /// <returns>The input def array.</returns>
        public static InputDef[] FromTensors(Tensor[] tensors) => tensors.Select(FromTensor).ToArray();

        /// <summary>
        /// Initializes and returns an instance of `InputDef` with float data type and tensor shape.
        /// </summary>
        /// <param name="shape">The shape of the input.</param>
        /// <returns>The input def.</returns>
        public static InputDef Float(TensorShape shape) => new(DataType.Float, shape);

        /// <summary>
        /// Initializes and returns an instance of `InputDef` with int data type and tensor shape.
        /// </summary>
        /// <param name="shape">The shape of the input.</param>
        /// <returns>The input def.</returns>
        public static InputDef Int(TensorShape shape) => new(DataType.Int, shape);
    }
}
