using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a constant in a model.
    /// </summary>
    public class Constant
    {
        /// <summary>
        /// The name of the constant.
        /// </summary>
        public string name;
        /// <summary>
        /// The shape of the constant as a `TensorShape`.
        /// </summary>
        public TensorShape shape;
        /// <summary>
        /// The number of elements in the constant.
        /// </summary>
        public Int32 length;
        /// <summary>
        /// The offset of the first element of the constant in the `weights` array.
        /// </summary>
        public Int64 offset;
        /// <summary>
        /// The data type of the constant as a `DataType`.
        /// </summary>
        public DataType dataType;
        /// <summary>
        /// The elements of the constant as a `NativeTensorArray`.
        /// </summary>
        public NativeTensorArray weights;

        /// <summary>
        /// Initializes and returns an empty `Constant`.
        /// </summary>
        public Constant() { }

        /// <summary>
        /// Initializes and returns a `Constant` from a given name and tensor.
        /// </summary>
        /// <param name="name">The name to use for the constant.</param>
        /// <param name="tensor">The tensor to take the shape, dataType and weights of the constant from.</param>
        public Constant(string name, Tensor tensor)
        {
            this.name = name;
            this.offset = 0;
            this.shape = tensor.shape;
            this.length = tensor.shape.length;
            this.dataType = tensor.dataType;
            weights = new NativeTensorArray(tensor.shape.length);

            switch (dataType)
            {
                case DataType.Float:
                {
                    (tensor as TensorFloat).ToReadOnlyArray().CopyToNativeTensorArray(weights, 0);
                    break;
                }
                case DataType.Int:
                {
                    weights = new NativeTensorArray(tensor.shape.length);
                    (tensor as TensorInt).ToReadOnlyArray().CopyToNativeTensorArray(weights, 0);
                    break;
                }
                default:
                    throw new NotImplementedException($"DataType {dataType} not supported");
            }
        }

        /// <summary>
        /// Returns a string that represents the `Constant`.
        /// </summary>
        public override string ToString()
        {
            return $"Constant{dataType.ToString()} - name: {name}, weights: [{name}, {shape}]";
        }

        /// <summary>
        /// Creates and returns a CPU `Tensor` of the constant.
        /// </summary>
        public Tensor DataSetToTensor()
        {
            switch (dataType)
            {
                case DataType.Float:
                {
                    var array = new float[shape.length];
                    NativeTensorArray.Copy(weights, (int)offset, array, 0, shape.length);
                    return new TensorFloat(shape, array);
                }
                case DataType.Int:
                {
                    var array = new int[shape.length];
                    NativeTensorArray.Copy(weights, (int)offset, array, 0, shape.length);
                    return new TensorInt(shape, array);
                }
                default:
                    throw new NotImplementedException($"DataType {dataType} not supported");
            }
        }

        /// <summary>
        /// Initializes the constant with the shape, dataType and weights from a given `Tensor`.
        /// </summary>
        public void TensorToDataSet(Tensor X)
        {
            this.shape = X.shape;
            this.length = X.shape.length;
            this.dataType = X.dataType;
            weights = new NativeTensorArray(X.shape.length);
            switch (dataType)
            {
                case DataType.Float:
                {
                    NativeTensorArray.Copy((X as TensorFloat).ToReadOnlyArray(), 0, weights, (int)offset, shape.length);
                    break;
                }
                case DataType.Int:
                {
                    NativeTensorArray.Copy((X as TensorInt).ToReadOnlyArray(), 0, weights, (int)offset, shape.length);
                    break;
                }
                default:
                    throw new NotImplementedException($"DataType {dataType} not supported");
            }
        }
    }
}
