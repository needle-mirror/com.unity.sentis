using System;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents a constant in a model.
    /// </summary>
    public class Constant
    {
        /// <summary>
        /// The index of the constant.
        /// </summary>
        public int index;

        /// <summary>
        /// The shape of the constant as a `TensorShape`.
        /// </summary>
        public TensorShape shape;

        /// <summary>
        /// The size of the constant in bytes.
        /// </summary>
        public int lengthBytes;

        /// <summary>
        /// The data type of the constant as a `DataType`.
        /// </summary>
        public DataType dataType;

        /// <summary>
        /// The elements of the constant as a `NativeTensorArrayFromManagedArray`.
        /// </summary>
        public NativeTensorArray weights {
            get { return m_Weights; }
            [Obsolete("Setting constant weights has been deprecated", false)]
            set
            {
                if (value == null || value is NativeTensorArrayFromManagedArray)
                    m_Weights = (value as NativeTensorArrayFromManagedArray);
                else
                    throw new NotImplementedException($"Cannot assign weights to constant, must be of type NativeTensorArrayFromManagedArray");
            }
        }
        internal NativeTensorArrayFromManagedArray m_Weights;

        internal Constant(int index, TensorShape shape, DataType dataType, NativeTensorArrayFromManagedArray array)
        {
            this.index = index;
            this.shape = shape;
            this.lengthBytes = array.Length * sizeof(float);
            this.dataType = dataType;
            this.m_Weights = array;
        }

        internal Constant(int index, TensorShape shape, DataType dataType, int lengthBytes)
        {
            this.index = index;
            this.shape = shape;
            this.lengthBytes = lengthBytes;
            this.dataType = dataType;
        }

        internal Constant(int index, TensorShape shape, DataType dataType, NativeTensorArray array)
        {
            this.index = index;
            this.shape = shape;
            this.lengthBytes = array.Length * sizeof(float);
            this.dataType = dataType;
            this.m_Weights = new NativeTensorArrayFromManagedArray(array.ToArray<float>(array.Length), 0, sizeof(float), array.Length);
        }

        /// <summary>
        /// Initializes and returns a vector `Constant` from a given index, shape and float array.
        /// </summary>
        /// <param name="index">The index to use for the constant.</param>
        /// <param name="shape">The shape to use for the constant.</param>
        /// <param name="value">The float array of values.</param>
        public Constant(int index, TensorShape shape, float[] value)
        {
            this.index = index;
            this.shape = shape;
            this.lengthBytes = value.Length * sizeof(float);
            this.dataType = DataType.Float;
            if (value.Length == 0)
                return;
            m_Weights = new NativeTensorArrayFromManagedArray(value, 0, sizeof(float), value.Length);
        }

        /// <summary>
        /// Initializes and returns a vector `Constant` from a given index, shape and int array.
        /// </summary>
        /// <param name="index">The index to use for the constant.</param>
        /// <param name="shape">The shape to use for the constant.</param>
        /// <param name="value">The int array of values.</param>
        internal Constant(int index, TensorShape shape, int[] value)
        {
            this.index = index;
            this.shape = shape;
            this.lengthBytes = value.Length * sizeof(int);
            this.dataType = DataType.Int;
            if (value.Length == 0)
                return;
            m_Weights = new NativeTensorArrayFromManagedArray(value, 0, sizeof(int), value.Length);
        }

        /// <summary>
        /// Returns a string that represents the `Constant`.
        /// </summary>
        /// <returns>A string representation of the `Constant`.</returns>
        public override string ToString()
        {
            return $"Constant{dataType.ToString()} - index: {index}, shape: {shape}, dataType: {dataType}";
        }

        /// <summary>
        /// Creates and returns a CPU `Tensor` of the constant.
        /// </summary>
        /// <returns>The created tensor.</returns>
        /// <exception cref="NotImplementedException">Thrown when a given data type is not supported.</exception>
        internal Tensor WeightsToTensor()
        {
            switch (dataType)
            {
                case DataType.Float:
                {
                    var tensor = new Tensor<float>(shape, clearOnInit: false);
                    NativeTensorArray.Copy(weights, 0, (tensor.dataOnBackend as CPUTensorData).array, 0, shape.length);
                    return tensor;
                }
                case DataType.Int:
                {
                    var tensor = new Tensor<int>(shape, clearOnInit: false);
                    NativeTensorArray.Copy(weights, 0, (tensor.dataOnBackend as CPUTensorData).array, 0, shape.length);
                    return tensor;
                }
                default:
                    throw new NotImplementedException();
            }
        }

        internal Tensor WeightsToTensorWithSharedTensorData()
        {
            Tensor output;
            switch (dataType)
            {
                case DataType.Float:
                {
                    output = new Tensor<float>(shape, data: null);
                    break;
                }
                case DataType.Int:
                {
                    output = new Tensor<int>(shape, data: null);
                    break;
                }
                default:
                    throw new NotImplementedException();
            }

            output.dataOnBackend = new CPUTensorData(weights);
            return output;
        }

        /// <summary>
        /// Initializes the constant with the shape, dataType and weights from a given `Tensor`.
        /// </summary>
        /// <param name="X">The tensor to use for initialization.</param>
        /// <exception cref="NotImplementedException">Thrown when a given data type is not supported.</exception>
        // TODO move this to LayerFusingHelper
        internal void TensorToDataSet(Tensor X)
        {
            X.CompleteAllPendingOperations();
            this.shape = X.shape;
            this.dataType = X.dataType;
            if (X.shape.HasZeroDims())
                return;
            switch (dataType)
            {
                case DataType.Float:
                {
                    this.lengthBytes = shape.length * sizeof(float);
                    m_Weights = new NativeTensorArrayFromManagedArray(X.AsReadOnlyNativeArray<float>().ToArray(), 0, sizeof(float), X.count);
                    break;
                }
                case DataType.Int:
                {
                    this.lengthBytes = shape.length * sizeof(int);
                    m_Weights = new NativeTensorArrayFromManagedArray(X.AsReadOnlyNativeArray<int>().ToArray(), 0, sizeof(int), X.count);
                    break;
                }
                default:
                    throw new NotImplementedException($"DataType {dataType} not supported");
            }
        }
    }
}
