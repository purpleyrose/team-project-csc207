package torch;

/**
 * Creates tensors without naming a concrete {@link Tensor} class.
 *
 * <p>Code that builds tensors (layers, loaders, tests) should depend on this
 * interface, so the implementation can be swapped in one place, for example
 * from {@link NaiveTensorFactory} to {@link FastTensorFactory}.</p>
 */
public interface TensorFactory {

    /**
     * Creates a tensor from row-major data and a shape. The data is copied, so
     * later changes to the array do not affect the tensor.
     *
     * @param data  the elements in row-major order (the last index varies fastest)
     * @param shape the size of each dimension
     * @return a new tensor
     * @throws IllegalArgumentException if {@code data.length} does not equal the
     *                                  product of the dimensions in {@code shape}
     */
    Tensor create(float[] data, int[] shape);

    /**
     * Creates a tensor with every element set to the same value.
     *
     * @param value the value of every element
     * @param shape the size of each dimension
     * @return a new tensor
     * @throws IllegalArgumentException if any dimension is not positive
     */
    default Tensor full(float value, int... shape) {
        int size = 1;
        for (int d : shape) {
            if (d <= 0) {
                throw new IllegalArgumentException("Dimensions must be positive, got " + d + ".");
            }
            size *= d;
        }
        float[] data = new float[size];
        java.util.Arrays.fill(data, value);
        return create(data, shape);
    }

    /**
     * Creates a tensor of zeros.
     *
     * @param shape the size of each dimension
     * @return a new tensor
     * @throws IllegalArgumentException if any dimension is not positive
     */
    default Tensor zeros(int... shape) {
        return full(0f, shape);
    }

    /**
     * Creates a tensor of ones.
     *
     * @param shape the size of each dimension
     * @return a new tensor
     * @throws IllegalArgumentException if any dimension is not positive
     */
    default Tensor ones(int... shape) {
        return full(1f, shape);
    }
}
