package torch;

/**
 * A {@link Tensor} that speeds up the operations that dominate training time,
 * and otherwise behaves exactly like {@link NaiveTensor}.
 *
 * <p>Every result is a {@code FastTensor}, because it overrides
 * {@link NaiveTensor#create(float[], int[])}. So far, only
 * {@link #matmul(Tensor)} is optimized; further optimizations should override
 * single methods here without changing their results.</p>
 */
public class FastTensor extends NaiveTensor {

    /**
     * Creates a tensor from row-major data and a shape. Both arrays are copied.
     *
     * @param data  the elements in row-major order
     * @param shape the size of each dimension
     * @throws IllegalArgumentException if {@code data.length} does not equal the
     *                                  product of the dimensions in {@code shape}
     */
    public FastTensor(float[] data, int[] shape) {
        super(data, shape, true);
    }

    /**
     * Creates a tensor that takes ownership of {@code data} without copying it.
     *
     * @param data     the elements in row-major order
     * @param shape    the size of each dimension
     * @param copyData whether to copy {@code data}
     */
    private FastTensor(float[] data, int[] shape, boolean copyData) {
        super(data, shape, copyData);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected Tensor create(float[] data, int[] shape) {
        return new FastTensor(data, shape, false);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Reads both operands as flat arrays and loops in i-k-j order. The
     * innermost loop then walks one row of {@code other} and one row of the
     * result left to right, which uses the CPU cache far better than the
     * column-wise reads of the i-j-k order. Each output element still adds
     * its products in increasing {@code k}, so results are identical to
     * {@link NaiveTensor#matmul(Tensor)}.</p>
     */
    @Override
    public Tensor matmul(Tensor other) {
        int[] a = shape();
        int[] b = other.shape();
        checkMatmulShapes(a, b);

        float[] left = rawData();
        float[] right = other instanceof NaiveTensor ? ((NaiveTensor) other).rawData() : other.toArray();
        int rows = a[0];
        int inner = a[1];
        int cols = b[1];
        float[] result = new float[rows * cols];

        for (int i = 0; i < rows; i++) {
            int outRow = i * cols;
            for (int k = 0; k < inner; k++) {
                float aik = left[i * inner + k];
                int rightRow = k * cols;
                for (int j = 0; j < cols; j++) {
                    result[outRow + j] += aik * right[rightRow + j];
                }
            }
        }
        return create(result, new int[]{rows, cols});
    }
}
