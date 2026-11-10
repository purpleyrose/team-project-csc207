package torch;

import java.util.Arrays;

/**
 * A straightforward, loop-based implementation of {@link Tensor}.
 *
 * <p>Elements are stored in a flat {@code float} array in row-major order, so
 * the last index varies fastest. Every operation allocates and returns a new
 * {@code NaiveTensor}; instances are never modified after construction.</p>
 *
 * <p>This implementation favours clarity over speed. Matrix multiplication uses
 * the standard O(n<sup>3</sup>) triple loop, and elementwise operations access
 * the other operand through {@link Tensor#get(int...)}, so any {@code Tensor}
 * implementation can be mixed in.</p>
 */
public class NaiveTensor implements Tensor {
    /** The elements in row-major order. */
    private final float[] data;
    /** The size of each dimension. */
    private final int[] shape;
    /** The number of dimensions, equal to {@code shape.length}. */
    private final int rank;

    /**
     * A binary operation on two {@code float} values, used to share one loop
     * between elementwise operations and between reductions. A dedicated
     * interface avoids boxing every value into a {@link Float}.
     */
    @FunctionalInterface
    private interface FloatOp {
        /**
         * Applies this operation.
         *
         * @param a the first operand (or the running accumulator, in a reduction)
         * @param b the second operand (or the next element, in a reduction)
         * @return the result of the operation
         */
        float apply(float a, float b);
    }

    /**
     * Creates a tensor from row-major data and a shape. The data array is used
     * directly rather than copied, so callers should not modify it afterwards.
     *
     * @param data  the elements in row-major order
     * @param shape the size of each dimension; this array is copied
     * @throws IllegalArgumentException if {@code data.length} does not equal the
     *                                  product of the dimensions in {@code shape}
     */
    public NaiveTensor(float[] data, int[] shape) {
        if (data.length != product(shape)) {
            throw new IllegalArgumentException("Data length " + data.length
                    + " does not match shape " + Arrays.toString(shape) + ".");
        }
        this.data = data;
        this.shape = shape.clone();
        this.rank = shape.length;
    }

    // ------------------------------------------------------------------ basics

    /**
     * {@inheritDoc}
     */
    @Override
    public int[] shape() {
        return shape.clone();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int rank() {
        return rank;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Computes the flat offset by accumulating {@code index * stride} from the
     * last dimension to the first.</p>
     */
    @Override
    public float get(int... indices) {
        if (indices.length != rank) {
            throw new IllegalArgumentException("Expected " + rank + " indices, got " + indices.length + ".");
        }
        int flatIndex = 0;
        int stride = 1;
        for (int i = rank - 1; i >= 0; i--) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " out of bounds for dim "
                        + i + " with size " + shape[i] + ".");
            }
            flatIndex += indices[i] * stride;
            stride *= shape[i];
        }
        return data[flatIndex];
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor clone() {
        return new NaiveTensor(data.clone(), shape);
    }

    // ------------------------------------------------------------------ matmul

    /**
     * {@inheritDoc}
     *
     * <p>Uses the naive O(m &middot; n &middot; p) triple loop.</p>
     */
    @Override
    public Tensor matmul(Tensor other) {
        int[] otherShape = other.shape();
        if (this.rank != 2 || other.rank() != 2) {
            throw new IllegalArgumentException("Both tensors must be rank 2 for matmul.");
        }
        if (shape[1] != otherShape[0]) {
            throw new IllegalArgumentException("Inner dimensions must match for matmul: "
                    + Arrays.toString(shape) + " x " + Arrays.toString(otherShape) + ".");
        }

        int rows = shape[0];
        int cols = otherShape[1];
        int inner = shape[1];
        float[] result = new float[rows * cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float sum = 0;
                for (int k = 0; k < inner; k++) {
                    sum += data[i * inner + k] * other.get(k, j);
                }
                result[i * cols + j] = sum;
            }
        }
        return new NaiveTensor(result, new int[]{rows, cols});
    }

    // ------------------------------------------------------- elementwise (tensor)

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor add(Tensor other) {
        return broadcast(other, (a, b) -> a + b, "add");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor sub(Tensor other) {
        return broadcast(other, (a, b) -> a - b, "sub");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor mul(Tensor other) {
        return broadcast(other, (a, b) -> a * b, "mul");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor div(Tensor other) {
        return broadcast(other, (a, b) -> a / b, "div");
    }

    /**
     * Applies a binary operation elementwise using NumPy-style broadcasting.
     *
     * <p>The shapes are aligned from the right, missing leading dimensions are
     * treated as size {@code 1}, and each aligned pair must be equal or contain a
     * {@code 1}. A size-{@code 1} dimension is stretched by reusing index
     * {@code 0} for every position along it.</p>
     *
     * @param other the right-hand operand
     * @param op    the operation to apply to each pair of elements
     * @param name  the operation's name, used in error messages
     * @return a new tensor in the broadcast shape holding {@code op(this, other)}
     * @throws IllegalArgumentException if the shapes cannot be broadcast together
     */
    private Tensor broadcast(Tensor other, FloatOp op, String name) {
        int[] a = this.shape;
        int[] b = other.shape();
        int outRank = Math.max(a.length, b.length);
        int offA = outRank - a.length;
        int offB = outRank - b.length;

        int[] outShape = new int[outRank];
        for (int i = 0; i < outRank; i++) {
            int da = i >= offA ? a[i - offA] : 1;
            int db = i >= offB ? b[i - offB] : 1;
            if (da != db && da != 1 && db != 1) {
                throw new IllegalArgumentException("Cannot broadcast shapes " + Arrays.toString(a)
                        + " and " + Arrays.toString(b) + " for " + name + ".");
            }
            outShape[i] = Math.max(da, db);
        }

        int size = product(outShape);
        float[] result = new float[size];
        int[] outIdx = new int[outRank];
        int[] idxA = new int[a.length];
        int[] idxB = new int[b.length];

        for (int flat = 0; flat < size; flat++) {
            unravel(flat, outShape, outIdx);
            for (int i = 0; i < a.length; i++) {
                idxA[i] = a[i] == 1 ? 0 : outIdx[i + offA];
            }
            for (int i = 0; i < b.length; i++) {
                idxB[i] = b[i] == 1 ? 0 : outIdx[i + offB];
            }
            result[flat] = op.apply(this.get(idxA), other.get(idxB));
        }
        return new NaiveTensor(result, outShape);
    }

    // ------------------------------------------------------- elementwise (scalar)

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor mul(float scalar) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] * scalar;
        }
        return new NaiveTensor(result, shape);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor div(float scalar) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] / scalar;
        }
        return new NaiveTensor(result, shape);
    }

    // ------------------------------------------------------------------- unary

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor exp() {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) Math.exp(data[i]);
        }
        return new NaiveTensor(result, shape);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor tanh() {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) Math.tanh(data[i]);
        }
        return new NaiveTensor(result, shape);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor sqrt() {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (float) Math.sqrt(data[i]);
        }
        return new NaiveTensor(result, shape);
    }

    // -------------------------------------------------------------- reductions

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor sum(int axis) {
        return reduce(axis, 0f, (acc, x) -> acc + x);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor max(int axis) {
        return reduce(axis, Float.NEGATIVE_INFINITY, Math::max);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor min(int axis) {
        return reduce(axis, Float.POSITIVE_INFINITY, Math::min);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Computed as {@code sum(axis)} divided by the size of {@code axis}.</p>
     */
    @Override
    public Tensor mean(int axis) {
        return sum(axis).div((float) shape[axis]);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Computed as the mean of the squared deviations from {@code mean(axis)},
     * relying on broadcasting to subtract the mean from every element.</p>
     */
    @Override
    public Tensor variance(int axis) {
        Tensor diff = this.sub(mean(axis));
        return diff.mul(diff).mean(axis);
    }

    /**
     * Folds a binary operation across one axis.
     *
     * <p>For each position in the result, starts from {@code init} and combines it
     * with every element along {@code axis} in order.</p>
     *
     * @param axis the dimension to reduce
     * @param init the starting value of each accumulator, such as {@code 0} for a sum
     * @param op   the operation combining the accumulator with the next element
     * @return a new tensor with the same shape except {@code axis} has size {@code 1}
     * @throws IllegalArgumentException if {@code axis} is out of range
     */
    private Tensor reduce(int axis, float init, FloatOp op) {
        checkAxis(axis);
        int[] newShape = shape.clone();
        newShape[axis] = 1;

        int size = product(newShape);
        float[] result = new float[size];
        int[] idx = new int[rank];

        for (int i = 0; i < size; i++) {
            unravel(i, newShape, idx);
            float acc = init;
            for (int a = 0; a < shape[axis]; a++) {
                idx[axis] = a;
                acc = op.apply(acc, get(idx));
            }
            result[i] = acc;
        }
        return new NaiveTensor(result, newShape);
    }

    // ---------------------------------------------------------- shape ops

    /**
     * {@inheritDoc}
     *
     * <p>The data is copied into the new layout, so the result is contiguous.</p>
     */
    @Override
    public Tensor transpose(int dim0, int dim1) {
        checkAxis(dim0);
        checkAxis(dim1);
        int[] newShape = shape.clone();
        newShape[dim0] = shape[dim1];
        newShape[dim1] = shape[dim0];

        float[] result = new float[data.length];
        int[] idx = new int[rank];
        for (int flat = 0; flat < data.length; flat++) {
            unravel(flat, newShape, idx);
            int tmp = idx[dim0];
            idx[dim0] = idx[dim1];
            idx[dim1] = tmp;
            result[flat] = get(idx);
        }
        return new NaiveTensor(result, newShape);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor reshape(int... newShape) {
        int[] s = newShape.clone();
        int inferred = -1;
        int known = 1;
        for (int i = 0; i < s.length; i++) {
            if (s[i] == -1) {
                if (inferred != -1) {
                    throw new IllegalArgumentException("Only one dimension can be -1 in reshape.");
                }
                inferred = i;
            } else if (s[i] <= 0) {
                throw new IllegalArgumentException("Invalid dimension " + s[i] + " in reshape.");
            } else {
                known *= s[i];
            }
        }
        if (inferred != -1) {
            if (data.length % known != 0) {
                throw new IllegalArgumentException("Cannot infer -1 for size " + data.length
                        + " into " + Arrays.toString(newShape) + ".");
            }
            s[inferred] = data.length / known;
        }
        if (product(s) != data.length) {
            throw new IllegalArgumentException("Cannot reshape " + Arrays.toString(shape)
                    + " into " + Arrays.toString(newShape) + ".");
        }
        return new NaiveTensor(data.clone(), s);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor slice(int dim, int start, int end) {
        checkAxis(dim);
        if (start < 0 || end > shape[dim] || start >= end) {
            throw new IllegalArgumentException("Invalid slice [" + start + ", " + end
                    + ") for dim " + dim + " with size " + shape[dim] + ".");
        }
        int[] newShape = shape.clone();
        newShape[dim] = end - start;

        int size = product(newShape);
        float[] result = new float[size];
        int[] idx = new int[rank];
        for (int flat = 0; flat < size; flat++) {
            unravel(flat, newShape, idx);
            idx[dim] += start;
            result[flat] = get(idx);
        }
        return new NaiveTensor(result, newShape);
    }

    // ---------------------------------------------------------------- helpers

    /**
     * Multiplies the dimensions of a shape together.
     *
     * @param s the shape
     * @return the total number of elements a tensor of that shape holds
     */
    private static int product(int[] s) {
        int p = 1;
        for (int x : s) {
            p *= x;
        }
        return p;
    }

    /**
     * Converts a flat row-major index into one index per dimension.
     *
     * @param flat  the flat index, from {@code 0} to {@code product(shape) - 1}
     * @param shape the shape to interpret the index against
     * @param out   the array to write the per-dimension indices into; must have
     *              length {@code shape.length}
     */
    private static void unravel(int flat, int[] shape, int[] out) {
        for (int d = shape.length - 1; d >= 0; d--) {
            out[d] = flat % shape[d];
            flat /= shape[d];
        }
    }

    /**
     * Checks that an axis refers to a dimension of this tensor.
     *
     * @param axis the axis to check
     * @throws IllegalArgumentException if {@code axis} is negative or at least {@link #rank()}
     */
    private void checkAxis(int axis) {
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Axis " + axis + " out of range for rank " + rank + ".");
        }
    }

    /**
     * Returns a debugging representation showing the shape and the flat data.
     *
     * @return a string of the form {@code NaiveTensor(shape=[...], data=[...])}
     */
    @Override
    public String toString() {
        return "NaiveTensor(shape=" + Arrays.toString(shape) + ", data=" + Arrays.toString(data) + ")";
    }
}