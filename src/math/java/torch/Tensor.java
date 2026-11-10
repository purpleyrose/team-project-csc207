package torch;

/**
 * An immutable n-dimensional array of {@code float} values.
 *
 * <p>Every operation returns a new {@code Tensor} and leaves the receiver
 * unchanged. Implementations may differ in how they store data or compute
 * results (for example, a naive loop-based version versus an optimized one),
 * but all must honour the contracts described here so they can be swapped
 * freely.</p>
 *
 * <p>Elementwise binary operations follow NumPy-style broadcasting: shapes are
 * aligned from the rightmost dimension, and each pair of aligned dimensions
 * must either be equal or have one of them equal to {@code 1}. For example,
 * {@code [4, 3]} and {@code [4, 1]} broadcast to {@code [4, 3]}, as do
 * {@code [4, 3]} and {@code [3]}.</p>
 *
 * <p>Reductions ({@link #sum}, {@link #mean}, {@link #max}, {@link #min},
 * {@link #variance}) keep the reduced dimension with size {@code 1}, so their
 * results broadcast directly back against the original tensor.</p>
 */
public interface Tensor {

    /**
     * Returns the size of each dimension of this tensor.
     *
     * @return a copy of this tensor's shape; modifying it does not affect the tensor
     */
    int[] shape();

    /**
     * Returns the number of dimensions of this tensor.
     *
     * @return the rank, equal to {@code shape().length}
     */
    int rank();

    /**
     * Computes the matrix product of this tensor and another.
     *
     * @param other the right-hand operand, of shape {@code [n, p]} when this tensor
     *              has shape {@code [m, n]}
     * @return a new tensor of shape {@code [m, p]}
     * @throws IllegalArgumentException if either tensor is not rank 2, or if the
     *                                  inner dimensions do not match
     */
    Tensor matmul(Tensor other);

    /**
     * Adds another tensor to this one elementwise, with broadcasting.
     *
     * @param other the tensor to add
     * @return a new tensor holding {@code this + other}, in the broadcast shape
     * @throws IllegalArgumentException if the shapes cannot be broadcast together
     */
    Tensor add(Tensor other);

    /**
     * Subtracts another tensor from this one elementwise, with broadcasting.
     *
     * @param other the tensor to subtract
     * @return a new tensor holding {@code this - other}, in the broadcast shape
     * @throws IllegalArgumentException if the shapes cannot be broadcast together
     */
    Tensor sub(Tensor other);

    /**
     * Multiplies every element of this tensor by a scalar.
     *
     * @param scalar the value to multiply by
     * @return a new tensor of the same shape holding {@code this * scalar}
     */
    Tensor mul(float scalar);

    /**
     * Multiplies this tensor by another elementwise (Hadamard product), with
     * broadcasting. This is not matrix multiplication; see {@link #matmul}.
     *
     * @param other the tensor to multiply by
     * @return a new tensor holding {@code this * other}, in the broadcast shape
     * @throws IllegalArgumentException if the shapes cannot be broadcast together
     */
    Tensor mul(Tensor other);

    /**
     * Divides this tensor by another elementwise, with broadcasting.
     *
     * @param other the divisor
     * @return a new tensor holding {@code this / other}, in the broadcast shape
     * @throws IllegalArgumentException if the shapes cannot be broadcast together
     */
    Tensor div(Tensor other);

    /**
     * Divides every element of this tensor by a scalar.
     *
     * @param scalar the divisor
     * @return a new tensor of the same shape holding {@code this / scalar}
     */
    Tensor div(float scalar);

    /**
     * Applies the exponential function to every element.
     *
     * @return a new tensor of the same shape holding {@code e^x} for each element {@code x}
     */
    Tensor exp();

    /**
     * Applies the hyperbolic tangent to every element.
     *
     * @return a new tensor of the same shape holding {@code tanh(x)} for each element {@code x}
     */
    Tensor tanh();

    /**
     * Applies the square root to every element. Negative elements produce {@code NaN}.
     *
     * @return a new tensor of the same shape holding {@code sqrt(x)} for each element {@code x}
     */
    Tensor sqrt();

    /**
     * Sums the elements along one axis.
     *
     * @param axis the dimension to reduce, from {@code 0} to {@code rank() - 1}
     * @return a new tensor with the same shape except {@code axis} has size {@code 1}
     * @throws IllegalArgumentException if {@code axis} is out of range
     */
    Tensor sum(int axis);

    /**
     * Computes the arithmetic mean of the elements along one axis.
     *
     * @param axis the dimension to reduce, from {@code 0} to {@code rank() - 1}
     * @return a new tensor with the same shape except {@code axis} has size {@code 1}
     * @throws IllegalArgumentException if {@code axis} is out of range
     */
    Tensor mean(int axis);

    /**
     * Finds the largest element along one axis.
     *
     * @param axis the dimension to reduce, from {@code 0} to {@code rank() - 1}
     * @return a new tensor with the same shape except {@code axis} has size {@code 1}
     * @throws IllegalArgumentException if {@code axis} is out of range
     */
    Tensor max(int axis);

    /**
     * Finds the smallest element along one axis.
     *
     * @param axis the dimension to reduce, from {@code 0} to {@code rank() - 1}
     * @return a new tensor with the same shape except {@code axis} has size {@code 1}
     * @throws IllegalArgumentException if {@code axis} is out of range
     */
    Tensor min(int axis);

    /**
     * Computes the population variance (dividing by {@code n}, not {@code n - 1})
     * of the elements along one axis, as used by layer normalization.
     *
     * @param axis the dimension to reduce, from {@code 0} to {@code rank() - 1}
     * @return a new tensor with the same shape except {@code axis} has size {@code 1}
     * @throws IllegalArgumentException if {@code axis} is out of range
     */
    Tensor variance(int axis);

    /**
     * Swaps two dimensions of this tensor. For a rank-2 tensor,
     * {@code transpose(0, 1)} is the ordinary matrix transpose.
     *
     * @param dim0 the first dimension to swap
     * @param dim1 the second dimension to swap
     * @return a new tensor whose shape has {@code dim0} and {@code dim1} exchanged
     * @throws IllegalArgumentException if either dimension is out of range
     */
    Tensor transpose(int dim0, int dim1);

    /**
     * Reinterprets this tensor's elements, in row-major order, under a new shape.
     * At most one dimension may be {@code -1}, in which case its size is inferred
     * from the total number of elements.
     *
     * @param newShape the desired shape, for example {@code reshape(-1, 64)}
     * @return a new tensor with the same elements in the requested shape
     * @throws IllegalArgumentException if more than one dimension is {@code -1}, a
     *                                  dimension is zero or otherwise negative, or
     *                                  the total size does not match
     */
    Tensor reshape(int... newShape);

    /**
     * Extracts the elements whose index along one dimension lies in
     * {@code [start, end)}.
     *
     * @param dim   the dimension to slice along
     * @param start the first index to include
     * @param end   the index to stop before (exclusive)
     * @return a new tensor whose size along {@code dim} is {@code end - start}
     * @throws IllegalArgumentException if {@code dim} is out of range, or the range
     *                                  is empty or exceeds the dimension's size
     */
    Tensor slice(int dim, int start, int end);

    /**
     * Returns the element at the given position.
     *
     * @param indices one index per dimension, in order
     * @return the element at that position
     * @throws IllegalArgumentException  if the number of indices does not equal {@link #rank()}
     * @throws IndexOutOfBoundsException if any index is outside its dimension
     */
    float get(int... indices);

    /**
     * Returns an independent copy of this tensor.
     *
     * @return a new tensor with the same shape and elements that shares no storage with this one
     */
    Tensor clone();
}