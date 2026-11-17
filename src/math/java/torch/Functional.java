package torch;

import java.util.Arrays;

/**
 * Neural-network functions built only from {@link Tensor} operations, in the
 * style of PyTorch's {@code torch.nn.functional}.
 *
 * <p>Because these functions use nothing but the {@code Tensor} interface,
 * they work unchanged with every implementation.</p>
 */
public final class Functional {

    /** The constant sqrt(2 / pi), used by {@link #gelu(Tensor)}. */
    private static final float SQRT_2_OVER_PI = (float) Math.sqrt(2.0 / Math.PI);

    /** The cubic coefficient in the tanh approximation of GELU. */
    private static final float GELU_COEFF = 0.044715f;

    /** Not instantiable; all methods are static. */
    private Functional() {
    }

    /**
     * Computes the softmax along one axis, turning each slice into a
     * probability distribution: {@code exp(x_i) / sum_j exp(x_j)}.
     *
     * <p>The maximum of each slice is subtracted before exponentiating. This
     * leaves the result unchanged but keeps every exponent at most {@code 0},
     * so large inputs cannot overflow to infinity.</p>
     *
     * @param x    the input
     * @param axis the axis to normalize over
     * @return a tensor of the same shape whose slices along {@code axis} are
     *         positive and sum to {@code 1}
     * @throws IllegalArgumentException if {@code axis} is out of range
     */
    public static Tensor softmax(Tensor x, int axis) {
        Tensor exps = x.sub(x.max(axis)).exp();
        return exps.div(exps.sum(axis));
    }

    /**
     * Computes the logarithm of the softmax along one axis:
     * {@code x_i - log(sum_j exp(x_j))}.
     *
     * <p>Equivalent to {@code softmax(x, axis).log()} but numerically stable,
     * because it never takes the logarithm of a probability that has rounded to
     * zero. Use it for cross-entropy loss.</p>
     *
     * @param x    the input
     * @param axis the axis to normalize over
     * @return a tensor of the same shape holding the log-probabilities
     * @throws IllegalArgumentException if {@code axis} is out of range
     */
    public static Tensor logSoftmax(Tensor x, int axis) {
        Tensor shifted = x.sub(x.max(axis));
        return shifted.sub(shifted.exp().sum(axis).log());
    }

    /**
     * Applies layer normalization over the last axis: each row is shifted to
     * mean {@code 0} and scaled to variance {@code 1}, then multiplied by
     * {@code gamma} and shifted by {@code beta}:
     * {@code gamma * (x - mean) / sqrt(variance + eps) + beta}.
     *
     * <p>The variance is the population variance (dividing by {@code n}), as in
     * PyTorch's {@code layer_norm}.</p>
     *
     * @param x     the input, of any rank at least {@code 1}
     * @param gamma the learnable gain, of shape {@code [d]} where {@code d} is the
     *              size of the last axis of {@code x}
     * @param beta  the learnable bias, of shape {@code [d]}
     * @param eps   a small positive constant added to the variance to avoid
     *              dividing by zero, typically {@code 1e-5}
     * @return a tensor of the same shape as {@code x}
     * @throws IllegalArgumentException if {@code gamma} or {@code beta} does not
     *                                  have shape {@code [d]}
     */
    public static Tensor layerNorm(Tensor x, Tensor gamma, Tensor beta, float eps) {
        int last = x.rank() - 1;
        int[] expected = {x.shape()[last]};
        checkShape("gamma", gamma, expected);
        checkShape("beta", beta, expected);

        Tensor centred = x.sub(x.mean(last));
        Tensor std = x.variance(last).add(eps).sqrt();
        return centred.div(std).mul(gamma).add(beta);
    }

    /**
     * Applies the Gaussian Error Linear Unit using the tanh approximation:
     * {@code 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))}.
     *
     * <p>This matches PyTorch's {@code gelu(x, approximate="tanh")} and GPT-2.</p>
     *
     * @param x the input
     * @return a new tensor of the same shape
     */
    public static Tensor gelu(Tensor x) {
        Tensor inner = x.add(x.pow(3).mul(GELU_COEFF)).mul(SQRT_2_OVER_PI);
        return x.mul(inner.tanh().add(1f)).mul(0.5f);
    }

    /**
     * Checks that a tensor has an expected shape.
     *
     * @param name     the parameter name, for the error message
     * @param t        the tensor to check
     * @param expected the required shape
     * @throws IllegalArgumentException if the shapes differ
     */
    private static void checkShape(String name, Tensor t, int[] expected) {
        if (!Arrays.equals(t.shape(), expected)) {
            throw new IllegalArgumentException(name + " must have shape " + Arrays.toString(expected)
                    + ", got " + Arrays.toString(t.shape()) + ".");
        }
    }
}
