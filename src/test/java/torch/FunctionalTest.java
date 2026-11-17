package torch;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

/**
 * Tests for {@link Functional}. Expected values were computed in NumPy (float64)
 * with the same formulas PyTorch uses: max-shifted softmax, population variance
 * in layer norm, and the tanh approximation of GELU.
 */
class FunctionalTest {
    private static final float EPS = 1e-5f;
    private final TensorFactory f = new NaiveTensorFactory();

    /** [[1, 2, 3], [-1, 0.5, 4]]. */
    private Tensor x() {
        return f.create(new float[]{1, 2, 3, -1, 0.5f, 4}, new int[]{2, 3});
    }

    private static void assertValues(float[] expected, Tensor actual, float eps) {
        assertArrayEquals(expected, actual.toArray(), eps);
    }

    // ---- softmax ----

    @Test
    void softmaxOverLastAxisMatchesReference() {
        Tensor s = Functional.softmax(x(), 1);
        assertArrayEquals(new int[]{2, 3}, s.shape());
        assertValues(new float[]{0.09003057f, 0.2447285f, 0.665241f,
            0.006497943f, 0.02912176f, 0.9643803f}, s, EPS);
    }

    @Test
    void softmaxOverFirstAxisMatchesReference() {
        assertValues(new float[]{0.8807971f, 0.8175745f, 0.2689414f,
            0.1192029f, 0.1824255f, 0.7310586f}, Functional.softmax(x(), 0), EPS);
    }

    @Test
    void softmaxRowsSumToOne() {
        assertValues(new float[]{1f, 1f}, Functional.softmax(x(), 1).sum(1), EPS);
    }

    @Test
    void softmaxIsStableForLargeInputs() {
        Tensor big = f.create(new float[]{1000, 1001, 1002}, new int[]{1, 3});
        assertValues(new float[]{0.09003057f, 0.2447285f, 0.665241f}, Functional.softmax(big, 1), EPS);
    }

    @Test
    void softmaxRejectsBadAxis() {
        assertThrows(IllegalArgumentException.class, () -> Functional.softmax(x(), 2));
    }

    // ---- logSoftmax ----

    @Test
    void logSoftmaxMatchesReference() {
        assertValues(new float[]{-2.407606f, -1.407606f, -0.407606f,
            -5.03627f, -3.53627f, -0.03626957f}, Functional.logSoftmax(x(), 1), 1e-4f);
    }

    @Test
    void logSoftmaxIsFiniteWhereSoftmaxUnderflows() {
        Tensor wide = f.create(new float[]{0, 200}, new int[]{1, 2});
        float lp = Functional.logSoftmax(wide, 1).get(0, 0);
        assertEquals(-200f, lp, 1e-3f);
    }

    // ---- layerNorm ----

    @Test
    void layerNormMatchesReference() {
        Tensor gamma = f.create(new float[]{1, 0.5f, 2}, new int[]{3});
        Tensor beta = f.create(new float[]{0, 1, -1}, new int[]{3});
        assertValues(new float[]{-1.224736f, 1f, 1.449471f,
            -1.034223f, 0.8408887f, 1.704892f}, Functional.layerNorm(x(), gamma, beta, 1e-5f), 1e-4f);
    }

    @Test
    void layerNormWithIdentityParamsHasZeroMeanUnitVariance() {
        Tensor y = Functional.layerNorm(x(), f.ones(3), f.zeros(3), 1e-5f);
        assertValues(new float[]{0f, 0f}, y.mean(1), 1e-5f);
        assertValues(new float[]{1f, 1f}, y.variance(1), 1e-3f);
    }

    @Test
    void layerNormOfConstantRowIsBeta() {
        Tensor c = f.full(5f, 1, 3);
        Tensor beta = f.create(new float[]{1, 2, 3}, new int[]{3});
        assertValues(new float[]{1, 2, 3}, Functional.layerNorm(c, f.ones(3), beta, 1e-5f), EPS);
    }

    @Test
    void layerNormRejectsWrongParamShape() {
        assertThrows(IllegalArgumentException.class,
            () -> Functional.layerNorm(x(), f.ones(2), f.zeros(3), 1e-5f));
        assertThrows(IllegalArgumentException.class,
            () -> Functional.layerNorm(x(), f.ones(3), f.zeros(1, 3), 1e-5f));
    }

    // ---- gelu ----

    @Test
    void geluMatchesReference() {
        Tensor in = f.create(new float[]{-3, -1, -0.5f, 0, 0.5f, 1, 3}, new int[]{7});
        assertValues(new float[]{-0.003637392f, -0.158808f, -0.154286f, 0f,
            0.345714f, 0.841192f, 2.996363f}, Functional.gelu(in), 1e-5f);
    }

    @Test
    void geluKeepsShape() {
        assertArrayEquals(new int[]{2, 3}, Functional.gelu(x()).shape());
    }

    // ---- works with any implementation ----

    @Test
    void functionsReturnFastTensorsForFastInput() {
        Tensor fx = new FastTensorFactory().create(new float[]{1, 2, 3}, new int[]{1, 3});
        assertTrue(Functional.softmax(fx, 1) instanceof FastTensor);
        assertTrue(Functional.gelu(fx) instanceof FastTensor);
    }
}
