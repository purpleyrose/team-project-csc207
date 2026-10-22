package torch;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Random;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class NaiveTensorTest {
    private static final float EPS = 1e-5f;

    // ---- helpers ----

    private static Tensor t(float[] data, int... shape) {
        return new NaiveTensor(data, shape);
    }

    /**
     * Asserts {@code actual} has the given shape and row-major contents.
     * Reads elements through {@code get} only, so it does not depend on {@code reshape}.
     */
    private static void assertTensor(int[] expectedShape, float[] expected, Tensor actual) {
        assertNotNull(actual, "tensor was null");
        assertArrayEquals(expectedShape, actual.shape(), "shape");
        assertEquals(expectedShape.length, actual.rank(), "rank");
        int[] idx = new int[expectedShape.length];
        for (int flat = 0; flat < expected.length; flat++) {
            int rem = flat;
            for (int d = expectedShape.length - 1; d >= 0; d--) {
                idx[d] = rem % expectedShape[d];
                rem /= expectedShape[d];
            }
            assertEquals(expected[flat], actual.get(idx), EPS, "element " + flat);
        }
    }

    /** 2x3 matrix [[1, 5, -3], [4, 2, 6]]. */
    private static Tensor matrix() {
        return t(new float[]{1, 5, -3, 4, 2, 6}, 2, 3);
    }

    /** Vector [1, 2, 3, 4]. */
    private static Tensor vector() {
        return t(new float[]{1, 2, 3, 4}, 4);
    }

    // ---- shape / rank ----

    @Test
    void shapeReturnsConstructorShape() {
        assertArrayEquals(new int[]{2, 3}, t(new float[6], 2, 3).shape());
    }

    @Test
    void rankIsNumberOfDimensions() {
        assertEquals(1, t(new float[3], 3).rank());
        assertEquals(2, t(new float[6], 2, 3).rank());
        assertEquals(3, t(new float[24], 2, 3, 4).rank());
    }

    // ---- get ----

    @Test
    void getReadsRowMajorElement() {
        Tensor a = t(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        assertEquals(1f, a.get(0, 0), EPS);
        assertEquals(3f, a.get(0, 2), EPS);
        assertEquals(4f, a.get(1, 0), EPS);
        assertEquals(6f, a.get(1, 2), EPS);
    }

    @Test
    void getWorksForHigherRank() {
        float[] data = new float[24];
        for (int i = 0; i < data.length; i++) {
            data[i] = i;
        }
        Tensor a = t(data, 2, 3, 4);
        assertEquals(0f, a.get(0, 0, 0), EPS);
        assertEquals(5f, a.get(0, 1, 1), EPS);
        assertEquals(12f, a.get(1, 0, 0), EPS);
        assertEquals(23f, a.get(1, 2, 3), EPS);
    }

    @Test
    void getWithOutOfRangeIndexThrows() {
        assertThrows(IndexOutOfBoundsException.class, () -> matrix().get(2, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> matrix().get(0, 3));
        assertThrows(IndexOutOfBoundsException.class, () -> matrix().get(-1, 0));
    }

    @Test
    void getWithWrongIndexCountThrows() {
        assertThrows(IllegalArgumentException.class, () -> matrix().get(0));
        assertThrows(IllegalArgumentException.class, () -> matrix().get(0, 0, 0));
    }


    @Test
    void getWithWrongNumberOfIndicesThrows() {
        assertThrows(IllegalArgumentException.class, () -> matrix().get(0));
        assertThrows(IllegalArgumentException.class, () -> matrix().get(0, 0, 0));
    }

    // ---- element-wise binary ops ----

    @Test
    void addIsElementWise() {
        Tensor r = t(new float[]{1, 2, 3, 4}, 2, 2).add(t(new float[]{10, 20, 30, 40}, 2, 2));
        assertTensor(new int[]{2, 2}, new float[]{11, 22, 33, 44}, r);
    }

    @Test
    void subIsElementWise() {
        Tensor r = t(new float[]{5, 5}, 2).sub(t(new float[]{1, 2}, 2));
        assertTensor(new int[]{2}, new float[]{4, 3}, r);
    }

    @Test
    void divByTensorIsElementWise() {
        Tensor r = t(new float[]{10, 9}, 2).div(t(new float[]{2, 3}, 2));
        assertTensor(new int[]{2}, new float[]{5, 3}, r);
    }

    @Test
    void equalShapesInDifferentArraysAreAccepted() {
        // shapes must be compared by value, not by array identity
        Tensor a = t(new float[]{1, 2}, 2);
        Tensor b = t(new float[]{3, 4}, new int[]{2});
        assertDoesNotThrow(() -> a.add(b));
        assertDoesNotThrow(() -> a.sub(b));
        assertDoesNotThrow(() -> a.div(b));
    }

    @Test
    void elementWiseOpsWithMismatchedShapesThrow() {
        Tensor a = t(new float[]{1, 2}, 2);
        Tensor b = t(new float[]{1, 2, 3}, 3);
        assertThrows(IllegalArgumentException.class, () -> a.add(b));
        assertThrows(IllegalArgumentException.class, () -> a.sub(b));
        assertThrows(IllegalArgumentException.class, () -> a.div(b));
    }

    @Test
    void elementWiseOpsDoNotMutateOperands() {
        Tensor a = t(new float[]{1, 2}, 2);
        Tensor b = t(new float[]{3, 4}, 2);
        a.add(b);
        a.sub(b);
        a.div(b);
        assertTensor(new int[]{2}, new float[]{1, 2}, a);
        assertTensor(new int[]{2}, new float[]{3, 4}, b);
    }

    // ---- scalar ops ----

    @Test
    void mulScalarScalesEveryElement() {
        assertTensor(new int[]{3}, new float[]{2, -4, 6}, t(new float[]{1, -2, 3}, 3).mul(2f));
    }

    @Test
    void divScalarDividesEveryElement() {
        assertTensor(new int[]{3}, new float[]{0.5f, -1, 1.5f}, t(new float[]{1, -2, 3}, 3).div(2f));
    }

    @Test
    void scalarOpsKeepShapeAndDoNotMutate() {
        Tensor a = matrix();
        Tensor r = a.mul(3f);
        assertArrayEquals(new int[]{2, 3}, r.shape());
        a.div(2f);
        assertTensor(new int[]{2, 3}, new float[]{1, 5, -3, 4, 2, 6}, a);
    }

    // ---- matmul ----

    @Test
    void matmulOfTwoMatrices() {
        Tensor a = t(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor b = t(new float[]{7, 8, 9, 10, 11, 12}, 3, 2);
        assertTensor(new int[]{2, 2}, new float[]{58, 64, 139, 154}, a.matmul(b));
    }

    @Test
    void matmulWithIdentityReturnsSameValues() {
        Tensor identity = t(new float[]{1, 0, 0, 0, 1, 0, 0, 0, 1}, 3, 3);
        assertTensor(new int[]{2, 3}, new float[]{1, 5, -3, 4, 2, 6}, matrix().matmul(identity));
    }

    @Test
    void matmulWithIncompatibleShapesThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> t(new float[6], 2, 3).matmul(t(new float[6], 2, 3)));
    }

    @Test
    void matmulRequiresRankTwo() {
        assertThrows(IllegalArgumentException.class, () -> vector().matmul(matrix()));
        assertThrows(IllegalArgumentException.class, () -> matrix().matmul(vector()));
    }


    // ---- broadcasting ----

    @Test
    void mulTensorIsElementWise() {
        assertTensor(new int[]{2, 3}, new float[]{1, 25, 9, 16, 4, 36}, matrix().mul(matrix()));
    }

    @Test
    void addBroadcastsLowerRankRowAcrossRows() {
        Tensor bias = t(new float[]{10, 20, 30}, 3);
        assertTensor(new int[]{2, 3}, new float[]{11, 25, 27, 14, 22, 36}, matrix().add(bias));
    }

    @Test
    void subBroadcastsKeepdimReductionBack() {
        // row means are [1, 4]
        assertTensor(new int[]{2, 3}, new float[]{0, 4, -4, 0, -2, 2}, matrix().sub(matrix().mean(1)));
    }

    @Test
    void broadcastingStretchesBothOperands() {
        Tensor col = t(new float[]{1, 2}, 2, 1);
        Tensor row = t(new float[]{10, 20, 30}, 1, 3);
        assertTensor(new int[]{2, 3}, new float[]{10, 20, 30, 20, 40, 60}, col.mul(row));
    }

    @Test
    void broadcastingIncompatibleShapesThrows() {
        assertThrows(IllegalArgumentException.class, () -> matrix().add(t(new float[]{1, 2}, 2)));
        assertThrows(IllegalArgumentException.class, () -> matrix().mul(t(new float[4], 2, 2)));
    }


    // ---- unary ops ----

    @Test
    void expAppliesElementWise() {
        assertTensor(new int[]{2}, new float[]{1f, (float) Math.E}, t(new float[]{0, 1}, 2).exp());
    }

    @Test
    void tanhAppliesElementWise() {
        assertTensor(new int[]{3},
                new float[]{0f, (float) Math.tanh(1), (float) Math.tanh(-1)},
                t(new float[]{0, 1, -1}, 3).tanh());
    }

    @Test
    void sqrtAppliesElementWise() {
        assertTensor(new int[]{2}, new float[]{2, 3}, t(new float[]{4, 9}, 2).sqrt());
    }

    @Test
    void sqrtOfNegativeIsNaN() {
        assertTrue(Float.isNaN(t(new float[]{-1}, 1).sqrt().get(0)));
    }

    @Test
    void unaryOpsKeepShapeAndDoNotMutate() {
        Tensor a = matrix();
        assertArrayEquals(new int[]{2, 3}, a.exp().shape());
        assertArrayEquals(new int[]{2, 3}, a.tanh().shape());
        assertTensor(new int[]{2, 3}, new float[]{1, 5, -3, 4, 2, 6}, a);
    }

    // ---- reductions (along an axis; the reduced dimension is kept with size 1, i.e. keepdim) ----

    @Test
    void sumAlongAxis() {
        assertTensor(new int[]{1, 3}, new float[]{5, 7, 3}, matrix().sum(0));
        assertTensor(new int[]{2, 1}, new float[]{3, 12}, matrix().sum(1));
    }

    @Test
    void meanAlongAxis() {
        assertTensor(new int[]{1, 3}, new float[]{2.5f, 3.5f, 1.5f}, matrix().mean(0));
        assertTensor(new int[]{2, 1}, new float[]{1f, 4f}, matrix().mean(1));
    }

    @Test
    void maxAlongAxis() {
        assertTensor(new int[]{1, 3}, new float[]{4, 5, 6}, matrix().max(0));
        assertTensor(new int[]{2, 1}, new float[]{5, 6}, matrix().max(1));
    }

    @Test
    void minAlongAxis() {
        assertTensor(new int[]{1, 3}, new float[]{1, 2, -3}, matrix().min(0));
        assertTensor(new int[]{2, 1}, new float[]{-3, 2}, matrix().min(1));
    }

    @Test
    void varianceAlongAxisIsPopulationVariance() {
        // columns: [1,4] [5,2] [-3,6] -> 2.25, 2.25, 20.25
        assertTensor(new int[]{1, 3}, new float[]{2.25f, 2.25f, 20.25f}, matrix().variance(0));
        // rows: mean 1 -> (0+16+16)/3 ; mean 4 -> (0+4+4)/3
        assertTensor(new int[]{2, 1}, new float[]{32f / 3, 8f / 3}, matrix().variance(1));
    }

    @Test
    void reducingAOneDimensionalTensorKeepsRankOne() {
        assertTensor(new int[]{1}, new float[]{10f}, vector().sum(0));
        assertTensor(new int[]{1}, new float[]{2.5f}, vector().mean(0));
        assertTensor(new int[]{1}, new float[]{4f}, vector().max(0));
        assertTensor(new int[]{1}, new float[]{1f}, vector().min(0));
        assertTensor(new int[]{1}, new float[]{1.25f}, vector().variance(0));
    }

    @Test
    void reductionsPreserveRank() {
        assertEquals(2, matrix().sum(0).rank());
        assertEquals(2, matrix().variance(1).rank());
    }

    @Test
    void reductionOfConstantTensorHasZeroVariance() {
        assertTensor(new int[]{2, 1}, new float[]{0, 0}, t(new float[]{3, 3, 3, 3}, 2, 2).variance(1));
    }

    @Test
    void reductionsRejectInvalidAxis() {
        assertThrows(IllegalArgumentException.class, () -> matrix().sum(2));
        assertThrows(IllegalArgumentException.class, () -> matrix().mean(-1));
        assertThrows(IllegalArgumentException.class, () -> matrix().max(5));
        assertThrows(IllegalArgumentException.class, () -> matrix().min(-1));
        assertThrows(IllegalArgumentException.class, () -> matrix().variance(2));
    }

    @Test
    void reductionsDoNotMutateInput() {
        Tensor m = matrix();
        m.sum(0);
        m.mean(1);
        m.max(0);
        m.min(1);
        m.variance(0);
        assertTensor(new int[]{2, 3}, new float[]{1, 5, -3, 4, 2, 6}, m);
    }

    // ---- shape manipulation ----

    @Test
    void transposeSwapsDimensions() {
        Tensor r = matrix().transpose(0, 1);
        assertTensor(new int[]{3, 2}, new float[]{1, 4, 5, 2, -3, 6}, r);
    }

    @Test
    void transposeTwiceRestoresOriginal() {
        assertTensor(new int[]{2, 3}, new float[]{1, 5, -3, 4, 2, 6},
                matrix().transpose(0, 1).transpose(0, 1));
    }

    @Test
    void transposeOfSameDimensionIsUnchanged() {
        assertTensor(new int[]{2, 3}, new float[]{1, 5, -3, 4, 2, 6}, matrix().transpose(1, 1));
    }

    @Test
    void transposeWithInvalidDimensionThrows() {
        assertThrows(IllegalArgumentException.class, () -> matrix().transpose(0, 2));
        assertThrows(IllegalArgumentException.class, () -> matrix().transpose(-1, 1));
    }

    @Test
    void reshapePreservesRowMajorOrder() {
        assertTensor(new int[]{3, 2}, new float[]{1, 5, -3, 4, 2, 6}, matrix().reshape(3, 2));
        assertTensor(new int[]{6}, new float[]{1, 5, -3, 4, 2, 6}, matrix().reshape(6));
    }

    @Test
    void reshapeWithWrongElementCountThrows() {
        assertThrows(IllegalArgumentException.class, () -> matrix().reshape(4, 2));
    }

    @Test
    void reshapeDoesNotMutateOriginal() {
        Tensor a = matrix();
        a.reshape(3, 2);
        assertArrayEquals(new int[]{2, 3}, a.shape());
    }

    // ---- slice ----

    @Test
    void sliceTakesHalfOpenRangeAlongRows() {
        Tensor a = t(new float[]{1, 2, 3, 4, 5, 6}, 3, 2);
        assertTensor(new int[]{2, 2}, new float[]{3, 4, 5, 6}, a.slice(0, 1, 3));
    }

    @Test
    void sliceAlongColumns() {
        assertTensor(new int[]{2, 2}, new float[]{5, -3, 2, 6}, matrix().slice(1, 1, 3));
    }

    @Test
    void sliceOfFullRangeEqualsOriginal() {
        assertTensor(new int[]{2, 3}, new float[]{1, 5, -3, 4, 2, 6}, matrix().slice(0, 0, 2));
    }

    @Test
    void sliceWithInvalidArgumentsThrows() {
        assertThrows(IllegalArgumentException.class, () -> matrix().slice(2, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> matrix().slice(0, -1, 1));
        assertThrows(IllegalArgumentException.class, () -> matrix().slice(0, 0, 3));
        assertThrows(IllegalArgumentException.class, () -> matrix().slice(0, 2, 1));
    }

    // ---- clone ----

    @Test
    void cloneIsEqualButIndependent() {
        float[] data = {1, 2, 3, 4};
        Tensor a = t(data, 2, 2);
        Tensor c = a.clone();
        assertNotSame(a, c);
        assertTensor(new int[]{2, 2}, new float[]{1, 2, 3, 4}, c);
        data[0] = 99f;
        assertEquals(1f, c.get(0, 0), EPS);
    }

    // =====================================================================
    // Extended coverage: higher rank, edge cases, parameterized and property tests
    // =====================================================================

    /** 2x2x3 tensor holding 1..12 in row-major order. */
    private static Tensor cube() {
        return t(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 2, 3);
    }

    /** Reads every element of {@code tensor} in row-major order using only {@code get}. */
    private static float[] flatten(Tensor tensor) {
        int[] shape = tensor.shape();
        int n = 1;
        for (int d : shape) {
            n *= d;
        }
        float[] out = new float[n];
        int[] idx = new int[shape.length];
        for (int flat = 0; flat < n; flat++) {
            int rem = flat;
            for (int d = shape.length - 1; d >= 0; d--) {
                idx[d] = rem % shape[d];
                rem /= shape[d];
            }
            out[flat] = tensor.get(idx);
        }
        return out;
    }

    private static void assertSameTensor(Tensor expected, Tensor actual, float eps) {
        assertNotNull(actual);
        assertArrayEquals(expected.shape(), actual.shape(), "shape");
        assertArrayEquals(flatten(expected), flatten(actual), eps);
    }

    private static Tensor randomTensor(long seed, int... shape) {
        int n = 1;
        for (int d : shape) {
            n *= d;
        }
        Random rng = new Random(seed);
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = rng.nextFloat() * 20f - 10f;
        }
        return t(data, shape);
    }

    // ---- reductions on a rank-3 tensor ----

    @Test
    void sumOfCubeAlongAxis0() {
        assertTensor(new int[]{1, 2, 3}, new float[]{8, 10, 12, 14, 16, 18}, cube().sum(0));
    }

    @Test
    void sumOfCubeAlongAxis1() {
        assertTensor(new int[]{2, 1, 3}, new float[]{5, 7, 9, 17, 19, 21}, cube().sum(1));
    }

    @Test
    void sumOfCubeAlongAxis2() {
        assertTensor(new int[]{2, 2, 1}, new float[]{6, 15, 24, 33}, cube().sum(2));
    }

    @Test
    void meanOfCubeAlongAxis0() {
        assertTensor(new int[]{1, 2, 3}, new float[]{4, 5, 6, 7, 8, 9}, cube().mean(0));
    }

    @Test
    void meanOfCubeAlongAxis1() {
        assertTensor(new int[]{2, 1, 3}, new float[]{2.5f, 3.5f, 4.5f, 8.5f, 9.5f, 10.5f}, cube().mean(1));
    }

    @Test
    void meanOfCubeAlongAxis2() {
        assertTensor(new int[]{2, 2, 1}, new float[]{2, 5, 8, 11}, cube().mean(2));
    }

    @Test
    void maxOfCubeAlongAxis0() {
        assertTensor(new int[]{1, 2, 3}, new float[]{7, 8, 9, 10, 11, 12}, cube().max(0));
    }

    @Test
    void maxOfCubeAlongAxis1() {
        assertTensor(new int[]{2, 1, 3}, new float[]{4, 5, 6, 10, 11, 12}, cube().max(1));
    }

    @Test
    void maxOfCubeAlongAxis2() {
        assertTensor(new int[]{2, 2, 1}, new float[]{3, 6, 9, 12}, cube().max(2));
    }

    @Test
    void minOfCubeAlongAxis0() {
        assertTensor(new int[]{1, 2, 3}, new float[]{1, 2, 3, 4, 5, 6}, cube().min(0));
    }

    @Test
    void minOfCubeAlongAxis1() {
        assertTensor(new int[]{2, 1, 3}, new float[]{1, 2, 3, 7, 8, 9}, cube().min(1));
    }

    @Test
    void minOfCubeAlongAxis2() {
        assertTensor(new int[]{2, 2, 1}, new float[]{1, 4, 7, 10}, cube().min(2));
    }

    @Test
    void varianceOfCubeAlongAxis0() {
        // every pair differs by 6 -> population variance 9
        assertTensor(new int[]{1, 2, 3}, new float[]{9, 9, 9, 9, 9, 9}, cube().variance(0));
    }

    @Test
    void varianceOfCubeAlongAxis1() {
        // every pair differs by 3 -> population variance 2.25
        assertTensor(new int[]{2, 1, 3}, new float[]{2.25f, 2.25f, 2.25f, 2.25f, 2.25f, 2.25f},
                cube().variance(1));
    }

    @Test
    void varianceOfCubeAlongAxis2() {
        // every run is {n, n+1, n+2} -> 2/3
        float v = 2f / 3f;
        assertTensor(new int[]{2, 2, 1}, new float[]{v, v, v, v}, cube().variance(2));
    }

    @Test
    void cubeReductionsRejectAxisEqualToRank() {
        assertThrows(IllegalArgumentException.class, () -> cube().sum(3));
        assertThrows(IllegalArgumentException.class, () -> cube().variance(3));
    }

    // ---- transpose / slice / reshape on rank 3 ----

    @Test
    void transposeCubeOuterAndInnerDimensions() {
        assertTensor(new int[]{3, 2, 2},
                new float[]{1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12}, cube().transpose(0, 2));
    }

    @Test
    void transposeCubeInnerDimensions() {
        assertTensor(new int[]{2, 3, 2},
                new float[]{1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12}, cube().transpose(1, 2));
    }

    @Test
    void sliceCubeAlongLastDimension() {
        assertTensor(new int[]{2, 2, 2}, new float[]{2, 3, 5, 6, 8, 9, 11, 12}, cube().slice(2, 1, 3));
    }

    @Test
    void sliceCubeAlongFirstDimension() {
        assertTensor(new int[]{1, 2, 3}, new float[]{7, 8, 9, 10, 11, 12}, cube().slice(0, 1, 2));
    }

    @Test
    void sliceCubeAlongMiddleDimension() {
        assertTensor(new int[]{2, 1, 3}, new float[]{1, 2, 3, 7, 8, 9}, cube().slice(1, 0, 1));
    }

    @Test
    void reshapeCubeToMatrix() {
        assertTensor(new int[]{4, 3},
                new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, cube().reshape(4, 3));
    }

    @Test
    void reshapeMatrixToHigherRank() {
        assertTensor(new int[]{1, 2, 3}, new float[]{1, 5, -3, 4, 2, 6}, matrix().reshape(1, 2, 3));
        assertTensor(new int[]{6, 1}, new float[]{1, 5, -3, 4, 2, 6}, matrix().reshape(6, 1));
    }

    // ---- matmul ----

    @Test
    void matmulRowVectorByColumnVectorIsDotProduct() {
        Tensor row = t(new float[]{1, 2, 3}, 1, 3);
        Tensor col = t(new float[]{4, 5, 6}, 3, 1);
        assertTensor(new int[]{1, 1}, new float[]{32}, row.matmul(col));
    }

    @Test
    void matmulColumnByRowIsOuterProduct() {
        Tensor col = t(new float[]{1, 2, 3}, 3, 1);
        Tensor row = t(new float[]{4, 5, 6}, 1, 3);
        assertTensor(new int[]{3, 3}, new float[]{4, 5, 6, 8, 10, 12, 12, 15, 18}, col.matmul(row));
    }

    @Test
    void matmulWithZeroMatrixIsZero() {
        assertTensor(new int[]{2, 2}, new float[]{0, 0, 0, 0}, matrix().matmul(t(new float[6], 3, 2)));
    }

    @Test
    void matmulIsAssociative() {
        Tensor a = t(new float[]{1, 2, 3, 4}, 2, 2);
        Tensor b = t(new float[]{0, 1, 1, 0}, 2, 2);
        Tensor c = t(new float[]{2, 0, 0, 2}, 2, 2);
        assertTensor(new int[]{2, 2}, new float[]{4, 2, 8, 6}, a.matmul(b).matmul(c));
        assertSameTensor(a.matmul(b).matmul(c), a.matmul(b.matmul(c)), EPS);
    }

    @Test
    void matmulTransposeOfProductEqualsProductOfTransposes() {
        Tensor a = randomTensor(1, 2, 3);
        Tensor b = randomTensor(2, 3, 4);
        assertSameTensor(a.matmul(b).transpose(0, 1),
                b.transpose(0, 1).matmul(a.transpose(0, 1)), 1e-3f);
    }

    @Test
    void matmulDoesNotMutateOperands() {
        Tensor a = matrix();
        Tensor b = t(new float[]{1, 2, 3, 4, 5, 6}, 3, 2);
        a.matmul(b);
        assertTensor(new int[]{2, 3}, new float[]{1, 5, -3, 4, 2, 6}, a);
        assertTensor(new int[]{3, 2}, new float[]{1, 2, 3, 4, 5, 6}, b);
    }

    // ---- unary ops over a range of inputs ----

    @ParameterizedTest
    @ValueSource(floats = {-3f, -1f, -0.5f, 0f, 0.5f, 1f, 2f, 10f})
    void expMatchesMathExp(float x) {
        assertEquals((float) Math.exp(x), t(new float[]{x}, 1).exp().get(0), 1e-4f);
    }

    @ParameterizedTest
    @ValueSource(floats = {-10f, -1f, -0.5f, 0f, 0.5f, 1f, 10f})
    void tanhMatchesMathTanh(float x) {
        assertEquals((float) Math.tanh(x), t(new float[]{x}, 1).tanh().get(0), EPS);
    }

    @ParameterizedTest
    @ValueSource(floats = {0f, 0.25f, 1f, 2f, 16f, 100f})
    void sqrtMatchesMathSqrt(float x) {
        assertEquals((float) Math.sqrt(x), t(new float[]{x}, 1).sqrt().get(0), EPS);
    }

    @Test
    void expSaturatesForLargeInputs() {
        Tensor r = t(new float[]{1000f, -1000f}, 2).exp();
        assertEquals(Float.POSITIVE_INFINITY, r.get(0));
        assertEquals(0f, r.get(1), EPS);
    }

    @Test
    void tanhSaturatesAtPlusAndMinusOne() {
        Tensor r = t(new float[]{50f, -50f}, 2).tanh();
        assertEquals(1f, r.get(0), EPS);
        assertEquals(-1f, r.get(1), EPS);
    }

    // ---- scalar ops over a range of inputs ----

    @ParameterizedTest
    @ValueSource(floats = {0f, 1f, -1f, 0.5f, 3f})
    void mulScalarMatchesPlainMultiplication(float k) {
        Tensor r = vector().mul(k);
        assertTensor(new int[]{4}, new float[]{1 * k, 2 * k, 3 * k, 4 * k}, r);
    }

    @ParameterizedTest
    @ValueSource(floats = {1f, -1f, 0.5f, 4f})
    void divScalarMatchesPlainDivision(float k) {
        Tensor r = vector().div(k);
        assertTensor(new int[]{4}, new float[]{1 / k, 2 / k, 3 / k, 4 / k}, r);
    }

    @Test
    void divScalarByZeroGivesInfinities() {
        Tensor r = t(new float[]{1, -1, 0}, 3).div(0f);
        assertEquals(Float.POSITIVE_INFINITY, r.get(0));
        assertEquals(Float.NEGATIVE_INFINITY, r.get(1));
        assertTrue(Float.isNaN(r.get(2)));
    }

    // ---- element-wise binary ops: identities and special values ----

    @Test
    void addingZeroTensorIsIdentity() {
        assertTensor(new int[]{2, 3}, new float[]{1, 5, -3, 4, 2, 6}, matrix().add(t(new float[6], 2, 3)));
    }

    @Test
    void subtractingSelfGivesZeros() {
        assertTensor(new int[]{2, 3}, new float[6], matrix().sub(matrix()));
    }

    @Test
    void dividingByOnesIsIdentity() {
        Tensor ones = t(new float[]{1, 1, 1, 1, 1, 1}, 2, 3);
        assertTensor(new int[]{2, 3}, new float[]{1, 5, -3, 4, 2, 6}, matrix().div(ones));
    }

    @Test
    void dividingTensorByItselfGivesOnes() {
        assertTensor(new int[]{4}, new float[]{1, 1, 1, 1}, vector().div(vector()));
    }

    @Test
    void divByTensorWithZeroElementGivesInfinity() {
        Tensor r = t(new float[]{1, 2}, 2).div(t(new float[]{0, 2}, 2));
        assertEquals(Float.POSITIVE_INFINITY, r.get(0));
        assertEquals(1f, r.get(1), EPS);
    }

    @Test
    void elementWiseOpsWorkOnRankThree() {
        Tensor r = cube().add(cube());
        assertArrayEquals(new int[]{2, 2, 3}, r.shape());
        assertEquals(24f, r.get(1, 1, 2), EPS);
    }

    @Test
    void nanPropagatesThroughAdd() {
        Tensor r = t(new float[]{Float.NaN, 1}, 2).add(t(new float[]{1, 1}, 2));
        assertTrue(Float.isNaN(r.get(0)));
        assertEquals(2f, r.get(1), EPS);
    }

    @Test
    void infinityPropagatesThroughSum() {
        Tensor r = t(new float[]{Float.POSITIVE_INFINITY, 1}, 2).sum(0);
        assertEquals(Float.POSITIVE_INFINITY, r.get(0));
    }

    // ---- algebraic properties on random data ----

    @Test
    void addIsCommutative() {
        Tensor a = randomTensor(10, 3, 4);
        Tensor b = randomTensor(11, 3, 4);
        assertSameTensor(a.add(b), b.add(a), EPS);
    }

    @Test
    void addThenSubRestoresOriginal() {
        Tensor a = randomTensor(12, 3, 4);
        Tensor b = randomTensor(13, 3, 4);
        assertSameTensor(a, a.add(b).sub(b), 1e-4f);
    }

    @Test
    void mulThenDivByScalarRestoresOriginal() {
        Tensor a = randomTensor(14, 4, 3);
        assertSameTensor(a, a.mul(2.5f).div(2.5f), 1e-4f);
    }

    @Test
    void transposeTwiceIsIdentityForRandomCube() {
        Tensor a = randomTensor(15, 2, 3, 4);
        assertSameTensor(a, a.transpose(0, 2).transpose(0, 2), EPS);
    }

    @Test
    void reshapeRoundTripIsIdentity() {
        Tensor a = randomTensor(16, 2, 3, 4);
        assertSameTensor(a, a.reshape(6, 4).reshape(2, 3, 4), EPS);
    }

    @Test
    void meanTimesCountEqualsSum() {
        Tensor a = randomTensor(17, 2, 3, 4);
        int[] shape = a.shape();
        for (int axis = 0; axis < 3; axis++) {
            assertSameTensor(a.sum(axis), a.mean(axis).mul(shape[axis]), 1e-3f);
        }
    }

    @Test
    void minIsAtMostMeanIsAtMostMax() {
        Tensor a = randomTensor(18, 2, 3, 4);
        for (int axis = 0; axis < 3; axis++) {
            float[] lo = flatten(a.min(axis));
            float[] mid = flatten(a.mean(axis));
            float[] hi = flatten(a.max(axis));
            for (int i = 0; i < lo.length; i++) {
                assertTrue(lo[i] <= mid[i] + EPS, "min <= mean at " + i);
                assertTrue(mid[i] <= hi[i] + EPS, "mean <= max at " + i);
            }
        }
    }

    @Test
    void varianceIsNeverNegative() {
        Tensor a = randomTensor(19, 2, 3, 4);
        for (int axis = 0; axis < 3; axis++) {
            for (float v : flatten(a.variance(axis))) {
                assertTrue(v >= -EPS, "variance was " + v);
            }
        }
    }

    @Test
    void varianceEqualsMeanOfSquaresMinusSquareOfMean() {
        Tensor a = randomTensor(20, 3, 4);
        // Var = E[x^2] - E[x]^2, computed by hand per row
        float[] x = flatten(a);
        float[] want = new float[3];
        for (int r = 0; r < 3; r++) {
            float mean = 0;
            float meanSq = 0;
            for (int c = 0; c < 4; c++) {
                mean += x[r * 4 + c] / 4;
                meanSq += x[r * 4 + c] * x[r * 4 + c] / 4;
            }
            want[r] = meanSq - mean * mean;
        }
        assertTensor(new int[]{3, 1}, want, a.variance(1));
    }

    // ---- dimensions of size one ----

    @Test
    void reducingARowVectorAlongSizeOneAxisKeepsValues() {
        Tensor row = t(new float[]{3, 1, 2}, 1, 3);
        assertTensor(new int[]{1, 3}, new float[]{3, 1, 2}, row.sum(0));
        assertTensor(new int[]{1, 3}, new float[]{3, 1, 2}, row.max(0));
        assertTensor(new int[]{1, 3}, new float[]{0, 0, 0}, row.variance(0));
    }

    @Test
    void reducingARowVectorAlongLongAxis() {
        Tensor row = t(new float[]{3, 1, 2}, 1, 3);
        assertTensor(new int[]{1, 1}, new float[]{6}, row.sum(1));
        assertTensor(new int[]{1, 1}, new float[]{3}, row.max(1));
        assertTensor(new int[]{1, 1}, new float[]{1}, row.min(1));
    }

    @Test
    void reducingAColumnVector() {
        Tensor col = t(new float[]{3, 1, 2}, 3, 1);
        assertTensor(new int[]{1, 1}, new float[]{6}, col.sum(0));
        assertTensor(new int[]{3, 1}, new float[]{3, 1, 2}, col.sum(1));
    }

    @Test
    void singleElementTensorBehavesLikeScalar() {
        Tensor one = t(new float[]{7}, 1, 1);
        assertEquals(7f, one.get(0, 0), EPS);
        assertEquals(14f, one.add(one).get(0, 0), EPS);
        assertEquals(49f, one.matmul(one).get(0, 0), EPS);
        assertTensor(new int[]{1, 1}, new float[]{7}, one.transpose(0, 1));
    }

    // ---- clone / toString / misc ----

    @Test
    void cloneOfCubeMatchesOriginal() {
        assertSameTensor(cube(), cube().clone(), EPS);
    }

    @Test
    void cloneHasIndependentShape() {
        Tensor a = matrix();
        Tensor c = a.clone();
        assertNotSame(a.shape(), c.shape());
        assertArrayEquals(new int[]{2, 3}, c.shape());
    }

    @Test
    void operationsReturnNewInstances() {
        Tensor a = matrix();
        assertNotSame(a, a.add(a));
        assertNotSame(a, a.mul(1f));
        assertNotSame(a, a.exp());
        assertNotSame(a, a.reshape(2, 3));
        assertNotSame(a, a.slice(0, 0, 2));
    }

    @Test
    void toStringIsNonEmpty() {
        String text = matrix().toString();
        assertNotNull(text);
        assertFalse(text.isEmpty());
    }

    @Test
    void reshapeInfersMinusOne() {
        assertTensor(new int[]{3, 2}, new float[]{1, 5, -3, 4, 2, 6}, matrix().reshape(-1, 2));
    }

    @Test
    void reshapeRejectsBadMinusOneUsage() {
        assertThrows(IllegalArgumentException.class, () -> matrix().reshape(-1, -1));
        assertThrows(IllegalArgumentException.class, () -> matrix().reshape(-1, 4));
        assertThrows(IllegalArgumentException.class, () -> matrix().reshape(0, 6));
    }

    @Test
    void constructorRejectsDataShapeMismatch() {
        assertThrows(IllegalArgumentException.class, () -> t(new float[5], 2, 3));
    }

    @Test
    void mutatingReturnedShapeDoesNotAffectTensor() {
        Tensor a = matrix();
        a.shape()[0] = 99;
        assertArrayEquals(new int[]{2, 3}, a.shape());
    }

    @Test
    void emptySliceThrows() {
        assertThrows(IllegalArgumentException.class, () -> matrix().slice(0, 1, 1));
    }
}
