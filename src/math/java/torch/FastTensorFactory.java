package torch;

/**
 * A {@link TensorFactory} that creates {@link FastTensor}s.
 */
public class FastTensorFactory implements TensorFactory {

    /**
     * Creates a factory for {@link FastTensor}s.
     */
    public FastTensorFactory() {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor create(float[] data, int[] shape) {
        return new FastTensor(data, shape);
    }
}
