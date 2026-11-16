package torch;

/**
 * A {@link TensorFactory} that creates {@link NaiveTensor}s.
 */
public class NaiveTensorFactory implements TensorFactory {

    /**
     * Creates a factory for {@link NaiveTensor}s.
     */
    public NaiveTensorFactory() {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor create(float[] data, int[] shape) {
        return new NaiveTensor(data, shape);
    }
}
