package torch;

/**
 * Runs the shared {@link AbstractTensorTest} suite against {@link NaiveTensor}.
 */
class NaiveTensorTest extends AbstractTensorTest {

    @Override
    protected TensorFactory factory() {
        return new NaiveTensorFactory();
    }
}
