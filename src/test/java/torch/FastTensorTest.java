package torch;

/**
 * Runs the shared {@link AbstractTensorTest} suite against {@link FastTensor}.
 */
class FastTensorTest extends AbstractTensorTest {

    @Override
    protected TensorFactory factory() {
        return new FastTensorFactory();
    }
}
