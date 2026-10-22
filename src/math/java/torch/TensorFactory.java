package torch;

public interface TensorFactory {

    Tensor create(float[] data, int[] shape);
}
