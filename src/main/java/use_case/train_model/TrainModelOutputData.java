package use_case.train_model;

/**
 * Output data representing the progress and current training loss of a model
 * training operation.
 *
 * <p> Stores the current training step, the total number of planned training steps,
 * and the loss value reported for the current step. </p>
 */
public class TrainModelOutputData {

    private final int step;

    private final int totalSteps;

    private final float loss;

    public TrainModelOutputData(int step, int totalSteps, float loss) {
        this.step = step;
        this.totalSteps = totalSteps;
        this.loss = loss;
    }

    public int getStep() {
        return step;
    }

    public int getTotalSteps() {
        return totalSteps;
    }

    public float getLoss() {
        return loss;
    }
}
