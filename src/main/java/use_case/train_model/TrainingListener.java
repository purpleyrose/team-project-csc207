package use_case.train_model;

/**
 * Listener interface for receiving progress updates during a model training process.
 *
 * <p>Implementations of this interface can be used to observe training progress,
 * including the current step, total number of steps, and the loss value reported
 * at each step.</p>
 */
public interface TrainingListener {

    /**
     * Called when a training step has completed or when progress is reported.
     *
     * @param step the current training step
     * @param totalSteps the total number of training steps expected
     * @param loss the loss value measured or reported at the current step
     */
    void onStep(int step, int totalSteps, float loss);
}
