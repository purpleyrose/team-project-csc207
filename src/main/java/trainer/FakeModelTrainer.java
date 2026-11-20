package trainer;

import entities.Corpus;
import use_case.train_model.ModelTrainer;
import use_case.train_model.TrainModelInputData;
import use_case.train_model.TrainingListener;

/**
 * A stand-in {@link ModelTrainer} that simulates training without touching a real model.
 *
 * <p>It ignores the corpus and model hyperparameters and instead produces a synthetic,
 * exponentially decaying loss curve, reporting each step to the {@link TrainingListener}
 * and pausing between steps. This makes it useful for developing and demoing the
 * train-model UI and presenter before the real trainer exists.</p>
 */
public class FakeModelTrainer implements ModelTrainer {

    /** How long to pause after each simulated step, in milliseconds. */
    private final long delayMillis;

    /**
     * Constructs a {@code FakeModelTrainer} that pauses between simulated steps.
     *
     * @param delayMillis the delay after each step, in milliseconds; {@code 0} runs
     *                    with no pause
     */
    public FakeModelTrainer(long delayMillis) {
        this.delayMillis = delayMillis;
    }

    /**
     * Simulates a training run and reports a synthetic loss at every step.
     *
     * <p>Steps run from {@code 0} to {@code inputData.getTrainingSteps()} inclusive, so the
     * listener is called {@code totalSteps + 1} times. The loss at each step is
     * {@code 1.3 + 3.1 * exp(-step / (0.15 * max(1, totalSteps)))}, starting near 4.4 and
     * levelling off toward 1.3; with zero training steps it reports a single step 0 with
     * a loss of 4.4. The thread sleeps for the configured delay after each
     * step. If the thread is interrupted, the interrupt flag is restored and the run
     * stops early, returning the most recent loss.</p>
     *
     * @param corpus           ignored by this implementation
     * @param inputData        supplies the number of training steps; other fields are ignored
     * @param trainingListener notified after each simulated step with the step, total
     *                         steps and loss
     * @return the loss from the last step that ran
     */
    @Override
    public float train(Corpus corpus, TrainModelInputData inputData, TrainingListener trainingListener) {
        int totalSteps = inputData.getTrainingSteps();
        // Guard the decay scale so zero steps doesn't compute 0 / 0.0 = NaN.
        double decayScale = 0.15 * Math.max(1, totalSteps);
        float loss = 0;
        for (int step = 0; step <= totalSteps; step++) {
            loss = (float) (1.3 + 3.1 * Math.exp(-step / decayScale));
            trainingListener.onStep(step, totalSteps, loss);
            try {
                Thread.sleep(delayMillis);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        return loss;

    }


}
