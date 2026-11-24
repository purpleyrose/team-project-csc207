package interface_adapter.train_model;

import java.util.ArrayList;
import java.util.List;

/**
 * Holds the current state of a model training run, so the UI can display
 * progress, loss values, and any errors that occur.
 *
 * <p>This is a mutable state object used by the training interface adapter.
 * A new instance starts in the {@link TrainingStatus#IDLE} state with all counters
 * at zero.</p>
 */
public class TrainingState {

    /** The number of training steps completed so far. */
    private int steps;

    /** The total number of training steps planned for the run. */
    private int totalSteps;

    /** The most recently recorded loss value. */
    private float loss;

    /** The loss values recorded at each step, in order. */
    private List<Float> lossHistory;

    /** The current status of the training run. */
    private TrainingStatus status;

    /** A message describing the most recent error, or an empty string if there is none. */
    private String errorMessage;

    /**
     * Creates a new {@code TrainingState} with all counters set to zero,
     * an empty loss history, the status set to {@link TrainingStatus#IDLE}, and an
     * empty error message.
     */
    public TrainingState() {
        this.steps = 0;
        this.totalSteps = 0;
        this.loss = 0;
        this.lossHistory = new ArrayList<>();
        this.status = TrainingStatus.IDLE;
        this.errorMessage = "";
    }

    /**
     * Returns the history of loss values recorded during training.
     *
     * @return the list of loss values, in the order they were recorded
     */
    public List<Float> getLossHistory() {
        return lossHistory;
    }


    /**
     * Returns the number of training steps completed so far.
     *
     * @return the number of completed steps
     */
    public int getSteps() {
        return steps;
    }

    /**
     * Sets the number of training steps completed so far.
     *
     * @param steps the number of completed steps
     */
    public void setSteps(int steps) {
        this.steps = steps;
    }

    /**
     * Returns the total number of training steps planned for the run.
     *
     * @return the total number of steps
     */
    public int getTotalSteps() {
        return totalSteps;
    }

    /**
     * Sets the total number of training steps planned for the run.
     *
     * @param totalSteps the total number of steps
     */
    public void setTotalSteps(int totalSteps) {
        this.totalSteps = totalSteps;
    }

    /**
     * Returns the most recently recorded loss value.
     *
     * @return the current loss
     */
    public float getLoss() {
        return loss;
    }

    /**
     * Sets the most recently recorded loss value.
     *
     * @param loss the current loss
     */
    public void setLoss(float loss) {
        this.loss = loss;
    }

    /**
     * Returns the current status of the training run.
     *
     * @return the current {@link TrainingStatus}
     */
    public TrainingStatus getStatus() {
        return status;
    }

    /**
     * Sets the current status of the training run.
     *
     * @param status the new {@link TrainingStatus}
     */
    public void setStatus(TrainingStatus status) {
        this.status = status;
    }

    /**
     * Returns the message describing the most recent error.
     *
     * @return the error message, or an empty string if no error has occurred
     */
    public String getErrorMessage() {
        return errorMessage;
    }

    /**
     * Sets the message describing the most recent error.
     *
     * @param errorMessage the error message to store
     */
    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }


    /**
     * Add a new loss to the loss history.
     *
     * @param loss the new loss to add
     */
    public void addLoss(float loss) {
        this.lossHistory.add(loss);
    }

    /**
     * Clear the entire loss history.
     */
    public void clearLossHistory() {
        this.lossHistory.clear();
    }




}