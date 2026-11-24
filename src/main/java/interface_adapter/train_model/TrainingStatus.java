package interface_adapter.train_model;

/** The phases of a training run, as shown on the training screen. */
public enum TrainingStatus {
    /** No training has started yet. */
    IDLE,
    /** Training is in progress. */
    RUNNING,
    /** Training finished successfully. */
    DONE,
    /** Training stopped because of an error. */
    FAILED
}