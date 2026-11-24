package use_case.train_model;

/**
 * Output boundary for the train model use case.
 * <p>
 * Defines the methods used by the interactor to communicate training progress,
 * successful completion, or failure to the presenter layer.
 */
public interface TrainModelOutputBoundary {

    /**
     * Prepares the progress view while the model training process is ongoing.
     *
     * @param outputData the output data containing the current training progress information
     */
    void prepareProgressView(TrainModelOutputData outputData);

    /**
     * Prepares the success view after the model training process completes successfully.
     *
     * @param outputData the output data containing the results of the completed training process
     */
    void prepareSuccessView(TrainModelOutputData outputData);

    /**
     * Prepares the failure view when the model training process cannot be completed.
     *
     * @param errorMessage a message describing the reason for the training failure
     */
    void prepareFailureView(String errorMessage);
}