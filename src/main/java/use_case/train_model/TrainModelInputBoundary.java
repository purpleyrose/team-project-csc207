package use_case.train_model;

/**
 * Defines the input boundary for the train model use case.
 *
 * <p>This interface represents the entry point through which external layers,
 * such as controllers, can request that a model training operation be performed.
 * Implementations are responsible for handling the provided input data and
 * coordinating the training process.</p>
 */
public interface TrainModelInputBoundary {

    /**
     * Executes the train model use case using the provided input data.
     *
     * @param inputData the data required to train the model
     */
    void execute(TrainModelInputData inputData);
}