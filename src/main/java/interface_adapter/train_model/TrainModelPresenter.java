package interface_adapter.train_model;

import use_case.train_model.TrainModelOutputBoundary;
import use_case.train_model.TrainModelOutputData;

/**
 * Presenter responsible for updating the training view model in response to
 * output data produced by the train model use case.
 *
 * <p>This class implements {@link TrainModelOutputBoundary} and converts
 * training progress, success, and failure results into updates on the
 * {@link TrainingState}. After each update, it notifies observers by firing a
 * state change on the {@link TrainingViewModel}.</p>
 */
public class TrainModelPresenter implements TrainModelOutputBoundary {
    private final TrainingViewModel trainingViewModel;

    /**
     * Constructs a {@code TrainModelPresenter} with the given training view model.
     *
     * @param trainingViewModel the view model that stores and publishes training state updates
     */
    public TrainModelPresenter(TrainingViewModel trainingViewModel) {
        this.trainingViewModel = trainingViewModel;
    }

    /**
     * Updates the training state to reflect ongoing training progress.
     *
     * <p>The state is marked as {@link TrainingStatus#RUNNING}, and the latest
     * loss, current step, total number of steps, and loss history are updated
     * from the provided output data. Observers of the view model are then notified.</p>
     *
     * @param outputData the output data containing the current training progress
     */
    @Override
    public void prepareProgressView(TrainModelOutputData outputData) {
        TrainingState state = trainingViewModel.getState();
        if (state.getStatus() != TrainingStatus.RUNNING) {
            state.clearLossHistory();
            state.setErrorMessage("");
        }
        state.setStatus(TrainingStatus.RUNNING);
        copyProgress(outputData, state);
        state.addLoss(outputData.getLoss()); // Adds to the loss history, not the actual loss
        trainingViewModel.fireStateChanged();

    }

    private void copyProgress(TrainModelOutputData outputData, TrainingState state) {
        state.setLoss(outputData.getLoss());
        state.setSteps(outputData.getStep());
        state.setTotalSteps(outputData.getTotalSteps());
    }

    /**
     * Updates the training state to reflect successful completion of training.
     *
     * <p>The state is marked as {@link TrainingStatus#DONE}, and the final loss,
     * current step, and total number of steps are updated from the
     * provided output data. Observers of the view model are then notified.</p>
     *
     * @param outputData the output data containing the final training results
     */
    @Override
    public void prepareSuccessView(TrainModelOutputData outputData) {
        TrainingState state = trainingViewModel.getState();
        state.setStatus(TrainingStatus.DONE);
        copyProgress(outputData, state);
        trainingViewModel.fireStateChanged();
    }

    /**
     * Updates the training state to reflect a training failure.
     *
     * <p>The provided error message is stored in the training state, and
     * observers of the view model are notified of the change.</p>
     *
     * @param errorMessage the message describing the training error
     */
    @Override
    public void prepareFailureView(String errorMessage) {
        TrainingState state = trainingViewModel.getState();
        state.setStatus(TrainingStatus.FAILED);
        state.setErrorMessage(errorMessage);
        trainingViewModel.fireStateChanged();
    }
}
