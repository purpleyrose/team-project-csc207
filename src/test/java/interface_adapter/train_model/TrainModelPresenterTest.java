package interface_adapter.train_model;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import use_case.train_model.TrainModelOutputData;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link TrainModelPresenter}.
 *
 * <p>Uses a real {@link TrainingViewModel} and counts {@code stateChanged()}
 * notifications with a simple listener.</p>
 */
class TrainModelPresenterTest {

    private static final float EPS = 1e-6f;

    private TrainingViewModel viewModel;
    private TrainModelPresenter presenter;
    private int notifications;

    @BeforeEach
    void setUp() {
        viewModel = new TrainingViewModel();
        presenter = new TrainModelPresenter(viewModel);
        notifications = 0;
        viewModel.addListener(() -> notifications++);
    }

    private TrainingState state() {
        return viewModel.getState();
    }

    // ---------------------------------------------------------------------
    // Progress
    // ---------------------------------------------------------------------

    @Test
    void progressSetsRunningAndCopiesStepTotalAndLoss() {
        presenter.prepareProgressView(new TrainModelOutputData(3, 10, 2.5f));

        assertEquals(TrainingStatus.RUNNING, state().getStatus());
        assertEquals(3, state().getSteps());
        assertEquals(10, state().getTotalSteps());
        assertEquals(2.5f, state().getLoss(), EPS);
    }

    @Test
    void progressAppendsEachLossToHistoryInOrder() {
        presenter.prepareProgressView(new TrainModelOutputData(1, 3, 3.0f));
        presenter.prepareProgressView(new TrainModelOutputData(2, 3, 2.0f));
        presenter.prepareProgressView(new TrainModelOutputData(3, 3, 1.0f));

        assertEquals(List.of(3.0f, 2.0f, 1.0f), state().getLossHistory());
        assertEquals(1.0f, state().getLoss(), EPS);
        assertEquals(3, state().getSteps());
    }

    @Test
    void progressNotifiesListenersOncePerCall() {
        presenter.prepareProgressView(new TrainModelOutputData(1, 2, 1.0f));
        presenter.prepareProgressView(new TrainModelOutputData(2, 2, 0.5f));

        assertEquals(2, notifications);
    }

    @Test
    void firstProgressOfNewRunClearsOldHistoryAndError() {
        TrainingState s = state();
        s.setStatus(TrainingStatus.FAILED);
        s.setErrorMessage("old error");
        s.addLoss(9.9f);
        s.addLoss(8.8f);

        presenter.prepareProgressView(new TrainModelOutputData(1, 5, 4.0f));

        assertEquals(TrainingStatus.RUNNING, s.getStatus());
        assertEquals("", s.getErrorMessage());
        assertEquals(List.of(4.0f), s.getLossHistory());
    }

    @Test
    void progressAfterDoneStartsAFreshHistory() {
        presenter.prepareProgressView(new TrainModelOutputData(1, 1, 2.0f));
        presenter.prepareSuccessView(new TrainModelOutputData(1, 1, 2.0f));

        presenter.prepareProgressView(new TrainModelOutputData(1, 4, 5.0f));

        assertEquals(TrainingStatus.RUNNING, state().getStatus());
        assertEquals(List.of(5.0f), state().getLossHistory());
        assertEquals(4, state().getTotalSteps());
    }

    @Test
    void progressWhileAlreadyRunningKeepsHistory() {
        TrainingState s = state();
        s.setStatus(TrainingStatus.RUNNING);
        s.addLoss(7.0f);

        presenter.prepareProgressView(new TrainModelOutputData(2, 5, 6.0f));

        assertEquals(List.of(7.0f, 6.0f), s.getLossHistory());
    }

    // ---------------------------------------------------------------------
    // Success
    // ---------------------------------------------------------------------

    @Test
    void successSetsDoneAndCopiesFinalValues() {
        presenter.prepareSuccessView(new TrainModelOutputData(10, 10, 0.75f));

        assertEquals(TrainingStatus.DONE, state().getStatus());
        assertEquals(10, state().getSteps());
        assertEquals(10, state().getTotalSteps());
        assertEquals(0.75f, state().getLoss(), EPS);
        assertEquals(1, notifications);
    }

    @Test
    void successKeepsHistoryAndDoesNotAddTheFinalLossAgain() {
        presenter.prepareProgressView(new TrainModelOutputData(1, 2, 2.0f));
        presenter.prepareProgressView(new TrainModelOutputData(2, 2, 1.0f));

        presenter.prepareSuccessView(new TrainModelOutputData(2, 2, 1.0f));

        assertEquals(List.of(2.0f, 1.0f), state().getLossHistory());
        assertEquals(3, notifications);
    }

    // ---------------------------------------------------------------------
    // Failure
    // ---------------------------------------------------------------------

    @Test
    void failureSetsFailedAndStoresMessage() {
        presenter.prepareFailureView("Corpus is too small to train a model.");

        assertEquals(TrainingStatus.FAILED, state().getStatus());
        assertEquals("Corpus is too small to train a model.", state().getErrorMessage());
        assertEquals(1, notifications);
    }

    @Test
    void failureMidRunKeepsProgressSoFar() {
        presenter.prepareProgressView(new TrainModelOutputData(4, 10, 1.5f));

        presenter.prepareFailureView("boom");

        assertEquals(TrainingStatus.FAILED, state().getStatus());
        assertEquals(4, state().getSteps());
        assertEquals(10, state().getTotalSteps());
        assertEquals(1.5f, state().getLoss(), EPS);
        assertEquals(List.of(1.5f), state().getLossHistory());
    }

    // ---------------------------------------------------------------------
    // View model wiring
    // ---------------------------------------------------------------------

    @Test
    void presenterUpdatesWhateverStateTheViewModelCurrentlyHolds() {
        TrainingState replacement = new TrainingState();
        viewModel.setState(replacement);

        presenter.prepareProgressView(new TrainModelOutputData(1, 2, 3.0f));

        assertSame(replacement, viewModel.getState());
        assertEquals(TrainingStatus.RUNNING, replacement.getStatus());
    }

    @Test
    void stateIsAlreadyUpdatedWhenListenersAreNotified() {
        TrainingStatus[] seen = new TrainingStatus[1];
        viewModel.addListener(() -> seen[0] = viewModel.getState().getStatus());

        presenter.prepareFailureView("nope");

        assertEquals(TrainingStatus.FAILED, seen[0]);
    }
}
