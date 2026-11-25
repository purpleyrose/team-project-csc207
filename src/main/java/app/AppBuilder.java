package app;

import data_access.PlaceholderCorpusDataAccessObject;
import interface_adapter.train_model.TrainModelController;
import interface_adapter.train_model.TrainModelPresenter;
import interface_adapter.train_model.TrainingViewModel;
import trainer.FakeModelTrainer;
import use_case.train_model.ModelTrainer;
import use_case.train_model.TrainModelDataAccessInterface;
import use_case.train_model.TrainModelInputBoundary;
import use_case.train_model.TrainModelInteractor;
import use_case.train_model.TrainModelOutputBoundary;
import view.Theme;
import view.TrainingView;

import javax.swing.JFrame;
import javax.swing.JPanel;
import java.awt.CardLayout;
import java.awt.Dimension;

/**
 * Builds the application one view and one use case at a time. Each use case
 * owner adds an {@code addXView()} and {@code addXUseCase()} pair here, and
 * {@link Main} chains the calls.
 *
 * <p>Add a view before its use case: the use case connects its controller to
 * the view.</p>
 */
public class AppBuilder {

    /** Milliseconds the fake trainer waits between steps, so runs are visible. */
    private static final long FAKE_STEP_DELAY_MS = 2;

    /** Holds every view; one is shown at a time. */
    private final CardLayout cardLayout = new CardLayout();
    /** The panel that the views are added to. */
    private final JPanel cardPanel = new JPanel(cardLayout);

    /** Where Train Model reads the corpus from. */
    private final TrainModelDataAccessInterface corpusDataAccess = new PlaceholderCorpusDataAccessObject();
    /** The trainer Train Model uses. Swap in the real trainer here. */
    private final ModelTrainer modelTrainer = new FakeModelTrainer(FAKE_STEP_DELAY_MS);

    /** The training view model, created by {@link #addTrainingView()}. */
    private TrainingViewModel trainingViewModel;
    /** The training view, created by {@link #addTrainingView()}. */
    private TrainingView trainingView;

    /** Creates an empty builder. */
    public AppBuilder() {
        cardPanel.setBackground(Theme.BACKGROUND);
    }

    /**
     * Adds the Train Model screen.
     *
     * @return this builder
     */
    public AppBuilder addTrainingView() {
        trainingViewModel = new TrainingViewModel();
        trainingView = new TrainingView(trainingViewModel);
        cardPanel.add(trainingView, trainingView.getViewName());
        return this;
    }

    /**
     * Adds the Train Model use case and connects it to the training view.
     *
     * @return this builder
     * @throws IllegalStateException if {@link #addTrainingView()} has not been called
     */
    public AppBuilder addTrainModelUseCase() {
        if (trainingView == null) {
            throw new IllegalStateException("Call addTrainingView() before addTrainModelUseCase()");
        }
        TrainModelOutputBoundary presenter = new TrainModelPresenter(trainingViewModel);
        TrainModelInteractor interactor = new TrainModelInteractor(corpusDataAccess, presenter, modelTrainer);

        // TEMPORARY until #24: run training off the Swing thread so the window stays responsive.
        TrainModelInputBoundary offSwingThread = inputData ->
                new Thread(() -> interactor.execute(inputData), "training").start();

        trainingView.setTrainModelController(new TrainModelController(offSwingThread));
        return this;
    }

    /**
     * Creates the window, showing the training view first.
     *
     * @return the window, not yet visible
     */
    public JFrame build() {
        JFrame frame = new JFrame("Text Generation Studio");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setContentPane(cardPanel);
        frame.setMinimumSize(new Dimension(1000, 680));
        if (trainingView != null) {
            cardLayout.show(cardPanel, trainingView.getViewName());
        }
        frame.pack();
        frame.setLocationRelativeTo(null);
        return frame;
    }
}