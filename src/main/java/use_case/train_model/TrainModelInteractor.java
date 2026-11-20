package use_case.train_model;

import entities.Corpus;

/**
 * Interactor for the "Train Model" use case.
 *
 * <p>This class holds the application-specific business logic for training a model.
 * It retrieves the {@link Corpus} from the data access layer, checks that the corpus is
 * large enough to train on, delegates the training itself to a {@link ModelTrainer},
 * and reports progress, success or failure to the {@link TrainModelOutputBoundary}
 * (the presenter).</p>
 *
 * <p>It depends only on abstractions (the data access interface, the output boundary and
 * the trainer), which keeps it independent of frameworks, UI and persistence details.</p>
 */
public class TrainModelInteractor implements TrainModelInputBoundary {

    /** Data access object used to retrieve the corpus the model is trained on. */
    private final TrainModelDataAccessInterface dataAccess;

    /** Presenter notified of training progress, success and failure. */
    private final TrainModelOutputBoundary presenter;

    /** Component that performs the actual model training. */
    private final ModelTrainer trainer;

    /**
     * Constructs a {@code TrainModelInteractor} with its required collaborators.
     *
     * @param dataAccess the data access object used to obtain the {@link Corpus} for training
     * @param presenter  the output boundary used to present progress, success and failure views
     * @param trainer    the trainer that runs the training loop over the corpus
     */
    public TrainModelInteractor(TrainModelDataAccessInterface dataAccess,
                                TrainModelOutputBoundary presenter, ModelTrainer trainer) {
        this.dataAccess = dataAccess;
        this.presenter = presenter;
        this.trainer = trainer;
    }

    /**
     * Executes the train-model use case.
     *
     * <p>The workflow is:</p>
     * <ol>
     *   <li>Fetch the corpus from the data access layer.</li>
     *   <li>Verify the corpus is large enough via {@link Corpus#isLargeEnoughToTrain()};
     *       if not, a failure view is prepared with an explanatory message.</li>
     *   <li>Train the model using the {@link ModelTrainer}. After each step, the trainer's
     *       callback triggers {@link TrainModelOutputBoundary#prepareProgressView} with
     *       the current step, total steps and loss.</li>
     *   <li>When training completes, prepare the success view containing the total number
     *       of steps and the final loss.</li>
     * </ol>
     *
     * @param inputData the training parameters (e.g. the number of training steps)
     *                  supplied by the caller
     */
    @Override
    public void execute(TrainModelInputData inputData) {
        Corpus corpus = dataAccess.getCorpus();
        if (!corpus.isLargeEnoughToTrain()) {
            presenter.prepareFailureView("Corpus is too small to train a model.");
            return;
        }

        float finalLoss = trainer.train(
                corpus,
                inputData,
                (step, totalSteps, loss) ->
                        presenter.prepareProgressView(new TrainModelOutputData(step, totalSteps, loss))
        );

        TrainModelOutputData outputData = new TrainModelOutputData(
                inputData.getTrainingSteps(),
                inputData.getTrainingSteps(),
                finalLoss);
        presenter.prepareSuccessView(outputData);
    }
}
