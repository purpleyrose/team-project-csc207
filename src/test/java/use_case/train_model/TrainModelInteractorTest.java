package use_case.train_model;

import entities.Corpus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link TrainModelInteractor}.
 *
 * <p>All collaborators (data access, presenter, trainer) are replaced with
 * hand-written fakes so the interactor's logic is tested in isolation.</p>
 */
class TrainModelInteractorTest {

    private static final float EPS = 1e-6f;

    private TrainModelInputData inputData;
    private RecordingPresenter presenter;

    @BeforeEach
    void setUp() {
        inputData = new TrainModelInputData(2, 4, 64, 32, 8, 3, 42, 0.001f);
        presenter = new RecordingPresenter();
    }

    @Test
    void corpusTooSmallPreparesFailureViewAndDoesNotTrain() {
        StubCorpus corpus = new StubCorpus(false);
        FakeTrainer trainer = new FakeTrainer(new float[]{1.0f}, 0.5f);
        TrainModelInteractor interactor =
                new TrainModelInteractor(() -> corpus, presenter, trainer);

        interactor.execute(inputData);

        assertEquals(List.of("Corpus is too small to train a model."), presenter.failures);
        assertTrue(presenter.progress.isEmpty());
        assertTrue(presenter.successes.isEmpty());
        assertEquals(0, trainer.callCount);
    }

    @Test
    void largeEnoughCorpusPreparesSuccessViewWithFinalLoss() {
        StubCorpus corpus = new StubCorpus(true);
        FakeTrainer trainer = new FakeTrainer(new float[0], 0.25f);
        TrainModelInteractor interactor =
                new TrainModelInteractor(() -> corpus, presenter, trainer);

        interactor.execute(inputData);

        assertTrue(presenter.failures.isEmpty());
        assertEquals(1, presenter.successes.size());
        TrainModelOutputData success = presenter.successes.get(0);
        assertEquals(inputData.getTrainingSteps(), success.getStep());
        assertEquals(inputData.getTrainingSteps(), success.getTotalSteps());
        assertEquals(0.25f, success.getLoss(), EPS);
    }

    @Test
    void trainerReceivesCorpusAndInputDataFromInteractor() {
        StubCorpus corpus = new StubCorpus(true);
        FakeTrainer trainer = new FakeTrainer(new float[0], 0.1f);
        TrainModelInteractor interactor =
                new TrainModelInteractor(() -> corpus, presenter, trainer);

        interactor.execute(inputData);

        assertEquals(1, trainer.callCount);
        assertSame(corpus, trainer.receivedCorpus);
        assertSame(inputData, trainer.receivedInputData);
        assertNotNull(trainer.receivedListener);
    }

    @Test
    void eachTrainerStepIsForwardedToPresenterAsProgress() {
        StubCorpus corpus = new StubCorpus(true);
        float[] stepLosses = {2.0f, 1.5f, 0.9f};
        FakeTrainer trainer = new FakeTrainer(stepLosses, 0.9f);
        TrainModelInteractor interactor =
                new TrainModelInteractor(() -> corpus, presenter, trainer);

        interactor.execute(inputData);

        assertEquals(stepLosses.length, presenter.progress.size());
        for (int i = 0; i < stepLosses.length; i++) {
            TrainModelOutputData p = presenter.progress.get(i);
            assertEquals(i + 1, p.getStep());
            assertEquals(stepLosses.length, p.getTotalSteps());
            assertEquals(stepLosses[i], p.getLoss(), EPS);
        }
    }

    @Test
    void progressViewsArePreparedBeforeSuccessView() {
        StubCorpus corpus = new StubCorpus(true);
        FakeTrainer trainer = new FakeTrainer(new float[]{3.0f, 2.0f}, 2.0f);
        TrainModelInteractor interactor =
                new TrainModelInteractor(() -> corpus, presenter, trainer);

        interactor.execute(inputData);

        assertEquals(List.of("progress", "progress", "success"), presenter.callOrder);
    }

    @Test
    void dataAccessIsQueriedForCorpusExactlyOnce() {
        StubCorpus corpus = new StubCorpus(true);
        int[] getCorpusCalls = {0};
        TrainModelDataAccessInterface dataAccess = () -> {
            getCorpusCalls[0]++;
            return corpus;
        };
        TrainModelInteractor interactor = new TrainModelInteractor(
                dataAccess, presenter, new FakeTrainer(new float[0], 0f));

        interactor.execute(inputData);

        assertEquals(1, getCorpusCalls[0]);
    }

    // ---------------------------------------------------------------------
    // Input validation (validate() is private, so it is tested through execute())
    // ---------------------------------------------------------------------

    /**
     * Runs the interactor on the given input with a trainable corpus and asserts that
     * it fails with {@code expectedMessage} before touching the corpus or the trainer.
     */
    private void assertRejected(TrainModelInputData input, String expectedMessage) {
        int[] getCorpusCalls = {0};
        StubCorpus corpus = new StubCorpus(true);
        FakeTrainer trainer = new FakeTrainer(new float[]{1.0f}, 0.5f);
        TrainModelInteractor interactor = new TrainModelInteractor(() -> {
            getCorpusCalls[0]++;
            return corpus;
        }, presenter, trainer);

        interactor.execute(input);

        assertEquals(List.of(expectedMessage), presenter.failures);
        assertTrue(presenter.progress.isEmpty());
        assertTrue(presenter.successes.isEmpty());
        assertEquals(0, trainer.callCount, "trainer should not run on invalid input");
        assertEquals(0, getCorpusCalls[0], "corpus should not be loaded on invalid input");
    }

    /** Runs the interactor and asserts the input passed validation and training succeeded. */
    private void assertAccepted(TrainModelInputData input) {
        TrainModelInteractor interactor = new TrainModelInteractor(
                () -> new StubCorpus(true), presenter, new FakeTrainer(new float[0], 0.5f));

        interactor.execute(input);

        assertTrue(presenter.failures.isEmpty(), "unexpected failure: " + presenter.failures);
        assertEquals(1, presenter.successes.size());
    }

    // Argument order: layers, heads, embeddingSize, contextLength, batchSize,
    // trainingSteps, seed, learningRate

    @Test
    void validateRejectsNegativeLayers() {
        assertRejected(new TrainModelInputData(-1, 4, 64, 32, 8, 3, 42, 0.001f),
                "Number of layers must be non-negative");
    }

    @Test
    void validateRejectsNegativeTrainingSteps() {
        assertRejected(new TrainModelInputData(2, 4, 64, 32, 8, -1, 42, 0.001f),
                "Number of training steps must be non-negative");
    }

    @Test
    void validateRejectsNegativeHeads() {
        assertRejected(new TrainModelInputData(2, -4, 64, 32, 8, 3, 42, 0.001f),
                "Number of heads must be positive");
    }

    @Test
    void validateRejectsNegativeEmbeddingSize() {
        assertRejected(new TrainModelInputData(2, 4, -64, 32, 8, 3, 42, 0.001f),
                "Embedding size must be non-negative");
    }

    @Test
    void validateRejectsNegativeContextLength() {
        assertRejected(new TrainModelInputData(2, 4, 64, -32, 8, 3, 42, 0.001f),
                "Context length must be non-negative");
    }

    @Test
    void validateRejectsNegativeBatchSize() {
        assertRejected(new TrainModelInputData(2, 4, 64, 32, -8, 3, 42, 0.001f),
                "Batch size must be non-negative");
    }

    @Test
    void validateRejectsZeroLearningRate() {
        assertRejected(new TrainModelInputData(2, 4, 64, 32, 8, 3, 42, 0f),
                "Learning rate must be positive");
    }

    @Test
    void validateRejectsNegativeLearningRate() {
        assertRejected(new TrainModelInputData(2, 4, 64, 32, 8, 3, 42, -0.01f),
                "Learning rate must be positive");
    }

    @Test
    void validateRejectsEmbeddingSizeNotDivisibleByHeads() {
        assertRejected(new TrainModelInputData(2, 3, 64, 32, 8, 3, 42, 0.001f),
                "Embedding size must be a multiple of heads");
    }

    @Test
    void validateRejectsZeroHeadsWithAFailureViewInsteadOfCrashing() {
        // Regression: zero heads used to reach embeddingSize % 0 and throw ArithmeticException.
        assertDoesNotThrow(() ->
                assertRejected(new TrainModelInputData(2, 0, 64, 32, 8, 3, 42, 0.001f),
                        "Number of heads must be positive"));
    }

    @Test
    void validateReportsOnlyTheFirstProblemFound() {
        // Layers is checked first, so its message wins over the bad learning rate.
        assertRejected(new TrainModelInputData(-1, 4, 64, 32, 8, 3, 42, -1f),
                "Number of layers must be non-negative");
    }

    @Test
    void validateAcceptsZeroForNonNegativeFields() {
        // Zero is allowed for everything except heads and learning rate.
        assertAccepted(new TrainModelInputData(0, 4, 0, 0, 0, 0, 42, 0.001f));
    }

    @Test
    void validateAcceptsNegativeSeed() {
        // Seed is not validated, so any int is fine.
        assertAccepted(new TrainModelInputData(2, 4, 64, 32, 8, 3, -42, 0.001f));
    }

    @Test
    void validateAcceptsEmbeddingSizeEqualToHeads() {
        assertAccepted(new TrainModelInputData(1, 8, 8, 16, 1, 1, 0, 0.5f));
    }

    // ---------------------------------------------------------------------
    // Test doubles
    // ---------------------------------------------------------------------

    /** Corpus whose trainability is fixed, avoiding the unimplemented entity logic. */
    private static class StubCorpus extends Corpus {
        private final boolean largeEnough;

        StubCorpus(boolean largeEnough) {
            super(Collections.emptyList());
            this.largeEnough = largeEnough;
        }

        @Override
        public boolean isLargeEnoughToTrain() {
            return largeEnough;
        }
    }

    /** Trainer that records its arguments and emits a scripted series of step losses. */
    private static class FakeTrainer implements ModelTrainer {
        private final float[] stepLosses;
        private final float finalLoss;
        int callCount;
        Corpus receivedCorpus;
        TrainModelInputData receivedInputData;
        TrainingListener receivedListener;

        FakeTrainer(float[] stepLosses, float finalLoss) {
            this.stepLosses = stepLosses;
            this.finalLoss = finalLoss;
        }

        @Override
        public float train(Corpus corpus, TrainModelInputData inputData, TrainingListener listener) {
            callCount++;
            receivedCorpus = corpus;
            receivedInputData = inputData;
            receivedListener = listener;
            for (int i = 0; i < stepLosses.length; i++) {
                listener.onStep(i + 1, stepLosses.length, stepLosses[i]);
            }
            return finalLoss;
        }
    }

    /** Presenter that records every view it is asked to prepare, in order. */
    private static class RecordingPresenter implements TrainModelOutputBoundary {
        final List<TrainModelOutputData> progress = new ArrayList<>();
        final List<TrainModelOutputData> successes = new ArrayList<>();
        final List<String> failures = new ArrayList<>();
        final List<String> callOrder = new ArrayList<>();

        @Override
        public void prepareProgressView(TrainModelOutputData outputData) {
            progress.add(outputData);
            callOrder.add("progress");
        }

        @Override
        public void prepareSuccessView(TrainModelOutputData outputData) {
            successes.add(outputData);
            callOrder.add("success");
        }

        @Override
        public void prepareFailureView(String errorMessage) {
            failures.add(errorMessage);
            callOrder.add("failure");
        }
    }
}
