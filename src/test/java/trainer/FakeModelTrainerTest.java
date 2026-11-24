package trainer;

import entities.Corpus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import use_case.train_model.TrainModelInputData;
import use_case.train_model.TrainingListener;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Unit tests for {@link FakeModelTrainer}. */
class FakeModelTrainerTest {

    private static final float EPS = 1e-5f;

    /** Records every callback the trainer makes. */
    private static class RecordingListener implements TrainingListener {
        final List<Integer> steps = new ArrayList<>();
        final List<Integer> totals = new ArrayList<>();
        final List<Float> losses = new ArrayList<>();

        @Override
        public void onStep(int step, int totalSteps, float loss) {
            steps.add(step);
            totals.add(totalSteps);
            losses.add(loss);
        }
    }

    private static TrainModelInputData inputWithSteps(int steps) {
        return new TrainModelInputData(1, 1, 8, 8, 1, steps, 0, 0.01f);
    }

    private static Corpus emptyCorpus() {
        return new Corpus(Collections.emptyList());
    }

    /** The loss formula documented on {@link FakeModelTrainer#train}. */
    private static float expectedLoss(int step, int totalSteps) {
        return (float) (1.3 + 3.1 * Math.exp(-step / (0.15 * Math.max(1, totalSteps))));
    }

    @AfterEach
    void clearInterruptFlag() {
        // Make sure an interrupt test can't leak into other tests on this thread.
        Thread.interrupted();
    }

    @Test
    void zeroStepsReportsSingleFiniteLoss() {
        RecordingListener listener = new RecordingListener();
        float result = new FakeModelTrainer(0).train(emptyCorpus(), inputWithSteps(0), listener);

        assertEquals(List.of(0), listener.steps);
        assertEquals(List.of(0), listener.totals);
        assertFalse(Float.isNaN(result));
        assertEquals(4.4f, result, EPS);
    }

    @Test
    void reportsEveryStepFromZeroToTotalInclusive() {
        RecordingListener listener = new RecordingListener();
        new FakeModelTrainer(0).train(emptyCorpus(), inputWithSteps(10), listener);

        assertEquals(List.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), listener.steps);
    }

    @Test
    void everyCallbackReportsTheConfiguredTotalSteps() {
        RecordingListener listener = new RecordingListener();
        new FakeModelTrainer(0).train(emptyCorpus(), inputWithSteps(5), listener);

        assertEquals(Collections.nCopies(6, 5), listener.totals);
    }

    @Test
    void lossMatchesDocumentedFormulaAtEveryStep() {
        int total = 20;
        RecordingListener listener = new RecordingListener();
        new FakeModelTrainer(0).train(emptyCorpus(), inputWithSteps(total), listener);

        for (int i = 0; i < listener.steps.size(); i++) {
            assertEquals(expectedLoss(listener.steps.get(i), total), listener.losses.get(i), EPS,
                    "loss at step " + listener.steps.get(i));
        }
    }

    @Test
    void lossStartsNearFourPointFourAndDecreasesTowardOnePointThree() {
        RecordingListener listener = new RecordingListener();
        new FakeModelTrainer(0).train(emptyCorpus(), inputWithSteps(100), listener);

        assertEquals(4.4f, listener.losses.get(0), EPS);
        for (int i = 1; i < listener.losses.size(); i++) {
            assertTrue(listener.losses.get(i) < listener.losses.get(i - 1),
                    "loss should drop at step " + i);
        }
        float last = listener.losses.get(listener.losses.size() - 1);
        assertTrue(last > 1.3f && last < 1.4f, "final loss should level off near 1.3, was " + last);
    }

    @Test
    void returnsTheLastReportedLoss() {
        RecordingListener listener = new RecordingListener();
        float result = new FakeModelTrainer(0).train(emptyCorpus(), inputWithSteps(7), listener);

        assertEquals(listener.losses.get(listener.losses.size() - 1), result, EPS);
    }

    @Test
    void corpusIsIgnored() {
        RecordingListener withNull = new RecordingListener();
        RecordingListener withEmpty = new RecordingListener();

        float nullResult = new FakeModelTrainer(0).train(null, inputWithSteps(4), withNull);
        float emptyResult = new FakeModelTrainer(0).train(emptyCorpus(), inputWithSteps(4), withEmpty);

        assertEquals(withEmpty.losses, withNull.losses);
        assertEquals(emptyResult, nullResult, EPS);
    }

    @Test
    void pausesForTheConfiguredDelayAfterEachStep() {
        long delay = 20;
        int total = 2; // 3 steps, so at least 3 sleeps
        long start = System.nanoTime();
        new FakeModelTrainer(delay).train(emptyCorpus(), inputWithSteps(total), new RecordingListener());
        long elapsedMillis = (System.nanoTime() - start) / 1_000_000;

        assertTrue(elapsedMillis >= delay * (total + 1),
                "expected at least " + delay * (total + 1) + "ms, took " + elapsedMillis + "ms");
    }

    @Test
    void interruptStopsEarlyRestoresFlagAndReturnsLastLoss() {
        RecordingListener listener = new RecordingListener();
        Thread.currentThread().interrupt(); // the first sleep will throw immediately

        float result = new FakeModelTrainer(1000).train(emptyCorpus(), inputWithSteps(50), listener);

        assertEquals(List.of(0), listener.steps, "should stop after the first step");
        assertTrue(Thread.currentThread().isInterrupted(), "interrupt flag should be restored");
        assertEquals(listener.losses.get(0), result, EPS);
    }
}
