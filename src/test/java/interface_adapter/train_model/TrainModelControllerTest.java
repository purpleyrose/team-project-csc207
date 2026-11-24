package interface_adapter.train_model;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import use_case.train_model.TrainModelInputBoundary;
import use_case.train_model.TrainModelInputData;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link TrainModelController}.
 *
 * <p>The interactor is replaced with a fake input boundary that records every
 * {@link TrainModelInputData} it receives.</p>
 */
class TrainModelControllerTest {

    private static final float EPS = 1e-6f;

    /** Input boundary that records each request instead of training anything. */
    private static class RecordingInteractor implements TrainModelInputBoundary {
        final List<TrainModelInputData> received = new ArrayList<>();

        @Override
        public void execute(TrainModelInputData inputData) {
            received.add(inputData);
        }
    }

    private RecordingInteractor interactor;
    private TrainModelController controller;

    @BeforeEach
    void setUp() {
        interactor = new RecordingInteractor();
        controller = new TrainModelController(interactor);
    }

    @Test
    void executeCallsInteractorExactlyOnce() {
        controller.execute(2, 4, 64, 32, 8, 100, 42, 0.001f);

        assertEquals(1, interactor.received.size());
        assertNotNull(interactor.received.get(0));
    }

    @Test
    void everyArgumentEndsUpInTheMatchingInputDataField() {
        // All values are different so a swapped pair of arguments would be caught.
        controller.execute(3, 5, 7, 11, 13, 17, 19, 0.023f);

        TrainModelInputData data = interactor.received.get(0);
        assertEquals(3, data.getLayers());
        assertEquals(5, data.getHeads());
        assertEquals(7, data.getEmbeddingSize());
        assertEquals(11, data.getContextLength());
        assertEquals(13, data.getBatchSize());
        assertEquals(17, data.getTrainingSteps());
        assertEquals(19, data.getSeed());
        assertEquals(0.023f, data.getLearningRate(), EPS);
    }

    @Test
    void eachCallSendsItsOwnInputData() {
        controller.execute(1, 1, 16, 8, 2, 10, 0, 0.1f);
        controller.execute(6, 8, 256, 128, 32, 5000, 99, 0.0003f);

        assertEquals(2, interactor.received.size());
        TrainModelInputData first = interactor.received.get(0);
        TrainModelInputData second = interactor.received.get(1);
        assertNotSame(first, second);
        assertEquals(10, first.getTrainingSteps());
        assertEquals(5000, second.getTrainingSteps());
        assertEquals(99, second.getSeed());
    }

    @Test
    void valuesArePassedThroughWithoutValidation() {
        // The controller only packages input; checking values is not its job.
        controller.execute(0, -1, 0, 0, 0, 0, -7, -0.5f);

        TrainModelInputData data = interactor.received.get(0);
        assertEquals(0, data.getLayers());
        assertEquals(-1, data.getHeads());
        assertEquals(0, data.getTrainingSteps());
        assertEquals(-7, data.getSeed());
        assertEquals(-0.5f, data.getLearningRate(), EPS);
    }

    @Test
    void noInteractorCallBeforeExecute() {
        assertTrue(interactor.received.isEmpty());
    }
}
