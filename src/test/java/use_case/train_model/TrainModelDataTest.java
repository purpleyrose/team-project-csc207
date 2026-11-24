package use_case.train_model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Unit tests for the train model use case's data transfer objects:
 * {@link TrainModelInputData} and {@link TrainModelOutputData}.
 */
class TrainModelDataTest {

    private static final float EPS = 1e-6f;

    @Test
    void inputDataGettersReturnConstructorValues() {
        TrainModelInputData input =
                new TrainModelInputData(6, 8, 256, 128, 16, 1000, 7, 0.0003f);

        assertEquals(6, input.getLayers());
        assertEquals(8, input.getHeads());
        assertEquals(256, input.getEmbeddingSize());
        assertEquals(128, input.getContextLength());
        assertEquals(16, input.getBatchSize());
        assertEquals(1000, input.getTrainingSteps());
        assertEquals(7, input.getSeed());
        assertEquals(0.0003f, input.getLearningRate(), EPS);
    }

    @Test
    void outputDataGettersReturnConstructorValues() {
        TrainModelOutputData output = new TrainModelOutputData(42, 100, 1.234f);

        assertEquals(42, output.getStep());
        assertEquals(100, output.getTotalSteps());
        assertEquals(1.234f, output.getLoss(), EPS);
    }
}
