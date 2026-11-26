package use_case.generate_text;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import torch.NaiveTensor;
import torch.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link GenerateTextInteractor}.
 *
 * <p>A two-token model and a deterministic sampler make each prediction
 * repeatable. The presenter records each decoded chunk and the callback order.</p>
 */
class GenerateTextInteractorTest {

    private RecordingPresenter presenter;
    private List<Float> temperatures;

    @BeforeEach
    void setUp() {
        presenter = new RecordingPresenter();
        temperatures = new ArrayList<>();
    }

    // --- Generation and streaming ---

    @Test
    void streamsTwoNewTokensInOrderUsingThePromptSuffix() {
        PredictableModel model = new PredictableModel(2, 1, 0, 1);

        executeWith(model, new GenerateTextInputData("My prompt ", 2, 1.5f));

        // The prompt encodes to [1, 0, 1]. Only its final two IDs fit in the first prediction.
        assertEquals(List.of("My prompt "), model.encodedPrompts);
        assertEquals(2, model.contexts.size());
        assertArrayEquals(new int[]{0, 1}, model.contexts.get(0));
        assertArrayEquals(new int[]{1, 0}, model.contexts.get(1));
        assertEquals(List.of(1.5f, 1.5f), temperatures);
        assertEquals(List.of(" thee", "\n"), presenter.progress);
        assertEquals(List.of("progress", "progress", "success"), presenter.callOrder);
    }

    @Test
    void shortPromptGrowsUntilTheContextIsFull() {
        PredictableModel model = new PredictableModel(3, 0);

        executeWith(model, new GenerateTextInputData("A", 3, 1f));

        assertEquals(3, model.contexts.size());
        assertArrayEquals(new int[]{0}, model.contexts.get(0));
        assertArrayEquals(new int[]{0, 1}, model.contexts.get(1));
        assertArrayEquals(new int[]{0, 1, 0}, model.contexts.get(2));
        assertEquals(List.of("\n", " thee", "\n"), presenter.progress);
        assertEquals(List.of("progress", "progress", "progress", "success"), presenter.callOrder);
    }

    @Test
    void oneTokenContextUsesOnlyTheMostRecentToken() {
        PredictableModel model = new PredictableModel(1, 0, 1);

        executeWith(model, new GenerateTextInputData("AB", 2, 1f));

        assertEquals(2, model.contexts.size());
        assertArrayEquals(new int[]{1}, model.contexts.get(0));
        assertArrayEquals(new int[]{0}, model.contexts.get(1));
    }

    // --- Model and sampler failures ---

    @Test
    void missingModelReportsOnlyFailure() {
        GenerateTextInteractor interactor =
                new GenerateTextInteractor(Optional::empty, presenter, this::sample);

        interactor.execute(new GenerateTextInputData("Hello", 2, 1f));

        assertOnlyFailureAbout("model");
        assertTrue(temperatures.isEmpty());
    }

    @Test
    void unusableContextLengthFailsBeforePrediction() {
        PredictableModel model = new PredictableModel(0, 1);

        executeWith(model, new GenerateTextInputData("A", 2, 1f));

        assertOnlyFailureAbout("context length");
        assertTrue(model.contexts.isEmpty());
    }

    @Test
    void promptThatEncodesToNoTokensFailsBeforePrediction() {
        PredictableModel model = new PredictableModel(2);

        executeWith(model, new GenerateTextInputData("A", 2, 1f));

        assertOnlyFailureAbout("token");
        assertTrue(model.contexts.isEmpty());
    }

    @Test
    void samplerFailureAfterOneTokenEndsTheStream() {
        PredictableModel model = new PredictableModel(2, 1);
        // The model favors token 0 on the first call and token 1 on the second.
        TokenSampler failingSampler = (scores, temperature) -> {
            if (scores.get(1) > scores.get(0)) {
                throw new GenerationException("Sampling failed.");
            }
            return 0;
        };
        GenerateTextInteractor interactor =
                new GenerateTextInteractor(() -> Optional.of(model), presenter, failingSampler);

        interactor.execute(new GenerateTextInputData("A", 3, 1f));

        assertEquals(List.of(" thee"), presenter.progress);
        assertEquals(List.of("Sampling failed."), presenter.failures);
        assertEquals(List.of("progress", "failure"), presenter.callOrder);
        assertEquals(2, model.contexts.size());
    }

    // --- Input validation through execute() ---

    @ParameterizedTest
    @MethodSource("invalidRequests")
    void invalidSettingsFailBeforeModelLookup(GenerateTextInputData request, String reason) {
        GenerateTextDataAccessInterface dataAccess = () -> fail("Unexpected model lookup");
        GenerateTextInteractor interactor =
                new GenerateTextInteractor(dataAccess, presenter, this::sample);

        interactor.execute(request);

        assertOnlyFailureAbout(reason);
        assertTrue(temperatures.isEmpty());
    }

    private static Stream<Arguments> invalidRequests() {
        return Stream.of(
                Arguments.of(null, "prompt"),
                Arguments.of(new GenerateTextInputData(null, 1, 1f), "prompt"),
                Arguments.of(new GenerateTextInputData("", 1, 1f), "prompt"),
                Arguments.of(new GenerateTextInputData("A", 0, 1f), "token"),
                Arguments.of(new GenerateTextInputData("A", -1, 1f), "token"),
                Arguments.of(new GenerateTextInputData("A", 1, 0f), "temperature"),
                Arguments.of(new GenerateTextInputData("A", 1, -1f), "temperature"),
                Arguments.of(new GenerateTextInputData("A", 1, Float.NaN), "temperature"),
                Arguments.of(new GenerateTextInputData("A", 1, Float.POSITIVE_INFINITY), "temperature"),
                Arguments.of(new GenerateTextInputData("A", 1, Float.NEGATIVE_INFINITY), "temperature")
        );
    }

    // --- Test helpers ---

    private void executeWith(ModelGeneration model, GenerateTextInputData request) {
        GenerateTextInputBoundary interactor =
                new GenerateTextInteractor(() -> Optional.of(model), presenter, this::sample);
        interactor.execute(request);
    }

    /** Selects the higher-scoring token and records the temperature passed by the interactor. */
    private int sample(Tensor scores, float temperature) {
        temperatures.add(temperature);
        return scores.get(0) > scores.get(1) ? 0 : 1;
    }

    /** Checks that failure was the only callback and its message identifies the problem. */
    private void assertOnlyFailureAbout(String reason) {
        assertEquals(List.of("failure"), presenter.callOrder);
        assertEquals(1, presenter.failures.size());
        String message = presenter.failures.get(0);
        assertTrue(message.contains(reason), message);
    }

    // --- Test doubles ---

    /** Encodes to fixed IDs and gives the highest score to the opposite of the last token. */
    private static class PredictableModel implements ModelGeneration {
        private final int contextLength;
        private final int[] promptTokens;
        private final List<String> encodedPrompts = new ArrayList<>();
        private final List<int[]> contexts = new ArrayList<>();

        PredictableModel(int contextLength, int... promptTokens) {
            this.contextLength = contextLength;
            this.promptTokens = promptTokens;
        }

        @Override
        public int getContextLength() {
            return contextLength;
        }

        @Override
        public int[] encode(String prompt) {
            encodedPrompts.add(prompt);
            return promptTokens.clone();
        }

        @Override
        public Tensor nextTokenScore(int[] context) {
            // Keep the IDs seen at prediction time, even if the caller later reuses the array.
            contexts.add(context.clone());
            int lastToken = context[context.length - 1];
            float[] scores = lastToken == 0 ? new float[]{0f, 1f} : new float[]{1f, 0f};
            return new NaiveTensor(scores, new int[]{2});
        }

        @Override
        public String decode(int token) {
            return List.of(" thee", "\n").get(token);
        }
    }

    /** Records the order and content of every output-boundary callback. */
    private static class RecordingPresenter implements GenerateTextOutputBoundary {
        final List<String> progress = new ArrayList<>();
        final List<String> failures = new ArrayList<>();
        final List<String> callOrder = new ArrayList<>();

        @Override
        public void prepareProgressView(String text) {
            progress.add(text);
            callOrder.add("progress");
        }

        @Override
        public void prepareSuccessView() {
            callOrder.add("success");
        }

        @Override
        public void prepareFailureView(String message) {
            failures.add(message);
            callOrder.add("failure");
        }
    }
}
