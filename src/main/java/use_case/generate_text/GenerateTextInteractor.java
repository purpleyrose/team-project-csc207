package use_case.generate_text;

import java.util.Arrays;
import torch.Tensor;

/**
 * Runs the Generate Text use case for one prompt at a time.
 *
 * <p>The active {@link ModelGeneration} encodes the prompt and predicts scores for
 * the next token. The {@link TokenSampler} selects a token from those scores at the
 * requested temperature. The interactor sends each decoded piece of text to the
 * {@link GenerateTextOutputBoundary} in generation order.</p>
 */
public class GenerateTextInteractor implements GenerateTextInputBoundary{

    /** Source of the active model and its matching tokenizer. */
    private final GenerateTextDataAccessInterface dataAccess;

    /** Presenter that receives decoded text and the final result of each request. */
    private final GenerateTextOutputBoundary presenter;

    /** Sampler that selects a token ID from the model's next-token scores. */
    private final TokenSampler sampler;

    /**
     * Creates the interactor with its model source, presenter, and token sampler.
     *
     * @param dataAccess provides the active model for each request
     * @param presenter receives generated text, completion, and failure messages
     * @param sampler selects each new token from the model's scores
     */
    public GenerateTextInteractor(GenerateTextDataAccessInterface dataAccess,
                                  GenerateTextOutputBoundary presenter,
                                  TokenSampler sampler) {
        this.dataAccess = dataAccess;
        this.presenter = presenter;
        this.sampler = sampler;
    }

    /**
     * Executes the interactor with the given input data.
     * Validates the request, looks up the active model once, and streams its continuation.
     *
     * <p>Invalid settings or a missing model produce a single failure callback before
     * any text is sent. If generation fails after some text has been sent, the failure
     * callback ends the stream. Success is reported after the last decoded token.</p>
     *
     * @param inputData the prompt, maximum number of new tokens, and sampling temperature
     */
    @Override
    public void execute(GenerateTextInputData inputData){
        try {
            validate(inputData);

            ModelGeneration model = dataAccess.getActiveModel().orElseThrow(
                    () -> new GenerationException("Train or load a model before generating text.")
            );

            generate(inputData, model);
        } catch (GenerationException failure) {
            presenter.prepareFailureView(failure.getMessage());
            return;
        }

        presenter.prepareSuccessView();
    }

    /**
     * Checks the request before accessing the model. Whitespace in the prompt is
     * preserved because spaces and line breaks can be meaningful to the tokenizer.
     *
     * @param inputData the generation request to check
     * @throws GenerationException if the request or prompt is missing, the prompt is
     *         empty, the token limit is below one, or the temperature is nonpositive
     *         or nonfinite
     */
    private static void validate(GenerateTextInputData inputData) throws GenerationException {

        if (inputData == null || inputData.getPrompt() == null || inputData.getPrompt().isEmpty()){
            throw new GenerationException("Enter a prompt before generating text.");
        }
        if (inputData.getMaxTokens() < 1){
            throw new GenerationException("Choose at least one new token.");
        }
        float temperature = inputData.getTemperature();
        if (!Float.isFinite(temperature) || temperature <= 0){
            throw new GenerationException("Choose a finite generation temperature greater than 0");
        }

    }

    /**
     * Encodes the prompt once and streams one decoded token per prediction.
     * The prompt seeds the first prediction. Only newly sampled tokens count
     * toward the requested output limit.
     *
     * @param inputData the validated generation request
     * @param model the model and tokenizer selected for this request
     * @throws GenerationException if the context length is invalid, the prompt
     *         produces no tokens, or encoding, prediction, sampling, or decoding fails
     */
    private void generate(GenerateTextInputData inputData, ModelGeneration model) throws GenerationException {

        int contextLength = model.getContextLength();
        if (contextLength <= 0){
            throw new GenerationException("Load a model with a positive context length");
        }

        int[] promptTokens = model.encode(inputData.getPrompt());
        if (promptTokens.length == 0){
            throw new GenerationException("Enter a prompt that produces at least one token.");
        }

        // For a long prompt, the first prediction sees the most recent tokens that fit.
        int start = Math.max(0, promptTokens.length - contextLength);
        int[] context = Arrays.copyOfRange(promptTokens, start, promptTokens.length);

        for (int generated = 0; generated < inputData.getMaxTokens(); generated++) {
            Tensor scores = model.nextTokenScore(context);
            int nextToken = sampler.sample(scores, inputData.getTemperature());
            String text = model.decode(nextToken);
            presenter.prepareProgressView(text);

            context = appendToContext(context, nextToken, contextLength);
        }
    }

    /**
     * Builds the context for the next prediction by appending the sampled token.
     * For example, at a context length of three, appending 40 to
     * {@code [10, 20, 30]} yields {@code [20, 30, 40]}.
     *
     * @param context the current token IDs, oldest first
     * @param nextToken the sampled token ID to append
     * @param contextLength the maximum number of tokens allowed for a prediction
     * @return a new array containing the most recent tokens in generation order
     */
    private static int[] appendToContext(int[] context, int nextToken, int contextLength) {
        int retained = Math.min(context.length, contextLength - 1);
        int[] nextContext = new int[retained + 1];

        System.arraycopy(context, context.length - retained, nextContext, 0, retained);
        nextContext[retained] = nextToken;

        return nextContext;
    }
}
