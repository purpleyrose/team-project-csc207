package use_case.generate_text;

/**
 * Lets the controller request text generation from the active model
 */
public interface GenerateTextInputBoundary {
    /**
     * Starts a generation request. Generated text, completion, and failures are
     * reported through {@link GenerateTextOutputBoundary}.
     * <p>
     * The final callback is either {@code onComplete()} or {@code onFailure(message)},
     * exactly one per request.
     * <p>
     * If the model is missing or the settings are invalid, {@code onFailure(message)}
     * will be the only callback.
     *
     * @param inputData the prompt, maximum number of new tokens, and temperature
     */
    void execute(GenerateTextInputData inputData);
}
