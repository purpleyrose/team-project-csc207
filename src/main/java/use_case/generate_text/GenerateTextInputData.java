package use_case.generate_text;

/**
 * Holds the prompt and sampling configurations for one text generation request.
 */
public class GenerateTextInputData {

    private final String prompt;
    private final int maxTokens;
    private final float temperature;

    /**
     * Creates a text generation request.
     *
     * @param prompt the starting text supplied
     * @param maxTokens the maximum number of tokens to generate, excluding the prompt
     * @param temperature the value controlling randomness when sampling the next token
     */

    public GenerateTextInputData(String prompt, int maxTokens, float temperature) {
        this.prompt = prompt;
        this.maxTokens = maxTokens;
        this.temperature = temperature;
    }

    /**
     * Returns the text used to start generation.
     *
     * @return the prompt supplied by the user
     */
    public String getPrompt() {
        return prompt;
    }

    /**
     * Returns the maximum number of new tokens to generate.
     *
     * @return the token limit, excluding tokens in the prompt
     */
    public int getMaxTokens() {
        return maxTokens;
    }

    /**
     * Returns the temperature used to sample each new token.
     *
     * @return the temperature used to sample the next token
     */
    public float getTemperature() {
        return temperature;
    }

}
