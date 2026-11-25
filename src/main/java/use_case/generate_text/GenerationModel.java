package use_case.generate_text;

import torch.Tensor;

/**
 * Provides the model and tokenizer operations needed to generate text.
 */
public interface GenerationModel {
    /**
     * Returns the model's context length.
     *
     * @return the maximum number of tokens the model accepts for one prediction
     */
    int getContextLength();

    /**
     * Encodes the prompt using this model's vocabulary.
     *
     * @param prompt the original prompt, including spaces and line breaks
     * @return the token IDs in reading order
     * @throws GenerationException if the tokenizer cannot encode the prompt
     */
    int[] encode(String prompt) throws GenerationException;

    /**
     * Compares the raw score of the next token using the provided context.
     *
     * @param context the recent token IDs, oldest first; treated as read-only
     * @return a one-dimensional tensor where index i holds the score of token ID i
     * @throws GenerationException if the model cannot compute the next token score
     */
    Tensor nextTokenScore(int[] context) throws GenerationException;

    /**
     * Decodes one generated token using this model's vocabulary.
     *
     * @param token the generated token ID
     * @return the text to append to the output, including any spaces or line breaks
     * @throws GenerationException if the tokenizer cannot decode the token
     */
    String decode(int token) throws GenerationException;
}
