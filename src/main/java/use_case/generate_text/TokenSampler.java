package use_case.generate_text;

import torch.Tensor;

/**
 * Selects the next token from a model's scores.
 */
@FunctionalInterface
public interface TokenSampler {
    /**
     * Samples one token from the provided scores.
     *
     * @param scores a non-empty tensor of raw scores, indexed by token ID
     * @param temperature the positive, finite sampling temperature
     * @return the selected token ID
     * @throws GenerationException if the scores or the temperature cannot be used for sampling
     */
    int sample(Tensor scores, float temperature) throws GenerationException;
}
