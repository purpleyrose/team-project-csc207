package entities;

import java.util.LinkedHashMap;

import torch.Tensor;

/**
 * A trained (or trainable) transformer language model.
 *
 * <p>Exposes everything needed to save the model and rebuild it later: its
 * learnable parameters, the configuration that determines their shapes, and the
 * tokenizer that maps text to token IDs.</p>
 */
public interface Model {

    /**
     * Returns every learnable parameter of this model, keyed by a dot-separated
     * path that mirrors the model's structure (for example
     * {@code "blocks.0.attn.q.weight"}).
     *
     * <p>Iteration order must be stable and deterministic for a given model
     * architecture, so that saving the same model twice produces identical files.
     * Names must be unique and non-empty.</p>
     *
     * @return the parameters in a stable order; never {@code null}
     */
    LinkedHashMap<String, Tensor> parameters();

    /**
     * Returns the configuration this model was built with.
     *
     * @return the model configuration; never {@code null}
     */
    ModelConfig config();

    /**
     * Returns the tokenizer this model was trained with.
     *
     * @return the tokenizer; never {@code null}
     */
    Tokenizer tokenizer();
}
