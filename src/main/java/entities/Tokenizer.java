package entities;

import java.util.List;

/**
 * Converts between text and sequences of integer token IDs, and holds the
 * vocabulary.
 */
public interface Tokenizer {

    /**
     * Returns the kind of tokenizer.
     *
     * @return {@code "char"} for a character-level tokenizer, or {@code "bpe"}
     *         for byte-pair encoding
     */
    String type();

    /**
     * Returns the vocabulary in ID order: the token at index {@code i} has ID
     * {@code i}.
     *
     * @return the token strings; never {@code null}
     */
    List<String> tokens();

    /**
     * Returns the BPE merge rules in the order they were learned. Each element
     * is a two-element array holding the IDs of the two tokens merged.
     *
     * @return the merge rules; empty for a character-level tokenizer, never
     *         {@code null}
     */
    List<int[]> merges();
}
