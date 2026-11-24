package entities;

import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * The hyperparameters that define a model's shape, plus optional details about
 * how it was trained.
 *
 * <p>The required values are enough to construct every layer of the model. The
 * optional values are for display only and are not needed to rebuild it.</p>
 */
public interface ModelConfig {

    /**
     * Returns the number of tokens in the vocabulary.
     *
     * @return the vocabulary size; equal to the tokenizer's vocabulary size
     */
    int getVocabSize();

    /**
     * Returns the maximum sequence length the model was trained with.
     *
     * @return the context length
     */
    int getContextLength();

    /**
     * Returns the embedding (hidden) dimension.
     *
     * @return the model dimension; divisible by {@link #getNHeads()}
     */
    int getDModel();

    /**
     * Returns the number of transformer blocks.
     *
     * @return the number of layers
     */
    int getNLayers();

    /**
     * Returns the number of attention heads per block.
     *
     * @return the number of heads
     */
    int getNHeads();

    /**
     * Returns the hidden size of each block's feed-forward layer.
     *
     * @return the feed-forward dimension
     */
    int getDFf();

    /**
     * Returns the learning rate used during training, if known.
     *
     * @return the learning rate, or empty if not recorded
     */
    OptionalDouble getLearningRate();

    /**
     * Returns the number of training steps completed, if known.
     *
     * @return the step count, or empty if not recorded
     */
    OptionalInt getTrainSteps();

    /**
     * Returns the loss at the end of training, if known.
     *
     * @return the final loss, or empty if not recorded
     */
    OptionalDouble getFinalLoss();

    /**
     * Returns a human-readable description of the training corpus, if known.
     *
     * @return the corpus name, or empty if not recorded
     */
    Optional<String> getCorpusName();
}
