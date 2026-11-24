package use_case.train_model;

import entities.Corpus;

/**
 * Defines the data access operations required by the train model use case.
 * <p>
 * Implementations of this interface are responsible for providing access to the
 * {@link Corpus} used during model training.
 * </p>
 */
public interface TrainModelDataAccessInterface {

    /**
     * Retrieves the corpus used as input for training the model.
     *
     * @return the {@link Corpus} containing the training data
     */
    Corpus getCorpus();
}
