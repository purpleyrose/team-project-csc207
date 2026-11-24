package use_case.train_model;

import entities.Corpus;

/**
 * Defines a trainer capable of training a model using a provided corpus and
 * training configuration data.
 */
public interface ModelTrainer {

    /**
     * Trains a model using the given corpus and training input data,
     * reporting progress to the provided listener.
     *
     * @param corpus the corpus used as the training data source
     * @param inputData the input data containing training configuration or parameters
     * @param trainingListener a listener for training progress updates
     * @return the final loss value produced by the training process
     */
    float train(Corpus corpus, TrainModelInputData inputData, TrainingListener trainingListener);
}
