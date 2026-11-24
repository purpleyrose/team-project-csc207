package interface_adapter.train_model;

import use_case.train_model.TrainModelInputBoundary;
import use_case.train_model.TrainModelInputData;

/**
 * Controller responsible for handling requests to train a model.
 * <p>
 * This class receives training configuration parameters from the interface layer,
 * packages them into a {@link TrainModelInputData} object, and passes the input
 * data to the train model use case through the {@link TrainModelInputBoundary}.
 */
public class TrainModelController {
    private final TrainModelInputBoundary trainModelInteractor;

    /**
     * Constructs a {@code TrainModelController} with the given train model interactor.
     *
     * @param trainModelInteractor the input boundary used to execute the train model use case
     */
    public TrainModelController(TrainModelInputBoundary trainModelInteractor) {
        this.trainModelInteractor = trainModelInteractor;
    }

    /**
     * Executes the train model use case with the provided training configuration.
     *
     * @param layers the number of layers in the model
     * @param heads the number of attention heads in the model
     * @param embeddingSize the size of the model embeddings
     * @param contextLength the maximum context length used during training
     * @param batchSize the number of samples processed in each training batch
     * @param trainingSteps the total number of training steps to perform
     * @param seed the random seed used for reproducible training
     * @param learningRate the learning rate used during training
     */
    public void execute(int layers, int heads, int embeddingSize,
                        int contextLength, int batchSize, int trainingSteps,
                        int seed, float learningRate) {
        TrainModelInputData inputData = new TrainModelInputData(layers, heads,
                embeddingSize, contextLength, batchSize, trainingSteps, seed, learningRate);
        trainModelInteractor.execute(inputData);
    }
}
