package use_case.train_model;

/**
 * Input data for configuring a model training request.
 *
 * <p>This class stores the hyperparameters and training settings needed to
 * start training a model, including the model architecture, training duration,
 * random seed, and learning rate.</p>
 */
public class TrainModelInputData {

    private final int layers;

    private final int heads;

    private final int embeddingSize;


    private final int contextLength;

    private final int batchSize;

    private final int trainingSteps;

    private final int seed;

    private final float learningRate;

    public TrainModelInputData(int layers, int heads, int embeddingSize, int contextLength, int batchSize, int trainingSteps, int seed, float learningRate) {
        this.layers = layers;
        this.heads = heads;
        this.embeddingSize = embeddingSize;
        this.contextLength = contextLength;
        this.batchSize = batchSize;
        this.trainingSteps = trainingSteps;
        this.seed = seed;
        this.learningRate = learningRate;
    }

    public int getLayers() {
        return layers;
    }

    public int getHeads() {
        return heads;
    }

    public int getEmbeddingSize() {
        return embeddingSize;
    }

    public int getContextLength() {
        return contextLength;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getTrainingSteps() {
        return trainingSteps;
    }

    public int getSeed() {
        return seed;
    }

    public float getLearningRate() {
        return learningRate;
    }


}
