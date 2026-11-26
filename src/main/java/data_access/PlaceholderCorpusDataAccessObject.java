package data_access;

import entities.Corpus;
import use_case.train_model.TrainModelDataAccessInterface;

import java.util.Collections;

/**
 * TEMPORARY stand-in until Load Corpus is finished: hands Train Model an empty
 * corpus that always reports it is large enough, so the training screen can be
 * run end to end. Replace it with the real corpus data access object.
 */
public class PlaceholderCorpusDataAccessObject implements TrainModelDataAccessInterface {

    /** The corpus returned on every call. */
    private final Corpus corpus = new Corpus(Collections.emptyList()) {
        @Override
        public boolean isLargeEnoughToTrain() {
            return true;
        }
    };

    /**
     * Returns the placeholder corpus.
     *
     * @return a corpus that is always large enough to train on
     */
    @Override
    public Corpus getCorpus() {
        return corpus;
    }
}