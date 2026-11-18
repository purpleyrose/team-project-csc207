package entities;

import java.util.List;

public class Corpus {

    private List<CorpusSource> sources;


    public Corpus(List<CorpusSource> sources) {
        this.sources = sources;
    }

    public List<CorpusSource> getSources() {
        return sources;
    };

    public String getFullText() {
        throw new UnsupportedOperationException("TODO: implement");
    }

    public int size() {
        throw new UnsupportedOperationException("TODO: implement");
    }

    public boolean isLargeEnoughToTrain() {
        throw new UnsupportedOperationException("TODO: implement");
    }

}
