package entities;



public class CorpusSource {
    private final String title;

    private final SourceType sourceType;

    private final String text;

    public CorpusSource(String title, SourceType sourceType, String text) {
        this.title = title;
        this.sourceType = sourceType;
        this.text = text;
    }


}
