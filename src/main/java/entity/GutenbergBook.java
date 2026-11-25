package entity;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Metadata about a book available through Project Gutenberg. */
public final class GutenbergBook {
    private final int id;
    private final String title;
    private final List<String> authors;
    private final Map<String, String> formats;

    public GutenbergBook(int id, String title,
                         List<String> authors, Map<String, String> formats) {
        this.id = id;
        this.title = Objects.requireNonNull(title);
        this.authors = List.copyOf(Objects.requireNonNull(authors));
        this.formats = Map.copyOf(Objects.requireNonNull(formats));
    }

    public int getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public List<String> getAuthors() {
        return authors;
    }

    public Map<String, String> getFormats() {
        return formats;
    }
}
