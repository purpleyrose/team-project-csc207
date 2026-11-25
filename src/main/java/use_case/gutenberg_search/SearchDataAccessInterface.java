package use_case.gutenberg_search;

import entity.GutenbergBook;
import java.util.List;

/** Port used by the search interactor; it knows nothing about HTTP or JSON. */
public interface SearchGateway {
    /**
     * Search one page (up to 32 books in Gutendex) by title/author or subject.
     * Null author-year values mean no corresponding filter.
     *
     * @param bySubject true to search subjects/bookshelves, false for title/author
     */
    List<GutenbergBook> searchBooks(String query, Integer authorYearStart,
                                   Integer authorYearEnd, boolean bySubject);
}
