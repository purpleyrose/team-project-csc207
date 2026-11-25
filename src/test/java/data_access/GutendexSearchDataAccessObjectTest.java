package data_access;

import com.sun.net.httpserver.HttpServer;
import entity.GutenbergBook;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import use_case.gutenberg_search.SearchGatewayException;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class GutendexSearchDataAccessObjectTest {
    private HttpServer server;
    private GutendexSearchDataAccessObject gateway;
    private volatile String rawQuery;
    private volatile int statusCode;
    private volatile String responseBody;

    @BeforeEach
    void setUp() throws IOException {
        statusCode = 200;
        responseBody = "{\"count\":0,\"next\":null,\"previous\":null,\"results\":[]}";
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/books/", exchange -> {
            rawQuery = exchange.getRequestURI().getRawQuery();
            byte[] body = responseBody.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(statusCode, body.length);
            try (OutputStream output = exchange.getResponseBody()) {
                output.write(body);
            }
        });
        server.start();
        URI endpoint = URI.create("http://127.0.0.1:"
                + server.getAddress().getPort() + "/books/");
        gateway = new GutendexSearchDataAccessObject(HttpClient.newHttpClient(), endpoint);
    }

    @AfterEach
    void tearDown() {
        server.stop(0);
    }

    @Test
    void parsesBookMetadataWithoutDownloadingText() {
        responseBody = "{\"results\":[{\"id\":1513,\"title\":\"Romeo and Juliet\","
                + "\"authors\":[{\"name\":\"Shakespeare, William\"}],"
                + "\"formats\":{\"text/plain; charset=utf-8\":\"https://example.org/1513.txt\"}}]}";

        List<GutenbergBook> books = gateway.searchBooks("romeo", 1500, 1750, false);

        assertEquals(1, books.size());
        assertEquals(1513, books.get(0).getId());
        assertEquals("Romeo and Juliet", books.get(0).getTitle());
        assertEquals("Shakespeare, William", books.get(0).getAuthors().get(0));
        assertEquals("https://example.org/1513.txt",
                books.get(0).getFormats().get("text/plain; charset=utf-8"));
        assertTrue(rawQuery.contains("search=romeo"));
        assertTrue(rawQuery.contains("author_year_start=1500"));
        assertTrue(rawQuery.contains("author_year_end=1750"));
    }

    @Test
    void usesTopicParameterForSubjectSearch() {
        gateway.searchBooks("dramatic poetry", null, null, true);
        assertTrue(rawQuery.contains("topic=dramatic+poetry"));
        assertFalse(rawQuery.contains("search="));
    }

    @Test
    void reportsHttpErrors() {
        statusCode = 503;
        SearchGatewayException error = assertThrows(SearchGatewayException.class,
                () -> gateway.searchBooks("romeo", null, null, false));
        assertTrue(error.getMessage().contains("503"));
    }

    @Test
    void reportsInvalidJson() {
        responseBody = "this is not JSON";
        assertThrows(SearchGatewayException.class,
                () -> gateway.searchBooks("romeo", null, null, false));
    }

    @Test
    void returnsEmptyListForNoResults() {
        assertTrue(gateway.searchBooks("no matching books", null, null, false).isEmpty());
    }
}
