package data_access;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import entity.GutenbergBook;
import use_case.gutenberg_search.SearchGateway;
import use_case.gutenberg_search.SearchGatewayException;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;

/** HTTP/Jackson adapter for the Gutendex catalogue. Does not download books. */
public class GutendexSearchDataAccessObject implements SearchGateway {
    private static final URI DEFAULT_ENDPOINT = URI.create("https://gutendex.com/books/");

    private final HttpClient client;
    private final URI endpoint;
    private final ObjectMapper mapper;

    public GutendexSearchDataAccessObject() {
        this(HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build(), DEFAULT_ENDPOINT);
    }

    /** Injectable endpoint and HTTP client allow tests without external network calls. */
    public GutendexSearchDataAccessObject(HttpClient client, URI endpoint) {
        this.client = Objects.requireNonNull(client);
        this.endpoint = Objects.requireNonNull(endpoint);
        this.mapper = new ObjectMapper();
    }

    @Override
    public List<GutenbergBook> searchBooks(String query, Integer authorYearStart,
                                          Integer authorYearEnd, boolean bySubject) {
        if (authorYearStart != null && authorYearEnd != null
                && authorYearStart > authorYearEnd) {
            throw new IllegalArgumentException("Start year must not exceed end year");
        }

        URI uri = buildSearchUri(query, authorYearStart, authorYearEnd, bySubject);
        HttpRequest request = HttpRequest.newBuilder(uri)
                .timeout(Duration.ofSeconds(20))
                .header("Accept", "application/json")
                .header("User-Agent", "TextGenerationStudio/0.1 (student project)")
                .GET()
                .build();

        try {
            HttpResponse<String> response = client.send(request,
                    HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            if (response.statusCode() != 200) {
                throw new SearchGatewayException(
                        "Gutendex returned HTTP " + response.statusCode());
            }
            return parseBooks(response.body());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new SearchGatewayException("Gutendex search was interrupted", e);
        } catch (IOException e) {
            throw new SearchGatewayException("Could not contact Gutendex", e);
        }
    }

    URI buildSearchUri(String query, Integer startYear, Integer endYear,
                       boolean bySubject) {
        List<String> parameters = new ArrayList<>();
        if (query != null && !query.trim().isEmpty()) {
            parameters.add((bySubject ? "topic=" : "search=")
                    + URLEncoder.encode(query.trim(), StandardCharsets.UTF_8));
        }
        if (startYear != null) {
            parameters.add("author_year_start=" + startYear);
        }
        if (endYear != null) {
            parameters.add("author_year_end=" + endYear);
        }

        if (parameters.isEmpty()) {
            return endpoint;
        }
        String separator = endpoint.toString().contains("?") ? "&" : "?";
        return URI.create(endpoint + separator + String.join("&", parameters));
    }

    List<GutenbergBook> parseBooks(String json) {
        final JsonNode root;
        try {
            root = mapper.readTree(json);
        } catch (JsonProcessingException e) {
            throw new SearchGatewayException("Gutendex returned invalid JSON", e);
        }
        if (root == null || !root.has("results") || !root.get("results").isArray()) {
            throw new SearchGatewayException("Gutendex JSON is missing a results array");
        }

        List<GutenbergBook> books = new ArrayList<>();
        for (JsonNode item : root.get("results")) {
            if (!item.path("id").isIntegralNumber()
                    || !item.path("title").isTextual()
                    || !item.path("authors").isArray()
                    || !item.path("formats").isObject()) {
                throw new SearchGatewayException("Gutendex returned incomplete book metadata");
            }

            List<String> authors = new ArrayList<>();
            for (JsonNode author : item.get("authors")) {
                if (author.path("name").isTextual()) {
                    authors.add(author.get("name").asText());
                }
            }

            Map<String, String> formats = new HashMap<>();
            Iterator<Entry<String, JsonNode>> fields = item.get("formats").fields();
            while (fields.hasNext()) {
                Entry<String, JsonNode> format = fields.next();
                if (format.getValue().isTextual()) {
                    formats.put(format.getKey(), format.getValue().asText());
                }
            }

            books.add(new GutenbergBook(item.get("id").asInt(),
                    item.get("title").asText(), authors, formats));
        }
        return books;
    }
}
