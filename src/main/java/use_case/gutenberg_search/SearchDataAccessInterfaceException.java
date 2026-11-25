package use_case.gutenberg_search;

/** A failed search at the external catalogue (HTTP, network, or invalid data). */
public class SearchGatewayException extends RuntimeException {
    public SearchGatewayException(String message) {
        super(message);
    }

    public SearchGatewayException(String message, Throwable cause) {
        super(message, cause);
    }
}
