package use_case.generate_text;

/**
 * A generation failure exception with a message which the presenter can
 * show to the user.
 */
public class GenerationException extends Exception {
    private static final long serialVersionUID = 1L;

    /**
     * Creates a generation failure.
     *
     * @param message an explanation of the failure for the user
     */
    public GenerationException(String message) {
        super(message);
    }
}
