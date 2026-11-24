package use_case.generate_text;

/**
 * Lets the interactor report the generation results to the presenter.
 */
public interface GenerateTextOutputBoundary {

    /**
     * Deliver the next piece of decoded generated text in order.
     *
     * @param text the newly generated text to add to the output
     */
    void prepareProgressView(String text);

    /**
     * Signals that the generation request has finished successfully and all
     * generated text has been delivered.
     */
    void prepareSuccessView();

    /**
     * Signals that the generation request has failed.
     *
     * @param message an explanation the presenter can show to the user
     */
    void prepareFailureView(String message);

}
