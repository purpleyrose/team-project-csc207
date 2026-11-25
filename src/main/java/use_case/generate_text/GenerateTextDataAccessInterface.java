package use_case.generate_text;

import java.util.Optional;

/**
 * Provides access to the active model for text generation.
 */
@FunctionalInterface
public interface GenerateTextDataAccessInterface {

    /**
     * Looks up the model currently available for generation.
     *
     * @return the active model, or an empty Optional if none is trained or loaded
     */
    Optional<GenerationModel> getActiveModel();
}
