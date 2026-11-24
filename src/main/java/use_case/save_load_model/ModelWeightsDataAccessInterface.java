package use_case.save_load_model;

import java.io.IOException;

import entities.Model;

/**
 * Persists trained models so they can be reloaded without retraining.
 *
 * <p>Interactors depend only on this interface and never see the file format.
 * Implementations decide how the model is encoded on disk.</p>
 */
public interface ModelWeightsDataAccessInterface {

    /**
     * Saves a model's configuration, tokenizer, and parameters to a file,
     * replacing any existing file at that path.
     *
     * @param path  the file to write
     * @param model the model to save
     * @throws IOException              if the file cannot be written
     * @throws IllegalArgumentException if the model is internally inconsistent
     *                                  (for example, its tokenizer size does not
     *                                  match its configured vocabulary size)
     */
    void save(String path, Model model) throws IOException;
}
