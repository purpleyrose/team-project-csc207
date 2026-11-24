package interface_adapter.train_model;

/**
 * Listener interface for receiving notifications when the training view model's
 * state changes.
 *
 * <p>Implementations of this interface can be registered with a training view
 * model to react to updates, such as refreshing the UI or handling newly
 * available training state.</p>
 */
public interface TrainingViewModelListener {

    /**
     * Called when the training view model's state has changed.
     *
     * <p>Implementations should use this callback to update any dependent
     * components, such as views or controllers, based on the latest state.</p>
     */
    void stateChanged();
}
