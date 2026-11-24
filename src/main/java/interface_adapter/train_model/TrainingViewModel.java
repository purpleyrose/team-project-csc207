package interface_adapter.train_model;

import java.util.ArrayList;
import java.util.List;

/**
 * View model for the training screen.
 *
 * <p>This class stores the current {@link TrainingState} for the training view
 * and notifies registered {@link TrainingViewModelListener} instances whenever
 * the state changes.</p>
 */
public class TrainingViewModel {
    /**
     * The display name of the training view.
     */
    public static final String VIEW_NAME = "Training";

    private TrainingState state = new TrainingState();
    private final List<TrainingViewModelListener> listeners = new ArrayList<>();

    /**
     * Returns the current state of the training view.
     *
     * @return the current {@link TrainingState}
     */
    public TrainingState getState() {
        return state;
    }

    /**
     * Updates the current state of the training view.
     *
     * <p>This method only replaces the stored state. To notify registered
     * listeners after updating the state, call {@link #fireStateChanged()}.</p>
     *
     * @param state the new {@link TrainingState} to store
     */
    public void setState(TrainingState state) {
        this.state = state;
    }

    /**
     * Registers a listener to be notified when the training view model's state changes.
     *
     * @param listener the listener to add
     */
    public void addListener(TrainingViewModelListener listener) {
        listeners.add(listener);
    }

    /**
     * Notifies all registered listeners that the training state has changed.
     *
     * <p>Each registered {@link TrainingViewModelListener} has its
     * {@link TrainingViewModelListener#stateChanged()} method invoked.</p>
     */
    public void fireStateChanged() {
        for (TrainingViewModelListener listener : listeners) {
            listener.stateChanged();
        }
    }
}