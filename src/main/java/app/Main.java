package app;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;

/** Starts Text Generation Studio. */
public final class Main {

    /** Not instantiable. */
    private Main() {
    }

    /**
     * Builds the app and shows its window, on the Swing thread.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame app = new AppBuilder()
                    .addTrainingView()
                    .addTrainModelUseCase()
                    .build();
            app.setVisible(true);
        });
    }
}