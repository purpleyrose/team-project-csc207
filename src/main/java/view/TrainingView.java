package view;

import interface_adapter.train_model.TrainModelController;
import interface_adapter.train_model.TrainingState;
import interface_adapter.train_model.TrainingStatus;
import interface_adapter.train_model.TrainingViewModel;
import interface_adapter.train_model.TrainingViewModelListener;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.ButtonGroup;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.JToggleButton;
import javax.swing.SwingUtilities;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.RenderingHints;
import java.util.ArrayList;
import java.util.List;

/**
 * The Train tab: a hyperparameter form on the left, and on the right the
 * run's status, progress, loss statistics and a live loss chart.
 *
 * <p>The view only parses the form's text into numbers. Rules such as "must be
 * positive" belong to the use case, which reports problems back through the
 * presenter. Everything shown on the right is read from the
 * {@link TrainingViewModel} whenever it reports a change.</p>
 */
public class TrainingView extends JPanel implements TrainingViewModelListener {

    /** Width of the hyperparameter card. */
    private static final int FORM_WIDTH = 380;

    /** The view model this view displays. */
    private final TrainingViewModel viewModel;
    /** The controller the Train button calls; set after construction. */
    private TrainModelController controller;

    /** Hyperparameter inputs. */
    private final JTextField layersField = Theme.field("4");
    private final JTextField headsField = Theme.field("4");
    private final JTextField embeddingField = Theme.field("128");
    private final JTextField contextField = Theme.field("256");
    private final JTextField learningRateField = Theme.field("3e-4");
    private final JTextField batchField = Theme.field("32");
    private final JTextField stepsField = Theme.field("5000");
    private final JTextField seedField = Theme.field("207");

    /** Starts a run. */
    private final FlatButton trainButton = new FlatButton("Train", true);
    /** Stops a run; enabled once training runs in the background (#24). */
    private final FlatButton stopButton = new FlatButton("Stop", false);

    /** The coloured status pill and its text. */
    private final RoundedPanel statusPill =
            new RoundedPanel(new FlowLayout(FlowLayout.CENTER, 0, 0), Theme.BACKGROUND, null, 13);
    private final JLabel statusLabel = new JLabel();

    /** "Step 2,140 of 5,000". */
    private final JLabel stepLabel = new JLabel();
    /** "43%". */
    private final JLabel percentLabel = new JLabel();
    /** The progress bar. */
    private final ProgressTrack progress = new ProgressTrack();

    /** Values of the three stat tiles. */
    private final JLabel lossValue = statValue();
    private final JLabel lowestValue = statValue();
    private final JLabel startValue = statValue();

    /** The error banner and its text. */
    private final RoundedPanel errorBanner =
            new RoundedPanel(new BorderLayout(), Theme.DANGER_SOFT, null, Theme.CONTROL_RADIUS);
    private final JLabel errorLabel = new JLabel();

    /** The loss chart. */
    private final LossChartPanel chart = new LossChartPanel();

    /**
     * Creates the view and starts listening to the view model.
     *
     * @param viewModel the training view model to display
     */
    public TrainingView(TrainingViewModel viewModel) {
        this.viewModel = viewModel;
        setLayout(new BorderLayout(24, 0));
        setBackground(Theme.BACKGROUND);
        setBorder(BorderFactory.createEmptyBorder(24, 24, 24, 24));

        add(buildForm(), BorderLayout.WEST);
        add(buildRunPanel(), BorderLayout.CENTER);

        trainButton.addActionListener(e -> onTrain());
        stopButton.setEnabled(false);
        stopButton.setToolTipText("Available once training runs in the background");

        viewModel.addListener(this);
        render(Snapshot.of(viewModel.getState()));
    }

    /**
     * Sets the controller the Train button calls.
     *
     * @param trainModelController the controller
     */
    public void setTrainModelController(TrainModelController trainModelController) {
        this.controller = trainModelController;
    }

    /**
     * Returns this view's name, used to switch between views.
     *
     * @return the view name
     */
    public String getViewName() {
        return TrainingViewModel.VIEW_NAME;
    }

    // ------------------------------------------------------------------ layout

    /**
     * Builds the hyperparameter card.
     *
     * @return the card
     */
    private JComponent buildForm() {
        RoundedPanel card = card();
        card.setLayout(new BoxLayout(card, BoxLayout.Y_AXIS));
        card.setPreferredSize(new Dimension(FORM_WIDTH, 10));

        card.add(Theme.fixHeight(Theme.title("Hyperparameters")));
        card.add(Box.createVerticalStrut(16));
        card.add(Theme.fixHeight(Theme.overline("Model size")));
        card.add(Box.createVerticalStrut(10));
        card.add(fieldRow("Layers", layersField, "Attention heads", headsField));
        card.add(Box.createVerticalStrut(12));
        card.add(fieldRow("Embedding size", embeddingField, "Context length", contextField));
        card.add(Box.createVerticalStrut(20));
        card.add(Theme.fixHeight(Theme.overline("Training")));
        card.add(Box.createVerticalStrut(10));
        card.add(fieldRow("Learning rate", learningRateField, "Batch size", batchField));
        card.add(Box.createVerticalStrut(12));
        card.add(fieldRow("Training steps", stepsField, "Random seed", seedField));
        card.add(Box.createVerticalStrut(20));
        card.add(Theme.fixHeight(Theme.overline("Tokenizer")));
        card.add(Box.createVerticalStrut(10));
        card.add(tokenizerToggle());
        card.add(Box.createVerticalGlue());

        JPanel buttons = new JPanel(new GridLayout(1, 2, 12, 0));
        buttons.setOpaque(false);
        buttons.add(stopButton);
        buttons.add(trainButton);
        card.add(Theme.fixHeight(buttons));
        return card;
    }

    /**
     * Builds the right-hand card: header, error banner, progress, stats and chart.
     *
     * @return the card
     */
    private JComponent buildRunPanel() {
        RoundedPanel card = card();
        card.setLayout(new BorderLayout(0, 20));

        JPanel top = new JPanel();
        top.setOpaque(false);
        top.setLayout(new BoxLayout(top, BoxLayout.Y_AXIS));

        JPanel header = new JPanel(new BorderLayout());
        header.setOpaque(false);
        header.add(Theme.title("Training run"), BorderLayout.WEST);
        statusLabel.setFont(Theme.sans(Font.BOLD, 12f));
        statusLabel.setBorder(BorderFactory.createEmptyBorder(5, 12, 5, 12));
        statusPill.add(statusLabel);
        header.add(statusPill, BorderLayout.EAST);
        top.add(Theme.fixHeight(header));
        top.add(Box.createVerticalStrut(16));

        errorLabel.setText(" ");
        errorLabel.setFont(Theme.sans(Font.PLAIN, 13f));
        errorLabel.setForeground(Theme.DANGER);
        errorBanner.setBorder(BorderFactory.createEmptyBorder(10, 14, 10, 14));
        errorBanner.add(errorLabel, BorderLayout.CENTER);
        top.add(Theme.fixHeight(errorBanner));
        top.add(Box.createVerticalStrut(16));

        JPanel stepRow = new JPanel(new BorderLayout());
        stepRow.setOpaque(false);
        stepLabel.setText("Not started");
        percentLabel.setText(" ");
        stepLabel.setFont(Theme.mono(20f));
        stepLabel.setForeground(Theme.INK);
        percentLabel.setFont(Theme.mono(14f));
        percentLabel.setForeground(Theme.MUTED);
        stepRow.add(stepLabel, BorderLayout.WEST);
        stepRow.add(percentLabel, BorderLayout.EAST);
        top.add(Theme.fixHeight(stepRow));
        top.add(Box.createVerticalStrut(10));
        top.add(Theme.fixHeight(progress));
        top.add(Box.createVerticalStrut(20));

        JPanel stats = new JPanel(new GridLayout(1, 3, 16, 0));
        stats.setOpaque(false);
        stats.add(statTile(lossValue, "loss"));
        stats.add(statTile(lowestValue, "lowest loss"));
        stats.add(statTile(startValue, "starting loss"));
        top.add(Theme.fixHeight(stats));

        card.add(top, BorderLayout.NORTH);

        RoundedPanel chartCard = new RoundedPanel(new BorderLayout(0, 10), Theme.SURFACE,
                Theme.BORDER, Theme.CONTROL_RADIUS);
        chartCard.setBorder(BorderFactory.createEmptyBorder(14, 16, 12, 16));
        chartCard.add(Theme.overline("Loss over time"), BorderLayout.NORTH);
        chart.setOpaque(false);
        chartCard.add(chart, BorderLayout.CENTER);
        card.add(chartCard, BorderLayout.CENTER);
        return card;
    }

    /**
     * Creates an empty white card with padding.
     *
     * @return the card
     */
    private static RoundedPanel card() {
        RoundedPanel card = new RoundedPanel(null, Theme.SURFACE, Theme.BORDER, Theme.CARD_RADIUS);
        card.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));
        return card;
    }

    /**
     * Creates a row of two captioned fields.
     *
     * @param leftCaption  caption of the left field
     * @param left         the left field
     * @param rightCaption caption of the right field
     * @param right        the right field
     * @return the row
     */
    private static JPanel fieldRow(String leftCaption, JTextField left,
                                   String rightCaption, JTextField right) {
        JPanel row = new JPanel(new GridLayout(1, 2, 12, 0));
        row.setOpaque(false);
        row.add(captioned(leftCaption, left));
        row.add(captioned(rightCaption, right));
        return Theme.fixHeight(row);
    }

    /**
     * Stacks a caption above a field.
     *
     * @param caption the caption
     * @param field   the field
     * @return the panel
     */
    private static JPanel captioned(String caption, JTextField field) {
        JPanel panel = new JPanel(new BorderLayout(0, 6));
        panel.setOpaque(false);
        panel.add(Theme.caption(caption), BorderLayout.NORTH);
        panel.add(field, BorderLayout.CENTER);
        return panel;
    }

    /**
     * Builds the Character / BPE toggle. BPE is a stretch goal, so it is disabled.
     *
     * @return the toggle
     */
    private static JComponent tokenizerToggle() {
        RoundedPanel track = new RoundedPanel(new GridLayout(1, 2, 3, 0), Theme.BACKGROUND, null,
                Theme.CONTROL_RADIUS);
        track.setBorder(BorderFactory.createEmptyBorder(3, 3, 3, 3));
        Segment character = new Segment("Character");
        Segment bpe = new Segment("BPE (stretch)");
        bpe.setEnabled(false);
        ButtonGroup group = new ButtonGroup();
        group.add(character);
        group.add(bpe);
        character.setSelected(true);
        track.add(character);
        track.add(bpe);
        return Theme.fixHeight(track);
    }

    /**
     * Creates the large number label of a stat tile.
     *
     * @return the label
     */
    private static JLabel statValue() {
        JLabel label = new JLabel("–");
        label.setFont(Theme.mono(22f));
        label.setForeground(Theme.INK);
        return label;
    }

    /**
     * Creates a grey tile with a value above a caption.
     *
     * @param value   the value label
     * @param caption the caption
     * @return the tile
     */
    private static JComponent statTile(JLabel value, String caption) {
        RoundedPanel tile = new RoundedPanel(new BorderLayout(0, 2), Theme.BACKGROUND, null, 10);
        tile.setBorder(BorderFactory.createEmptyBorder(12, 16, 12, 16));
        tile.add(value, BorderLayout.NORTH);
        tile.add(Theme.caption(caption), BorderLayout.SOUTH);
        return tile;
    }

    // ------------------------------------------------------------------ input

    /**
     * Reads the form and asks the controller to train. Text that is not a
     * number is reported here; every other rule is checked by the use case.
     */
    private void onTrain() {
        if (controller == null) {
            return;
        }
        int layers;
        int heads;
        int embedding;
        int context;
        int batch;
        int steps;
        int seed;
        float learningRate;
        try {
            layers = parseInt(layersField, "Layers");
            heads = parseInt(headsField, "Attention heads");
            embedding = parseInt(embeddingField, "Embedding size");
            context = parseInt(contextField, "Context length");
            batch = parseInt(batchField, "Batch size");
            steps = parseInt(stepsField, "Training steps");
            seed = parseInt(seedField, "Random seed");
            learningRate = parseFloat(learningRateField, "Learning rate");
        } catch (IllegalArgumentException e) {
            showError(e.getMessage());
            return;
        }
        showError(null);
        controller.execute(layers, heads, embedding, context, batch, steps, seed, learningRate);
    }

    /**
     * Parses a whole number from a field.
     *
     * @param field the field
     * @param name  the field's name, for the error message
     * @return the number
     * @throws IllegalArgumentException if the text is not a whole number
     */
    private static int parseInt(JTextField field, String name) {
        try {
            return Integer.parseInt(field.getText().trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(name + " must be a whole number.");
        }
    }

    /**
     * Parses a finite decimal number from a field. {@link Float#parseFloat}
     * also accepts "NaN", "Infinity" and overflowing values such as 1e39,
     * so those are rejected here.
     *
     * @param field the field
     * @param name  the field's name, for the error message
     * @return the number
     * @throws IllegalArgumentException if the text is not a finite number
     */
    private static float parseFloat(JTextField field, String name) {
        float value;
        try {
            value = Float.parseFloat(field.getText().trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(name + " must be a number.");
        }
        if (!Float.isFinite(value)) {
            throw new IllegalArgumentException(name + " must be a finite number.");
        }
        return value;
    }

    // ------------------------------------------------------------------ output

    /**
     * Called by the view model after the presenter updates the state. A copy
     * of the state is taken immediately, on the calling thread, and drawn on
     * Swing's event dispatch thread, so this stays safe once training runs on
     * a background thread (#24).
     */
    @Override
    public void stateChanged() {
        Snapshot snapshot = Snapshot.of(viewModel.getState());
        if (SwingUtilities.isEventDispatchThread()) {
            render(snapshot);
        } else {
            SwingUtilities.invokeLater(() -> render(snapshot));
        }
    }

    /**
     * Updates every widget from a snapshot of the state.
     *
     * @param s the snapshot
     */
    private void render(Snapshot s) {
        renderStatus(s.status);

        boolean running = s.status == TrainingStatus.RUNNING;
        trainButton.setEnabled(!running);
        for (JTextField field : new JTextField[]{layersField, headsField, embeddingField,
                contextField, learningRateField, batchField, stepsField, seedField}) {
            field.setEnabled(!running);
        }

        if (s.status == TrainingStatus.IDLE || s.totalSteps == 0) {
            stepLabel.setText("Not started");
            percentLabel.setText(" ");
            progress.setFraction(0);
        } else {
            double fraction = s.totalSteps > 0 ? (double) s.steps / s.totalSteps : 0;
            stepLabel.setText(String.format("Step %,d of %,d", s.steps, s.totalSteps));
            percentLabel.setText(Math.round(fraction * 100) + "%");
            progress.setFraction(fraction);
        }

        boolean hasLosses = !s.history.isEmpty();
        lossValue.setText(hasLosses ? format(s.loss) : "–");
        lowestValue.setText(hasLosses ? format(min(s.history)) : "–");
        startValue.setText(hasLosses ? format(s.history.get(0)) : "–");
        chart.setData(s.history, s.steps, s.totalSteps);

        showError(s.status == TrainingStatus.FAILED ? s.errorMessage : null);
    }

    /**
     * Sets the pill's text and colours for a status.
     *
     * @param status the status
     */
    private void renderStatus(TrainingStatus status) {
        Color fg;
        Color bg;
        String text;
        switch (status) {
            case RUNNING:
                text = "Running";
                fg = Theme.ACCENT;
                bg = Theme.ACCENT_SOFT;
                break;
            case DONE:
                text = "Done";
                fg = Theme.SUCCESS;
                bg = Theme.SUCCESS_SOFT;
                break;
            case FAILED:
                text = "Failed";
                fg = Theme.DANGER;
                bg = Theme.DANGER_SOFT;
                break;
            default:
                text = "Idle";
                fg = Theme.MUTED;
                bg = Theme.BACKGROUND;
                break;
        }
        statusLabel.setText(text);
        statusLabel.setForeground(fg);
        statusPill.setColors(bg, null);
    }

    /**
     * Shows a message in the error banner, or hides the banner.
     *
     * @param message the message, or {@code null} or empty to hide the banner
     */
    private void showError(String message) {
        boolean show = message != null && !message.isEmpty();
        errorLabel.setText(show ? message : "");
        errorBanner.setVisible(show);
        revalidate();
    }

    /**
     * Formats a loss to three decimal places.
     *
     * @param loss the loss
     * @return the text
     */
    private static String format(float loss) {
        return String.format("%.3f", loss);
    }

    /**
     * Returns the smallest value in a non-empty list.
     *
     * @param values the values
     * @return the minimum
     */
    private static float min(List<Float> values) {
        float m = Float.POSITIVE_INFINITY;
        for (float v : values) {
            m = Math.min(m, v);
        }
        return m;
    }

    // ------------------------------------------------------------------ helpers

    /** An immutable copy of the state at one moment. */
    private static final class Snapshot {
        /** The status. */
        private final TrainingStatus status;
        /** The latest step. */
        private final int steps;
        /** The total number of steps. */
        private final int totalSteps;
        /** The latest loss. */
        private final float loss;
        /** A copy of the loss history. */
        private final List<Float> history;
        /** The error message. */
        private final String errorMessage;

        /**
         * Copies a state.
         *
         * @param state the state to copy
         */
        private Snapshot(TrainingState state) {
            this.status = state.getStatus();
            this.steps = state.getSteps();
            this.totalSteps = state.getTotalSteps();
            this.loss = state.getLoss();
            this.history = new ArrayList<>(state.getLossHistory());
            this.errorMessage = state.getErrorMessage();
        }

        /**
         * Copies a state.
         *
         * @param state the state to copy
         * @return the snapshot
         */
        private static Snapshot of(TrainingState state) {
            return new Snapshot(state);
        }
    }

    /** One option of the segmented tokenizer toggle. */
    private static final class Segment extends JToggleButton {

        /**
         * Creates an option.
         *
         * @param text the label
         */
        private Segment(String text) {
            super(text);
            setFont(Theme.sans(Font.BOLD, 13f));
            setContentAreaFilled(false);
            setFocusPainted(false);
            setBorderPainted(false);
            setOpaque(false);
            setBorder(BorderFactory.createEmptyBorder(8, 10, 8, 10));
        }

        /**
         * Paints a white rounded background when selected.
         *
         * @param g the graphics context
         */
        @Override
        protected void paintComponent(Graphics g) {
            if (isSelected()) {
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(Theme.SURFACE);
                g2.fillRoundRect(0, 0, getWidth() - 1, getHeight() - 1, 12, 12);
                g2.setColor(Theme.BORDER);
                g2.drawRoundRect(0, 0, getWidth() - 1, getHeight() - 1, 12, 12);
                g2.dispose();
            }
            setForeground(isSelected() ? Theme.INK : Theme.MUTED);
            super.paintComponent(g);
        }
    }
}