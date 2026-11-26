package view;

import javax.swing.JComponent;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

/**
 * A thin rounded progress bar. Swing's {@link javax.swing.JProgressBar} ignores
 * custom colours on some operating systems (notably macOS), so this draws its own.
 */
public class ProgressTrack extends JComponent {

    /** Bar height in pixels. */
    private static final int HEIGHT = 8;
    /** Completed fraction, from 0 to 1. */
    private double fraction;

    /** Creates an empty progress bar. */
    public ProgressTrack() {
        setPreferredSize(new Dimension(100, HEIGHT));
    }

    /**
     * Sets how much of the bar is filled. Values are clamped to [0, 1].
     *
     * @param value the completed fraction
     */
    public void setFraction(double value) {
        this.fraction = Math.max(0, Math.min(1, value));
        repaint();
    }

    /**
     * Paints the track and the filled part.
     *
     * @param g the graphics context
     */
    @Override
    protected void paintComponent(Graphics g) {
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        int y = (getHeight() - HEIGHT) / 2;
        g2.setColor(Theme.BACKGROUND);
        g2.fillRoundRect(0, y, getWidth(), HEIGHT, HEIGHT, HEIGHT);
        int filled = (int) Math.round(getWidth() * fraction);
        if (filled > 0) {
            g2.setColor(Theme.ACCENT);
            g2.fillRoundRect(0, y, Math.max(filled, HEIGHT), HEIGHT, HEIGHT, HEIGHT);
        }
        g2.dispose();
    }
}