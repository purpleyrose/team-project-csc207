package view;

import javax.swing.JPanel;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.LayoutManager;
import java.awt.RenderingHints;

/**
 * A panel with a rounded, optionally outlined background, used for cards,
 * stat tiles and banners.
 */
public class RoundedPanel extends JPanel {

    /** Corner radius in pixels. */
    private final int radius;
    /** Outline colour, or {@code null} for none. */
    private Color outline;

    /**
     * Creates a rounded panel.
     *
     * @param layout     the layout manager
     * @param background the fill colour
     * @param outline    the outline colour, or {@code null} for none
     * @param radius     the corner radius in pixels
     */
    public RoundedPanel(LayoutManager layout, Color background, Color outline, int radius) {
        super(layout);
        this.radius = radius;
        this.outline = outline;
        setBackground(background);
        setOpaque(false);
    }

    /**
     * Changes the fill and outline colours.
     *
     * @param background the new fill colour
     * @param newOutline the new outline colour, or {@code null} for none
     */
    public void setColors(Color background, Color newOutline) {
        setBackground(background);
        this.outline = newOutline;
        repaint();
    }

    /**
     * Paints the rounded background and outline, then the children.
     *
     * @param g the graphics context
     */
    @Override
    protected void paintComponent(Graphics g) {
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setColor(getBackground());
        g2.fillRoundRect(0, 0, getWidth() - 1, getHeight() - 1, radius * 2, radius * 2);
        if (outline != null) {
            g2.setColor(outline);
            g2.drawRoundRect(0, 0, getWidth() - 1, getHeight() - 1, radius * 2, radius * 2);
        }
        g2.dispose();
        super.paintComponent(g);
    }
}