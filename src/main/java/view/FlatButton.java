package view;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

/**
 * A flat, rounded button in either the primary (filled accent) or secondary
 * (white with an outline) style from the mockup. It looks the same on every
 * operating system, unlike the default Swing button.
 */
public class FlatButton extends JButton {

    /** Whether this is the filled, primary style. */
    private final boolean primary;

    /**
     * Creates a button.
     *
     * @param text    the label
     * @param primary {@code true} for the filled accent style, {@code false} for the outlined style
     */
    public FlatButton(String text, boolean primary) {
        super(text);
        this.primary = primary;
        setFont(Theme.sans(Font.BOLD, 14f));
        setForeground(primary ? Color.WHITE : Theme.INK);
        setContentAreaFilled(false);
        setFocusPainted(false);
        setBorderPainted(false);
        setOpaque(false);
        setBorder(BorderFactory.createEmptyBorder(10, 18, 10, 18));
        setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
    }

    /**
     * Paints the rounded background (dimmed when disabled, darker when pressed), then the label.
     *
     * @param g the graphics context
     */
    @Override
    protected void paintComponent(Graphics g) {
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        int arc = Theme.CONTROL_RADIUS * 2;
        if (primary) {
            Color fill = getModel().isPressed() ? Theme.ACCENT.darker() : Theme.ACCENT;
            g2.setColor(isEnabled() ? fill : Theme.BORDER);
            g2.fillRoundRect(0, 0, getWidth() - 1, getHeight() - 1, arc, arc);
        } else {
            g2.setColor(getModel().isPressed() ? Theme.BACKGROUND : Theme.SURFACE);
            g2.fillRoundRect(0, 0, getWidth() - 1, getHeight() - 1, arc, arc);
            g2.setColor(Theme.BORDER);
            g2.drawRoundRect(0, 0, getWidth() - 1, getHeight() - 1, arc, arc);
        }
        g2.dispose();
        setForeground(isEnabled() ? (primary ? Color.WHITE : Theme.INK) : Theme.MUTED);
        super.paintComponent(g);
    }
}