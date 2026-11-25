package view;

import javax.swing.BorderFactory;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JTextField;
import javax.swing.border.Border;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GraphicsEnvironment;
import java.util.Arrays;
import java.util.List;

/**
 * Colours, fonts and small component factories shared by every view, so the
 * Swing screens match the Figma mockup and each other.
 */
public final class Theme {

    /** Window background behind the cards. */
    public static final Color BACKGROUND = new Color(0xF4F5F7);
    /** Card and input background. */
    public static final Color SURFACE = Color.WHITE;
    /** Card, input and divider outlines. */
    public static final Color BORDER = new Color(0xE2E5EA);
    /** Main text colour. */
    public static final Color INK = new Color(0x17191E);
    /** Secondary text: labels, captions, axis text. */
    public static final Color MUTED = new Color(0x6B7280);
    /** The single accent colour: primary buttons, progress, the loss curve. */
    public static final Color ACCENT = new Color(0x2F5BEA);
    /** Pale accent, for the "running" pill and selected toggles. */
    public static final Color ACCENT_SOFT = new Color(0xE8EEFD);
    /** Text colour for success. */
    public static final Color SUCCESS = new Color(0x2E7D4F);
    /** Background for success. */
    public static final Color SUCCESS_SOFT = new Color(0xE3F4EA);
    /** Text colour for errors. */
    public static final Color DANGER = new Color(0xB42318);
    /** Background for errors. */
    public static final Color DANGER_SOFT = new Color(0xFEE4E2);

    /** Corner radius of cards. */
    public static final int CARD_RADIUS = 12;
    /** Corner radius of buttons and inputs. */
    public static final int CONTROL_RADIUS = 8;

    /** The UI font family: Inter when installed, otherwise the default sans-serif. */
    private static final String SANS = pick("Inter", Font.SANS_SERIF);
    /** The number font family: JetBrains Mono when installed, otherwise the default monospace. */
    private static final String MONO = pick("JetBrains Mono", Font.MONOSPACED);

    /** Not instantiable. */
    private Theme() {
    }

    /**
     * Returns {@code preferred} if that font family is installed, otherwise {@code fallback}.
     *
     * @param preferred the family to use when available
     * @param fallback  a logical family that always exists
     * @return the family name to use
     */
    private static String pick(String preferred, String fallback) {
        List<String> installed = Arrays.asList(
                GraphicsEnvironment.getLocalGraphicsEnvironment().getAvailableFontFamilyNames());
        return installed.contains(preferred) ? preferred : fallback;
    }

    /**
     * Returns the UI font.
     *
     * @param style a {@link Font} style such as {@link Font#BOLD}
     * @param size  the point size
     * @return the font
     */
    public static Font sans(int style, float size) {
        return new Font(SANS, style, 1).deriveFont(size);
    }

    /**
     * Returns the monospace font used for numbers.
     *
     * @param size the point size
     * @return the font
     */
    public static Font mono(float size) {
        return new Font(MONO, Font.PLAIN, 1).deriveFont(size);
    }

    /**
     * Creates a card title, such as "Hyperparameters".
     *
     * @param text the title
     * @return the label
     */
    public static JLabel title(String text) {
        JLabel label = new JLabel(text);
        label.setFont(sans(Font.BOLD, 17f));
        label.setForeground(INK);
        return label;
    }

    /**
     * Creates a small upper-case section heading, such as "MODEL SIZE".
     *
     * @param text the heading
     * @return the label
     */
    public static JLabel overline(String text) {
        JLabel label = new JLabel(text.toUpperCase());
        label.setFont(sans(Font.BOLD, 11f));
        label.setForeground(MUTED);
        return label;
    }

    /**
     * Creates a field caption, such as "Learning rate".
     *
     * @param text the caption
     * @return the label
     */
    public static JLabel caption(String text) {
        JLabel label = new JLabel(text);
        label.setFont(sans(Font.PLAIN, 12f));
        label.setForeground(MUTED);
        return label;
    }

    /**
     * Creates a single-line text input styled like the mockup's fields.
     *
     * @param initial the initial text
     * @return the text field
     */
    public static JTextField field(String initial) {
        JTextField field = new JTextField(initial);
        field.setFont(mono(13f));
        field.setForeground(INK);
        field.setBackground(SURFACE);
        field.setCaretColor(INK);
        field.setDisabledTextColor(MUTED);
        field.setBorder(inputBorder(BORDER));
        return field;
    }

    /**
     * Returns the outline-plus-padding border of an input.
     *
     * @param outline the outline colour
     * @return the border
     */
    public static Border inputBorder(Color outline) {
        return BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(outline, 1, true),
                BorderFactory.createEmptyBorder(8, 10, 8, 10));
    }

    /**
     * Lets a component stretch horizontally but never vertically, which
     * {@link javax.swing.BoxLayout} otherwise does.
     *
     * @param component the component
     * @param <T>       the component's type
     * @return the same component
     */
    public static <T extends JComponent> T fixHeight(T component) {
        component.setMaximumSize(new Dimension(Integer.MAX_VALUE, component.getPreferredSize().height));
        component.setAlignmentX(JComponent.LEFT_ALIGNMENT);
        return component;
    }
}