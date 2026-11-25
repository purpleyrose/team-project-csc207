package view;

import javax.swing.JComponent;
import java.awt.BasicStroke;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Path2D;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Draws the loss curve of a training run: dashed grid lines with loss values
 * on the left, step labels underneath, and the curve in the accent colour.
 *
 * <p>The curve's right end sits at the run's progress (step / total steps),
 * so it grows across the chart as training goes on.</p>
 */
public class LossChartPanel extends JComponent {

    /** Space for the loss labels on the left. */
    private static final int LEFT = 44;
    /** Space above the plot. */
    private static final int TOP = 8;
    /** Space for the step labels underneath. */
    private static final int BOTTOM = 24;
    /** Number of horizontal grid lines. */
    private static final int GRID_LINES = 4;

    /** The losses to draw, in order. */
    private List<Float> losses = Collections.emptyList();
    /** How far through the run the last loss is, from 0 to 1. */
    private double progress;
    /** The run's total number of steps, for the right-hand label. */
    private int totalSteps;

    /** Creates an empty chart. */
    public LossChartPanel() {
        setPreferredSize(new Dimension(400, 220));
    }

    /**
     * Replaces the data shown. The list is copied.
     *
     * @param newLosses     the losses so far, in order
     * @param step          the latest step
     * @param newTotalSteps the total number of steps in the run
     */
    public void setData(List<Float> newLosses, int step, int newTotalSteps) {
        this.losses = new ArrayList<>(newLosses);
        this.totalSteps = newTotalSteps;
        this.progress = newTotalSteps > 0 ? Math.min(1.0, (double) step / newTotalSteps) : 1.0;
        repaint();
    }

    /**
     * Paints the grid, labels and curve, or a hint when there is no data yet.
     *
     * @param g the graphics context
     */
    @Override
    protected void paintComponent(Graphics g) {
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        int w = getWidth() - LEFT;
        int h = getHeight() - TOP - BOTTOM;
        if (w <= 0 || h <= 0) {
            g2.dispose();
            return;
        }
        if (losses.isEmpty()) {
            drawCentered(g2, "Press Train to start a run");
            g2.dispose();
            return;
        }

        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float v : losses) {
            if (Float.isFinite(v)) {
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
        }
        if (min > max) {
            drawCentered(g2, "Loss is not a number");
            g2.dispose();
            return;
        }
        float pad = Math.max((max - min) * 0.08f, 0.05f);
        float lo = Math.max(0f, min - pad);
        float hi = max + pad;

        drawGrid(g2, w, h, lo, hi);
        drawCurve(g2, w, h, lo, hi);
        g2.dispose();
    }

    /**
     * Draws the dashed grid lines, their loss labels, and the step labels.
     *
     * @param g2 the graphics context
     * @param w  plot width
     * @param h  plot height
     * @param lo loss at the bottom of the plot
     * @param hi loss at the top of the plot
     */
    private void drawGrid(Graphics2D g2, int w, int h, float lo, float hi) {
        g2.setFont(Theme.mono(11f));
        FontMetrics fm = g2.getFontMetrics();
        g2.setStroke(new BasicStroke(1f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER,
                10f, new float[]{3f, 5f}, 0f));
        for (int i = 0; i <= GRID_LINES; i++) {
            int y = TOP + Math.round(h * i / (float) GRID_LINES);
            g2.setColor(Theme.BORDER);
            g2.drawLine(LEFT, y, LEFT + w, y);
            float value = hi - (hi - lo) * i / GRID_LINES;
            String label = String.format("%.2f", value);
            g2.setColor(Theme.MUTED);
            g2.drawString(label, LEFT - 8 - fm.stringWidth(label), y + fm.getAscent() / 2 - 1);
        }
        int baseline = TOP + h + fm.getAscent() + 8;
        g2.drawString("step 0", LEFT, baseline);
        String end = String.format("step %,d", totalSteps);
        g2.drawString(end, LEFT + w - fm.stringWidth(end), baseline);
    }

    /**
     * Draws the loss curve. Very long runs are thinned to about two points per
     * pixel, which looks the same and keeps repainting fast.
     *
     * @param g2 the graphics context
     * @param w  plot width
     * @param h  plot height
     * @param lo loss at the bottom of the plot
     * @param hi loss at the top of the plot
     */
    private void drawCurve(Graphics2D g2, int w, int h, float lo, float hi) {
        int n = losses.size();
        double right = w * progress;
        int stride = Math.max(1, n / Math.max(1, w * 2));
        Path2D.Float path = new Path2D.Float();
        for (int i = 0; i < n; i += stride) {
            addPoint(path, i, n, right, h, lo, hi);
        }
        if ((n - 1) % stride != 0) {
            addPoint(path, n - 1, n, right, h, lo, hi);
        }
        g2.setColor(Theme.ACCENT);
        g2.setStroke(new BasicStroke(2.5f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        if (n == 1) {
            float y = toY(losses.get(0), h, lo, hi);
            g2.fillOval(LEFT - 3, Math.round(y) - 3, 6, 6);
        } else {
            g2.draw(path);
        }
    }

    /**
     * Adds point {@code i} of {@code n} to the curve.
     *
     * @param path  the curve being built
     * @param i     the point's index
     * @param n     the number of points
     * @param right x offset of the last point
     * @param h     plot height
     * @param lo    loss at the bottom of the plot
     * @param hi    loss at the top of the plot
     */
    private void addPoint(Path2D.Float path, int i, int n, double right, int h, float lo, float hi) {
        float loss = losses.get(i);
        if (!Float.isFinite(loss)) {
            return;
        }
        float x = (float) (LEFT + (n == 1 ? 0 : right * i / (n - 1)));
        float y = toY(loss, h, lo, hi);
        if (path.getCurrentPoint() == null) {
            path.moveTo(x, y);
        } else {
            path.lineTo(x, y);
        }
    }

    /**
     * Converts a loss to a y coordinate.
     *
     * @param loss the loss
     * @param h    plot height
     * @param lo   loss at the bottom of the plot
     * @param hi   loss at the top of the plot
     * @return the y coordinate
     */
    private float toY(float loss, int h, float lo, float hi) {
        return TOP + h * (hi - loss) / (hi - lo);
    }

    /**
     * Draws a muted message in the middle of the chart.
     *
     * @param g2   the graphics context
     * @param text the message
     */
    private void drawCentered(Graphics2D g2, String text) {
        g2.setFont(Theme.sans(Font.PLAIN, 13f));
        g2.setColor(Theme.MUTED);
        FontMetrics fm = g2.getFontMetrics();
        g2.drawString(text, (getWidth() - fm.stringWidth(text)) / 2, getHeight() / 2);
    }
}