package com.tweakr;

import javax.swing.*;
import java.awt.*;

/**
 * <p>A swing component capable of displaying an image, resizing it to fit within its bounds
 * and keeping the aspect ratio</p>
 */
public class ImagePanel extends JPanel {

    private Image image;
    private boolean centered;

    /**
     * <p>Creates a new ImagePanel instance that will try to display an image, scaling it to fit in the space of the component.</p>
     * <p>No image is defined so nothing will be displayed by default</p>
     */
    public ImagePanel() {
        this(null, false);
    }

    /**
     * <p>Creates a new ImagePanel instance that will try to display an image, scaling it to fit in the space of the component.</p>
     * <p>No image is defined so nothing will be displayed by default</p>
     * @param centered - If the image should be centered relative to the components size
     */
    public ImagePanel(boolean centered) {
        this(null, centered);
    }

    /**
     * <p>Creates a new ImagePanel instance that will try to display an image, scaling it to fit in the space of the component</p>
     * @param image - The image to be displayed, or null for nothing
     */
    public ImagePanel(Image image) {
        this(image, false);
    }

    /**
     * <p>Creates a new ImagePanel instance that will try to display an image, scaling it to fit in the space of the component</p>
     * @param image - The image to be displayed, or null for nothing
     * @param centered - If the image should be centered relative to the components size
     */
    public ImagePanel(Image image, boolean centered) {
        this.image = image;
        this.centered = centered;
        if (image != null) setPreferredSize(new Dimension(image.getWidth(null), image.getHeight(null)));
    }

    public Image getImage() {
        return image;
    }

    public void setImage(Image image) {
        this.image = image;
        if (image != null) setPreferredSize(new Dimension(image.getWidth(null), image.getHeight(null)));
        repaint();
    }

    public boolean isCentered() {
        return centered;
    }

    public void setCentered(boolean centered) {
        this.centered = centered;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        if (image != null) {

            // Gets the size of the component and image
            Dimension size = getSize();
            int imgWidth = image.getWidth(null);
            int imgHeight = image.getHeight(null);
            if (imgWidth == -1 || imgHeight == -1) throw new RuntimeException("The size of the image is unknown");

            double imgRatio = (double) imgWidth / imgHeight;
            double compRatio = size.getWidth() / size.getHeight();

            // Calculates the width and height of the image to fit inside the component
            int dWidth = 0;
            int dHeight = 0;
            if (imgRatio >= compRatio) {
                dWidth = (int) size.getWidth();
                dHeight = (int) (dWidth / imgRatio);
            } else {
                dHeight = (int) size.getHeight();
                dWidth = (int) (imgRatio * dHeight);
            }

            // Draws the image
            g.drawImage(
                    image.getScaledInstance(dWidth, dHeight, Image.SCALE_AREA_AVERAGING),
                    centered ? size.width / 2 - dWidth / 2 : 0,
                    centered ? size.height / 2 - dHeight / 2 : 0,
                    null);
        }
    }
}
