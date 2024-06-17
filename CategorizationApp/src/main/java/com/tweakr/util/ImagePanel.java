package com.tweakr.util;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;

/**
 * <p>A swing component capable of displaying an image, resizing it to fit within its bounds
 * and keeping the aspect ratio</p>
 */
public class ImagePanel extends JPanel {

    public static final long SWIPE_DURATION = 500;
    public static final long SWIPE_FPS = 60;

    private Image image;
    private boolean centered;

    private Image prev = null;
    private long swipeTime = 0;
    private boolean swipeToLeft = true;

    Timer timer = null;

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
        this.centered = centered;
        if (image != null) {
            Dimension fit = containedImageDimension(image);
            setPreferredSize(new Dimension(image.getWidth(null), image.getHeight(null)));
            this.image = image.getScaledInstance(fit.width, fit.height, Image.SCALE_AREA_AVERAGING);
        } else {
            this.image = image;
        }
    }

    public Image getImage() {
        return image;
    }

    public void setImage(Image image) {
        if (image != null) {
            Dimension fit = containedImageDimension(image);
            setPreferredSize(new Dimension(image.getWidth(null), image.getHeight(null)));
            this.image = image.getScaledInstance(fit.width, fit.height, Image.SCALE_AREA_AVERAGING);
        } else {
            this.image = image;
        }
        repaint();
    }

    public void swipeImage(Image newImage, boolean toLeft) {
        prev = image;
        swipeTime = System.currentTimeMillis();
        swipeToLeft = toLeft;
        setImage(newImage);
        if (timer != null && timer.isRunning()) timer.stop();
        timer = new Timer((int) (1000 / SWIPE_FPS), (e)->{
            this.repaint(1000 / SWIPE_FPS);
            if (System.currentTimeMillis() - swipeTime > SWIPE_DURATION) {
                ((Timer) e.getSource()).stop();
            }
        });
        timer.start();
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

        Graphics2D g2d = (Graphics2D) g;
        Dimension size = getSize();

        double delta = Math.min((double) (System.currentTimeMillis() - swipeTime) / SWIPE_DURATION, 1);

        // Handle fading swipe
        if (prev != null && System.currentTimeMillis() - swipeTime < SWIPE_DURATION) {

            Dimension imgSize = containedImageDimension(prev);

            AffineTransform ogAf = g2d.getTransform();
            Composite ogComp = g2d.getComposite();

            //AffineTransform af = new AffineTransform();
            //af.setToTranslation(200 * delta * (swipeToLeft ? 1 : -1), 50);
            //af.setToRotation(Math.PI / 6 * delta * (swipeToLeft ? 1 : -1), (double) this.getWidth() / 2, this.getHeight());
            g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, (float) (1 - delta)));
            //g2d.setTransform(af);
            g2d.drawImage(
                    prev,//.getScaledInstance(imgSize.width, imgSize.height, Image.SCALE_AREA_AVERAGING),
                    (centered ? size.width / 2 - imgSize.width / 2 : 0) + (int)(200 * delta * (swipeToLeft ? 1 : -1)),
                    (centered ? size.height / 2 - imgSize.height / 2 : 0) + (int)(50 * delta),
                    null);

            g2d.setTransform(ogAf);
            g2d.setComposite(ogComp);
        }

        if (image != null) {

            Dimension imgSize = containedImageDimension(image);

            if (delta < 1) g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, (float) delta));

            // Draws the image
            g.drawImage(
                    image,//.getScaledInstance(imgSize.width, imgSize.height, Image.SCALE_AREA_AVERAGING),
                    centered ? size.width / 2 - image.getWidth(null) / 2 : 0,
                    centered ? size.height / 2 - image.getHeight(null) / 2 : 0,
                    null);
        }
    }

    private Dimension containedImageDimension(Image inner) {

        Dimension size = getSize();
        double compRatio = size.getWidth() / size.getHeight();

        // Gets the size of the component and image
        int imgWidth = inner.getWidth(null);
        int imgHeight = inner.getHeight(null);
        if (imgWidth == -1 || imgHeight == -1) throw new RuntimeException("The size of the image is unknown");
        double imgRatio = (double) imgWidth / imgHeight;

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

        return new Dimension(dWidth, dHeight);
    }
}
