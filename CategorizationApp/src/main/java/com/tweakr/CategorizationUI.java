package com.tweakr;

import javax.swing.*;
import java.awt.image.BufferedImage;

public class CategorizationUI extends Box {

    JLabel imageLabel;
    BufferedImage currentImage;

    public CategorizationUI() {
        super(BoxLayout.Y_AXIS);

        setCurrentImage(null);
    }

    public void setCurrentImage(BufferedImage img) {
        if (imageLabel != null) remove(imageLabel);
        currentImage = img;
        if (currentImage == null) {
            imageLabel = new JLabel("Press [File > Open Folder] to start categorizing images");
        } else {
            imageLabel = new JLabel(new ImageIcon(currentImage));
        }
        imageLabel.setAlignmentX(0.5f);
        imageLabel.setAlignmentY(0.5f);
        add(imageLabel);
    }
}
