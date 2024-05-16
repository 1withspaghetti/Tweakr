package com.tweakr;

import com.tweakr.util.ImagePanel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ContainerEvent;
import java.awt.event.ContainerListener;
import java.awt.image.BufferedImage;

public class CategorizationUI extends Box {

    ImagePanel imagePanel;
    JTextField noImageText;

    public CategorizationUI() {
        super(BoxLayout.Y_AXIS);
        setAlignmentX(0.5f);
        setAlignmentY(0.5f);

        imagePanel = new ImagePanel(true);
        noImageText = new JTextField("Press [File > Open Folder] to start categorizing images");
        setCurrentImage(null);

        add(imagePanel);
    }

    public void setCurrentImage(BufferedImage img) {
        imagePanel.setImage(img);

        imagePanel.setVisible(img != null);
        noImageText.setVisible(img == null);
    }
}
