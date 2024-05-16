package com.tweakr;

import com.tweakr.util.FileManager;
import com.tweakr.util.ImagePanel;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

public class CategorizationUI extends Box {

    FileManager fileManager;

    BufferedImage currentImage;
    ImagePanel imagePanel;
    JLabel noImageText;

    JFileChooser fileChooser;

    public CategorizationUI() {
        super(BoxLayout.Y_AXIS);
        setAlignmentX(0.5f);
        setAlignmentY(0.5f);

        imagePanel = new ImagePanel(true);
        imagePanel.setVisible(false);
        noImageText = new JLabel("Press [File > Open Folder] to start categorizing images");
        noImageText.setFont(noImageText.getFont().deriveFont(16f));
        noImageText.setAlignmentX(0.5f);
        noImageText.setAlignmentY(0.5f);

        fileChooser = new JFileChooser();
        fileChooser.setCurrentDirectory(new File("."));
        fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

        fileManager = null;

        this.addMouseListener(new MouseListener() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (fileManager != null) nextImage();
            }

            @Override
            public void mousePressed(MouseEvent e) {}
            @Override
            public void mouseReleased(MouseEvent e) {}
            @Override
            public void mouseEntered(MouseEvent e) {}
            @Override
            public void mouseExited(MouseEvent e) {}
        });

        add(imagePanel);
        add(noImageText);
    }

    public void openFolder() {
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            System.out.println(fileChooser.getSelectedFile().toPath());
            fileManager = new FileManager(fileChooser.getSelectedFile().toPath());
            nextImage();
        }
    }

    public void nextImage() {
        if (fileManager == null) throw new RuntimeException("Cannot increment image with no folder selected");
        try {
            if (fileManager.hasNext()) {
                Path next = fileManager.next();
                setCurrentImage(ImageIO.read(next.toFile()));
            } else {
                fileManager = null;
                setCurrentImage(null);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void setCurrentImage(BufferedImage img) {
        currentImage = img;
        imagePanel.setImage(currentImage);

        imagePanel.setVisible(currentImage != null);
        noImageText.setVisible(currentImage == null);
    }
}
