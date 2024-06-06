package com.tweakr;

import com.tweakr.util.FileManager;
import com.tweakr.util.ImagePanel;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.Flow;

public class CategorizationUI extends JPanel {

    // This FileManager is used as an iterator through the selected folder and classify/move the images to the output directories
    FileManager fileManager;

    // Children components
    JPanel centerPanel;
    ImagePanel imagePanel;
    JPanel noImagePanel;
    Box noImageInfo;
    JLabel noImageText;
    JButton fileChooserButton;
    JFileChooser fileChooser;

    JPanel buttonPanel;
    JButton lockedInButton;
    JButton tweakingButton;

    public CategorizationUI() {
        super(new BorderLayout());
        setAlignmentX(0.5f);
        setAlignmentY(0.5f);

        Font font = getFont().deriveFont(18f);

        centerPanel = new JPanel(new CardLayout());


        // Screen for when no folder is selected
        // Create a panel that centers a box containing the button and label
        noImagePanel = new JPanel(new GridBagLayout());
        noImageInfo = new Box(BoxLayout.Y_AXIS);

        fileChooserButton = new JButton("Open Folder");
        fileChooserButton.setFont(font);
        fileChooserButton.addActionListener(e->openFolder());
        fileChooserButton.setAlignmentX(0.5f);
        noImageInfo.add(fileChooserButton);

        noImageInfo.add(Box.createRigidArea(new Dimension(0, 5))); // Margin

        noImageText = new JLabel("Select a folder to start categorizing images");
        noImageText.setFont(font);
        noImageText.setAlignmentX(0.5f);
        noImageInfo.add(noImageText);

        noImagePanel.add(noImageInfo);
        centerPanel.add(noImagePanel, "noImagePanel");

        // Image panel displaying the image
        imagePanel = new ImagePanel(true);
        imagePanel.setVisible(false);
        centerPanel.add(imagePanel, "imagePanel");

        // Invisible file chooser for opening the gui
        fileChooser = new JFileChooser();
        fileChooser.setCurrentDirectory(new File("."));
        fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

        // File management
        fileManager = null;


        // Bottom buttons for categorizing
        buttonPanel = new JPanel();
        buttonPanel.setBackground(Color.WHITE);
        buttonPanel.setBorder(new EmptyBorder(10, 10, 10, 10));
        ((FlowLayout)buttonPanel.getLayout()).setHgap(30);
        buttonPanel.setVisible(false);

        lockedInButton = new JButton("← Locked In");
        lockedInButton.setMnemonic(KeyEvent.VK_LEFT);
        lockedInButton.setToolTipText("[ALT + LEFT]");
        lockedInButton.setFont(font);
        lockedInButton.setBackground(Color.GREEN);
        buttonPanel.add(lockedInButton);

        tweakingButton = new JButton("Tweaking →");
        tweakingButton.setMnemonic(KeyEvent.VK_RIGHT);
        tweakingButton.setToolTipText("[ALT + RIGHT]");
        tweakingButton.setFont(font);
        tweakingButton.setBackground(Color.ORANGE);
        buttonPanel.add(tweakingButton);

        lockedInButton.addActionListener(e->{
            try {
                fileManager.move(false);
                nextImage(false);
            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
        });

        tweakingButton.addActionListener(e->{
            try {
                fileManager.move(true);
                nextImage(true);
            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
        });

        // Add everything to the main layout
        add(centerPanel, BorderLayout.CENTER);
        add(buttonPanel, BorderLayout.SOUTH);
    }

    public void openFolder() {
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            System.out.println(fileChooser.getSelectedFile().toPath());
            fileManager = new FileManager(fileChooser.getSelectedFile().toPath());
            nextImage(false);
        }
    }

    public void closeFolder() {
        fileManager = null;
        setCurrentImage(null);
    }

    public void nextImage(boolean isTweaking) {
        if (fileManager == null) throw new RuntimeException("Cannot increment image with no folder selected");
        try {
            if (fileManager.hasNext()) {
                Path next = fileManager.next();
                System.out.println("Incremented to "+next);
                setCurrentImage(ImageIO.read(next.toFile()), isTweaking);
            } else {
                closeFolder();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void setCurrentImage(BufferedImage newImage) {
        imagePanel.setImage(newImage);

        CardLayout centerCardLayout = (CardLayout) centerPanel.getLayout();
        centerCardLayout.show(centerPanel, newImage != null ? "imagePanel" : "noImagePanel");

        buttonPanel.setVisible(newImage != null);
    }

    private void setCurrentImage(BufferedImage newImage, boolean wasTweaking) {
        imagePanel.swipeImage(newImage, wasTweaking);

        CardLayout centerCardLayout = (CardLayout) centerPanel.getLayout();
        centerCardLayout.show(centerPanel, newImage != null ? "imagePanel" : "noImagePanel");

        buttonPanel.setVisible(newImage != null);
    }
}
