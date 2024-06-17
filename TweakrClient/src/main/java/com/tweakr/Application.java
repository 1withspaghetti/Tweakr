package com.tweakr;
import com.tweakr.util.PythonInterface;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class Application extends JFrame {

    PythonInterface pythonInterface;

    // initializing UI components
    JFileChooser fileChooser;

    JPanel panelTop;
    JPanel panelBottom;     // :3
    JPanel bottomGrid;

    JButton fileButton;
    JButton exitButton;
    JButton sendButton;

    JTextArea textOutput;
    JTextArea confidenceOutput;

    File imagePath;
    Image currentIcon;
    ImagePanel displayedImage;
    Random rand = new Random();
    Color bg = new Color(150, 150, 150);

    public Application() throws UnsupportedLookAndFeelException, ClassNotFoundException, InstantiationException, IllegalAccessException {

        super("Tweakr");

        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());

        pythonInterface = new PythonInterface();

        // setting some variables for the application window
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(true);
        setSize(640, 512);

        // setting the layout of JFrame
        setLayout(new BorderLayout());

        //setting layout and background color of different panels
        panelTop = new JPanel(new FlowLayout(FlowLayout.CENTER, 5, 10));
        panelTop.setBackground(bg);

        panelBottom = new JPanel(new FlowLayout(FlowLayout.CENTER, 5, 10));
        panelBottom.setBackground(bg);

        bottomGrid = new JPanel(new GridLayout(3,3,0,10));
        bottomGrid.setBackground(bg);
        panelBottom.add(bottomGrid);


        // setting up all the other components being used
        fileChooser = new JFileChooser(System.getProperty("user.dir"));

        sendButton = new JButton("send");
        fileButton = new JButton("Choose image");
        exitButton = new JButton("Exit");

        displayedImage = new ImagePanel(true);

        // a couple more things need to be set up for the output box
        textOutput = new JTextArea(1, 15);
        textOutput.setEditable(false);
        textOutput.setFont(textOutput.getFont().deriveFont(15f));
        textOutput.setAlignmentX(JTextArea.CENTER_ALIGNMENT);

        confidenceOutput = new JTextArea();
        confidenceOutput.setEditable(false);
        confidenceOutput.setFont(confidenceOutput.getFont().deriveFont(15f));

        // adding buttons and labels to their proper panels
        panelTop.add(fileButton);
        panelTop.add(exitButton);

        // boxlayout wasn't working, so I just made a 3x3 grid with empty containers on the sides
        // turns out the width of the struts don't actually matter
        bottomGrid.add(Box.createHorizontalStrut(6));
        bottomGrid.add(sendButton);
        bottomGrid.add(Box.createHorizontalStrut(6));

        bottomGrid.add(Box.createHorizontalStrut(6));
        bottomGrid.add(textOutput);
        bottomGrid.add(Box.createHorizontalStrut(6));

        bottomGrid.add(Box.createHorizontalStrut(6));
        bottomGrid.add(confidenceOutput);
        bottomGrid.add(Box.createHorizontalStrut(6));

        // adding panels to JFrame
        add(panelTop, BorderLayout.NORTH);
        add(displayedImage, BorderLayout.CENTER);
        add(panelBottom, BorderLayout.SOUTH);

        // adding listener event to fileButton to open file explorer and display the selected image
        fileButton.addActionListener(e -> {

            fileChooser.showSaveDialog(null);
            imagePath = fileChooser.getSelectedFile();

            try {
                currentIcon = ImageIO.read(imagePath);
            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }

            displayedImage.setImage(currentIcon);
        });

        // making exitButton close the application when pressed
        exitButton.addActionListener(e -> dispose());


        sendButton.addActionListener(e -> {

            textOutput.setText(null);
            confidenceOutput.setText(null);

            if (currentIcon != null) {

                textOutput.setText("Loading...");

                // Submits to the python interface and provides callbacks that are ran on result and error
                pythonInterface.runModelOnImage(imagePath, result->{
                    textOutput.setText("this fish is "+(result.isTweaking() ? "tweakin" : "locked in"));
                    confidenceOutput.setText("Confidence: " + (result.getConfidence() + "%"));
                }, err->{
                    textOutput.setText("Error :c (check console)");
                    err.printStackTrace(System.err);
                });

            } else {

                textOutput.append("please select an image to inspect");
            }
        });

        // setting the JFrame visible so it actually displays everything
        setVisible(true);
    }
}
