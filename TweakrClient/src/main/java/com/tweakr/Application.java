package com.tweakr;
import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class Application extends JFrame {

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
    SpringLayout spring = new SpringLayout();


    public Application() throws UnsupportedLookAndFeelException, ClassNotFoundException, InstantiationException, IllegalAccessException {

        super("Tweakr");

        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());

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
        //displayedImage.setSize(new Dimension(1, 1));


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

        // boxlayout wasnt working so i just made a 3x3 grid with empty containers on the sides
        bottomGrid.add(Box.createHorizontalStrut(40));
        bottomGrid.add(sendButton);

        bottomGrid.add(Box.createHorizontalStrut(40));
        bottomGrid.add(Box.createHorizontalStrut(40));
        bottomGrid.add(textOutput);
        bottomGrid.add(Box.createHorizontalStrut(40));

        bottomGrid.add(Box.createHorizontalStrut(40));
        bottomGrid.add(confidenceOutput);
        bottomGrid.add(Box.createHorizontalStrut(40));

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
        exitButton.addActionListener(e -> {
            dispose();
        });

        // currently the application doesn't have any AI app to talk with, so it just gets a random bool
        sendButton.addActionListener(e -> {

            textOutput.setText(null);
            confidenceOutput.setText(null);

            if (currentIcon != null) {

                boolean tweakin = rand.nextBoolean();

                textOutput.append("this fish is ");
                if (tweakin) {
                    textOutput.append("tweakin");
                } else {
                    textOutput.append("locked in");
                }

                confidenceOutput.append("Confidence: " + (rand.nextFloat() * 100) + "%");

            } else {

                textOutput.append("please select an image to inspect");
            }
        });

        // setting the JFrame visible so it actually displays everything
        setVisible(true);
    }
}
