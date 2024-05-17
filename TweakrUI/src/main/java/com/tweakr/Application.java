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
    JTextArea textOutput;
    JButton fileButton;
    JButton exitButton;
    File imagePath;
    Image currentIcon;
    ImagePanel displayedImage;
    JButton sendButton;
    Random rand = new Random();


    public Application() throws UnsupportedLookAndFeelException, ClassNotFoundException, InstantiationException, IllegalAccessException {

        super("Tweakr");

        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());

        // setting some variables for the application window
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(true);
        setSize(512, 512);

        // setting the layout of JFrame
        setLayout(new BorderLayout());

        //setting layout of different panels
        panelTop = new JPanel(new FlowLayout(FlowLayout.CENTER, 5, 5));
        panelBottom = new JPanel(new FlowLayout(FlowLayout.CENTER, 5, 5));

        // setting up all the other components being used
        fileChooser = new JFileChooser(System.getProperty("user.dir"));
        fileButton = new JButton("Choose files");
        exitButton = new JButton("Exit");
        displayedImage = new ImagePanel(true);
        displayedImage.setSize(new Dimension(1, 1));
        sendButton = new JButton("send");

        // a couple more things need to be set up for the output box
        textOutput = new JTextArea(1, 15 );
        textOutput.setEditable(false);
        textOutput.setFont(textOutput.getFont().deriveFont(15f));

        // adding buttons and labels to their proper panels
        panelTop.add(fileButton);
        panelTop.add(exitButton);

        panelBottom.add(sendButton);
        panelBottom.add(textOutput);

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

        // currently the application doesnt have any AI app to talk with, so it just gets a random bool
        sendButton.addActionListener(e -> {

            textOutput.setText(null);

            if (currentIcon != null) {

                boolean tweakin = rand.nextBoolean();

                //textOutput.
                textOutput.append("this fish is ");
                if (tweakin) {
                    textOutput.append("tweakin");
                } else {

                    textOutput.append("locked in");
                }
            } else {

                textOutput.append("please select an image to inspect");
            }
        });

        // setting the JFrame visible so it actually displays everything
        setVisible(true);
    }
}
