package com.tweakr;
import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;

public class Application extends JFrame {

    JFileChooser fileChooser;
    JPanel panel;
    JTextArea text;
    JButton fileButton;
    JButton exitButton;
    File imagePath;
    ImageIcon currentIcon; // = new ImageIcon("C:\\Users\\willi\\OneDrive\\probably very important stuff\\Pictures\\Camera Roll\\captures and misc\\discord pfp\\goofy_dog.png");
    JLabel image;

    public Application() throws UnsupportedLookAndFeelException, ClassNotFoundException, InstantiationException, IllegalAccessException {

        super("Tweakr");

        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);
        setSize(512, 512);


        fileChooser = new JFileChooser(System.getProperty("user.dir"));
        panel = new JPanel();
        fileButton = new JButton("Choose files");
        exitButton = new JButton("Exit");
        image = new JLabel(currentIcon);

        setLayout(new BorderLayout());

        panel.add(fileButton);
        panel.add(exitButton);
        panel.add(image);
        add(panel);


        fileButton.addActionListener(e -> {

            if (image != null) panel.remove(image);
            fileChooser.showSaveDialog(null);
            imagePath = fileChooser.getSelectedFile().getAbsoluteFile();
            currentIcon = new ImageIcon(String.valueOf(imagePath));
            image = new JLabel(currentIcon);

            panel.add(image);
            panel.revalidate();
            panel.repaint();
        });

        exitButton.addActionListener(e -> {
            dispose();
        });

        setVisible(true);
    }
}
