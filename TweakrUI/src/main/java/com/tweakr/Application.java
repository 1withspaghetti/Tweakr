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
    ImageIcon currentIcon = new ImageIcon("C:\\Users\\willi\\OneDrive\\probably very important stuff\\Pictures\\Camera Roll\\captures and misc\\discord pfp");
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
        text = new JTextArea("haiii",1, 10);
        exitButton = new JButton("Exit");
        //image = new JLabel()

        //fileButton.setlo
        setLayout(new GridBagLayout());

        panel.add(fileButton);
        //panel.add(text);
        panel.add(exitButton);
        //panel.setLayout(new )
        add(panel);
        setVisible(true);

        fileButton.addActionListener(e -> {
            //fileChooser.showSaveDialog(null);
            //image = new JLabel()
            System.out.println("A");
            panel.add(new JButton("jdiejdjiw"));
            System.out.println("B");
        });

        exitButton.addActionListener(e -> {
            //dispatchEvent(new WindowEvent(Application, WindowEvent.WINDOW_CLOSED));
            dispose();
        });
    }
}
