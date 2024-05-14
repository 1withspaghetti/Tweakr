package com.tweakr;
import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class Application extends JFrame {

    JFileChooser fileChooser;
    JPanel panel;
    JButton fileButton;


    public Application() {

        super("Tweakr");

        fileChooser = new JFileChooser("C:");
        panel = new JPanel();
        fileButton = new JButton("Choose files");

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(true);
        setSize(512, 512);

        panel.add(fileButton);
        add(panel);
        setVisible(true);

        fileButton.addActionListener(e -> {
            fileChooser.showSaveDialog(null);
        });
    }
}
