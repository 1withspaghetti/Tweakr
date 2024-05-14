package com.tweakr;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;

public class Application extends JFrame {

    JMenuBar menu;
    JMenu fileMenu;
    JMenuItem openFolderButton;
    JMenuItem exitButton;

    CategorizationUI categorizationUI;

    public Application() throws UnsupportedLookAndFeelException, ClassNotFoundException, InstantiationException, IllegalAccessException {
        super("Tweakr Categorization");
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        setSize(800, 800);
        setResizable(true);

        menu = new JMenuBar();


        fileMenu = new JMenu("File");
        fileMenu.add(openFolderButton = new JMenuItem("Open Folder"));
        fileMenu.addSeparator();
        fileMenu.add(exitButton = new JMenuItem("Exit"));

        menu.add(fileMenu);
        setJMenuBar(menu);

        categorizationUI = new CategorizationUI();
        add(categorizationUI, BorderLayout.CENTER);


        setVisible(true);
    }
}
