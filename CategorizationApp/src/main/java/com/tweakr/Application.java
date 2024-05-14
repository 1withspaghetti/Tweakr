package com.tweakr;

import javax.swing.*;

public class Application extends JFrame {

    JMenuBar menu;
    JMenu fileMenu;
    JMenuItem openFolderButton;
    JMenuItem exitButton;

    public Application() {
        super("Tweakr Categorization");

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(400, 800);
        setResizable(true);

        menu = new JMenuBar();


        fileMenu = new JMenu("File");
        fileMenu.add(openFolderButton = new JMenuItem("Open Folder"));
        fileMenu.addSeparator();
        fileMenu.add(exitButton = new JMenuItem("Exit"));

        menu.add(fileMenu);
        setJMenuBar(menu);


        setVisible(true);
    }
}
