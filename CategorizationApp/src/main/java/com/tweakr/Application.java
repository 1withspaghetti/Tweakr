package com.tweakr;

import javax.swing.*;
import java.awt.*;

public class Application extends JFrame {

    JMenuBar menu;
    JMenu fileMenu;
    JMenuItem openFolderButton;
    JMenuItem closeFolderButton;
    JMenuItem exitButton;

    CategorizationUI categorizationUI;

    public Application() throws UnsupportedLookAndFeelException, ClassNotFoundException, InstantiationException, IllegalAccessException {
        super("Tweakr Categorization");
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 800);
        setResizable(true);
        setLocationRelativeTo(null);

        menu = new JMenuBar();


        fileMenu = new JMenu("File");
        fileMenu.add(openFolderButton = new JMenuItem("Open Folder"));
        fileMenu.add(closeFolderButton = new JMenuItem("Close Current Folder"));
        fileMenu.addSeparator();
        fileMenu.add(exitButton = new JMenuItem("Exit"));

        menu.add(fileMenu);
        setJMenuBar(menu);

        categorizationUI = new CategorizationUI(this);
        add(categorizationUI);

        openFolderButton.addActionListener(e->categorizationUI.openFolder());
        closeFolderButton.addActionListener(e->categorizationUI.closeFolder());
        exitButton.addActionListener(e->dispose());

        setVisible(true);
    }

    public void showErrorMessage(Exception ex) {
        JOptionPane.showMessageDialog(this, ex.getClass().getName()+": "+ex.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
    }
}
