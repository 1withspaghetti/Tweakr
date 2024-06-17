package com.tweakr.util;

import java.awt.Color;
import java.awt.GridBagLayout;
import java.awt.Window;
import java.awt.geom.RoundRectangle2D;

import javax.swing.*;

public class ToastMessage extends JDialog {

    public ToastMessage(JComponent caller, String toastString) {
        setUndecorated(true);
        setAlwaysOnTop(true);
        setFocusableWindowState(false);
        setLayout(new GridBagLayout());

        JPanel panel = new JPanel();
        panel.setBorder(BorderFactory.createEmptyBorder(5, 10, 5, 10));
        panel.setBackground(Color.WHITE);
        JLabel toastLabel = new JLabel(toastString);
        JLabel starLabel = new JLabel("⭐");
        starLabel.setFont(this.getFont().deriveFont(32f));
        panel.add(starLabel);
        panel.add(toastLabel);
        add(panel);
        pack();

        Window window = SwingUtilities.getWindowAncestor(caller);
        int xcoord = window.getLocationOnScreen().x + window.getWidth() / 2 - getWidth() / 2;
        int ycoord = window.getLocationOnScreen().y + (int)((double)window.getHeight() * 0.75) - getHeight() / 2;
        setLocation(xcoord, ycoord);
        setShape(new RoundRectangle2D.Double(0, 0, getWidth(), getHeight(), 30, 30));
        setVisible(true);
    }
}
