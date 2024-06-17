package com.tweakr.util;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.LinkedHashMap;

public class AchievementManager extends JDialog {

    public static final double FPS = 30;
    public static final long TOAST_DURATION = 5000;
    public static final long SWIPE_DURATION = 500;

    private JComponent parent;

    // End time, message data
    private LinkedHashMap<ToastMessage, Long> messages = new LinkedHashMap<>();

    Timer timer = null;

    public AchievementManager(JComponent parent) {
        this.parent = parent;

        setUndecorated(true);
        setAlwaysOnTop(true);
        setFocusableWindowState(false);
        setLayout(new GridLayout(0, 1));

        timer = new Timer((int) (1000 / FPS), new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                messages.forEach((toast, expires)->{
                    if (expires < System.currentTimeMillis()) {
                        toast.dispose();
                        messages.remove(toast);
                        remove(toast);
                    } else if (expires - SWIPE_DURATION < System.currentTimeMillis()) {
                        toast.setOpacity((float)(expires - System.currentTimeMillis()) / SWIPE_DURATION);
                    }
                });
            }
        });
    }

    public void pushToastMessage(String toastString) {
        ToastMessage message = new ToastMessage(parent, toastString);
        messages.put(message, System.currentTimeMillis() + TOAST_DURATION);
        add(message);
    }

    @Override
    public void dispose() {
        super.dispose();
        if (timer != null) timer.stop();
        messages.clear();
    }
}
