package com.tweakr.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class FileManager implements Iterator<Path> {

    public static final Path OUTPUT_FOLDER_TWEAKING = Path.of(System.getProperty("user.dir"), "output", "tweaking");
    public static final Path OUTPUT_FOLDER_LOCKED_IN = Path.of(System.getProperty("user.dir"), "output", "lockedIn");

    private final Path inputFolder;
    private int currentIndex;
    private final Path[] files;

    /**
     * Creates a new FileManager that iterates through the given folder to categorize the image files as tweaking or lockedIn.
     * @param folder - The folder to iterate through
     */
    public FileManager(Path folder) {
        if (folder == null || !Files.isDirectory(folder))
            throw new IllegalArgumentException("Input folder must be a valid directory");
        this.inputFolder = folder;
        currentIndex = 0;

        files = listFiles().toArray(new Path[0]);

        try {
            if (!Files.isDirectory(OUTPUT_FOLDER_TWEAKING)) Files.createDirectories(OUTPUT_FOLDER_TWEAKING);
            if (!Files.isDirectory(OUTPUT_FOLDER_LOCKED_IN)) Files.createDirectories(OUTPUT_FOLDER_LOCKED_IN);
        } catch (IOException e) {
            throw new RuntimeException("Error while creating output directories", e);
        }
    }

    /**
     * If the folder has another file to process
     *
     * @return True if there is at least one file remaining in the folder
     */
    public boolean hasNext() {
        return currentIndex + 1 < files.length;
    }

    /**
     * Gets the next file in the folder, can only be called once per loop.
     *
     * @return The next file
     */
    public Path next() {
        if (!hasNext()) throw new NoSuchElementException("Cannot get next file when no files are remaining.");
        currentIndex++;
        return files[currentIndex];
    }

    /**
     * Moves the current file to a destination folder depending on the input.
     * <b>This can only be called once per {@link #next()} call</b>
     *
     * @param isTweaking - If the file should be moved to OUTPUT_FOLDER_TWEAKING, otherwise OUTPUT_FOLDER_LOCKED_IN
     * @throws IOException If an IOException occurs during the file move
     */
    public void move(boolean isTweaking) throws IOException {
        Path current = files[currentIndex];
        if (isTweaking)
            Files.move(current, OUTPUT_FOLDER_TWEAKING.resolve(current.getFileName()));
        else
            Files.move(current, OUTPUT_FOLDER_LOCKED_IN.resolve(current.getFileName()));
    }

    private List<Path> listFiles() {
        try (final Stream<Path> stream = Files.list(inputFolder)) {

            return stream.filter(Files::isRegularFile).collect(Collectors.toList());

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
