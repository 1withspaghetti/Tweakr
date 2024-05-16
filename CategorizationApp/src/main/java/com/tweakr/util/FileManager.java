package com.tweakr.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.stream.Stream;

public class FileManager implements Iterator<Path> {

    public static final Path OUTPUT_FOLDER_TWEAKING = Path.of(System.getProperty("user.dir"), "output", "tweaking");
    public static final Path OUTPUT_FOLDER_LOCKED_IN = Path.of(System.getProperty("user.dir"), "output", "lockedIn");

    private Path inputFolder;
    private Path current;

    /**
     * Creates a new FileManager that iterates through the given folder to categorize the image files as tweaking or lockedIn.
     * @param folder - The folder to iterate through
     */
    public FileManager(Path folder) {
        if (folder == null || Files.isDirectory(folder))
            throw new IllegalArgumentException("Input folder must be a valid directory");
        this.inputFolder = folder;
    }

    /**
     * If the folder has another file to process
     *
     * @return True if there is at least one file remaining in the folder
     */
    public boolean hasNext() {
        return listFiles().findAny().isPresent();
    }

    /**
     * Gets the next file in the folder, can only be called once per loop.
     *
     * @return The next file
     */
    public Path next() {
        return current = listFiles().findFirst().orElseThrow();
    }

    /**
     * Moves the current file to a destination folder depending on the input.
     * <b>This can only be called once per {@link #next()} call</b>
     *
     * @param isTweaking - If the file should be moved to OUTPUT_FOLDER_TWEAKING, otherwise OUTPUT_FOLDER_LOCKED_IN
     * @throws IOException If an IOException occurs during the file move
     */
    public void move(boolean isTweaking) throws IOException {
        if (isTweaking)
            Files.move(current, OUTPUT_FOLDER_TWEAKING.resolve(current.getFileName()));
        else
            Files.move(current, OUTPUT_FOLDER_LOCKED_IN.resolve(current.getFileName()));
    }

    private Stream<Path> listFiles() {
        try (final Stream<Path> stream = Files.list(inputFolder)) {

            return stream.filter(Files::isRegularFile);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
