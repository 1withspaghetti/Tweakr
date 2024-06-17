package com.tweakr.util;

import java.io.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A class for connecting with the python scripts in the ./Tweakr AI/ project folder.
 * Contains a single thread ExecutorService for multithreading results
 */
public class PythonInterface implements Closeable {

    Pattern resultsPattern = Pattern.compile("(true|false) (\\d+(.\\d+)?)", Pattern.CASE_INSENSITIVE);

    ExecutorService exe;
    Future task;

    /**
     * Creates the new PythonInterface with a new single thread executor service
     */
    public PythonInterface() {
        exe = Executors.newSingleThreadExecutor();
    }

    /**
     * Runs the run.py file with the argument of the absolute file path.
     * @param image - The image file to use
     * @param resultCallback - The callback when the python file has printed results
     * @param errCallback - The callback if there is any sort of error
     */
    public void runModelOnImage(File image,
                                       Consumer<PredictionResult> resultCallback,
                                       Consumer<Exception> errCallback) {
        task = exe.submit(()->{
            try {
                ProcessBuilder builder = new ProcessBuilder("python", "\"../Tweakr AI/run.py\"", "\""+image.getAbsolutePath()+"\"");
                builder.directory(new File("../Tweakr AI/"));
                builder.redirectErrorStream(true);
                Process p = builder.start();
                BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println("Python > "+line);
                    Matcher m = resultsPattern.matcher(line);
                    if (m.matches()) {
                        System.out.println("Found result: "+m.group(1)+" with "+m.group(2));
                        resultCallback.accept(
                                new PredictionResult(
                                        Boolean.parseBoolean(m.group(1).toLowerCase()),
                                        Double.parseDouble(m.group(2))
                                ));
                    }
                }
                p.onExit().thenRun(()->{
                   task.cancel(true);
                });
            } catch (Exception e) {
                errCallback.accept(e);
            }
        });
    }

    /**
     * Closes the executor service
     * @throws IOException
     */
    @Override
    public void close() throws IOException {
        exe.close();
    }

    /**
     * Class that describes the result of running the ai model on an image
     */
    public static class PredictionResult {

        private final boolean isTweaking;
        private final double confidence;

        protected PredictionResult(boolean isTweaking, double confidence) {
            this.isTweaking = isTweaking;
            this.confidence = confidence;
        }

        /**
         * @return "True" if the ai model thought the image was tweaking
         */
        public boolean isTweaking() {
            return isTweaking;
        }

        /**
         * @return A percent for how confident the model was for its classification, from 50.0 to 100.0
         */
        public double getConfidence() {
            return confidence;
        }

        @Override
        public String toString() {
            return "PredictionResult{" +
                    "isTweaking=" + isTweaking +
                    ", confidence=" + confidence +
                    '}';
        }
    }
}
