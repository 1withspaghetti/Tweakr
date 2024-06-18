# Tweakr Categorization App

This app emulates similar concepts to a dating app (such as Grindr, our name inspiration) where the user selects a folder of fish images, and then can swipe left to categorize that fish as locked in, or right for tweaking. The images are moved to two output folders when can then be copied into the AI project for the training.​

I highly recommend using [Intellij Idea Community Edition](https://www.jetbrains.com/idea/download/other.html) for editing and running, however you can still run the code with nothing but maven.


If you would like to run the code without an IDE, download [maven](https://maven.apache.org/download.cgi) and [add it to path](https://stackoverflow.com/questions/45119595/how-to-add-maven-to-the-path-variable).
Then, follow the commands below.

### To build the files:
```bash
mvn package
```

### To run the jar:
```bash
java -jar .\target\CategorizationApp-1.0-SNAPSHOT.jar
```