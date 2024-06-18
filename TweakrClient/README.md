# Tweakr Client App

The Tweakr app is the client-side application that users can submit images of fish. Tweakr app communicates with the Tweakr AI model by sending the users submitted image. The AI then processes the image and returns two variables, a true or false depending if the submitted fish image is tweaking or locked in, and a float value of the AIs confidence of its answer.

I highly recommend using [Intellij Idea Community Edition](https://www.jetbrains.com/idea/download/other.html) for editing and running, however you can still run the code with nothing but maven.


If you would like to run the code without an IDE, download [maven](https://maven.apache.org/download.cgi) and [add it to path](https://stackoverflow.com/questions/45119595/how-to-add-maven-to-the-path-variable).
Then, follow the commands below.

### To build the files:
```bash
mvn package
```

### To run the jar:
```bash
java -jar java -jar .\target\TweakrClient-1.0-SNAPSHOT.jar
```