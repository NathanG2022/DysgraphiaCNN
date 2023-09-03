import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class Main {

    public static void listOfFiles(File dirPath, FileWriter writer){
        File filesList[] = dirPath.listFiles();
        for(File file : filesList) {
            try {
                if (file.isFile()) {
                    writer.write(file.toString()+String.format("%n"));
                } else {
                    listOfFiles(file, writer);
                }
            } catch(IOException e) {
                System.out.println("An error occurred: " + e.getMessage());
            }
        }
    }

    public static void writeFileList(String filePath, FileWriter writer) {
        int[] userInfo = {150, 1, 101, 104, 107, 108, 109, 11, 110, 112, 113, 114, 115, 118, 119, 120, 121, 122, 124,
                125, 128, 129, 13, 130, 131, 133, 134, 135, 139, 14, 141, 148, 149, 15, 16, 17, 19, 21, 22, 26, 29, 32,
                36, 38, 39, 42, 44, 48, 49, 6, 7, 8, 91, 92, 93, 95, 99,};
        BufferedReader reader;
        try {
            File file = new File(filePath);
            reader = new BufferedReader(new FileReader(filePath));
            String line = reader.readLine();
            line = reader.readLine();
            String index = file.getName().substring(3,6);
            String type = "NORMAL";

            for (int i = 0; i < userInfo.length; i++) {
                if (userInfo[i]==Integer.parseInt(index)){
                    type = "DISGRA";
                    break;
                }
            }

            while (line != null) {
                line = line.replace(' ',',');
                writer.write(index + "," + type + "," + line + String.format("%n"));
                // read next line
                line = reader.readLine();
            }

            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {

        String folderPath = "C:/Repos/PreProcessing/src/Data";
        String outputFilePath = "data.csv";

        try {
            // Create the output file
            File outputFile = new File(outputFilePath);
            outputFile.createNewFile();
            FileWriter writer = new FileWriter(outputFile);

            FileWriter inputPath = new FileWriter("path.txt");
            File folder = new File(folderPath);
            listOfFiles(folder, inputPath);
            inputPath.close();

            BufferedReader reader;

            reader = new BufferedReader(new FileReader("path.txt"));
            String line = reader.readLine();

            while (line != null) {
                writeFileList(line, writer);
                // read next line
                line = reader.readLine();
            }

            reader.close();
            writer.close();

        } catch (IOException e) {
            System.out.println("An error occurred: " + e.getMessage());
        }
    }
}
