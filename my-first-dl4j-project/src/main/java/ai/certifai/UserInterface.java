package ai.certifai;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;

public class UserInterface {

    private static JLabel label;

    public static void main(String[] args) {
        JFrame frame= new JFrame("Acne Analysis");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(900, 600);
        GridLayout grid = new GridLayout(1,3,10,10);
        frame.setLayout(grid);
        label = new JLabel();
        frame.add(label);
        JTextArea text = new JTextArea();
        text.setText("Prediction: xxx \n\n Probability: xxx");
        text.setEditable(false);
        frame.add(text);
        JButton btnUpload = new JButton("Upload Image");
        frame.add(btnUpload);

        frame.setVisible(true);

//        frame.setResizable(false);

        btnUpload.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JFileChooser file = new JFileChooser();
                file.setCurrentDirectory(new File(System.getProperty("user.home")));
                //filter the files
                FileNameExtensionFilter filter = new FileNameExtensionFilter("*.Images", "jpg","gif","png");
                file.addChoosableFileFilter(filter);
                file.setDialogTitle("Select an Acne Image");
                int result = file.showSaveDialog(null);

                //if the user click on save in Jfilechooser
                if(result == JFileChooser.APPROVE_OPTION){
                    File selectedFile = file.getSelectedFile();
                    String path = selectedFile.getAbsolutePath();
                    label.setIcon(ResizeImage(path));

                    // Check if model exist
                    File modelSave =  new File("C:\\Users\\User\\Desktop\\TrainingLabs-main\\TrainingLabs-main\\Acne-Analysis-transfer-learning\\Acne-Analysis-transfer-learning\\acne-model-skin.zip");
                    if(modelSave.exists() == false)
                    {
                        System.out.println("Model not exist. Abort");
                        return;
                    }

                    //Load Image
                    File imageToTest = null;
                    NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
                    try {
                        //Load Model
                        ComputationGraph modelVGG16 = ModelSerializer.restoreComputationGraph(modelSave);
                        INDArray image = loader.asMatrix(selectedFile);

                        //Normalize Image
                        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
                        scaler.transform(image);

                        INDArray[] output = modelVGG16.output(image);

                        System.out.println("Label:         " + Nd4j.argMax(output[0], 1).getFloat(0));;
                        System.out.println("Probabilities: " + output[0].toString());
                        String txt1 = "Prediction:\nNormal: "+output[0].getFloat(3)+"\nLevel 0: "+output[0].getFloat(0)+"\nLevel 1: "+output[0].getFloat(1)+"\nLevel 2: "+output[0].getFloat(2);
                        String txt2 = "Level 0: Drink More Water\nLevel 1: Avoid Greasy Food\nLevel 2: Apply Retinol";
                        text.setText(txt1 +"\n\nSolution:\n"+txt2);

                    } catch (IOException ioException) {
                        ioException.printStackTrace();
                    }
                }
            }
        });
    }

    // Methode to resize imageIcon with the same size of a Jlabel
    public static ImageIcon ResizeImage(String ImagePath)
    {
        ImageIcon MyImage = new ImageIcon(ImagePath);
        Image img = MyImage.getImage();
        Image newImg = img.getScaledInstance(label.getWidth(), label.getHeight(), Image.SCALE_SMOOTH);
        ImageIcon image = new ImageIcon(newImg);
        return image;
    }
}
