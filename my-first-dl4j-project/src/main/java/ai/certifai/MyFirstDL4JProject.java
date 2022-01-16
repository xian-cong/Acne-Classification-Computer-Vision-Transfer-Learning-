/*
 * Copyright (c) 2019 Skymind AI Bhd.
 * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.certifai;

import org.datavec.image.transform.*;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


public class MyFirstDL4JProject
{
    public static void main( String[] args ) throws Exception {
        Random randNumGen = new Random(1234);

        // image augmentation
        ImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform verticalFlip = new FlipImageTransform(0);
        ImageTransform cropImage = new CropImageTransform(25);
        ImageTransform rotateImage15 = new RotateImageTransform(randNumGen, 15);
        ImageTransform rotateImage30 = new RotateImageTransform(randNumGen, 30);
        ImageTransform showImage = new ShowImageTransform("Image",100);
        boolean shuffle = false;
        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip, 1.0),
                new Pair<>(verticalFlip, 1.0),
                new Pair<>(rotateImage15, 1.0),
                new Pair<>(rotateImage30, 1.0),
                new Pair<>(cropImage,1.0)
//                ,new Pair<>(showImage,1.0) //uncomment this to show transform image
        );

        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);

        String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        File datasetDir = new File("C:\\Users\\Admin\\Desktop\\acne detection\\skin");
        FileSplit wholeDataset = new FileSplit(datasetDir, allowedExtensions, randNumGen);

        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
        BalancedPathFilter balancedPathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelGenerator);

        InputSplit[] trainTestSplit = wholeDataset.sample(balancedPathFilter, 80, 20);

        InputSplit trainData = trainTestSplit[0];
        InputSplit testData = trainTestSplit[1];

        int height = 224;
        int width = 224;
        int channel = 3;
        int batchSize = 40;
        int numOfClass = 4;

        ImageRecordReader trainRecordReader = new ImageRecordReader(height, width,channel,  labelGenerator);
        ImageRecordReader testRecordReader = new ImageRecordReader(height, width, channel, labelGenerator);

        trainRecordReader.initialize(trainData,transform);
        testRecordReader.initialize(testData,transform);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1, numOfClass);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1, numOfClass);

        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        //load vgg16 zoo model
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        System.out.println(vgg16.summary());

        // Override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Adam(5e-4))
                .seed(1234)
                .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneConf)
                //set which layer to freeze
                .setFeatureExtractor("fc2") //the specified layer and above are "frozen"
                .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
                .addLayer("fc3",
                        new DenseLayer.Builder()
                                .nIn(4096)
                                .nOut(1024)
                                .activation(Activation.RELU)
                                .build(),
                        "fc2")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(1024).nOut(numOfClass)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).build(),
                        "fc3")
                .setOutputs("predictions")
                .build();
        System.out.println(vgg16Transfer.summary());

        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);
        vgg16Transfer.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

        Evaluation eval;
        Evaluation eval2;
        for (int i=1; i <= 200; i++){
            vgg16Transfer.fit(trainIter);
            eval2 = vgg16Transfer.evaluate(trainIter);
            eval = vgg16Transfer.evaluate(testIter);
            System.out.println("EPOCH: " + i + " Train Acc "+ eval2.accuracy() +" Val Accuracy: " + eval.accuracy());
        }

        File locationToSave = new File("C:\\Users\\Admin\\Desktop\\Acne-Analysis\\acne-model-skin2.zip");
        System.out.println(locationToSave.toString());
        // boolean save Updater
        boolean saveUpdater = false;
        // ModelSerializer needs modelname, saveUpdater, Location
        ModelSerializer.writeModel(vgg16Transfer,locationToSave,saveUpdater);

        eval2 = vgg16Transfer.evaluate(trainIter);
        eval = vgg16Transfer.evaluate(testIter);

        System.out.println(eval2.stats());
        System.out.println(eval.stats());
    }
}
