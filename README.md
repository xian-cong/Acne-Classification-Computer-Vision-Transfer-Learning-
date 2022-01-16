# Acne Classification with Deep Learning
All fundamental DL4J dependencies are included in the [pom.xml](https://github.com/xian-cong/Deep-Learning-Acne-Classification/blob/main/my-first-dl4j-project/pom.xml).

## Introduction
Without a doubt, most of teenages face acne problem. However, there isn't a guideline on how serious the acne is and what proper steps should be taken in order to cure the acne and prevent scars. Hence, acne classification is developed using deep learning. It is carried out through transfer learning method using VGG-16 model in this project. It is able to **classify acne seriousness** into:

  1. Normal
  2. Level 0
  3. Level 1
  4. Level 2

and **provide different solutions** for user to cure the acne issue as shown below:

<h4 align="center"> <img src="https://user-images.githubusercontent.com/22144223/149659429-15aa181f-be1a-4ae8-ad17-3caf8eed1432.png" width="500"> </br>
Values shown is the confidence level of different classes in percentage

### Dataset
250 HD images is being hand-picked for each classes from various internet sources. 
<h4 align="center"> <img src="https://user-images.githubusercontent.com/22144223/149659973-5242ca18-e52c-491f-aabc-f1773b39cb21.png" width="500"> </br>

### Annotation
Data annotation is being carried out by separating dataset into 4 classes.
<h4 align="center"> <img src="https://user-images.githubusercontent.com/22144223/149660036-e69fb470-9e30-4249-9b83-6a678866c157.png" width="500"> </br>

### Data Preprocessing
To increase the size of the dataset for training, data preprocessing is being carried out which includes:
- Horizontal Flip
- Vertical Flip
- 15° rotation
- 30° rotation

## To run on IDE
1. Import project
2. Wait for IDE to resolve dependencies
3. Navigate to ```MyFirstDL4JProject.java``` 
4. Run program

## To run from command line
Firstly, the project needs to be compiled as a jar file. The command used will build an uber jar. This type of jar compiles all classes from this project with its dependencies.

### To build uber jar  
```
mvn clean package
```
The command will output .jar file in the ```target``` directory.

### Run program
```
cd target
java -cp my-first-dl4j-project-1.0-SNAPSHOT-bin.jar ai.certifai.MyFirstDL4JProject
```
MyFirstDL4JProject is the class to run which is located in ai.certifai package

## Result of AI Model
The AI Model from transfer learning is able to achieve up to 74% accuracy by training with only 250 HD images from each classes. 
### Training Result
<h4 align="center"> <img src="https://github.com/xian-cong/Deep-Learning-Acne-Classification/blob/main/train%20cm.PNG" width="500"> </br>


### Validation Result
<h4 align="center"> <img src="https://github.com/xian-cong/Deep-Learning-Acne-Classification/blob/main/validation%20cm.PNG" width="500"> </br>

### Future Improvements
1. Platform to discuss skin care products
2. Cross geographical skin samples
3. Develop smartphone app
4. More detailed classifier
5. Higher Accuracy

### Additional Information
Additional information about this project can be read [here](https://xcdiary.wordpress.com/2021/01/16/applied-deep-learning-bootcamp-forward-school/).
