

## Introduction

Fiber cut due to construction activities (e.g. digging) is one of the main causes of network outage in metropolitan area. Thus, predicting imminent fiber cuts and rerouting traffic accordingly are crucial in maintain network reliability. 

For this project, we focus on using machine learning (ML) to build such an “early warning” system, by using state-of-polarization (SOP) signatures associated with different mechanical movements.  For this project, we can consider an SOP signature as a multi-variate time series (MTS). As different types of mechanical perturbations will generate different MTS patterns, we first build a classifier of SOP MTS and thus detect those that could proceed a fiber break.

We note that MTS classification and anomaly detection have a lot of relevance in other industries, such as finance, manufacturing, and healthcare.
##  Key Files
* data\_processing\_and\_visualization.ipynb, Jupyter notebook for SOP time series (TS) processing and visualization;
* plot\_trace\_poincare\_sphere\_rotate.ipynb, Jupyter notebook for SOP rotation;
* sop_classification_svm.py, Python script for SOP TS classification using support vector machine (SVM).
* sop_classification_lstm.py, Python script for SOP TS classification using Long-Short-Term-Memory (LSTM).
* sop_classification_1d_cnn.py, Python script for SOP TS classification using 1-D Convolutional Neural Network (1-D CNN).



## Data Generation

For proof of concept, we first generate SOP traces in a lab environment. In particular, we use a robot arm to emulate four different types of typical mechanical disturbances to a fiber cable and record the corresponding SOP traces. These four types of mechanical disturbances mimic "hit", "rotation", "swing", and "stretch",  respectively.  In total, we generate ~10000 traces, with an average of 2500 traces per movement (label). The following animations show the robot movements and associated MTS signatures.

#### Movement 1 : "Hit"
![img](figs/mvt1.gif)  ![img](figs/sop_mvt1.gif)

#### Movement 2 : "Rotation"
![img](figs/mvt2.gif)  ![img](figs/sop_mvt2.gif)

#### Movement 3 : "Swing"
![img](figs/mvt3.gif)  ![img](figs/sop_mvt3.gif)

#### Movement 4 : "Stretch"
![img](figs/mvt4.gif)  ![img](figs/sop_mvt4.gif)


## Date Pre-processing and Visualization

For data processing, we only use down-sampling. Down-sampling not only reduces the computational complexity, but also prevents overfitting (ML algorithms could inadvertently learn the local noise). The input data has a dimension of (3, 10000), we decide on a factor of 100, as shown in the following plots.
![img](figs/down_sampling.png)  
  
Visualization is crucial in helping us gain intuition on what type of ML tools are appropriate for the classification tasks. If we plot several MTS on a sphere of unit radius (so called Poincare Sphere). We can observe that though the traces have different in starting/ending positions, lengths, as well as orientations,  they share some inherent trajectory structures. The observation shows that the task of classifying these trajectories is very similar to that of classifying handwritten digits.  This intuition help us select suitable ML classification models.
![img](figs/4_samples_s1_s2_s3.png)
![img](figs/sphere.png)

![img](figs/handwritten-digits.png)




Image reference [https://www.researchgate.net/figure/Sample-of-the-MNIST-dataset-of-handwritten-digits_fig1_311806756]
### Stokes Vector Rotation
Much like addressing the locality and translation invariance issue in image classification, we can "move" SOP traces to the same starting position on a Poincare Sphere. For example, as shown in the following figures, all the traces start form s1=0, s2 = -1, and s3 = 0 position. This operation can be accomplished by Stokes Vector Rotation.

![img](figs/4_samples_s1_s2_s3_rot.png)
![img](figs/sphere_rot.png)

## Classification
We focus on three promising ML algorithms for SOP TS classification: SVM, LSTM, and 1-D CNN.

### SVM

We use Scikit-learn library `SVC()` for SVM classification. In particular, we use kernel SVM, with radial basis function (RBF) as the kernel. In the training, we focus on tuning two hyperparameters C and gamma. The impacts of the values of C and gamma on the bias-variance tradeoff are summarized in the following tables. 

|Bias       |  Value is large              | Value is small         |
|---        |---                |---            |
| C         | low               |high           |
| gamma     | low               |high           |




|Variance   |  Value is large             |Value is small       |
|---        |---                |---         |
| C         | high              |low         |
| gamma     | high              |low         |

### Long-Short Term Memory (LSTM)
### 1-D Convolutional Neural Network (CNN)
