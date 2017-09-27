#!/bin/bash

mkdir bin

make

# Generate the example features (first and last characters of the
# first names) from the entire dataset. This shows an example of how
# the featurre files may be built. Note that don't necessarily have to
# use Java for this step.
java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.all ./../badges.example.arff

#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold1 ./../badges.1.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold2 ./../badges.2.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold3 ./../badges.3.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold4 ./../badges.4.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold5 ./../badges.5.arff

# Using the features generated above, train a decision tree classifier
# to predict the data. This is just an example code and in the
# homework, you should perform five fold cross-validation. 
java -cp lib/weka.jar:bin cs446.homework2.Stump ./../badges.1.arff
#java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.2.arff
#java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.3.arff
#java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.4.arff
#java -cp lib/weka.jar:bin cs446.homework2.WekaTesterWekaTesterWekaTesterWekaTesterWekaTester ./../badges.5.arff
#java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.5.arff
#java -cp lib/weka.jar:bin cs446.weka.classifiers.trees.Stump ./../badges.1.arff

#java -cp lib/weka.jar:bin cs446.homework2.Depth4 ./../badges.1.arff
#java -cp lib/weka.jar:bin cs446.homework2.Depth8 ./../badges.example.arff

#java -cp lib/weka.jar:bin cs446.homework2.WholeTree ./../badges.example.arff