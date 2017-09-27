package cs446.homework2;

import java.io.FileReader;
import java.util.*;
import java.util.Collections;
import java.io.IOException;
import java.io.File;
import java.lang.Boolean;
import java.lang.Exception;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.Enumeration;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.*;
import java.lang.*;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import cs446.weka.classifiers.trees.Id3;

public class WholeTree {

public static void main(String[] args)throws Exception
	{
		//same as SGDRrun except for the classifier
		File temp_file = new File(args[0]);
		FileReader temp_reader = new FileReader(temp_file);
	    Instances data = new Instances(temp_reader);
        //data.setClassIndex(data.numAttributes());
        data.setClassIndex(data.numAttributes() - 1);
             	     
        double right = 0;
        double sum = 0;
         
         System.out.println("Evaluation from the ID3 whole decision tree\n");
	    for(int i = 0; i < 5; i++)
        {
            Id3 training = new Id3();
            training.setMaxDepth(260);

	      	Instances train = data.trainCV(5,i);
            Instances test  = data.testCV(5,i);

	      	boolean reTest = false;
			if(costwhole(train) >= 0.001){
				reTest = true;
			}
			else{
				reTest = false;
			}
	      	training.buildClassifier(train);
	      
          
            Evaluation evaluation = new Evaluation(test);
	      	evaluation.evaluateModel(training,test);
	      	right = right + evaluation.correct();
	      	sum = sum + evaluation.correct() + evaluation.incorrect();
	      	System.out.println("Evaluation of data in Fold:" + (i+1));
	      	System.out.println(evaluation.toSummaryString());

	    }
	  double average = right / sum * 100;
	  System.out.println("Average of the ID3 wholetree:" + average + "%\n\n");
	}
	private static double costwhole(Instances data){
		double evaluation = 0;
		double result = 0;
		int limitaion = data.numInstances();
		for(int i=0; i < limitaion;i++)
		{
			Instance sample=data.instance(i);
			double output;
			if(sample.stringValue(i).equals("+"))
			{
				output = 1;
			}
			else
			{
				output = -1;
			}
			for(int j=0; j<limitaion; j++)
			{
				result = result + output;
			}

		}
		return evaluation + result;
	}
	
}
