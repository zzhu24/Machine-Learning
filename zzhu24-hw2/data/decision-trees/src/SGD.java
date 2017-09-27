package cs446.weka.classifiers.trees;

import weka.classifiers.*;
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

import java.util.*;
import java.util.Collections;
import java.io.IOException;
import java.io.File;
import java.lang.*;
import java.lang.Boolean;
import java.lang.Exception;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.Enumeration;
import java.util.Random;

public class SGD extends Classifier
{	
	double error = 0.0;
	//set to one and update until less than the threshold
	double delta = 1.0;
	//double delta = 0.5;
	//double delta = 0.1;
	//double delta[] = range(0.1, 1.0, 10);
	//threshold that specify where to stop and print the output
	double threshold = 0.0001;
	//double threshold = 0.001;
	//double threshold = 0.01;
	//double threshold[] = range(0.00001, 0.01, 10);

	boolean trained = false;
	private double[] weight;
	
	public double calculation(double a, double b, double c){
		return a*b*c;
	}

	
   public void buildClassifier(Instances temp_data) throws IOException
   {
   		double learning_rate = 0.00001;
   	   int num = temp_data.numAttributes() - 1;
	   weight = new double[num];
	   Random temp_random = new Random();
	   for(int i = 0; i < num ; i++)
	   {
		    weight[i] = temp_random.nextDouble();
	   }

	   while( delta > threshold)
	   {
	   		//since don't know the initial value, use the random to acheive goal faster
	   		double temp_error = 0.0;
	   		Random random_initial = new Random();
	   		temp_data.randomize(random_initial);
			double[]  x = new double[num];
		   	for(int i = 0; i < temp_data.numInstances(); i++)
		    {
			    Instance temp_instance = temp_data.instance(i);
			    double sum = 0;
			    double temp_sum = 0;
		        x[i] = temp_instance.classValue();
		        if(x[i] == 0){
		        	x[i] = -1;
		        }
		        
			    for(int j  = 0; j < num ; j++)
		        {
		        	temp_sum = calculation(1, temp_instance.value(j), weight[j]);
		        	sum = sum + temp_sum; 
		        }  
		        double temp_difference = 0.0;
		        temp_difference = x[i] - sum;
		        for(int k = 0; k < num; k++)
		        {
		        	weight[k] = weight[k] + calculation(learning_rate, temp_difference, temp_instance.value(k));
		        }
		        
		        temp_error += calculation(0.5, temp_difference, temp_difference);  
		    }
		    delta = Math.abs(temp_error - error);
		    error = temp_error;
	   }
	   trained = true;
	}
/*
	public double calculation(double a, double b, double c){
		return a*b*c;
	}
*/
	public double classifyInstance(Instance input_instance) throws java.lang.Exception 
	{
		if(trained == false)
		{
		    throw new Exception("Not yet Trained");
		}
		else{
			double temp_product = 0.0;
			double sum_product = 0.0;
			double temp_attributes = input_instance.numAttributes() -1;

			for(int i  = 0; i < temp_attributes ; i++)
	    	{
	    		temp_product = calculation(1, input_instance.value(i), weight[i]);
	    		sum_product =  sum_product + temp_product; 
	   		}
	   		if(sum_product >= 0.0){
	   			return 1.0;
	   		}
	   		else{
	   			return 0.0;
	   		}
		}
		
	}


}
