package cs446.homework2;

import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import cs446.weka.classifiers.trees.Id3;
import cs446.weka.classifiers.trees.SGD;

public class Stump
{	
	public static void main(String[] args) throws Exception 
    {
    	int temp_size = 2;
        double sum= 0;
	    double total = 0;
    	FastVector stump_lable= setVector(temp_size);
        FastVector all_lable = setVector(temp_size);
        temp_size = 100 + 1;
        FastVector attributes = new FastVector(temp_size);
    	Id3[] decision_tree= new Id3[100];
        
        for(int i = 0; i < 100; i++) 
        {
        	String temp_string = "Stump"+i;
        	addingattributes(attributes, temp_string, stump_lable);
    	}
    	//String temp_string = "First Class";
        //addingattributes(attributes, temp_string, all_lable);
        Attribute all_class =  new Attribute("Class", all_lable);
        attributes.addElement(all_class);
        
        //same as SGDRrun except for the classifier
		File temp_file = new File(args[0]);
		FileReader temp_reader = new FileReader(temp_file);
	    Instances data = new Instances(temp_reader);
        //data.setClassIndex(data.numAttributes());
        data.setClassIndex(data.numAttributes() - 1);
        
	    Evaluation evaluation = new Evaluation(data);
	    for(int i = 0; i < 5 ; i++)
        {
	           Instances train = data.trainCV(5,i);
	           Instances test = data.testCV(5,i);
	           Instances first_input = new Instances(train,(train.numInstances()/2));
		       for(int k= 0; k < 100; k++)
	           {
	              for(int m = 0; m < train.numInstances()/2 ; m++)
	              {
	                  first_input.add(train.instance(m));
	              }
	              decision_tree[k] = new Id3();
	              //decision_tree[k].setMaxDepth(14);

	              //decision_tree[k].setMaxDepth(8);

	              decision_tree[k].setMaxDepth(4);
	              decision_tree[k].buildClassifier(first_input);
		       }
	           //decision_tree is an array of 100 Decision Trees of maxDepth4
		       
		       System.out.println("Evaluation of data in Fold:" + (i+1));           
	           Instances Traning_Set= new Instances("Stump_Tree_Depth_4",attributes,train.numInstances());
	           //Instances Traning_Set = new Instances("Stump_Tree_Depth_8",attributes,train.numInstances());
	           //Instances Traning_Set = new Instances("Stump_Tree_Depth_14",attributes,train.numInstances());
	           Traning_Set.setClass(all_class);

	           Instances Testing_Set = new Instances("Stump_Tree_Depth_4",attributes,test.numInstances());
	           //Instances Testing_Set = new Instances("Stump_Tree_Depth_8",attributes,test.numInstances());
	           //Instances Testing_Set = new Instances("Stump_Tree_Depth_14",attributes,test.numInstances());
	           Testing_Set.setClass(all_class);
	         
	           int iteration_num = train.numInstances();
	           trainortest(iteration_num, decision_tree, train, attributes, Traning_Set);
	         
	        	iteration_num = test.numInstances();
	         	trainortest(iteration_num, decision_tree, test, attributes, Testing_Set);
	           
	           printevaluation(evaluation, Traning_Set, Testing_Set, sum, total);
        }
        double average_a = sum/total * 100;
		System.out.println("Average: "+ average_a + "%\n\n");
    }
    public static FastVector setVector(int limitsize){
    	FastVector temp = new FastVector(limitsize);
        temp.addElement("1");
        temp.addElement("0");
        return temp;
    }
    public static void addingattributes(FastVector added, String attriS, FastVector adding){
    	Attribute temp = new Attribute(attriS, adding);
    	added.addElement(temp);
    }
    public static void trainortest(int num, Id3[] the_tree, Instances train, FastVector attributes, Instances Input_Set)throws Exception { 
    	for(int t = 0; t < num; t++)
    		{
    			Instance return_instance = new Instance(101); 
	        	double stump_num;
	        	for (int j = 0; j < 100; j++)
	        	   {
	        		   stump_num =  the_tree[j].classifyInstance(train.instance(t));
	        		   return_instance.setValue((Attribute)attributes.elementAt(j),stump_num);
	               }
	        	   return_instance.setValue((Attribute)attributes.elementAt(100),train.instance(t).classValue());
	        	   Input_Set.add(return_instance);
	           }
    }
    public static void printevaluation(Evaluation evaluation, Instances training_temp, Instances testing_temp, double sum, double total)throws Exception {
    	SGD temp_classifier= new SGD();
    	temp_classifier.buildClassifier(training_temp);
    	evaluation = new Evaluation(testing_temp);
    	evaluation.evaluateModel(temp_classifier,testing_temp);
    	System.out.println(evaluation.toSummaryString());
    	sum = sum + evaluation.correct();
    	total = total + sum + evaluation.incorrect();
    }
}