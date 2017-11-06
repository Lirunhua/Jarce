/*
 * extLearnWeka.cpp
 *
 *  Created on: 17/09/2012
 *      Author: nayyar
 */

#include "extLearnWeka.h"
#include "correlationMeasures.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>

void extLearnWeka::printTrainingData(InstanceStream *instanceStream) {
	if (trainf == NULL) {
		printf("Creating %s\n",tmpTrain.c_str());
		trainf = fopen(tmpTrain.c_str(), "w");
		if (trainf == NULL) error("Cannot open output file %s", tmpTrain.c_str());
		writeHeader(instanceStream, trainf);
	}
        
        instance inst(*instanceStream);
        while (instanceStream->advance(inst)) {
          printInst(trainf, inst, instanceStream);
        }
}

void extLearnWeka::printTest(InstanceStream *instanceStream) {
	if (testf == NULL) {
		printf("Creating %s\n", tmpTest.c_str());
		testf = fopen(tmpTest.c_str(), "w");
		if (testf == NULL) error("Cannot open output file %s", tmpTest.c_str());
		writeHeader(instanceStream, testf);
	}
        instance inst(*instanceStream);
        while (instanceStream->advance(inst)) {
                printInst(testf, inst, instanceStream);
        }
}

void extLearnWeka::printInst(FILE *f, instance &inst, InstanceStream *instanceStream) {

	for (CategoricalAttribute a = 0; a < instanceStream->getNoCatAtts(); a++){
           if (streq(instanceStream->getCatAttValName(a,inst.getCatVal(a)),"?"))//This is not necessary
                fprintf(f, "MISSING,");
           else
                fprintf(f, "%s,", instanceStream->getCatAttValName(a,inst.getCatVal(a)));
        } 
        for (NumericAttribute a = 0; a < instanceStream->getNoNumAtts(); a++) {
            if (inst.isMissing(a)) {
                    fprintf(f,"?,");
            } else {
                    fprintf(f, "%0.*f ", instanceStream->getPrecision(a), inst.getNumVal(a));
            }
	}
        fputs(instanceStream->getClassName(inst.getClass()), f);
        fputc('\n',f);
}

void extLearnWeka::writeHeader(InstanceStream *instanceStream, FILE *f) {

	fprintf(f, "@relation PetalTempforWeka \n\n");
	for (CategoricalAttribute a = 0; a < instanceStream->getNoCatAtts(); a++){
		fprintf(f, "@attribute %s ", instanceStream->getCatAttName(a));
			fprintf(f, "{");
			for (int i = 0; i < instanceStream->getNoValues(a); i++) {
				bool last = (i == instanceStream->getNoValues(a) - 1) ? true : false;
				if (streq(instanceStream->getCatAttValName(a,i),"?")) {
					(last) ? fprintf(f, "MISSING") : fprintf(f, "MISSING,");
				} else {
					(last) ? fprintf(f, "%s", instanceStream->getCatAttValName(a,i)) : fprintf(f, "%s,", instanceStream->getCatAttValName(a,i));
				}
			}
			fprintf(f, "}\n");
        }
        for (NumericAttribute a = 0; a < instanceStream->getNoNumAtts(); a++) {
			fprintf(f, "@attribute %s real\n", instanceStream->getNumAttName(a));
        } 
        //the class
        fprintf(f, "@attribute %s {", instanceStream->getClassAttName());
        for (int i = 0; i < instanceStream->getNoClasses(); i++) {
                bool last = (i == instanceStream->getNoClasses() - 1) ? true : false;
                (last) ? fprintf(f, "%s", instanceStream->getClassName(i)) : fprintf(f, "%s,", instanceStream->getClassName(i));
        }
        fprintf(f, "}\n\n");
	fprintf(f, "@data\n");
}

void extLearnWeka::printMeta(InstanceStream *instanceStream) const {
	//instanceStream->printMetadata("tmp.meta");
}

void extLearnWeka::classify(InstanceStream *instanceStream, int fold) {
        printTest(instanceStream);
	fclose(testf);
	fclose(trainf);
	testf = NULL;
	trainf = NULL;

	std::string wekaCommand;
	wekaCommand = "java -cp ";
	wekaCommand += binDir;
	wekaCommand += "/weka.jar";

	if(argcnt!=0) {
		wekaCommand += ' ';
		wekaCommand += args[0];
	}
	wekaCommand += ' '; wekaCommand += "-t "; wekaCommand += tmpTrain.c_str();
	wekaCommand += ' '; wekaCommand += "-T "; wekaCommand += tmpTest.c_str();
        
        int i;
        for (i = 1; i < argcnt-1; i++) {
		wekaCommand += ' ';
		wekaCommand += args[i];
	}
        tmpOutput = workingDir;
        tmpOutput += "/";
        tmpOutput += args[i];         //Weka's output: the last argument is the name of the file.
        wekaCommand += ">> "+ tmpOutput;

	printf("Executing commnad - %s\n", wekaCommand.c_str());
	printf("Calling Weka\n"); system(wekaCommand.c_str()); printf("Finished calling Weka\n");

	//remove("tmp.meta");
	printf("Deleting files \n");
	remove(tmpTrain.c_str());
	remove(tmpTest.c_str());
	printf("Finished deleting files\n");
        
}

void extLearnWeka::printResults(InstanceStream *instanceStream){

        //Parse output file and combine loss functions
        
        std::vector<double> foldZOLoss;   ///< 0-1 loss from each fold
        std::vector<double> foldrmse;     ///< RMSE (petal) from each fold
        crosstab<InstanceCount> xtab(instanceStream->getNoClasses());
        
        unsigned int noFolds = 0;
        std::ifstream wekaOutput;  
        int offset; 
        
        wekaOutput.open(tmpOutput.c_str());
        if(wekaOutput){
          std::string item = "";
          while(!wekaOutput.eof()){              
            getline(wekaOutput,item);
              if((offset = item.find("Error on test", 0)) != std::string::npos) {
                noFolds++;
                std::getline(wekaOutput,item);
                
                //0-1 loss
                while((offset = item.find("Incorrectly", 0)) == std::string::npos)
                    std::getline(wekaOutput,item,' ');
                  std::getline(wekaOutput,item,' ');//Classified
                  std::getline(wekaOutput,item,' '); //Instances
                  std::getline(wekaOutput,item,' ');
                  while(streq(item.c_str(),""))
                    std::getline(wekaOutput,item,' ');  //absolute value
                  std::getline(wekaOutput,item,' ');
                  while(streq(item.c_str(),""))
                    std::getline(wekaOutput,item,' ');//0-1 loss percentage
                  foldZOLoss.push_back(atof(item.c_str())/100);
                  
                //RMSE (petal) 
                while((offset = item.find("(Petal)", 0)) == std::string::npos)
                    std::getline(wekaOutput,item,' ');
                  std::getline(wekaOutput,item,' ');
                  while(streq(item.c_str(),""))
                    std::getline(wekaOutput,item,' ');//RMSE percentage
                  item = item.substr(0,6);
                  foldrmse.push_back(atof(item.c_str()));

                //Confusion Matrix
                while((offset = item.find("Confusion", 0)) == std::string::npos)
                    std::getline(wekaOutput,item);
                  std::getline(wekaOutput,item);//blank line
                  std::getline(wekaOutput,item);//header line (classified as)
                  for(CatValue y = 0; y< instanceStream->getNoClasses(); y++){
                    CatValue predicted = 0;
                    std::getline(wekaOutput,item,' ');
                    for(CatValue p = 0; p< instanceStream->getNoClasses(); p++){
                      std::getline(wekaOutput,item,' ');
                      while(streq(item.c_str(),""))
                        std::getline(wekaOutput,item,' ');
                      xtab[y][predicted] += atoi(item.c_str());
                      predicted ++;
                    }
                    while((offset = item.find("\n", 0)) == std::string::npos)//Go to the next line
                      std::getline(wekaOutput,item,' ');
                 }
              }
            }
          }
          wekaOutput.close();
          //rmdir(dirTemp.c_str());
          
          printf("\n0-1 loss:\n");
          printf("%0.4f", mean(foldZOLoss));
          printf("\n+/-:");
          printf("%0.4f", stddev(foldZOLoss));
          printf("\nRMSE:\n");
          printf("%0.4f", mean(foldrmse));
          printf("\n+/-:");
          printf("%0.4f", stddev(foldrmse));
          
          
          for (CatValue predicted = 0; predicted < instanceStream->getNoClasses(); predicted++) {
            for (CatValue y = 0; y < instanceStream->getNoClasses(); y++) {
              xtab[y][predicted]/=noFolds;
            }
          }
          
        // Compute MCC for multi-class problems
        double MCC = 0.0;
        MCC = calcMCC(xtab);
        
        printf("\nMCC:\n");
        printf("%0.4f\n", MCC);
          
       
        // Print the confusion matrix
        // find the maximum value to determine how wide the output fields need to be
        InstanceCount maxval = 0;
        for (CatValue predicted = 0; predicted < instanceStream->getNoClasses(); predicted++) {
          for (CatValue y = 0; y < instanceStream->getNoClasses(); y++) {
            if (xtab[y][predicted] > maxval) maxval = xtab[y][predicted];
          }
        }

        // calculate how wide the output fields should be
        const int printwidth = max(4, printWidth(maxval));

        // print the heading line of class names
        printf("\n");
        for (CatValue y = 0; y < instanceStream->getNoClasses(); y++) {
          printf(" %*.*s", printwidth, printwidth, instanceStream->getClassName(y));
        }
        printf(" <- Actual class\n");

        // print the counts of instances classified as each class
        for (CatValue predicted = 0; predicted < instanceStream->getNoClasses(); predicted++) {
          for (CatValue y = 0; y < instanceStream->getNoClasses(); y++) {
            printf(" %*" ICFMT, printwidth, xtab[y][predicted]);
          }
          printf(" <- %s predicted\n", instanceStream->getClassName(predicted));
        }
        

}



