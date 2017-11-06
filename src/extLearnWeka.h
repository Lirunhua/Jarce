/*
 * extLearnWeka.h
 *
 *  Created on: 14/09/2012
 *      Author: nayyar
 */


#pragma once
#include "externLearner.h"
#include "instance.h"

#include "stdlib.h"
#include "utils.h"
#ifdef __linux__
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif

/*
 * Call to extLearnWeka (example, the name of the output weka file always at the end):
 * petal data.pmeta data.pdata -e weka weka.classifiers.bayes.NaiveBayes wekaFile.output
 */

class extLearnWeka : public externLearner {
private:
  FILE *testf;
  FILE *trainf;
  char*const* args;  // arguments to pass to executable
  int argcnt; // count of args
  std::string tmpTrain;
  std::string tmpTest;
  std::string tmpOutput;
  std::string dirTemp;
  
  template <typename T> std::string toString(T tmp)
  {
    std::ostringstream out;
    out << tmp;
    return out.str();
  }

public:  
  extLearnWeka(char *wd, char *bd, char*const argv[], int argc) : externLearner(wd, bd), testf(NULL), trainf(NULL) {
          args = argv;
          argcnt = argc;
          tmpTrain = workingDir;
          tmpTest = workingDir;
          #ifdef __linux__
          unsigned int dirNumber = 1;
          dirTemp = tmpTrain+"/temp"+toString(dirNumber);
          int status = mkdir(dirTemp.c_str(),0777);
          while(status==-1){ //Different directory for different experiments
            dirNumber++;
            dirTemp =  tmpTrain+"/temp"+toString(dirNumber);
            status = mkdir(dirTemp.c_str(),0777);
          }
          tmpTrain += "/temp"+toString(dirNumber)+"/tmp.train.arff";
          tmpTest += "/temp"+toString(dirNumber)+"/tmp.test.arff";
          #endif
        }
	~extLearnWeka() { fclose(testf); fclose(trainf); }

	void printTrainingData(InstanceStream *instanceStream);
	void printTest(InstanceStream *instanceStream);
	void printInst(FILE * f, instance &inst, InstanceStream *instanceStream);
	void printMeta(InstanceStream *instanceStream) const;
	void classify(InstanceStream *instanceStream, int fold);
	void writeHeader(InstanceStream *instanceStream, FILE *f);
        void printResults(InstanceStream *instanceStream);
};

