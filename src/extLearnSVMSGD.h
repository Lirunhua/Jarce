/*
 * extLearnLibSVM.h
 *
 *  Created on: 14/09/2012
 *      Author: nayyar
 */
#pragma once
#include "externLearner.h"
#include "instanceStream.h"

#include "stdlib.h"
#include "utils.h"

class extLearnSVMSGD : public externLearner {
public:
  extLearnSVMSGD(char *wd, char *bd, char*const argv[], int argc) : externLearner(wd, bd), testf(NULL), trainf(NULL) {
    args = argv;
    argcnt = argc;
    tmpTrain = workingDir;
    tmpTrain += "/tmp.train.txt";
    tmpTest = workingDir;
    tmpTest += "/tmp.test.txt";
  }
  ~extLearnSVMSGD() { fclose(testf); fclose(trainf); }

  void printTrainingData(InstanceStream *instanceStream);
  void printMeta(InstanceStream *instanceStream) const;
  void classify(InstanceStream *instanceStream, int fold);
  void printResults(InstanceStream *instanceStream);

private:
  void printData(const char* filename, InstanceStream *instanceStream);

  FILE *testf;
  FILE *trainf;
  char*const* args;  // arguments to pass to executable
  int argcnt; // count of args
  std::string tmpTrain;
  std::string tmpTest;
};

