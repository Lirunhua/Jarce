/*
 * extLearnOMCLPBoost.h
 *
 *  Created on: 14/09/2012
 *      Author: nayyar
 */
#pragma once
#include "externLearner.h"
#include "instanceStream.h"

#include "stdlib.h"
#include "utils.h"

class extLearnOMCLPBoost : public externLearner {
public:
  extLearnOMCLPBoost(char *wd, char *bd, char*const argv[], int argc) :
      externLearner(wd, bd), testf(NULL), trainf(NULL), testfl(NULL), trainfl(NULL) {
    args = argv;
    argcnt = argc;
    tmpTrain = workingDir;
    tmpTrain += "/data.train";
    tmpTest = workingDir;
    tmpTest += "/data.test";

    tmpTrainl = workingDir;
    tmpTrainl += "/label.train";
    tmpTestl = workingDir;
    tmpTestl += "/label.test";
  }
  ~extLearnOMCLPBoost() { fclose(testf); fclose(trainf); fclose(testfl); fclose(trainfl); }

  void printTrainingData(InstanceStream *instanceStream);
  void printMeta(InstanceStream *instanceStream) const;
  void classify(InstanceStream *instanceStream, int fold);
  void printResults(InstanceStream *instanceStream);

private:
  void createConfFile(int fold);
  void printData(const char* filename, const char* classfilename, InstanceStream *instanceStream);

private:
  FILE *testf;
  FILE *trainf;
  FILE *testfl;
  FILE *trainfl;
  char*const* args;  // arguments to pass to executable
  int argcnt; // count of args
  std::string tmpTrain;
  std::string tmpTest;
  std::string tmpTrainl;
  std::string tmpTestl;
};
