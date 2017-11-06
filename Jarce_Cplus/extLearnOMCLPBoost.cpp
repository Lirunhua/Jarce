/*
 * extLearnOMCLPBoost.cpp
 *
 *  Created on: 14/09/2012
 *      Author: nayyar
 */

#include "extLearnOMCLPBoost.h"

void extLearnOMCLPBoost::printTrainingData(InstanceStream *instanceStream) {
  printData(tmpTrain.c_str(), tmpTrainl.c_str(), instanceStream);
}


void extLearnOMCLPBoost::printData(const char* filename, const char* classfilename, InstanceStream *instanceStream) {
  FILE* f = fopen(filename, "w");

  if (f == NULL) error("Cannot open output file %s", filename);
  
  fprintf(f,"%d %d\n", instanceStream->size(), instanceStream->getNoNumAtts()+instanceStream->getNoCatAtts());

  FILE* classf = fopen(classfilename, "w");

  if (classf == NULL) error("Cannot open output file %s", classfilename);
  
  fprintf(classf,"%d %d\n", instanceStream->size(), 1);

  instance inst(*instanceStream);

  while (instanceStream->advance(inst)) {
    for (CategoricalAttribute a = 0; a < instanceStream->getNoCatAtts(); a++) {
      error("Cant' handle categorical attributes");
    }
    for (NumericAttribute a = 0; a < instanceStream->getNoNumAtts(); a++) {
      if (inst.isMissing(a)) {
        fprintf(f, "0.0 ");
      } else {
        fprintf(f, "%0.*f ", instanceStream->getPrecision(a), inst.getNumVal(a));
      }
    }

    fputc('\n',f);
    
    fprintf(classf, "%d\n", inst.getClass());
  }
}


void extLearnOMCLPBoost::printMeta(InstanceStream *instanceStream) const {
  // does not use a meta file
}

void extLearnOMCLPBoost::classify(InstanceStream *instanceStream, int fold) {
  printData(tmpTest.c_str(), tmpTestl.c_str(), instanceStream);

  createConfFile(fold);

  std::string tmpModel = workingDir;
  tmpModel += "/data.conf";

  std::string rfBoostStr;

  rfBoostStr = binDir;
  rfBoostStr += "/OMCBoost -c";
  rfBoostStr += ' ';
  rfBoostStr += tmpModel;
  for (int i = 0; i < argcnt; i++) {
    rfBoostStr += ' ';
    rfBoostStr += args[i];
  }
  rfBoostStr +=  " --train --test";

  printf("Executing command - %s\n", rfBoostStr.c_str());
  printf("Calling OMCLPBoost\n"); system(rfBoostStr.c_str()); printf("Finished calling OMCLPBoost\n");

  //remove("tmp.meta");
  printf("Deleting files \n");


  remove(tmpTrain.c_str());
  remove(tmpTest.c_str());
  remove(tmpTrainl.c_str());
  remove(tmpTestl.c_str());
  printf("Finished deleting files\n");
}

void extLearnOMCLPBoost::createConfFile(int fold) {
  FILE* confFile;

  printf("Creating data.conf\n");
  char tmpDirName[2000];
  sprintf(tmpDirName,"%s/data.conf",workingDir);
  confFile = fopen(tmpDirName, "w");
  if (confFile == NULL) error("Cannot open conf file data.conf");

  char trainOptions[2000];
  for (int i = 0; i < argcnt; i++) {
    sprintf(trainOptions, " %s", args[i]);
  }

  fprintf(confFile,"Data:\n{ \ntrainData = \"%s/data.train\";\ntrainLabels = \"%s/label.train\";"
    "\ntestData = \"%s/data.test\";\ntestLabels = \"%s/label.test\";\n};\n",
    workingDir,workingDir,workingDir,workingDir);

  fprintf(confFile,"Forest:\n{ \nmaxDepth = 20;\nnumRandomTests = 20;\ncounterThreshold = 200;\nnumTrees = 10;\n};\n");
  fprintf(confFile,"LaRank:\n{ \nlarankC = 1.0;\n};\n");
  fprintf(confFile,"Boosting: \n{ \nnumBases = 10; \nweakLearner = 0; \nshrinkage = 0.5; \nlossFunction = 0; "
    "\nC = 5.0; \ncacheSize = 1; \nnuD = 2.0; \nnuP = 1e-6; "
    "\nannealingRate = 0.9999999; \ntheta = 1.0; \nnumIterations = 1; \n};\n");
  fprintf(confFile,"Experimenter: \n{ \nfindTrainError = 1; \nnumEpochs = 10;\n};\n");
  fprintf(confFile,"Output:\n{ \nsavePath = \"%s/omclpBoost.output_Fold%d\";\nverbose = 1;\n};\n",workingDir,fold);

  fclose(confFile);
  confFile = NULL;
}

void extLearnOMCLPBoost::printResults(InstanceStream *instanceStream){
  
}
