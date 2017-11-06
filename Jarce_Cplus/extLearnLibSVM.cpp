/*
 * extLearnLibSVM.cpp
 *
 *  Created on: 14/09/2012
 *      Author: nayyar
 */

#include "extLearnLibSVM.h"

void extLearnLibSVM::printTrainingData(InstanceStream* instanceStream) {
  printData("tmpTrain.c_str()", instanceStream);
}

void extLearnLibSVM::printMeta(InstanceStream* /*instanceStream*/) const {
  // no metadata used
}

void extLearnLibSVM::printData(const char* filename, InstanceStream* instanceStream) {
  FILE* f = fopen(filename, "w");
  if (f == NULL) error("Cannot open output file %s", filename);
  
  instance inst(*instanceStream);

  while (instanceStream->advance(inst)) {
    if (instanceStream->getNoClasses() == 2) {
      if (inst.getClass() == 1) {
        fprintf(f,"%d",1);
      } else {
        fprintf(f,"%d",-1);
      }
    } else
      fprintf(f,"%d",inst.getClass());

    for (CategoricalAttribute a = 0; a < instanceStream->getNoCatAtts(); a++) {
      if ( (strcmp(instanceStream->getCatAttValName(a, inst.getCatVal(a)),"?") == 0) ||
        (strcmp(instanceStream->getCatAttValName(a, inst.getCatVal(a)),"0") == 0) ) {
          error("Encountered categorical value.");
      } else {
        fprintf(f, " %d:%s", a+1, instanceStream->getCatAttValName(a, inst.getCatVal(a)));
      }
    }
    for (NumericAttribute a = 0; a < instanceStream->getNoNumAtts(); a++) {
      if (inst.getNumVal(a) == 0 || inst.isMissing(a)) {
      } else {
        fprintf(f, " %d:%0.*f", a+1, instanceStream->getPrecision(a), inst.getNumVal(a));
      }
    }

    fputc('\n',f);
  }

  fclose(f);
}

void extLearnLibSVM::classify(InstanceStream *instanceStream, int fold) {
  printData(tmpTest.c_str(),  instanceStream);

  // the temporary files
  std::string tmpModel = workingDir;
  tmpModel += "/tmp.model";
  std::string tmpOutput = workingDir;
  tmpOutput += "/tmp.output";

  std::string svmTrain;

  //sprintf(svmTrain,"%s/svm-train", binDir);
  svmTrain = binDir;
  svmTrain += "/svm-train";
  for (int i = 0; i < argcnt; i++) {
    //sprintf(svmTrain+strlen(svmTrain), " %s", args[i]);
    svmTrain += ' ';
    svmTrain += args[i];
  }
  //sprintf(svmTrain+strlen(svmTrain)," %s/tmp.train %s/tmp.model",workingDir,workingDir);
  svmTrain += ' ';
  svmTrain += tmpTrain;
  svmTrain += ' ';
  svmTrain += tmpModel;

  std::string svmPredict;
  //sprintf(svmPredict,"%s/svm-predict %s/tmp.test %s/tmp.model %s/tmp.output\n",	binDir,workingDir,workingDir,workingDir,workingDir,fold);
  svmPredict = binDir;
  svmPredict += "/svm-predict ";
  svmPredict = tmpTest;
  svmPredict += ' ';
  svmPredict = tmpModel;
  svmPredict += ' ';
  svmPredict = tmpOutput;

  printf("Calling svm-train\n"); system(svmTrain.c_str()); printf("Finished calling svm-train\n");
  printf("Calling svm-predict\n"); system(svmPredict.c_str()); printf("Finished calling svm-predict\n");

  //remove("tmp.meta");
  printf("Deleting files \n");
  remove(tmpTrain.c_str());
  remove(tmpTest.c_str());
  remove(tmpModel.c_str());
  remove(tmpOutput.c_str());
  printf("Finished deleting files\n");
}

void extLearnLibSVM::printResults(InstanceStream *instanceStream){
  
}
