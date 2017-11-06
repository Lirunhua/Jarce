/* Petal: An open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
**
** This program is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program. If not, see <http://www.gnu.org/licenses/>.
**
** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
*/
#include "learningCurves.h"
#include "StoredInstanceStream.h"
#include "StoredIndirectInstanceStream.h"
#include "IndirectInstanceSubstream.h"
#include "utils.h"
#include "globals.h"
#include "crosstab.h"
#include "incrementalLearner.h"

#include <assert.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


// get settings from command line arguments
void LearningCurveArgs::getArgs(char*const*& argv, char*const* end) {
  while (argv != end) {
    if (*argv[0] != '-') {
      break;
    }
    else if (argv[0][1] == 'e') {
      getUIntFromStr(argv[0]+2, endingPoint_, "e");
    }
    else if (argv[0][1] == 'h') {
      getUIntFromStr(argv[0]+2, testSetSize_, "h");
    }
    else if (argv[0][1] == 'n') {
      logProgression_ = false;
      getUIntFromStr(argv[0]+2, noOfPoints_, "n");
    }
    else if (argv[0][1] == 's') {
      getUIntFromStr(argv[0]+2, startingPoint_, "s");
    }
    else if (argv[0][1] == 't') {
      getUIntFromStr(argv[0]+2, noOfTrials_, "t");
    }
    else {
      break;
    }

    ++argv;
  }
}


void genLearningCurves(std::vector<learner*> theLearners, InstanceStream &instStream, FilterSet &filters, LearningCurveArgs* args) {
  IndirectInstanceSubstream substream;    // the first n instances in the instanceOrder
  const InstanceCount testSetSize = args->testSetSize_;
  const unsigned int noOfTrials = args->noOfTrials_;
  const InstanceCount minSampleSize = args->startingPoint_;
  std::vector<unsigned int*> vals;
  const unsigned int noClasses = instStream.getNoClasses();
  std::vector<InstanceCount> sampleSizes;
  InstanceCount step;
  std::vector<double> classDist(noClasses);

  StoredInstanceStream store;

  store.setSource(*filters.apply(&instStream));

  StoredIndirectInstanceStream instanceOrder(store);

  if (testSetSize > store.size()) error("Cannot take %" ICFMT " test examples from %" ICFMT " examples", testSetSize, store.size());

  const InstanceCount availableForTraining =  min(store.size()-testSetSize, args->endingPoint_);

  unsigned int noOfSampleSizes = 0;

  // print the data set name and sample sizes
  printf("\n>>> begin learning curves >>>\ndatasetName = %s;\ntrainsize = [", instStream.getName());

  if (args->logProgression_) {
    for (InstanceCount sampleSize = minSampleSize; sampleSize <= availableForTraining; sampleSize *= 2) {
      printf(" %d", sampleSize);
      ++noOfSampleSizes;
    }
  }
  else {
    noOfSampleSizes = args->noOfPoints_;
    step = (availableForTraining-minSampleSize)/(noOfSampleSizes-1);
    for (InstanceCount sampleSize = minSampleSize; sampleSize <= availableForTraining; sampleSize += step) {
      printf(" %d", sampleSize);
    }
  }
  puts("];");

  /// the RMSE indexed by learner then sample size then trial
  std::vector<std::vector<std::vector<double> > > rmse(theLearners.size());
  std::vector<std::vector<std::vector<double> > > logLoss(theLearners.size());
  std::vector<std::vector<std::vector<double> > > zoLoss(theLearners.size()); // true positives
  std::vector<std::vector<std::vector<InstanceCount> > > tp(theLearners.size()); // true positives
  std::vector<std::vector<std::vector<InstanceCount> > > fp(theLearners.size()); // false positive
  std::vector<std::vector<std::vector<InstanceCount> > > tn(theLearners.size()); // true negatives
  std::vector<std::vector<std::vector<InstanceCount> > > fn(theLearners.size()); // false egatives
  std::vector<std::vector<std::vector<double> > > ptp(theLearners.size()); // proportonal true positives
  std::vector<std::vector<std::vector<double> > > pfp(theLearners.size()); // proportonal false positive
  std::vector<std::vector<std::vector<double> > > ptn(theLearners.size()); // proportonal true negatives
  std::vector<std::vector<std::vector<double> > > pfn(theLearners.size()); // proportonal false egatives

  for (unsigned int learner = 0; learner < theLearners.size(); learner++) {
    theLearners[learner]->testCapabilities(instanceOrder); //after filters    
    zoLoss[learner].resize(noOfSampleSizes);
    tp[learner].resize(noOfSampleSizes);
    fp[learner].resize(noOfSampleSizes);
    tn[learner].resize(noOfSampleSizes);
    fn[learner].resize(noOfSampleSizes);
    ptp[learner].resize(noOfSampleSizes);
    pfp[learner].resize(noOfSampleSizes);
    ptn[learner].resize(noOfSampleSizes);
    pfn[learner].resize(noOfSampleSizes);
    rmse[learner].resize(noOfSampleSizes);
    logLoss[learner].resize(noOfSampleSizes);
  }

  for (unsigned int trial = 0; trial < noOfTrials; trial++) {
    int ssIndex = 0;
    instanceOrder.shuffle();

    for (InstanceCount sampleSize = minSampleSize; sampleSize <= availableForTraining; sampleSize = (args->logProgression_ ? (2*sampleSize) : (sampleSize+step))) {
      instanceOrder.setIndirectInstanceSubstream(substream, 0, sampleSize);


      // for each learner train on each successive sample size and store the rmse
      for (unsigned int learner = 0; learner < theLearners.size(); learner++) {
        theLearners[learner]->train(substream);

        // test on the last testSetSize instances
        instanceOrder.goTo(store.size()-testSetSize+1);

        double squaredError = 0.0;
        InstanceCount thisZOLoss = 0;
        InstanceCount thisTP = 0;
        InstanceCount thisFP = 0;
        InstanceCount thisTN = 0;
        InstanceCount thisFN = 0;
        double thisPTP = 0.0;
        double thisPFP = 0.0;
        double thisPTN = 0.0;
        double thisPFN = 0.0;
        double thisLogLoss = 0.0;

        while (!instanceOrder.isAtEnd()) {
          theLearners[learner]->classify(*instanceOrder.current(), classDist);

          const CatValue trueClass = instanceOrder.current()->getClass();
          const CatValue predictedClass = indexOfMaxVal(classDist);

          const double error = 1.0-classDist[trueClass];
          squaredError += error * error;
          thisLogLoss += log2(classDist[trueClass]);

          if (trueClass != predictedClass) ++thisZOLoss;

          if (trueClass == 0) {
            if (predictedClass == 0) ++thisTP;
            else ++thisFN;

            thisPTP += classDist[0];
            thisPFN += classDist[1];
          }
          else {
            if (predictedClass == 0) ++thisFP;
            else ++thisTN;

            thisPTN += classDist[1];
            thisPFP += classDist[0];
          }

          instanceOrder.advance();
        }

        zoLoss[learner][ssIndex].push_back(thisZOLoss/static_cast<double>(testSetSize));
        rmse[learner][ssIndex].push_back(sqrt(squaredError/testSetSize));
        logLoss[learner][ssIndex].push_back(-thisLogLoss/testSetSize);
        tp[learner][ssIndex].push_back(thisTP);
        tn[learner][ssIndex].push_back(thisTN);
        fp[learner][ssIndex].push_back(thisFP);
        fn[learner][ssIndex].push_back(thisFN);
        ptp[learner][ssIndex].push_back(thisPTP);
        ptn[learner][ssIndex].push_back(thisPTN);
        pfp[learner][ssIndex].push_back(thisPFP);
        pfn[learner][ssIndex].push_back(thisPFN);
      }

      ++ssIndex;
    }
  }

  // output the learning curves
  printf("=== Zero-One Loss ===\n");
  for (unsigned int learner = 0; learner < theLearners.size(); learner++) {
    print_(stdout, *theLearners[learner]->getName());
    printf(" = [");
    for (unsigned int j = 0; j < noOfSampleSizes; j++) {
      if (j) printf("; [");
      else putchar('[');
      for (unsigned int k = 0; k < zoLoss[learner][j].size(); ++k) {
        if (k) printf(", ");
        printf("%f", zoLoss[learner][j][k]);
      }
      putchar(']');
    }
    puts("];");
  }
  printf("\n=== RMSE ===\n");
  for (unsigned int learner = 0; learner < theLearners.size(); learner++) {
    print_(stdout, *theLearners[learner]->getName());
    printf(" = [");
    for (unsigned int j = 0; j < noOfSampleSizes; j++) {
      if (j) printf("; [");
      else putchar('[');
      for (unsigned int k = 0; k < rmse[learner][j].size(); ++k) {
        if (k) printf(", ");
        printf("%f", rmse[learner][j][k]);
      }
      putchar(']');
    }
    puts("];");
  }
  printf("\n=== Log Loss ===\n");
  for (unsigned int learner = 0; learner < theLearners.size(); learner++) {
    print_(stdout, *theLearners[learner]->getName());
    printf(" = [");
    for (unsigned int j = 0; j < noOfSampleSizes; j++) {
      if (j) printf("; [");
      else putchar('[');
      for (unsigned int k = 0; k < logLoss[learner][j].size(); ++k) {
        if (k) printf(", ");
        printf("%f", logLoss[learner][j][k]);
      }
      putchar(']');
    }
    puts("];");
  }

  if (store.getNoClasses() == 2) {
    printf("\n=== Matthews Correlation Coefficient ===\n");
    for (unsigned int learner = 0; learner < theLearners.size(); learner++) {
      print_(stdout, *theLearners[learner]->getName());
      printf(" = [");
      for (unsigned int j = 0; j < noOfSampleSizes; j++) {
        bool comma = false; // true if a comma should be output before the next value
        if (j) printf("; [");
        else putchar('[');
        for (unsigned int k = 0; k < tp[learner][j].size(); ++k) {
          if (tp[learner][j][k]+fn[learner][j][k]==0 || tn[learner][j][k]+fp[learner][j][k]==0) {
            // no test instances for one class - do nothing
          }
          else if (tp[learner][j][k]+fp[learner][j][k]==0 || tn[learner][j][k]+fn[learner][j][k]==0) {
            // no predictions for one class
            if (comma) printf(", ");
            else comma = true;
            putchar('0');
          }
          else {
            if (comma) printf(", ");
            else comma = true;
            printf("%f", 
                    (static_cast<double>(tp[learner][j][k])*tn[learner][j][k]-static_cast<double>(fp[learner][j][k])*fn[learner][j][k])
                    / sqrt(static_cast<double>(tp[learner][j][k]+fp[learner][j][k])
                            * static_cast<double>(tp[learner][j][k]+fn[learner][j][k])
                            * static_cast<double>(tn[learner][j][k]+fp[learner][j][k])
                            * static_cast<double>(tn[learner][j][k]+fn[learner][j][k]))
                    );

          }
        }
        putchar(']');
      }
      puts("];");
    }

    printf("\n=== Proportional Matthews Correlation Coefficient ===\n");
    for (unsigned int learner = 0; learner < theLearners.size(); learner++) {
      print_(stdout, *theLearners[learner]->getName());
      printf(" = [");
      for (unsigned int j = 0; j < noOfSampleSizes; j++) {
        bool comma = false; // true if a comma should be output before the next value
        if (j) printf("; [");
        else putchar('[');
        for (unsigned int k = 0; k < ptp[learner][j].size(); ++k) {
          if (ptp[learner][j][k]+pfn[learner][j][k]==0 || ptn[learner][j][k]+pfp[learner][j][k]==0) {
            // no test instances for one class - do nothing
          }
          else if (ptp[learner][j][k]+pfp[learner][j][k]==0.0 || ptn[learner][j][k]+pfn[learner][j][k]==0.0) {
            // no predictions for one class
            if (comma) printf(", ");
            else comma = true;
            putchar('0');
          }
          else {
            if (comma) printf(", ");
            else comma = true;
            printf("%f", 
                    (static_cast<double>(ptp[learner][j][k])*ptn[learner][j][k]-static_cast<double>(pfp[learner][j][k])*pfn[learner][j][k])
                    / sqrt(static_cast<double>(ptp[learner][j][k]+pfp[learner][j][k])
                            * static_cast<double>(ptp[learner][j][k]+pfn[learner][j][k])
                            * static_cast<double>(ptn[learner][j][k]+pfp[learner][j][k])
                            * static_cast<double>(ptn[learner][j][k]+pfn[learner][j][k]))
                    );

          }
        }
        putchar(']');
      }
      puts("];");
    }
  }
  printf("<<< end learning curves <<<\n");
}
