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
#include "FeedBack.h"
#include "xValInstanceStream.h"
#include "utils.h"
#include "globals.h"
#include "crosstab.h"
#include "instanceStreamDiscretiser.h"
#include "correlationMeasures.h"

#include <assert.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __linux__
#include <sys/time.h>
#include <sys/resource.h>
#endif

void FeedBack(learner *theLearner, InstanceStream &instStream, FilterSet &filters, char* args) {
    unsigned int noFolds = 10;
    unsigned int noExperiments = 1;
    std::vector<unsigned int*> vals;

    vals.push_back(&noFolds);
    vals.push_back(&noExperiments);

    getUIntListFromStr(args, vals, "cross validation settings");

    const unsigned int noClasses = instStream.getNoClasses();

    std::vector<double> classDist(noClasses);
    std::vector<double> classDist_change(noClasses);

    std::vector<double> zOLoss; // 0-1 loss from each experiment
    std::vector<double> rmse; // rmse from each experiment
    std::vector<double> rmsea; // rmse for all classes from each experiment
    std::vector<double> logloss; // logarithmic loss for all classes from each experiment
    std::vector<double> zOLossSD; // standard deviation of 0-1 loss from each experiment
    std::vector<double> rmseSD; // standard deviation of rmse from each experiment
    std::vector<double> rmseaSD; // standard deviation of rmse for all classes from each experiment
    std::vector<double> loglossSD; // standard deviation of logarithmic loss for all classes from each experiment
    std::vector<long int> trainTimeM; //training time from each experiment
    std::vector<long int> testTimeM; //test time from each experiment

    for (unsigned int exp = 0; exp < noExperiments; exp++) {
        if (verbosity >= 1) printf("Cross validation experiment %d for %s\n", exp + 1, instStream.getName());

        InstanceCount count = 0;
        unsigned int zeroOneLoss = 0;

        double squaredError = 0.0;
        double squaredErrorAll = 0.0;
        double logLoss = 0.0;
        long int trainTime = 0;
        long int testTime = 0;


        std::vector<double> foldZOLoss; ///< 0-1 loss from each fold
        std::vector<double> foldrmse; ///< rmse from each fold
        std::vector<double> foldrmsea; ///< rmse for all classes from each fold
        std::vector<double> foldlogloss; ///< logarithmic loss for all classes from each fold

        crosstab<InstanceCount> xtab(instStream.getNoClasses());
        XValInstanceStream xValStream(&instStream, noFolds, exp, 1); //---------------------------0-normal -1

        for (unsigned int fold = 0; fold < noFolds; fold++) {
            InstanceCount foldcount = 0; ///< a count of the number of test instances in the fold
            unsigned int foldzeroOneLoss = 0;
            double foldsquaredError = 0.0;
            double foldsquaredErrorAll = 0.0;
            double foldlogLoss = 0.0;
            long int timeFold = 0;
#ifdef __linux__
            struct rusage usage;
#endif

            if (verbosity >= 2) printf("Fold %d\n", fold);
            //该函数是进行开始调用xValInstanceStream中进行拆分训练集的
            xValStream.startSubstream(fold, true, true); // start the cross validation training stream for the fold---得到训练集的90%
            //第一个false取训练集1.0，第二个false标识进一步划分训练集1.0为训练集2.0+反馈，训练集2.0占整个数据集的72%
            //调用训练集中作为训练集的一部分
            InstanceStream* filteredInstanceStream = filters.apply(&xValStream); // train the filters on the training stream
            //72%训练---18%反馈---10%测试
            //
#ifdef __linux__
            getrusage(RUSAGE_SELF, &usage);
            timeFold = usage.ru_utime.tv_sec + usage.ru_stime.tv_sec;
#endif
            //printf("训练之前 \n");
            theLearner->train(*filteredInstanceStream); // train the classifier on the filtered training stream
            //训练结束，已经走完算法cpp中的finalpass
            //printf("训练完毕 \n");
            //进行添加调用函数，调用change进行改变结构

#ifdef __linux__
            getrusage(RUSAGE_SELF, &usage);
            trainTime += ((usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) - timeFold);
#endif      
            /*********************************反馈流程*****************************************************************/
            if (!strncmp(theLearner->getName()->c_str(), "sortkdb", 7)) {
                printf("反馈ing...\n");
                unsigned int noCatAtts_;
                theLearner->getNoCatAtts_(noCatAtts_); //得到属性个数
                //printf("属性个数：%d\n", noCatAtts_);

                std::vector<std::vector<CategoricalAttribute> > parents_; //声明初始训练后结构
                parents_.resize(noCatAtts_);
                std::vector<std::vector<CategoricalAttribute> > parents_change; //声明改变后的结构
                parents_change.resize(noCatAtts_);
                std::vector<CategoricalAttribute> order; //声明初始训练的order序列

                theLearner->getStructure(parents_, order);

                //            printf("order:\t");
                //            for (CategoricalAttribute a = 0; a < order.size(); a++) {
                //                printf("%d\t", order[a]);
                //            }
                //            printf("\n");

                unsigned int zeroOneLoss_ = 0;
                unsigned int zeroOneLoss_change = 0;
                unsigned int foldcount_ = 0;
                bool decision = true;
                int pos = 0;

                while (decision) {
                    foldcount_ = 0;
                    zeroOneLoss_ = 0;
                    zeroOneLoss_change = 0;
                    xValStream.startSubstream(fold, true, false); //得到验证集18%,即训练集1.0的20%,训练集2.0(72%)+验证集(18%)=90%
                    filteredInstanceStream->rewind(); // rewind the filtered stream to the start

                    instance inst1(*filteredInstanceStream); // create a test instance

                    for (unsigned int i = 0; i < noCatAtts_; i++) {
                        if (i == order[pos])
                            continue;
                        else {
                            for (unsigned int j = 0; j < parents_[i].size(); j++) {
                                parents_change[i].push_back(parents_[i][j]);
                            }
                        }
                    }

                    theLearner->chang_parents(parents_);

                    //计算0-1loss
                    while (!filteredInstanceStream->isAtEnd()) {
                        if (filteredInstanceStream->advance(inst1)) {
                            foldcount_++;

                            theLearner->classify(inst1, classDist);

                            const CatValue prediction = indexOfMaxVal(classDist);
                            const CatValue trueClass = inst1.getClass();
                            if (prediction != trueClass) {
                                zeroOneLoss_++;
                            }

                            theLearner->classify_change(inst1, classDist_change, parents_change);
                            const CatValue prediction_change = indexOfMaxVal(classDist_change);
                            const CatValue trueClass_change = inst1.getClass();
                            if (prediction_change != trueClass_change) {
                                zeroOneLoss_change++;
                            }
                        }
                    }
                    //printf("feedback_count=%d\n", count);
                    double temp = zeroOneLoss_ / static_cast<double> (foldcount_);  //before
                    double temp_change = zeroOneLoss_change / static_cast<double> (foldcount_);//删末尾属性后结构

                    //printf("%lf\t%lf\t%lf\t%lf\n", temp, temp_change,(temp - temp_change) / temp,(temp_change - temp) / temp_change);

                    if ((temp - temp_change) / temp > 0.005) {
                        pos++;
                        for (unsigned int i = 0; i < noCatAtts_; i++)
                            parents_[i].clear();
                        for (unsigned int i = 0; i < noCatAtts_; i++) {
                            for (unsigned int j = 0; j < parents_change[i].size(); j++) {
                                parents_[i].push_back(parents_change[i][j]);
                            }
                        }
                        for (unsigned int i = 0; i < noCatAtts_; i++)
                            parents_change[i].clear();

                    } else {
                        decision = false;
                    }

                }
                //                printf("parents_final:\n");
                //                for (unsigned int i = 0; i < noCatAtts_; i++) {
                //                    if (parents_[i].size() == 0) {
                //                        printf("parents_[%d][0]\tY\n", i);
                //                    }
                //                    if (parents_[i].size() == 1) {
                //                        printf("parents_[%d][0]=%d\n", i, parents_[i][0]);
                //                    }
                //                    if (parents_[i].size() == 2) {
                //                        printf("parents_[%d][0]=%d\tparents_[%d][1]=%d\n", i, parents_[i][0], i, parents_[i][1]);
                //                    }
                //                }
                theLearner->chang_parents(parents_);
            }
	    //！！！！！！！！！！！！
            /*************************反馈结束********************************************************************/







            ///////////////////
            //                        xValStream.setflag(0);
            //                          xValStream.startSubstream(fold, true, true); // start the cross validation training stream for the fold---得到训练集的90%
            //                        //调用训练集中作为训练集的一部分
            //                         filteredInstanceStream = filters.apply(&xValStream); // train the filters on the training stream
            //            
            //                       theLearner->train(*filteredInstanceStream); // train the classifier on the filtered training stream
            //            theLearner->chang_parents(parents_);
            //          /////////////////////////////////////////////////////////////////////
            xValStream.startSubstream(fold, false, false); // reset the cross validation stream to the test stream for the fold, leaving the trained filters in place
            //第一个false取测试集，第二个false标识不进一步划分该测试集
            filteredInstanceStream->rewind(); // rewind the filtered stream to the start

            instance inst(*filteredInstanceStream); // create a test instance

            if (verbosity >= 3) printf("Fold %d testing\n", fold);

#ifdef __linux__
            getrusage(RUSAGE_SELF, &usage);
            timeFold = usage.ru_utime.tv_sec + usage.ru_stime.tv_sec;
#endif
            //            classDist.clear();
            //            classDist.resize(noClasses);
            if (!strncmp(theLearner->getName()->c_str(), "KDB-CondDisc", 12)) {
                while (!filteredInstanceStream->isAtEnd()) {
                    if (static_cast<InstanceStreamDiscretiser*> (filteredInstanceStream)->advanceNumeric(inst)) {
                        count++;
                        foldcount++;

                        theLearner->classify(inst, classDist);

                        const CatValue prediction = indexOfMaxVal(classDist);
                        const CatValue trueClass = inst.getClass();

                        if (prediction != trueClass) {
                            zeroOneLoss++;
                            foldzeroOneLoss++;
                        }

                        const double error = 1.0 - classDist[trueClass];
                        squaredError += error * error;
                        squaredErrorAll += error * error;
                        logLoss += log2(classDist[trueClass]);
                        foldsquaredError += error * error;
                        foldsquaredErrorAll += error * error;
                        foldlogLoss += log2(classDist[trueClass]);
                        for (CatValue y = 0; y < filteredInstanceStream->getNoClasses(); y++) {
                            if (y != trueClass) {
                                const double err = classDist[y];
                                squaredErrorAll += err * err;
                                foldsquaredErrorAll += err * err;
                            }
                        }

                        xtab[trueClass][prediction]++;
                    }
                }
            } else {
                //                printf("Here it is!\n");
                //                unsigned int noCatAtts_;
                //                theLearner->getNoCatAtts_(noCatAtts_); //得到属性个数
                //                std::vector<std::vector<CategoricalAttribute> > parents_sss; //声明初始训练后结构
                //                parents_sss.resize(noCatAtts_);
                //                std::vector<CategoricalAttribute> order; //声明初始训练的order序列
                //                if (!strncmp(theLearner->getName()->c_str(), "KDB", 3)) {
                //                    printf("kdb\n");
                //                    theLearner->getStructure_kdb(parents_sss);
                //                } else {
                //                    printf("else\n");
                //                    theLearner->getStructure(parents_sss, order);
                //                }
                //
                //
                //                printf("parents_sss:\n");
                //                for (unsigned int i = 0; i < noCatAtts_; i++) {
                //                    if (parents_sss[i].size() == 0) {
                //                        printf("parents_sss[%d][0]\tY\n", i);
                //                    }
                //                    if (parents_sss[i].size() == 1) {
                //                        printf("parents_sss[%d][0]=%d\n", i, parents_sss[i][0]);
                //                    }
                //                    if (parents_sss[i].size() == 2) {
                //                        printf("parents_sss[%d][0]=%d\tparents_[%d][1]=%d\n", i, parents_sss[i][0], i, parents_sss[i][1]);
                //                    }
                //                }

                while (!filteredInstanceStream->isAtEnd()) {
                    if (filteredInstanceStream->advance(inst)) {

                        count++;
                        foldcount++;

                        theLearner->classify(inst, classDist);

                        const CatValue prediction = indexOfMaxVal(classDist);
                        const CatValue trueClass = inst.getClass();

                        if (prediction != trueClass) {
                            zeroOneLoss++;
                            foldzeroOneLoss++;
                        }

                        const double error = 1.0 - classDist[trueClass];
                        squaredError += error * error;
                        squaredErrorAll += error * error;
                        logLoss += log2(classDist[trueClass]);
                        foldsquaredError += error * error;
                        foldsquaredErrorAll += error * error;
                        foldlogLoss += log2(classDist[trueClass]);
                        for (CatValue y = 0; y < filteredInstanceStream->getNoClasses(); y++) {
                            if (y != trueClass) {
                                const double err = classDist[y];
                                squaredErrorAll += err * err;
                                foldsquaredErrorAll += err * err;
                            }
                        }

                        xtab[trueClass][prediction]++;
                    }
                }
                //printf("test_count=%d\n", count);
            }

#ifdef __linux__
            getrusage(RUSAGE_SELF, &usage);
            testTime += ((usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) - timeFold);
#endif

            if (foldcount == 0) {
                printf("Fold %d is empty\n", fold);
            } else {
                foldZOLoss.push_back(foldzeroOneLoss / static_cast<double> (foldcount));
                foldrmse.push_back(sqrt(foldsquaredError / foldcount));
                foldrmsea.push_back(sqrt(foldsquaredErrorAll / (foldcount * xValStream.getNoClasses())));
                foldlogloss.push_back(-foldlogLoss / foldcount);
                if (verbosity >= 2) {
                    printf("\n0-1 loss (fold %d): %0.4f\n", fold, foldzeroOneLoss / static_cast<double> (foldcount));
                    printf("RMSE (fold %d): %0.4f\n", fold, sqrt(foldsquaredError / foldcount));
                    printf("RMSE All Classes (fold %d):  %0.4f\n", fold, sqrt(foldsquaredErrorAll / (foldcount * xValStream.getNoClasses())));
                    printf("Logarithmic Loss (fold %d):  %0.4f\n", fold, -foldlogLoss / foldcount);
                    printf("--------------------------------------------\n");
                }
            }
        }

        zOLoss.push_back(zeroOneLoss / static_cast<double> (count));
        assert(squaredError >= 0);
        rmse.push_back(sqrt(squaredError / count));
        rmsea.push_back(sqrt(squaredErrorAll / (count * xValStream.getNoClasses())));
        logloss.push_back(-logLoss / count);

        zOLossSD.push_back(stddev(foldZOLoss)); //标准差
        rmseSD.push_back(stddev(foldrmse));
        rmseaSD.push_back(stddev(foldrmsea));
        loglossSD.push_back(stddev(foldlogloss));

        trainTimeM.push_back(trainTime /= noFolds);
        testTimeM.push_back(testTime /= noFolds);

        if (verbosity >= 1) {
            if (verbosity >= 1) theLearner->printClassifier();

            printResults(xtab, xValStream);
            double MCC = calcMCC(xtab);
            printf("\nMCC:\n");
            printf("%0.4f\n", MCC);
        }
    }

    printf("\n0-1 loss:\n");
    print(zOLoss);
    printf("\n+/-:");
    print(zOLossSD);
    printf("\nRMSE:\n");
    print(rmse);
    printf("\n+/-:");
    print(rmseSD);
    printf("\nRMSE All Classes: ");
    print(rmsea);
    printf("\n             +/-: ");
    print(rmseaSD);
    printf("\nLogarithmic Loss: ");
    print(logloss);
    printf("\n             +/-: ");
    print(loglossSD);
    printf("\nTraining time: ", noFolds);
    print(trainTimeM);
    printf(" seconds");
    printf("\nClassification time: ", noFolds);
    print(testTimeM);
    printf(" seconds");

    if (noExperiments > 1) {
        printf("\nMean 0-1 loss: %0.4f + %0.4f\nMean RMSE: %0.4f + %0.4f\nMean RMSE All: %0.4f + %0.4f\n"
                "Mean Logarithmic Loss: %0.4f + %0.4f\nMean Training time: %ld\nMean Classification time: %ld\n",
                mean(zOLoss), stddev(zOLoss), mean(rmse), mean(rmse), stddev(rmsea), mean(logloss),
                stddev(logloss), mean(trainTimeM), mean(testTimeM));
    } else {
        putchar('\n');
    }
}
