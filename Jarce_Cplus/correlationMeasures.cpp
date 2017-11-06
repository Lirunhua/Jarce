#include <math.h>
#include <assert.h>
#include <set>

#include "ALGLIB_specialfunctions.h"
#include "utils.h"
#include "correlationMeasures.h"

double getEntropy(std::vector<InstanceCount>& classDist) {
    double h = 0.0;

    double s = sum(classDist);

    if (s == 0.0) return 0.0;

    for (CatValue y = 0; y < classDist.size(); y++) {
        const double py = classDist[y] / s;

        if (py > 0.0) {
            h -= py * log2(py);
        }
    }

    return h;
}

double getInfoGain(xyDist& dist, CategoricalAttribute a) {
    const double s = sum(dist.classCounts);

    if (s == 0.0) return 0.0;

    double g = getEntropy(dist.classCounts);

    for (CatValue v = 0; v < dist.getNoValues(a); v++) {
        const double cnt = dist.getCount(a, v);
        if (cnt) {
            double ch = 0.0; // H(y | a)

            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                const double cp = dist.getCount(a, v, y) / cnt;

                if (cp > 0.0) {
                    ch -= cp * log2(cp);
                }
            }

            g -= (cnt / s) * ch;
        }
    }

    return g;
}

double getInfoGain(std::vector<InstanceCount> &left, std::vector<InstanceCount> &right) {
    std::vector<InstanceCount> jointDistribution;

    for (CatValue y = 0; y < left.size(); y++) {
        jointDistribution.push_back(left[y] + right[y]);
    }

    const double s = sum(jointDistribution);

    if (s == 0.0) return 0.0;

    double g = getEntropy(jointDistribution);

    const double cntl = sum(left);

    assert(cntl > 0);

    g -= (sum(left) / s) * getEntropy(left);

    const double cntr = sum(right);

    if (cntr > 0.0) {
        g -= (cntr / s) * getEntropy(right);
    }

    return g;
}

double getInfoGain(std::vector<InstanceCount> &left, std::vector<InstanceCount> &right, std::vector<InstanceCount> &unknown) {
    std::vector<InstanceCount> jointDistribution;

    for (CatValue y = 0; y < left.size(); y++) {
        jointDistribution.push_back(left[y] + right[y] + unknown[y]);
    }

    const double s = sum(jointDistribution);

    if (s == 0.0) return 0.0;

    double g = getEntropy(jointDistribution);

    const double cntl = sum(left);

    assert(cntl > 0);

    g -= (sum(left) / s) * getEntropy(left);

    const double cntr = sum(right);

    if (cntr > 0.0) {
        g -= (cntr / s) * getEntropy(right);
    }

    const double cntu = sum(unknown);

    if (cntu > 0.0) {
        g -= (cntu / s) * getEntropy(unknown);
    }

    return g;
}

double getInformation(xyDist& dist, CategoricalAttribute a) {
    double i = 0.0;
    const double s = sum(dist.classCounts);

    if (s == 0.0) return 0.0;

    for (CatValue v = 0; v < dist.getNoValues(a); v++) {
        const double p = dist.getCount(a, v) / s;

        if (p > 0.0) {
            i -= p * log2(p);
        }
    }

    return i;
}

double getGainRatio(xyDist& dist, CategoricalAttribute a) {
    const double iv = getInformation(dist, a);

    if (iv == 0.0) return 0.0;
    else return getInfoGain(dist, a) / iv;
}

/**
 * Calculates the gain ratio for the split class distributions 
 * 
 * @param left the distribution for the first branch
 * @param right the distribution for the second branch
 */
double getGainRatio(std::vector<InstanceCount> &left, std::vector<InstanceCount> &right) {
    std::vector<InstanceCount> splitDistribution;

    splitDistribution.push_back(sum(left));
    splitDistribution.push_back(sum(right));

    const double iv = getEntropy(splitDistribution);

    if (iv == 0.0) return 0.0;
    else return getInfoGain(left, right) / iv;

}

/**
 * Calculates the gain ratio for the split class distributions 
 * 
 * @param left the distribution for the first branch
 * @param right the distribution for the second branch
 * @param unknown the distribution for the unknown branch
 */
double getGainRatio(std::vector<InstanceCount> &left, std::vector<InstanceCount> &right, std::vector<InstanceCount> &unknown) {
    std::vector<InstanceCount> splitDistribution;

    splitDistribution.push_back(sum(left));
    splitDistribution.push_back(sum(right));
    splitDistribution.push_back(sum(unknown));

    const double iv = getEntropy(splitDistribution);

    if (iv == 0.0) return 0.0;
    else return getInfoGain(left, right, unknown) / iv;

}

double H_loc(xxyDist &dist, CategoricalAttribute xi, CategoricalAttribute xj, int q, const instance & inst) {
    std::vector<double> p;
    p.resize(dist.getNoClasses());
    p.clear();
    p.assign(dist.getNoCatAtts(), 0.0);

    //printf("aaa\n");
    const double totalCount = dist.xyCounts.count;
    //printf("bbb\n");
    //NB联合概率
    if (q == 1) {
        //printf("ccc111\n");
        //P_nb(xi,xj,y) = p(y)p(xi|y)p(xj|y) = p(xj,y) * p(xi,y) / p(y) = count(xj,y) * count(xi,y) / (count(y) * totalCount)
        CatValue vi = inst.getCatVal(xi);
        CatValue vj = inst.getCatVal(xj);
        for (CatValue y = 0; y < dist.getNoClasses(); y++) {
            const double y_count = dist.xyCounts.getClassCount(y);
            const double xiy = dist.xyCounts.getCount(xi, vi, y);
            const double xjy = dist.xyCounts.getCount(xj, vj, y);
            p[y] = (xjy * xiy) / (y_count * totalCount);

            //printf("1 p[%d] =%lf \n", y, p[y]);
        }
        //printf("ccc222\n");

    }
    //KDB联合概率
    if (q == 2) {
        //printf("ddd111\n");
        //P_kdb(xi,xj,y) = p(y)p(xj|y)p(xi|xj,y) = p(xi,xj,y)
        CatValue vi = inst.getCatVal(xi);
        CatValue vj = inst.getCatVal(xj);
        for (CatValue y = 0; y < dist.getNoClasses(); y++) {
            const double xixjy = dist.getCount(xi, vi, xj, vj, y);
            p[y] = xixjy / totalCount;
        }
        //printf("ddd222\n");
    }


    double H = 0.0;
    double m = 0.0;
    for (CatValue y = 0; y < dist.getNoClasses(); y++) {
        //printf("2 p[%d] =%lf \n", y, p[y]);
        m += p[y];
    }
    for (CatValue y = 0; y < dist.getNoClasses(); y++) {
        H -= (p[y] / m) * log2(p[y] / m);
    }
    //printf("H_%d =%lf \n\n", q, H);
    return H;

}

void getH_xy(xyDist &dist, std::vector<float> &mi) {
    mi.assign(dist.getNoCatAtts(), 0.0);

    //const double totalCount = dist.count;

    for (CategoricalAttribute a = 0; a < dist.getNoCatAtts(); a++) {
        double m = 0.0;

        for (CatValue v = 0; v < dist.getNoValues(a); v++) {
            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                const InstanceCount avyCount = dist.getCount(a, v, y);
                const InstanceCount avCount = dist.getCount(a, v);
                if (avyCount) {
                    m += (avyCount / avCount) * log2((avyCount / avCount));
                }
            }
        }
    }
}

void getlocH_xy(xyDist &dist, std::vector<float> &mi, const instance & inst) {
    mi.assign(dist.getNoCatAtts(), 0.0);
    std::vector<double> p;
    p.resize(dist.getNoClasses());
    p.clear();
    p.assign(dist.getNoCatAtts(), 0.0);
    const double totalCount = dist.count;
    for (CategoricalAttribute a = 0; a < dist.getNoCatAtts(); a++) {
        double m = 0.0;
        CatValue v = inst.getCatVal(a);
        for (CatValue y = 0; y < dist.getNoClasses(); y++) {
            const InstanceCount avyCount = dist.getCount(a, v, y);
            p[y] = avyCount / totalCount;
        }
        double sum = 0.0;
        for (CatValue y = 0; y < dist.getNoClasses(); y++) {
            sum += p[y];
        }
        for (CatValue y = 0; y < dist.getNoClasses(); y++) {
            mi[a] -= (p[y] / sum) * log2(p[y] / sum);
        }
        //printf("a = %d  m[a] = %f\n", a, mi[a]);
    }
}

void displayClassify(yDist &classDist_, xxxyDist & dist_1, std::vector<std::vector<CategoricalAttribute> > &parents_1, const instance & inst) {
    double H = 0.0;
    H = H_standard_loc_k2(classDist_, dist_1, parents_1, inst);
    printf("H=%lle\t", H);
    unsigned int noCatAtts_ = dist_1.getNoCatAtts();
    unsigned int noClasses_ = dist_1.getNoClasses();
    std::vector<double> posteriorDist0;
    posteriorDist0.assign(noClasses_, 0);
    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist0[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents_1[x1].size() == 0) {
                // printf("PARent=0  \n");

                posteriorDist0[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate


            } else if (parents_1[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist0[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist0[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_1[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), parents_1[x1][1], inst.getCatVal(parents_1[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_1[x1][0], inst.getCatVal(parents_1[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist0[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist0[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), y);
                    }
                } else {
                    posteriorDist0[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_1[x1][0], inst.getCatVal(parents_1[x1][0]), parents_1[x1][1], inst.getCatVal(parents_1[x1][1]), y);
                }

            }
        }
    }
    normalise(posteriorDist0);
    for (CatValue y = 0; y < noClasses_; y++) {
        //printf("p[%d]=%lf\t", y, posteriorDist0[y]);
        printf("%lf\t", y, posteriorDist0[y]);
    }
    printf("\n");
}
//基于nb,k=2,测度为p*log(p/m)

double H_standard_rewrite(yDist &classDist, xxxyDist &dist, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst) {
    unsigned int noClasses_ = dist.getNoClasses();
    unsigned int noCatAtts_ = dist.getNoCatAtts();
    std::vector<double> posteriorDist;
    posteriorDist.resize(noClasses_);
    posteriorDist.clear();
    posteriorDist.assign(noClasses_, 0.0);

    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist.p(y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents[x1].size() == 0) {

                posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate

            } else if (parents[x1].size() == 1) {

                const InstanceCount totalCount1 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents[x1].size() == 2) {

                const InstanceCount totalCount1 = dist.xxyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]), y);
                }

            }

        }
    }

    double H_standard = 0.0;
    double m = 0.0;

    for (CatValue y = 0; y < noClasses_; y++) {
        m += posteriorDist[y];
    }
    printf("%lle\t", m);
    for (CatValue y = 0; y < noClasses_; y++) {

        H_standard -= (posteriorDist[y]) * log2(posteriorDist[y] / m);
    }
    return H_standard;
}
//非nb,k=2,测度为p*log(p/m)

double H_standard_rewrite_only_have_parents(yDist &classDist, xxxyDist &dist, CategoricalAttribute x0, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst) {
    unsigned int noClasses_ = dist.getNoClasses();
    unsigned int noCatAtts_ = dist.getNoCatAtts();
    std::vector<double> posteriorDist;
    posteriorDist.resize(noClasses_);
    posteriorDist.clear();
    posteriorDist.assign(noClasses_, 0.0);


    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist.p(y);
    }

    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x0, inst.getCatVal(x0), y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]), y);
                }

            }

        }
    }

    double H_standard = 0.0;
    double m = 0.0;
    for (CatValue y = 0; y < noClasses_; y++) {
        m += posteriorDist[y];
    }
    for (CatValue y = 0; y < noClasses_; y++) {
        H_standard -= (posteriorDist[y]) * log2(posteriorDist[y] / m);
    }
    return H_standard;
}


//基于nb,k=2,测度为(p/m)*log(p/m)

double H_standard_loc_k2(yDist &classDist, xxxyDist &dist, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst) {
    unsigned int noClasses_ = dist.getNoClasses();
    unsigned int noCatAtts_ = dist.getNoCatAtts();
    std::vector<double> posteriorDist;
    posteriorDist.resize(noClasses_);
    posteriorDist.clear();
    posteriorDist.assign(noClasses_, 0.0);


    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist.p(y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents[x1].size() == 0) {
                // printf("PARent=0  \n");

                posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate

            } else if (parents[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]), y);
                }

            }

        }
    }

    double H_standard = 0.0;
    double m = 0.0;

    for (CatValue y = 0; y < noClasses_; y++) {
        m += posteriorDist[y];
    }
    //printf("%lle\t", m);
    for (CatValue y = 0; y < noClasses_; y++) {

        H_standard -= (posteriorDist[y] / m) * log2(posteriorDist[y] / m);
    }
    return H_standard;
}

void show_p(yDist &classDist, xxxyDist &dist, std::vector<std::vector<CategoricalAttribute> > &parents, CategoricalAttribute xk, const instance & inst) {
    unsigned int noClasses_ = dist.getNoClasses();
    unsigned int noCatAtts_ = dist.getNoCatAtts();
    std::vector<double> posteriorDist;
    posteriorDist.resize(noClasses_);
    posteriorDist.clear();
    posteriorDist.assign(noClasses_, 0.0);


    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist.p(y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents[x1].size() == 0) {
                // printf("PARent=0  \n");

                posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate

            } else if (parents[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]), y);
                }

            }

        }
    }
    double m = 0.0;
    for (CatValue y = 0; y < noClasses_; y++) {
        m += posteriorDist[y];
    }

    //I(x1,...,xn;Y)
    double ret1 = 0.0;
    for (CatValue y = 0; y < noClasses_; y++) {
        ret1 += posteriorDist[y] * log2(posteriorDist[y] / (m * classDist.p(y)));
    }
    printf("I(x1,...,xn;Y)=%lle\t", ret1);


    std::vector<double> posteriorDist1;
    posteriorDist1.resize(noClasses_);
    posteriorDist1.clear();
    posteriorDist1.assign(noClasses_, 0.0);

    std::vector<double> ptemp;
    ptemp.resize(noClasses_);
    ptemp.clear();
    ptemp.assign(noClasses_, 0.0);

    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist1[y] = classDist.p(y);
    }
    for (CatValue y = 0; y < noClasses_; y++) {
        ptemp[y] = classDist.p(y);
    }
    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        if (x1 != xk) {
            for (CatValue y = 0; y < noClasses_; y++) {
                if (parents[x1].size() == 0) {
                    // printf("PARent=0  \n");

                    posteriorDist1[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate

                } else if (parents[x1].size() == 1) {
                    //  printf("PARent=1  \n");
                    const InstanceCount totalCount1 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                    if (totalCount1 == 0) {
                        posteriorDist1[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist1[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                    }
                } else if (parents[x1].size() == 2) {
                    // printf("PARent=2  \n");
                    const InstanceCount totalCount1 = dist.xxyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]));
                    if (totalCount1 == 0) {
                        const InstanceCount totalCount2 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                        if (totalCount2 == 0) {
                            posteriorDist1[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                        } else {
                            posteriorDist1[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y);
                        }
                    } else {
                        posteriorDist1[y] *= dist.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]), y);
                    }

                }

            }
        } else {
            for (CatValue y = 0; y < noClasses_; y++) {
                posteriorDist1[y] *= 1;

                if (parents[x1].size() == 0) {
                    ptemp[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
                } else if (parents[x1].size() == 1) {
                    const InstanceCount totalCount1 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                    if (totalCount1 == 0) {
                        ptemp[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        ptemp[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                    }
                } else if (parents[x1].size() == 2) {
                    const InstanceCount totalCount1 = dist.xxyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]));
                    if (totalCount1 == 0) {
                        const InstanceCount totalCount2 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                        if (totalCount2 == 0) {
                            ptemp[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                        } else {
                            ptemp[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y);
                        }
                    } else {
                        ptemp[y] *= dist.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]), y);
                    }
                }
            }
        }
    }

    //I(x1,...xn;xj|Y)
    double ret2 = 0.0;
    for (CatValue y = 0; y < noClasses_; y++) {
        ret2 += posteriorDist[y] * log2((posteriorDist[y] * classDist.p(y)) / (posteriorDist1[y] * ptemp[y]));
    }
    printf("I(x1,...xn;xj|Y)=%lle\t", ret2);
}

double re_show_p(yDist &classDist, xxxyDist &dist, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst) {
    unsigned int noClasses_ = dist.getNoClasses();
    unsigned int noCatAtts_ = dist.getNoCatAtts();
    std::vector<double> posteriorDist;
    posteriorDist.resize(noClasses_);
    posteriorDist.clear();
    posteriorDist.assign(noClasses_, 0.0);


    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist.p(y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents[x1].size() == 0) {
                // printf("PARent=0  \n");

                posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate

            } else if (parents[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]), y);
                }

            }

        }
    }

    double n = 0.0;
    for (CatValue y = 0; y < noClasses_; y++) {
        n += posteriorDist[y] / classDist.p(y);
    }
    return n;
}

//非nb,k=2,测度为(p/m)*log(p/m)

double H_standard_loc_k2_only_have_parents(yDist &classDist, xxxyDist &dist, std::vector<CategoricalAttribute> &parentIsC_order, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst) {
    unsigned int noClasses_ = dist.getNoClasses();
    unsigned int noCatAtts_ = dist.getNoCatAtts();
    std::vector<double> posteriorDist;
    posteriorDist.resize(noClasses_);
    posteriorDist.clear();
    posteriorDist.assign(noClasses_, 0.0);

    //    printf("parents:\n");
    //    printf("root = %d\n", x0);
    //    for (unsigned int i = 0; i < noCatAtts_; i++) {
    //        if (parents[i].size() == 1) {
    //            printf("parents[%d][0]=%d\n", i, parents[i][0]);
    //        }
    //        if (parents[i].size() == 2) {
    //            printf("parents[%d][0]=%d\tparents[%d][1]=%d\n", i, parents[i][0], i, parents[i][1]);
    //        }
    //        
    //    }
    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist.p(y) * (std::numeric_limits<double>::max() / 2.0);
    }
    for (int i = 0; i < parentIsC_order.size(); i++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            posteriorDist[y] *= dist.xxyCounts.xyCounts.p(parentIsC_order[i], inst.getCatVal(parentIsC_order[i]), y);
        }
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]), y);
                }

            }

        }
    }

    double H_standard = 0.0;
    double m = 0.0;
    for (CatValue y = 0; y < noClasses_; y++) {
        m += posteriorDist[y];
    }
    for (CatValue y = 0; y < noClasses_; y++) {
        H_standard -= (posteriorDist[y] / m) * log2(posteriorDist[y] / m);
    }
    return H_standard;
}
//***********************不用了
//非nb(成长型)，为属性找父节点过程，max{P(x1,x2,...,xn)} * max{H(Y) - H(Y|x1,x2,...,xn)}, H测度为(p/m)*log(p/m)
//x0为根，temp为候选父结点,parents为已生成的结构
//P(x1,x2,...,xn)*{H(Y) - H(Y|x1,x2,...,xn)}
//for kdb

double H_PH_only_have_parents(yDist &classDist, xxxyDist &dist, CategoricalAttribute x0, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst) {
    unsigned int noClasses_ = dist.getNoClasses();
    unsigned int noCatAtts_ = dist.getNoCatAtts();
    std::vector<double> posteriorDist;
    posteriorDist.resize(noClasses_);
    posteriorDist.clear();
    posteriorDist.assign(noClasses_, 0.0);

    //    printf("parents:\n");
    //    printf("root = %d\n", x0);
    //    for (unsigned int i = 0; i < noCatAtts_; i++) {
    //        if (parents[i].size() == 1) {
    //            printf("parents[%d][0]=%d\n", i, parents[i][0]);
    //        }
    //        if (parents[i].size() == 2) {
    //            printf("parents[%d][0]=%d\tparents[%d][1]=%d\n", i, parents[i][0], i, parents[i][1]);
    //        }
    //        
    //    }
    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist.p(y);
    }

    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x0, inst.getCatVal(x0), y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents[x1].size() == 0) {
                // printf("PARent=0  \n");

                posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate

            } else if (parents[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]), y);
                }

            }

        }
    }
    double H_ret = 0.0;
    double H_standard = 0.0;
    double m = 0.0;
    for (CatValue y = 0; y < noClasses_; y++) {
        m += posteriorDist[y];
    }
    for (CatValue y = 0; y < noClasses_; y++) {
        H_standard -= (posteriorDist[y] / m) * log2(posteriorDist[y] / m);
    }
    double H_Y = getEntropy(dist.xxyCounts.xyCounts.classCounts);

    printf("H_standard=%lle\t%lle\t%lle\t", H_standard, m, H_Y - H_standard);
    H_ret = m * (H_Y - H_standard);
    return H_ret;
}
//for norder local kdb
//至少两个属性的父结点仅为类标签节点    成长型

double H_PH_only_have_parents2(yDist &classDist, xxxyDist &dist, CategoricalAttribute x0, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst) {
    unsigned int noClasses_ = dist.getNoClasses();
    unsigned int noCatAtts_ = dist.getNoCatAtts();
    std::vector<double> posteriorDist;
    posteriorDist.resize(noClasses_);
    posteriorDist.clear();
    posteriorDist.assign(noClasses_, 0.0);

    //    printf("parents:\n");
    //    printf("root = %d\n", x0);
    //    for (unsigned int i = 0; i < noCatAtts_; i++) {
    //        if (parents[i].size() == 1) {
    //            printf("parents[%d][0]=%d\n", i, parents[i][0]);
    //        }
    //        if (parents[i].size() == 2) {
    //            printf("parents[%d][0]=%d\tparents[%d][1]=%d\n", i, parents[i][0], i, parents[i][1]);
    //        }
    //        
    //    }
    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist.p(y);
    }

    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x0, inst.getCatVal(x0), y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents[x1].size() == 0) {
                // printf("PARent=0  \n");

                posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate

            } else if (parents[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]), y);
                }

            }

        }
    }
    double H_ret = 0.0;
    double H_standard = 0.0;
    double m = 0.0;
    for (CatValue y = 0; y < noClasses_; y++) {
        m += posteriorDist[y];
    }
    for (CatValue y = 0; y < noClasses_; y++) {
        H_standard -= (posteriorDist[y] / m) * log2(posteriorDist[y] / m);
    }
    double H_Y = getEntropy(dist.xxyCounts.xyCounts.classCounts);

    printf("H_standard=%lle\t%lle\t%lle\t", H_standard, m, H_Y - H_standard);
    H_ret = m * (H_Y - H_standard);
    return H_ret;
}
//***********************
//nb
//P(x1,x2,...,xn)*{H(Y) - H(Y|x1,x2,...,xn)}

double H_PH(yDist &classDist, xxxyDist &dist, std::vector<std::vector<CategoricalAttribute> > &parents, const instance & inst) {
    unsigned int noClasses_ = dist.getNoClasses();
    unsigned int noCatAtts_ = dist.getNoCatAtts();
    std::vector<double> posteriorDist;
    posteriorDist.resize(noClasses_);
    posteriorDist.clear();
    posteriorDist.assign(noClasses_, 0.0);

    //    printf("parents:\n");
    //    printf("root = %d\n", x0);
    //    for (unsigned int i = 0; i < noCatAtts_; i++) {
    //        if (parents[i].size() == 1) {
    //            printf("parents[%d][0]=%d\n", i, parents[i][0]);
    //        }
    //        if (parents[i].size() == 2) {
    //            printf("parents[%d][0]=%d\tparents[%d][1]=%d\n", i, parents[i][0], i, parents[i][1]);
    //        }
    //        
    //    }
    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist.p(y);
    }

    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents[x1].size() == 0) {
                // printf("PARent=0  \n");

                posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate

            } else if (parents[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist.xxyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist.xxyCounts.xyCounts.getCount(parents[x1][0], inst.getCatVal(parents[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist[y] *= dist.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist[y] *= dist.xxyCounts.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), y);
                    }
                } else {
                    posteriorDist[y] *= dist.p(x1, inst.getCatVal(x1), parents[x1][0], inst.getCatVal(parents[x1][0]), parents[x1][1], inst.getCatVal(parents[x1][1]), y);
                }

            }

        }
    }
    double H_ret = 0.0;
    double H_standard = 0.0;
    double m = 0.0;
    for (CatValue y = 0; y < noClasses_; y++) {
        m += posteriorDist[y];
    }
    for (CatValue y = 0; y < noClasses_; y++) {
        H_standard -= (posteriorDist[y] / m) * log2(posteriorDist[y] / m);
    }
    double H_Y = getEntropy(dist.xxyCounts.xyCounts.classCounts);

    //printf("m=%lle\t%lle\t", m, H_Y - H_standard);
    H_ret = m * (H_Y - H_standard);
    return H_ret;
}


//非nb,k=2,为属性x找父节点过程

void findmink_k2_only_have_parents(yDist &classDist, xxxyDist & dist, std::vector<CategoricalAttribute> &parentIsC_order, CategoricalAttribute x, std::vector<CategoricalAttribute> &temp, std::vector<std::vector<CategoricalAttribute> > &parents,
        std::vector<CategoricalAttribute> &order, const instance & inst) {
    unsigned int noCatAtts_ = dist.getNoCatAtts();
    double H = 0.0;
    double H_0 = 0.0;
    CategoricalAttribute pos1 = 0xFFFFFFFFUL;
    CategoricalAttribute pos2 = 0xFFFFFFFFUL;
    std::vector<std::vector<CategoricalAttribute> > parents_1;
    parents_1.resize(noCatAtts_);

    for (unsigned int i = 0; i < noCatAtts_; i++) {
        for (unsigned int j = 0; j < parents[i].size(); j++) {
            parents_1[i].push_back(parents[i][j]);
        }
    }
    parents_1[x].push_back(temp[0]);
    parents_1[x].push_back(temp[1]);
    pos1 = temp[0];
    pos2 = temp[1];
    H_0 = H_standard_loc_k2_only_have_parents(classDist, dist, parentIsC_order, parents_1, inst); //非nb,k=2,测度为(p/m)*log(p/m)
    //H_0 = H_standard_rewrite_only_have_parents(classDist, dist, x0, parents_1, inst); //非nb,k=2,测度为p*log(p/m)
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_1[a].clear();
    }

    for (std::vector<CategoricalAttribute>::const_iterator it1 = temp.begin(); it1 != temp.end(); it1++) {
        for (std::vector<CategoricalAttribute>::const_iterator it2 = it1 + 1; it2 != temp.end(); it2++) {
            //printf("%d\t%d\n", *it1, *it2);
            for (unsigned int i = 0; i < noCatAtts_; i++) {
                for (unsigned int j = 0; j < parents[i].size(); j++) {
                    parents_1[i].push_back(parents[i][j]);
                }
            }
            parents_1[x].push_back(*it1);
            parents_1[x].push_back(*it2);
            H = H_standard_loc_k2_only_have_parents(classDist, dist, parentIsC_order, parents_1, inst);
            //H = H_standard_rewrite_only_have_parents(classDist, dist, x0, parents_1, inst);
            //printf("H=%lf   %d  %d\n",H,*it1,*it2);
            if (H < H_0) {
                H_0 = H;
                pos1 = *it1;
                pos2 = *it2;
            }

            for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                parents_1[a].clear();
            }

        }
    }
    //printf("%d  %d\n", pos1, pos2);
    order.push_back(pos1);
    order.push_back(pos2);
    //    printf("porder in function:\n");
    //    for (int i= 0; i < 2; i++) {
    //        printf("%d\t", order[i]);
    //    }
    //    printf("\n");

}
//基于nb,k=2,找父节点过程

void findmink_k2(yDist &classDist, xxxyDist & dist, CategoricalAttribute x, std::vector<CategoricalAttribute> &temp, std::vector<std::vector<CategoricalAttribute> > &parents,
        std::vector<CategoricalAttribute> &order, const instance & inst) {
    unsigned int noCatAtts_ = dist.getNoCatAtts();
    double H = 0.0;
    double H_0 = 0.0;
    CategoricalAttribute pos1 = 0xFFFFFFFFUL;
    CategoricalAttribute pos2 = 0xFFFFFFFFUL;
    std::vector<std::vector<CategoricalAttribute> > parents_1;
    parents_1.resize(noCatAtts_);

    for (unsigned int i = 0; i < noCatAtts_; i++) {
        for (unsigned int j = 0; j < parents[i].size(); j++) {
            parents_1[i].push_back(parents[i][j]);
        }
    }
    parents_1[x].push_back(temp[0]);
    parents_1[x].push_back(temp[1]);
    pos1 = temp[0];
    pos2 = temp[1];
    H_0 = H_standard_loc_k2(classDist, dist, parents_1, inst); //基于nb,k=2,测度为(p/m)*log(p/m)
    //H_0 = H_standard_rewrite(classDist, dist, parents_1, inst); //基于nb,k=2,测度为p*log(p/m)
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_1[a].clear();
    }

    for (std::vector<CategoricalAttribute>::const_iterator it1 = temp.begin(); it1 != temp.end(); it1++) {
        for (std::vector<CategoricalAttribute>::const_iterator it2 = it1 + 1; it2 != temp.end(); it2++) {
            //printf("%d\t%d\n", *it1, *it2);
            for (unsigned int i = 0; i < noCatAtts_; i++) {
                for (unsigned int j = 0; j < parents[i].size(); j++) {
                    parents_1[i].push_back(parents[i][j]);
                }
            }
            parents_1[x].push_back(*it1);
            parents_1[x].push_back(*it2);
            H = H_standard_loc_k2(classDist, dist, parents_1, inst); //测度p/m*log(p/m)
            //H = H_standard_rewrite(classDist, dist, parents_1, inst); //测度p/*log(p/m)
            //printf("H=%lf   %d  %d\n",H,*it1,*it2);
            if (H < H_0) {
                H_0 = H;
                pos1 = *it1;
                pos2 = *it2;
            }

            for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                parents_1[a].clear();
            }

        }
    }
    //printf("%d  %d\n", pos1, pos2);
    order.push_back(pos1);
    order.push_back(pos2);
    //    printf("porder in function:\n");
    //    for (int i= 0; i < 2; i++) {
    //        printf("%d\t", order[i]);
    //    }
    //    printf("\n");

}

//递归删弧，局部条件熵，动态标杆

void H_recursion_del(yDist &classDist, xxxyDist & dist, double H_standard, unsigned int arc, std::vector<std::vector<CategoricalAttribute> > &parents_,
        std::vector<std::vector<CategoricalAttribute> > &parents_after, const instance & inst) {
    //    if (arc == 0) {
    //        parents_after.resize(dist.getNoCatAtts());
    //        for (CategoricalAttribute a = 0; a < dist.getNoCatAtts(); a++) {
    //            parents_after[a].clear();
    //        }
    //    } else {
    //printf("%lf\n",H_standard);
    std::vector<std::vector<CategoricalAttribute> > parents_1;
    parents_1.resize(dist.getNoCatAtts());
    unsigned int k = 0;
    double gm = H_standard;
    unsigned int pos = 0;
    int noDel_count = 0;
    int Del_count = 0;
    while (k < arc) {
        for (CategoricalAttribute a = 0; a < dist.getNoCatAtts(); a++) {
            parents_1[a].clear();
        }
        int temp = 0;
        for (unsigned int i = 0; i < dist.getNoCatAtts(); i++) {
            for (unsigned int j = 0; j < parents_[i].size(); j++) {
                if (temp == k) {
                    temp++;
                    continue;
                } else {
                    temp++;
                    parents_1[i].push_back(parents_[i][j]);
                }
            }
        }

        double H_new = 0.0;
        H_new = H_standard_loc_k2(classDist, dist, parents_1, inst);
        if (H_new < gm) {
            Del_count++;
            gm = H_new;
            pos = k;
        } else
            noDel_count++;

        k++;
    }

    //出口

    if (noDel_count == arc) {

        //printf("arc=%d\tnoDel_count=%d\n", arc, noDel_count);
        for (unsigned int i = 0; i < dist.getNoCatAtts(); i++) {
            for (unsigned int j = 0; j < parents_[i].size(); j++) {
                parents_after[i].push_back(parents_[i][j]);
            }
        }

        //        printf("parents_after:~~~~~~~~~~~~~~~~~~~~\n");
        //        for (unsigned int i = 0; i < dist.getNoCatAtts(); i++) {
        //            if (parents_after[i].size() == 1) {
        //                printf("parents_after[%d][0]=%d\n", i, parents_after[i][0]);
        //            }
        //            if (parents_after[i].size() > 1) {
        //                printf("parents_after[%d][0]=%d\tparents_after[%d][1]=%d\n", i, parents_after[i][0], i, parents_after[i][1]);
        //            }
        //        }
        //        printf("~~~~~~~~~~~~~~~~~~~~\n");
    } else {
        //printf("arc=%d\tDel_count=%d\tpos=%d\n", arc, Del_count, pos);
        //计算新标杆
        for (CategoricalAttribute a = 0; a < dist.getNoCatAtts(); a++) {
            parents_1[a].clear();
        }
        int temp1 = 0;
        for (unsigned int i = 0; i < dist.getNoCatAtts(); i++) {
            for (unsigned int j = 0; j < parents_[i].size(); j++) {
                //printf("temp=%d\n", temp);
                if (temp1 == pos) {
                    temp1++;
                    continue;
                } else {
                    temp1++;
                    parents_1[i].push_back(parents_[i][j]);
                }
            }
        }

        //                printf("parents_temp:##############\n");
        //                for (unsigned int i = 0; i < dist.getNoCatAtts(); i++) {
        //                    if (parents_1[i].size() == 1) {
        //                        printf("parents_temp[%d][0]=%d\n", i, parents_1[i][0]);
        //                    }
        //                    if (parents_1[i].size() > 1) {
        //                        printf("parents_temp[%d][0]=%d\tparents_temp[%d][1]=%d\n", i, parents_1[i][0], i, parents_1[i][1]);
        //                    }
        //                }
        //                printf("##############\n");

        double H_new1 = 0.0;
        H_new1 = H_standard_loc_k2(classDist, dist, parents_1, inst);
        //        printf("H_new=%lf\n",H_new1);
        //        printf("##############\n");
        H_recursion_del(classDist, dist, H_new1, arc - 1, parents_1, parents_after, inst);
    }

    //}
}

void displayInfo(yDist &classDist_, xxxyDist & dist_1, unsigned int noCatAtts_, unsigned int noClasses_, const instance& inst, std::vector<std::vector<CategoricalAttribute> > parents_) {
    std::vector<double> posteriorDist1;
    posteriorDist1.assign(noClasses_, 0);

    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist1[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
    }
    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        for (CatValue y = 0; y < noClasses_; y++) {
            if (parents_[x1].size() == 0) {
                // printf("PARent=0  \n");
                posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
            } else if (parents_[x1].size() == 1) {
                //  printf("PARent=1  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                if (totalCount1 == 0) {
                    posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                } else {
                    posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                }
            } else if (parents_[x1].size() == 2) {
                // printf("PARent=2  \n");
                const InstanceCount totalCount1 = dist_1.xxyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]));
                if (totalCount1 == 0) {
                    const InstanceCount totalCount2 = dist_1.xxyCounts.xyCounts.getCount(parents_[x1][0], inst.getCatVal(parents_[x1][0]));
                    if (totalCount2 == 0) {
                        posteriorDist1[y] *= dist_1.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist1[y] *= dist_1.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), y);
                    }
                } else {
                    posteriorDist1[y] *= dist_1.p(x1, inst.getCatVal(x1), parents_[x1][0], inst.getCatVal(parents_[x1][0]), parents_[x1][1], inst.getCatVal(parents_[x1][1]), y);
                }

            }
        }
    }

    double H_standard = 0.0;
    double m = 0.0;
    for (CatValue y = 0; y < noClasses_; y++) {
        m += posteriorDist1[y];
    }
    for (CatValue y = 0; y < noClasses_; y++) {
        H_standard -= (posteriorDist1[y] / m) * log2(posteriorDist1[y] / m);
    }
    printf("%lf\t", H_standard);
    normalise(posteriorDist1);
    double maxposteriorDist1 = 0.0;
    CatValue pre1 = 0xFFFFFFFFUL;
    for (CatValue y = 0; y < noClasses_; y++) {
        if (posteriorDist1[y] > maxposteriorDist1) {
            maxposteriorDist1 = posteriorDist1[y];
            pre1 = y;
        }
    }
    printf("%d\t", pre1);
    for (CatValue y = 0; y < noClasses_; y++) {
        //printf("p[%d]=%lf\t", y, posteriorDist1[y]);
        printf("%lf\t", posteriorDist1[y]);
    }
    printf("\n");
}

void H_recursion_dp_k2(yDist &classDist, xxxyDist & dist, double H_standard, unsigned int arc, std::vector<CategoricalAttribute> &order,
        std::vector<std::vector<CategoricalAttribute> > &parents_,
        std::vector<std::vector<CategoricalAttribute> > &parents_after, const instance & inst) {
    unsigned int noCatAtts_ = dist.getNoCatAtts();
    unsigned int noClasses_ = dist.getNoClasses();

    std::vector<std::vector<CategoricalAttribute> > parents_1;
    parents_1.resize(noCatAtts_);
    unsigned int k = 0;
    double gm = H_standard;
    unsigned int pos = 0xFFFFFFFFUL;
    //找到最应当删的一条弧，其条件熵与原来相比最小
    //printf("arc_delete...\n");
    while (k < arc) {
        //printf("a\n", arc);
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            parents_1[a].clear();
        }
        int temp = 0;
        for (unsigned int i = 0; i < noCatAtts_; i++) {
            for (unsigned int j = 0; j < parents_[i].size(); j++) {
                if (temp == k) {
                    temp++;
                    continue;
                } else {
                    temp++;
                    parents_1[i].push_back(parents_[i][j]);
                }
            }
        }
        //printf("b\n");
        double H_new = 0.0;
        H_new = H_standard_loc_k2(classDist, dist, parents_1, inst);
        //printf("c\n");
        if (H_new < gm) {
            gm = H_new;
            pos = k;
        }
        //printf("d\n");
        k++;
    }
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_1[a].clear();
    }

    //printf("arc_plus...\n");
    //找到最应当加的一条弧，其条件熵与原来相比最小
    double gm_plus = H_standard;
    unsigned int pos_plus_father = 0xFFFFFFFFUL;
    unsigned int pos_plus_child = 0xFFFFFFFFUL;
    for (std::vector<CategoricalAttribute>::const_iterator it = order.begin(); it != order.end(); it++) {
        for (std::vector<CategoricalAttribute>::const_iterator it2 = it + 1; it2 != order.end(); it2++) {
            //printf("(%d %d)\n", *it, *it2);
            for (unsigned int i = 0; i < noCatAtts_; i++) {
                for (unsigned int j = 0; j < parents_[i].size(); j++) {
                    parents_1[i].push_back(parents_[i][j]);
                }
            }
            if (parents_1[*it].size() == 2) {
                //printf("have two arc\n");
            } else if (parents_1[*it].size() == 1) {
                //printf("have one arc\n");
                if (parents_1[*it][0] != *it2) {
                    parents_1[*it].push_back(*it2);
                    //printf("plus arc(%d->%d)", *it, *it2);
                    //                    printf("parents_1:\n");
                    //                    for (unsigned int i = 0; i < dist.getNoCatAtts(); i++) {
                    //                        if (parents_1[i].size() == 1) {
                    //                            printf("parents1[%d][0]=%d\n", i, parents_1[i][0]);
                    //                        }
                    //                        if (parents_1[i].size() > 1) {
                    //                            printf("parents1[%d][0]=%d\tparents1[%d][1]=%d\n", i, parents_1[i][0], i, parents_1[i][1]);
                    //                        }
                    //                    }

                    double H_1 = 0.0; //加一条弧后的局部条件熵
                    H_1 = H_standard_loc_k2(classDist, dist, parents_1, inst);
                    if (H_1 < gm_plus) {
                        //printf("%d  %d 's H < H_standard\n", *it, *it2);
                        gm_plus = H_1;
                        pos_plus_father = *it2;
                        pos_plus_child = *it;
                    }
                }
            } else if (parents_1[*it].size() == 0) {
                //printf("have zero arc\n");
                parents_1[*it].push_back(*it2);
                double H_1 = 0.0; //加一条弧后的局部条件熵
                H_1 = H_standard_loc_k2(classDist, dist, parents_1, inst);
                if (H_1 < gm_plus) {
                    //printf("%d  %d 's H < H_standard\n", *it, *it2);
                    gm_plus = H_1;
                    pos_plus_father = *it2;
                    pos_plus_child = *it;
                }
            }

            for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                parents_1[a].clear();
            }

        }
    }


    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_1[a].clear();
    }
    //double gm_plus = 100.0;
    double H_new = gm - gm_plus;

    CategoricalAttribute Ch = 0xFFFFFFFFUL;
    CategoricalAttribute Pa = 0xFFFFFFFFUL;



    //printf("H_standard=%lf  (gm=%lf) - (gm_plus=%lf) = (H_new=%lf)\n", H_standard, gm, gm_plus, H_new);
    //判断此次递归是删弧还是加弧，使得熵最小
    if (H_new < 0 && fabs(H_new) > 0.005) { //删弧
        int temp1 = 0;
        for (unsigned int i = 0; i < noCatAtts_; i++) {
            for (unsigned int j = 0; j < parents_[i].size(); j++) {
                //printf("temp=%d\n", temp);
                if (temp1 == pos) {
                    Pa = parents_[i][j];
                    Ch = i;
                    //printf("del %d->%d\t", parents_[i][j], i);
                    temp1++;
                    //printf("del parents[%d][%d]=%d\n", i, j, parents_[i][j]);
                    continue;
                } else {
                    temp1++;
                    parents_1[i].push_back(parents_[i][j]);
                }
            }
        }


        //show_p(classDist, dist, parents_1, inst);
        //displayInfo(classDist, dist, noCatAtts_, noClasses_, inst, parents_1);
        H_recursion_dp_k2(classDist, dist, gm, arc - 1, order, parents_1, parents_after, inst);
    } else if (H_new > 0 && fabs(H_new) > 0.005) { //加弧
        //pos_plus_parent = no_Atts;
        //pos_plus = order[no_Atts][i];
        for (unsigned int i = 0; i < noCatAtts_; i++) {
            for (unsigned int j = 0; j < parents_[i].size(); j++) {
                parents_1[i].push_back(parents_[i][j]);
            }

        }
        if (pos_plus_child != 0xFFFFFFFFUL && pos_plus_father != 0xFFFFFFFFUL) {
            parents_1[pos_plus_child].push_back(pos_plus_father);
            //printf("plus %d->%d\t", pos_plus_father, pos_plus_child);
            //printf("plus parents[%d]=%d\n", pos_plus_child, pos_plus_father);
        }
        //show_p(classDist, dist, parents_1, inst);
        //displayInfo(classDist, dist, noCatAtts_, noClasses_, inst, parents_1);
        H_recursion_dp_k2(classDist, dist, gm_plus, arc + 1, order, parents_1, parents_after, inst);
    } else if (fabs(H_new) < 0.005) {//出口
        //printf("*************out*************\n");
        for (unsigned int i = 0; i < noCatAtts_; i++) {
            for (unsigned int j = 0; j < parents_[i].size(); j++) {
                parents_after[i].push_back(parents_[i][j]);
            }
        }
        //printf("over\n");
        //displayInfo(classDist, dist, noCatAtts_, noClasses_, inst, parents_after);
    }

}

bool search1(std::vector<CategoricalAttribute> qwe, int k) {
    for (int i = 0; i < qwe.size(); i++) {
        if (qwe[i] == k)
            return true;
    }
    return false;
}

bool search2(std::vector<CategoricalAttribute> ch, std::vector<CategoricalAttribute> fa, CategoricalAttribute a, CategoricalAttribute b) {
    for (int i = 0; i < ch.size(); i++) {
        if (ch[i] == a && fa[i] == b)
            return true;
    }

    return false;
}

void H_rec_p(yDist &classDist, xxxyDist & dist, double H_standard, unsigned int arc, std::vector<CategoricalAttribute> &order,
        std::vector<std::vector<CategoricalAttribute> > &parents_,
        std::vector<std::vector<CategoricalAttribute> > &parents_after, const instance & inst, std::vector<CategoricalAttribute> qwe, std::vector<CategoricalAttribute> ch, std::vector<CategoricalAttribute> fa) {
    unsigned int noCatAtts_ = dist.getNoCatAtts();
    unsigned int noClasses_ = dist.getNoClasses();
    std::vector<CategoricalAttribute> asd;
    asd.resize(arc);
    asd.clear();
    std::vector<CategoricalAttribute> faa;
    faa.resize(noCatAtts_);
    faa.clear();
    std::vector<CategoricalAttribute> chh;
    chh.resize(noCatAtts_);
    chh.clear();


    std::vector<std::vector<CategoricalAttribute> > parents_1;
    parents_1.resize(noCatAtts_);
    unsigned int k = 0;
    double gm = H_standard;
    unsigned int pos = 0xFFFFFFFFUL;
    //找到最应当删的一条弧，其条件熵与原来相比最小
    //printf("arc_delete...\n");
    while (k < arc) {
        if (search1(qwe, k))
            k++;
        else {
            for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                parents_1[a].clear();
            }
            int temp = 0;
            for (unsigned int i = 0; i < noCatAtts_; i++) {
                for (unsigned int j = 0; j < parents_[i].size(); j++) {
                    if (temp == k) {
                        temp++;
                        continue;
                    } else {
                        temp++;
                        parents_1[i].push_back(parents_[i][j]);
                    }
                }
            }
            double H_new = 0.0;
            H_new = H_standard_loc_k2(classDist, dist, parents_1, inst);
            if (H_new < gm) {
                gm = H_new;
                pos = k;
            }
            k++;
        }
    }
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_1[a].clear();
    }

    //printf("arc_plus...\n");
    //找到最应当加的一条弧，其条件熵与原来相比最小
    double gm_plus = H_standard;
    unsigned int pos_plus_father = 0xFFFFFFFFUL;
    unsigned int pos_plus_child = 0xFFFFFFFFUL;
    for (std::vector<CategoricalAttribute>::const_iterator it = order.begin(); it != order.end(); it++) {
        for (std::vector<CategoricalAttribute>::const_iterator it2 = it + 1; it2 != order.end(); it2++) {
            //printf("(%d %d)\n", *it, *it2);
            if (search2(ch, fa, *it, *it2))
                continue;
            else {
                for (unsigned int i = 0; i < noCatAtts_; i++) {
                    for (unsigned int j = 0; j < parents_[i].size(); j++) {
                        parents_1[i].push_back(parents_[i][j]);
                    }
                }
                if (parents_1[*it].size() == 2) {
                } else if (parents_1[*it].size() == 1) {
                    if (parents_1[*it][0] != *it2) {
                        parents_1[*it].push_back(*it2);

                        double H_1 = 0.0; //加一条弧后的局部条件熵
                        H_1 = H_standard_loc_k2(classDist, dist, parents_1, inst);
                        if (H_1 < gm_plus) {
                            gm_plus = H_1;
                            pos_plus_father = *it2;
                            pos_plus_child = *it;
                        }
                    }
                } else if (parents_1[*it].size() == 0) {
                    parents_1[*it].push_back(*it2);
                    double H_1 = 0.0; //加一条弧后的局部条件熵
                    H_1 = H_standard_loc_k2(classDist, dist, parents_1, inst);
                    if (H_1 < gm_plus) {
                        gm_plus = H_1;
                        pos_plus_father = *it2;
                        pos_plus_child = *it;
                    }
                }
                for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
                    parents_1[a].clear();
                }
            }
        }
    }


    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_1[a].clear();
    }
    //double gm_plus = 100.0;
    double H_new = gm - gm_plus;

    CategoricalAttribute Ch = 0xFFFFFFFFUL;
    CategoricalAttribute Pa = 0xFFFFFFFFUL;
    double Hp_new = 0.0;

    //printf("H_standard=%lf  (gm=%lf) - (gm_plus=%lf) = (H_new=%lf)\n", H_standard, gm, gm_plus, H_new);
    //判断此次递归是删弧还是加弧，使得熵最小
    if (H_new < 0 && fabs(H_new) > 0.005) { //删弧
        int temp1 = 0;
        for (unsigned int i = 0; i < noCatAtts_; i++) {
            for (unsigned int j = 0; j < parents_[i].size(); j++) {
                if (temp1 == pos) {
                    Pa = parents_[i][j];
                    Ch = i;
                    printf("del %d->%d\t", parents_[i][j], i);
                    temp1++;
                    continue;
                } else {
                    temp1++;
                    parents_1[i].push_back(parents_[i][j]);
                }
            }
        }
        double p_old = re_show_p(classDist, dist, parents_, inst);
        double p_new = re_show_p(classDist, dist, parents_1, inst);
        Hp_new = p_old - p_new;
        if (Hp_new < 0) {
            //show_p(classDist, dist, parents_1, inst);
            displayInfo(classDist, dist, noCatAtts_, noClasses_, inst, parents_1);

            for (int i = 0; i < qwe.size(); i++) {
                if (qwe[i] == pos)
                    continue;
                else
                    asd.push_back(qwe[i]);
            }
            H_rec_p(classDist, dist, gm, arc - 1, order, parents_1, parents_after, inst, asd, ch, fa);
        } else {
            printf("del fail\n");
            qwe.push_back(pos);
            H_rec_p(classDist, dist, gm, arc, order, parents_, parents_after, inst, qwe, ch, fa);
        }
    } else if (H_new > 0 && fabs(H_new) > 0.005) { //加弧
        for (unsigned int i = 0; i < noCatAtts_; i++) {
            for (unsigned int j = 0; j < parents_[i].size(); j++) {
                parents_1[i].push_back(parents_[i][j]);
            }
        }
        if (pos_plus_child != 0xFFFFFFFFUL && pos_plus_father != 0xFFFFFFFFUL) {
            parents_1[pos_plus_child].push_back(pos_plus_father);
            printf("plus %d->%d\t", pos_plus_father, pos_plus_child);
        }
        double p_old = re_show_p(classDist, dist, parents_, inst);
        double p_new = re_show_p(classDist, dist, parents_1, inst);
        Hp_new = p_old - p_new;
        if (Hp_new < 0) {
            printf("plus success\t");
            //show_p(classDist, dist, parents_1, inst);
            displayInfo(classDist, dist, noCatAtts_, noClasses_, inst, parents_1);

            for (int i = 0; i < ch.size(); i++) {
                if (ch[i] == pos_plus_child && fa[i] == pos_plus_father)
                    continue;
                else {
                    chh.push_back(ch[i]);
                    faa.push_back(fa[i]);
                }
            }

            H_rec_p(classDist, dist, gm_plus, arc + 1, order, parents_1, parents_after, inst, qwe, chh, faa);
        } else {
            printf("plus fail\n");
            ch.push_back(pos_plus_child);
            fa.push_back(pos_plus_father);
            H_rec_p(classDist, dist, gm, arc, order, parents_, parents_after, inst, qwe, ch, fa);
        }
    } else if (fabs(H_new) < 0.005) {//出口
        //printf("*************out*************\n");
        for (unsigned int i = 0; i < noCatAtts_; i++) {
            for (unsigned int j = 0; j < parents_[i].size(); j++) {
                parents_after[i].push_back(parents_[i][j]);
            }
        }
        printf("over\n");
        //displayInfo(classDist, dist, noCatAtts_, noClasses_, inst, parents_after);
    }

}

//打算给value添加一个父结点node，检查该父节点是否为value的祖先节点，以防止形成环(回路)，pm是原数组，没加弧

bool depthParent(xxxyDist & dist, CategoricalAttribute node, CategoricalAttribute value, std::vector<std::vector<CategoricalAttribute> > pm) { //node 的祖先中是否有有 value这个节点

    //printf("stck lll\n");
    CategoricalAttribute NOPARENT = 0xFFFFFFFFUL;

    CategoricalAttribute *visited = new CategoricalAttribute[dist.getNoCatAtts()];
    //printf("node: %u value: %u\n", node, value);
    //    printf("pm:\n");
    //    for (unsigned int i = 0; i < dist.getNoCatAtts(); i++) {
    //        if (pm[i].size() == 1) {
    //            printf("pm[%d][0]=%d\n", i, pm[i][0]);
    //        }
    //        if (pm[i].size() > 1) {
    //            printf("pm[%d][0]=%d\tpm[%d][1]=%d\n", i, pm[i][0], i, pm[i][1]);
    //        }
    //    }
    //    printf("\n");
    for (CategoricalAttribute k = 0; k < dist.getNoCatAtts(); k++)
        visited[k] = 0;
    //stack<CategoricalAttribute> S;//=new stack<CategoricalAttribute>;
    std::vector<CategoricalAttribute>vstack;

    //S.push(node);
    vstack.push_back(node);
    CategoricalAttribute w, k;
    while (!vstack.empty()) {
        //printf("stck while22222222222\n");
        //w=S.top();
        w = vstack.back();
        //printf("pop: %u--\n", w);
        //S.pop();
        vstack.pop_back();
        if (visited[w] == 0) {
            visited[w] = 1;
            if (w == value) {
                delete[] visited;
                pm.clear();
                std::vector<CategoricalAttribute>().swap(vstack);
                return true;
            }
        }
        //if(pm[w].size()==0)continue;
        //k = pm[w][0]; // printf("neibufistrr  %u\n",k);
        /* bool cirlcount=0;
         while((k!=NOPARENT)&&(cirlcount<2)){
                 if(visited[k]==0){
                         //S.push(k);
                     vstack.push_back(k);
                         k=pm[w][1];
                         cirlcount++;
                            printf("neibucirlcout %u\n",k);
                 }*/
        if (pm[w].size() > 0) //if (k != NOPARENT)
        {
            k = pm[w][0];
            if (visited[k] == 0) {
                //S.push(k);
                vstack.push_back(k);
                if (pm[w].size() == 2) {
                    k = pm[w][1];
                    //  printf("k1ff %u  \n",k);
                    // if (k != NOPARENT) {
                    if (visited[k] == 0) {
                        //  printf("k2ff %u\n",k);
                        //S.push(k);
                        vstack.push_back(k);
                    }
                }
            }
        }

        // cirlcount++;
        //   printf("neibucirlcout %u\n",k);
    }

    delete[] visited;
    pm.clear();
    std::vector<CategoricalAttribute>().swap(vstack);
    // delete S;
    return false;
}

/**
 *
 *                  __
 *                  \                     P(x1,x2,xk|y)
 *  I(X1,X2;Xk|Y)=  /_  P(x1,x2,xk,y)log-------------------
 *                X1,X2,Xk,Y             P(x1,x2|y)P(xk|y)
 *
 *
 */
//xi->xj xi->xk
//xi->xj->xk   

double getIxxx(xxxyDist &dist, CategoricalAttribute xi, CategoricalAttribute xj, CategoricalAttribute xk, std::vector<CategoricalAttribute> parents_1) {
    unsigned int noCatAtts_ = dist.getNoCatAtts();
    unsigned int noClasses_ = dist.getNoClasses();
    const static CategoricalAttribute NOPARENT = 0xFFFFFFFFUL;
    double ret = 0.0;
    printf("parents_1:\n");
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        if (parents_1[i] != NOPARENT)
            printf("%d->%d\n", parents_1[i], i);
    }

    std::set<CategoricalAttribute> used3;
    used3.insert(xi);
    used3.insert(xj);
    used3.insert(xk);

    std::set<CategoricalAttribute> used2;
    used2.insert(xi);
    used2.insert(xj);

    for (CatValue y = 0; y < noClasses_; y++) {
        double xixjxk_ = dist.xxyCounts.xyCounts.p(y);

        for (std::set<CategoricalAttribute>::const_iterator it0 = used3.begin(); it0 != used3.end(); it0++) {
            CategoricalAttribute x1 = *it0;
            const CategoricalAttribute parent = parents_1[x1];
            if (parent == NOPARENT) {
                for (CatValue v1 = 0; v1 < dist.xxyCounts.getNoValues(x1); v1++)
                    xixjxk_ *= dist.xxyCounts.xyCounts.p(x1, v1, y);
            } else {
                for (CatValue v1 = 0; v1 < dist.xxyCounts.getNoValues(x1); v1++) {
                    for (CatValue v2 = 0; v2 < dist.xxyCounts.getNoValues(parent); v2++) {
                        xixjxk_ *= dist.xxyCounts.p(x1, v1, parent, v2, y);
                    }
                }
            }
        }
        double xixj_ = dist.xxyCounts.xyCounts.p(y);

        for (std::set<CategoricalAttribute>::const_iterator it0 = used2.begin(); it0 != used2.end(); it0++) {
            CategoricalAttribute x1 = *it0;
            const CategoricalAttribute parent = parents_1[x1];
            if (parent == NOPARENT) {
                for (CatValue v1 = 0; v1 < dist.xxyCounts.getNoValues(x1); v1++)
                    xixj_ *= dist.xxyCounts.xyCounts.p(x1, v1, y);
            } else {
                for (CatValue v1 = 0; v1 < dist.xxyCounts.getNoValues(x1); v1++) {
                    for (CatValue v2 = 0; v2 < dist.xxyCounts.getNoValues(parent); v2++) {
                        xixj_ *= dist.xxyCounts.p(x1, v1, parent, v2, y);
                    }
                }
            }
        }

        CategoricalAttribute x1 = xk;
        double xk_ = dist.xxyCounts.xyCounts.p(y);
        for (CatValue v1 = 0; v1 < dist.xxyCounts.getNoValues(x1); v1++)
            xk_ *= dist.xxyCounts.xyCounts.p(x1, v1, y);


        ret += xixjxk_ * log2((xixjxk_ * dist.xxyCounts.xyCounts.p(y) / (xixj_ * xk_)));
    }
    printf("ret = %lle\n", ret);
    printf("###\n");

    return ret;
}

/**
 *
 *             __
 *             \               P(x,y)
 *  MI(X,Y)=   /_  P(x,y)log------------
 *            x,y             P(x)P(y)
 *
 *
 */
void getMutualInformation(xyDist &dist, std::vector<float> &mi) {
    mi.assign(dist.getNoCatAtts(), 0.0);

    const double totalCount = dist.count;

    for (CategoricalAttribute a = 0; a < dist.getNoCatAtts(); a++) {
        double m = 0.0;

        for (CatValue v = 0; v < dist.getNoValues(a); v++) {
            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                const InstanceCount avyCount = dist.getCount(a, v, y);
                if (avyCount) {

                    m += (avyCount / totalCount) * log2(avyCount / ((dist.getCount(a, v) / totalCount)
                            * dist.getClassCount(y)));
                }
            }
        }
        //printf("a = %d  m[a] = %f\n",a,m);
        mi[a] = m;
    }
}

void getMutualInformationTC(xyDist &dist, crosstab<double> &mi) {

    const double totalCount = dist.count;
    //printf("aaa\n");
    for (CategoricalAttribute a = 0; a < dist.getNoClasses(); a++) {
        for (CategoricalAttribute b = 0; b < dist.getNoCatAtts(); b++) {
            //for (CategoricalAttribute b = 0; b < 5; b++) {
            double m = 0.0;
            //printf("bbb\n");
            for (CatValue v = 0; v < dist.getNoValues(b); v++) {
                CatValue y = a;
                //  printf("ccc\n");
                const InstanceCount avyCount = dist.getCount(b, v, y);
                // printf("ddd\n");
                if (avyCount) {
                    //   printf("eee\n");

                    m += (avyCount / totalCount) * log2(avyCount / ((dist.getCount(b, v) / totalCount)
                            * dist.getClassCount(y)));
                    //   printf("fff\n");
                }

            }

            mi[a][b] = m;
        }
    }
}

void getMutualInformationTCloc(xyDist &dist, crosstab<float> &mi, const instance & inst) {

    const double totalCount = dist.count;
    //printf("aaa\n");
    for (CategoricalAttribute a = 0; a < dist.getNoClasses(); a++) {
        for (CategoricalAttribute b = 0; b < dist.getNoCatAtts(); b++) {
            double m = 0.0;
            //  printf("bbb\n");
            CatValue v = inst.getCatVal(b);
            CatValue y = a;
            // printf("ccc\n");
            const InstanceCount avyCount = dist.getCount(b, v, y);
            // printf("ddd\n");
            if (avyCount) {
                //   printf("eee\n");

                m += (avyCount / totalCount) * log2(avyCount / ((dist.getCount(b, v) / totalCount)
                        * dist.getClassCount(y)));
                //   printf("fff\n");
            }
            //   printf("eee\n");
            mi[a][b] = m;
            //  printf("fff\n");
        }
    }
}

void getMutualInformationloc(xyDist &dist, std::vector<float> &mi, const instance & inst) {
    //dist.update(inst);
    mi.assign(dist.getNoCatAtts(), 0.0);

    const double totalCount = dist.count;

    for (CategoricalAttribute a = 0; a < dist.getNoCatAtts(); a++) {
        //for (CategoricalAttribute a = 0; a < 5; a++) {
        double m = 0.0;

        CatValue v = inst.getCatVal(a);
        for (CatValue y = 0; y < dist.getNoClasses(); y++) {
            const InstanceCount avyCount = dist.getCount(a, v, y);
            if (avyCount) {

                m += (avyCount / totalCount) * log2(avyCount / ((dist.getCount(a, v) / totalCount)
                        * dist.getClassCount(y)));
            }
        }

        mi[a] = m;
    }
}

double getInfoGain_loc(xyDist& dist, CategoricalAttribute a, const instance & inst) {
    const double s = sum(dist.classCounts);

    if (s == 0.0) return 0.0;

    double g = getEntropy(dist.classCounts);

    CatValue v = inst.getCatVal(a);
    const double cnt = dist.getCount(a, v);
    if (cnt) {
        double ch = 0.0; // H(y | a)

        for (CatValue y = 0; y < dist.getNoClasses(); y++) {
            const double cp = dist.getCount(a, v, y) / cnt;

            if (cp > 0.0) {
                ch -= cp * log2(cp);
            }
        }

        g -= (cnt / s) * ch;
    }
    return g;
}

double getInformation_loc(xyDist& dist, CategoricalAttribute a, const instance & inst) {
    double i = 0.0;
    const double s = sum(dist.classCounts);

    if (s == 0.0) return 0.0;

    CatValue v = inst.getCatVal(a);
    const double p = dist.getCount(a, v) / s;

    if (p > 0.0) {
        i -= p * log2(p);
    }
    return i;
}

double getGainRatio_loc(xyDist& dist, CategoricalAttribute a, const instance & inst) {
    const double iv = getInformation_loc(dist, a, inst);

    if (iv == 0.0) return 0.0;
    else return getInfoGain_loc(dist, a, inst) / iv;
}

/*互信息I(Xi,Xj;Y)矩阵
 *互信息增益I(Xi,Xj;Y)/H(Xi,Xj)
 *                 __
 *                 \                    P(x1,x2,y)
 * MI(X1,X2;Y)= = /_   P(x1,x2,y) log-------------
 *               X1,X2,Y              P(x1,x2)P(y)
 *
 */
void getUnionMI(xxyDist &dist, crosstab<float> &mi_xxy) {
    const double totalCount = dist.xyCounts.count;

    for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
            float m = 0.0;
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                    for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                        const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                        if (x1x2y) {
                            m += (x1x2y / totalCount) * log2(totalCount * x1x2y /
                                    (static_cast<double> (dist.getCount(x1, v1, x2, v2)) *
                                    dist.xyCounts.getClassCount(y)));
                        }
                    }
                }
            }
            assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

            mi_xxy[x1][x2] = m;
            mi_xxy[x2][x1] = m;
        }
    }
}

void getUnmi_loc(xxyDist &dist, crosstab<float> &mi_xxy, const instance & inst) {
    const double totalCount = dist.xyCounts.count;

    for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
            float m = 0.0;
            CatValue v1 = inst.getCatVal(x1);
            CatValue v2 = inst.getCatVal(x2);
            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                if (x1x2y) {
                    m += (x1x2y / totalCount) * log2(totalCount * x1x2y /
                            (static_cast<double> (dist.getCount(x1, v1, x2, v2)) *
                            dist.xyCounts.getClassCount(y)));
                }
            }
            assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

            mi_xxy[x1][x2] = m;
            mi_xxy[x2][x1] = m;

        }
    }
}

/*联合条件互信息I(Xi,Xj;Xk|Y)
 *                  __
 *                  \                         P(xi,xj,xk|y)
 * I(Xi,Xj;Xk|Y)= = /_    P(xi,xj,xk,y) log2------------------
 *               Xi,Xj,Xk,y                  P(xi,xj|y)P(xk|y)
 *
 */
void getUnionCmi(xxxyDist &dist, crosstab3D<float> &cmi) {
    const double totalCount = dist.xxyCounts.xyCounts.count;
    for (CategoricalAttribute xk = 0; xk < dist.getNoCatAtts(); xk++) {
        for (CategoricalAttribute xi = 1; xi < dist.getNoCatAtts(); xi++) {
            for (CategoricalAttribute xj = 0; xj < xi; xj++) {
                double m = 0.0;
                if (xi != xk && xj != xk) {
                    for (CatValue v1 = 0; v1 < dist.getNoValues(xi); v1++) {
                        for (CatValue v2 = 0; v2 < dist.getNoValues(xj); v2++) {
                            for (CatValue v3 = 0; v3 < dist.getNoValues(xk); v3++) {
                                for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                                    const double xixjxky = dist.getCount(xi, v1, xj, v2, xk, v3, y);
                                    const double xixjy = dist.xxyCounts.getCount(xi, v1, xj, v2, y);
                                    const double xky = dist.xxyCounts.xyCounts.getCount(xk, v3, y);
                                    if (xixjxky) {
                                        m += (xixjxky / totalCount) * log2((xixjxky * dist.xxyCounts.xyCounts.getClassCount(y)) / (xixjy * xky));
                                    }
                                }
                            }
                        }
                    }
                }
                cmi[xk][xi][xj] = m; //I(Xi,Xj;Xk|Y)
                cmi[xk][xj][xi] = m;
            }
        }
    }
}

void getUnionCmi_loc(xxxyDist &dist, crosstab3D<float> &cmi, const instance & inst) {
    const double totalCount = dist.xxyCounts.xyCounts.count;
    for (CategoricalAttribute xk = 0; xk < dist.getNoCatAtts(); xk++) {
        for (CategoricalAttribute xi = 1; xi < dist.getNoCatAtts(); xi++) {
            for (CategoricalAttribute xj = 0; xj < xi; xj++) {
                double m = 0.0;
                if (xi != xk && xj != xk) {
                    CatValue v1 = inst.getCatVal(xi);
                    CatValue v2 = inst.getCatVal(xj);
                    CatValue v3 = inst.getCatVal(xk);
                    for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                        const double xixjxky = dist.getCount(xi, v1, xj, v2, xk, v3, y);
                        const double xixjy = dist.xxyCounts.getCount(xi, v1, xj, v2, y);
                        const double xky = dist.xxyCounts.xyCounts.getCount(xk, v3, y);
                        if (xixjxky) {
                            m += (xixjxky / totalCount) * log2((xixjxky * dist.xxyCounts.xyCounts.getClassCount(y)) / (xixjy * xky));
                        }
                    }
                }
                cmi[xk][xi][xj] = m; //I(Xi,Xj;Xk|Y)
                cmi[xk][xj][xi] = m;
            }
        }
    }
}

/*三个变量的条件互信息I(Xi;Xj;Xk|Y)
 *                  __
 *                  \                         P(xi,xj|y)P(xi,xk|y)P(xj,xk|y)
 * I(Xi;Xj;Xk|Y)= = /_    P(xi,xj,xk,y) log2----------------------------------
 *               Xi,Xj,Xk,y                  P(xi|y)P(xj|y)P(xk|y)P(xi,xj,xk|y)
 *
 */
void getUnion3cmi(xxxyDist &dist, crosstab3D<float> &cmi) {
    const double totalCount = dist.xxyCounts.xyCounts.count;
    for (CategoricalAttribute xk = 0; xk < dist.getNoCatAtts(); xk++) {
        for (CategoricalAttribute xi = 1; xi < dist.getNoCatAtts(); xi++) {
            for (CategoricalAttribute xj = 0; xj < xi; xj++) {
                double m = 0.0;
                if (xi != xk && xj != xk) {
                    for (CatValue v1 = 0; v1 < dist.getNoValues(xi); v1++) {
                        for (CatValue v2 = 0; v2 < dist.getNoValues(xj); v2++) {
                            for (CatValue v3 = 0; v3 < dist.getNoValues(xk); v3++) {
                                for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                                    const double xixjxky = dist.getCount(xi, v1, xj, v2, xk, v3, y);
                                    const double xixjy = dist.xxyCounts.getCount(xi, v1, xj, v2, y);
                                    const double xixky = dist.xxyCounts.getCount(xi, v1, xk, v3, y);
                                    const double xjxky = dist.xxyCounts.getCount(xj, v2, xk, v3, y);
                                    const double xiy = dist.xxyCounts.xyCounts.getCount(xi, v1, y);
                                    const double xjy = dist.xxyCounts.xyCounts.getCount(xj, v2, y);
                                    const double xky = dist.xxyCounts.xyCounts.getCount(xk, v3, y);
                                    if (xixjxky && xixjy && xixky && xjxky) {
                                        m += (xixjxky / totalCount) * log2((xixjy * xixky * xjxky * dist.xxyCounts.xyCounts.getClassCount(y)) /
                                                (xiy * xjy * xky * xixjxky));
                                    }
                                }
                            }
                        }
                    }
                }
                //printf("%lf\t", m);
                //assert(m >= -0.00000001);
                cmi[xk][xi][xj] = m; //I(Xi,Xj;Xk|Y)
                cmi[xk][xj][xi] = m;

            }
            //printf("\n");
        }
        //printf("-------------\n");
    }
}

/*
 *                   __
 *                   \                         P(x1,y|X2=x2)
 * MI(X1,Y|X2=x2)=   /_   P(x1,X2=x2,y) log----------------------
 *                 X1,X2=x2,Y               P(y|X2=x2)P(x1|X2=x2)
 *
 */
void getXMI(xxyDist &dist, std::vector<float> &xmi, const CategoricalAttribute & attno, const unsigned int & attvalno) {
    xmi.assign(dist.getNoCatAtts(), 0.0);
    const double totalCount = dist.xyCounts.count;

    for (CategoricalAttribute x1 = 0; x1 < dist.getNoCatAtts(); x1++) {

        CategoricalAttribute x2 = attno;
        double xm = 0.0;

        if (x1 != x2) {
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                CatValue v2 = attvalno;
                for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                    const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                    if (x1x2y) {
                        xm += (x1x2y / totalCount) * log2(x1x2y * dist.xyCounts.getCount(x2, v2) /
                                (static_cast<double> (dist.xyCounts.getCount(x2, v2, y)) *
                                dist.getCount(x1, v1, x2, v2)));
                    }
                }

            }
        } else {

            xm = 0.0;
        }
        xmi[x1] = xm;
    }
}

void getXMI_loc(xxyDist &dist, std::vector<float> &xmi, const CategoricalAttribute & attno, const unsigned int & attvalno, const instance & inst) {
    xmi.assign(dist.getNoCatAtts(), 0.0);
    const double totalCount = dist.xyCounts.count;

    for (CategoricalAttribute x1 = 0; x1 < dist.getNoCatAtts(); x1++) {

        CategoricalAttribute x2 = attno;
        double xm = 0.0;

        if (x1 != x2) {
            CatValue v1 = inst.getCatVal(x1);
            CatValue v2 = attvalno;

            for (CatValue y = 0; y < dist.getNoClasses(); y++) {

                const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                if (x1x2y) {
                    xm += (x1x2y / totalCount) * log2(x1x2y * dist.xyCounts.getCount(x2, v2) /
                            (static_cast<double> (dist.xyCounts.getCount(x2, v2, y)) *
                            dist.getCount(x1, v1, x2, v2)));
                }
            }


        } else {

            xm = 0.0;
        }
        xmi[x1] = xm;
    }
}

/**
 *
 *             __
 *             \              P(x,y)
 *  GMI(X,Y)=  /_ P(x,y)log------------
 *              y            P(x)P(y)
 *
 *
 */
void getGeneralMutualInformation(xyDist &dist, std::vector<std::vector<float> > &gmi) {

    int noCatAtts = dist.getNoCatAtts();
    gmi.resize(noCatAtts);
    int yNo = dist.getNoClasses();
    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        gmi[a].resize(yNo);
    }
    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        gmi[a].assign(noCatAtts, 0.0);
    }

    const double totalCount = dist.count;

    for (CategoricalAttribute a = 0; a < dist.getNoCatAtts(); a++) {
        double m = 0.0;

        for (CatValue v = 0; v < dist.getNoValues(a); v++) {
            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                const InstanceCount avyCount = dist.getCount(a, v, y);
                if (avyCount) {

                    m += (avyCount / totalCount) * log2(avyCount / ((dist.getCount(a, v) / totalCount)
                            * dist.getClassCount(y)));
                }
            }
        }
        // gmi[a] = m;
    }
}

void getSymmetricalUncert(xyDist &dist, std::vector<float> &su) {

    su.assign(dist.getNoCatAtts(), 0.0);

    const double totalCount = dist.count;

    for (CategoricalAttribute a = 0; a < dist.getNoCatAtts(); a++) {
        double m = 0.0;
        double xEnt = 0.0;
        double yEnt = 0.0;

        for (CatValue y = 0; y < dist.getNoClasses(); y++) {
            double pvy = dist.getClassCount(y) / totalCount;
            if (pvy)
                yEnt += pvy * log2(pvy);
        }

        for (CatValue v = 0; v < dist.getNoValues(a); v++) {

            double pvx = dist.getCount(a, v) / totalCount;
            if (pvx)
                xEnt += pvx * log2(pvx);

            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                const InstanceCount avyCount = dist.getCount(a, v, y);

                if (avyCount) {

                    m += (avyCount / totalCount)
                            * log2(avyCount / (pvx * dist.getClassCount(y)));
                }
            }
        }

        su[a] = 2 * (m / (-xEnt - yEnt));
    }
}

/*
 *                 __
 *                 \                    P(x1,y|x2)
 * CMI(X1,Y|X2)= = /_   P(x1,x2,y) log-------------
 *               X1,X2,Y              P(y|x2)P(x1|x2)
 *
 */
void getXCondMutualInf(xxyDist &dist, crosstab<float> &DoubleMI) {
    const double totalCount = dist.xyCounts.count;

    for (CategoricalAttribute x1 = 0; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < dist.getNoCatAtts(); x2++) {
            if (x1 != x2) {
                DoubleMI[x1][x2] = 0.0;
                for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                    for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                        for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                            const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                            if (x1x2y) {
                                DoubleMI[x1][x2] += (x1x2y / totalCount) * log2(x1x2y * dist.xyCounts.getCount(x2, v2) /
                                        (static_cast<double> (dist.xyCounts.getCount(x2, v2, y)) *
                                        dist.getCount(x1, v1, x2, v2)));
                            }
                        }
                    }
                }
            } else {

                DoubleMI[x1][x2] = 0.0;
            }

        }
    }
}

/*
 *                 __
 *                 \                    P(x1,x2|y)
 * CMI(X1,X2|Y)= = /_   P(x1,x2,y) log-------------
 *               x1,x2,y              P(x1|y)P(x2|y)
 *
 */
void getCondMutualInf(xxyDist &dist, crosstab<float> &cmi) {
    const double totalCount = dist.xyCounts.count;
    for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
            float m = 0.0;
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                    for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                        const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                        if (x1x2y) {
                            //const unsigned int yCount = dist->xyCounts.getClassCount(y);
                            //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
                            //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);

                            m += (x1x2y / totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y /
                                    (static_cast<double> (dist.xyCounts.getCount(x1, v1, y)) *
                                    dist.xyCounts.getCount(x2, v2, y)));
                        }
                    }
                }
            }

            assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision
            cmi[x1][x2] = m;
            cmi[x2][x1] = m;
        }
    }
}

void getCMIxxy(xxyDist &dist, crosstab<float> &cmi_xxy, crosstab<float> &cmi_xxy_ratio) {
    const double totalCount = dist.xyCounts.count;
    for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
            float m = 0.0;
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                    for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                        const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                        if (x1x2y) {
                            //const unsigned int yCount = dist->xyCounts.getClassCount(y);
                            //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
                            //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);

                            m += (x1x2y / totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y /
                                    (static_cast<double> (dist.xyCounts.getCount(x1, v1, y)) *
                                    dist.xyCounts.getCount(x2, v2, y)));
                        }
                    }
                }
            }
            assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision
            cmi_xxy[x1][x2] = m;
            cmi_xxy[x2][x1] = m;

            double hx1x2 = 0.0;
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                    const double x1x2 = dist.getCount(x1, v1, x2, v2) / totalCount; //p(x1,x2)
                    if (x1x2)
                        hx1x2 -= x1x2 * log2(x1x2);
                }
            }

            assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

            cmi_xxy_ratio[x1][x2] = m / hx1x2;
            cmi_xxy_ratio[x2][x1] = m / hx1x2;

        }
    }
}

void getCMIxxy2(xxyDist &dist, crosstab<float> &cmi_xxy, crosstab<float> &cmi_xxy_Ixx) {
    const double totalCount = dist.xyCounts.count;
    for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
            float m = 0.0;
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                    for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                        const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                        if (x1x2y) {
                            //const unsigned int yCount = dist->xyCounts.getClassCount(y);
                            //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
                            //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);

                            m += (x1x2y / totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y /
                                    (static_cast<double> (dist.xyCounts.getCount(x1, v1, y)) *
                                    dist.xyCounts.getCount(x2, v2, y)));
                        }
                    }
                }
            }
            assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision
            cmi_xxy[x1][x2] = m;
            cmi_xxy[x2][x1] = m;

            double Ixx = 0.0; //I(x1,x2)
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                    const InstanceCount avyCount = dist.getCount(x1, v1, x2, v2);
                    if (avyCount) {
                        Ixx += (avyCount / totalCount) * log2(avyCount / ((dist.xyCounts.getCount(x1, v1) / totalCount)
                                * dist.xyCounts.getCount(x2, v2)));
                    }
                }
            }
            printf("%lle\t", Ixx);
            assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

            cmi_xxy_Ixx[x1][x2] = m / Ixx;
            cmi_xxy_Ixx[x2][x1] = m / Ixx;
        }
        printf("\n");
    }
}

/*条件互信息I(Xi;Y|Xj)
 *条件互信息增益I(Xi;Y|Xj)/H(Xi,Xj)
 *                 __
 *                 \                    P(x1,y|x2)
 * CMI(X1,Y|X2)= = /_   P(x1,x2,y) log-------------
 *               x1,x2,y              P(x1|x2)P(y|x2)
 *
 */
void getCMIxyx(xxyDist &dist, crosstab<float> &cmi_xyx, crosstab<float> &cmi_xyx_ratio) {
    const double totalCount = dist.xyCounts.count;
    for (CategoricalAttribute x1 = 0; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < dist.getNoCatAtts(); x2++) {
            if (x1 != x2) {
                float m = 0.0;
                for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                    for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                        for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                            const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                            if (x1x2y) {
                                //const unsigned int yCount = dist->xyCounts.getClassCount(y);
                                //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
                                //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);

                                m += (x1x2y / totalCount) * log2(dist.xyCounts.getCount(x2, v2) * x1x2y /
                                        (static_cast<double> (dist.getCount(x1, v1, x2, v2)) *
                                        dist.xyCounts.getCount(x2, v2, y)));
                            }
                        }
                    }
                }

                assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision
                cmi_xyx[x1][x2] = m;
                double hx1x2 = 0.0;
                for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                    for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                        const double x1x2 = dist.getCount(x1, v1, x2, v2) / totalCount; //p(x1,x2)
                        if (x1x2)
                            hx1x2 -= x1x2 * log2(x1x2);
                    }
                }

                assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

                cmi_xyx_ratio[x1][x2] = m / hx1x2;
            }
        }
    }
}

void getCondMutualInfTC(xxyDist &dist, crosstab3D<double> &cmi) {

    const double totalCount = dist.xyCounts.count;
    for (CategoricalAttribute c = 0; c < dist.getNoClasses(); c++) {
        for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
            //for (CategoricalAttribute x1 = 1; x1 < 5; x1++) {
            for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
                double m = 0.0;
                for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                    for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                        CatValue y = c;
                        const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                        if (x1x2y) {
                            //const unsigned int yCount = dist->xyCounts.getClassCount(y);
                            //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
                            //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);

                            m += (x1x2y / totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y /
                                    (static_cast<double> (dist.xyCounts.getCount(x1, v1, y)) *
                                    dist.xyCounts.getCount(x2, v2, y)));
                        }

                    }
                }

                //assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision
                // printf("c= %d  x1 = %d  x2 = %d  cmi[c][x1][x2] = %f\n", c, x1, x2, m);
                cmi[c][x1][x2] = m;
                cmi[c][x2][x1] = m;
                //cmi[c][x1].push_back(m);

            }
        }
    }
}

/*
 *                 __
 *                 \                     P(x1,x2|y)
 * CMI(x1,x2|Y)= = /_   P(x1,x2,y) log------------------
 *                  Y                   P(x1|y)P(x2|y)
 *
 */
void getCondMutualInfTCloc(xxyDist &dist, crosstab3D<float> &cmi, const instance & inst) {

    const double totalCount = dist.xyCounts.count;
    for (CategoricalAttribute c = 0; c < dist.getNoClasses(); c++) {
        for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
            for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
                double m = 0.0;
                CatValue v1 = inst.getCatVal(x1);
                CatValue v2 = inst.getCatVal(x2);
                CatValue y = c;
                const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                if (x1x2y) {
                    //const unsigned int yCount = dist->xyCounts.getClassCount(y);
                    //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
                    //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);

                    m += (x1x2y / totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y /
                            (static_cast<double> (dist.xyCounts.getCount(x1, v1, y)) *
                            dist.xyCounts.getCount(x2, v2, y)));
                }




                //assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision
                // printf("c= %d  x1 = %d  x2 = %d  cmi[c][x1][x2] = %f\n", c, x1, x2, m);
                cmi[c][x1][x2] = m;
                cmi[c][x2][x1] = m;
                //cmi[c][x1].push_back(m);

            }
        }
    }
}

void getCondMutualInfloc(xxyDist &dist, crosstab<float> &cmi, const instance & inst) {
    // dist.xyCounts.update(inst);
    const double totalCount = dist.xyCounts.count;

    for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
        //for (CategoricalAttribute x1 = 1; x1 < 5; x1++) {
        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
            float m = 0.0;
            CatValue v1 = inst.getCatVal(x1);
            CatValue v2 = inst.getCatVal(x2);
            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                const double x1x2y = dist.getCount(x1, v1, x2, v2, y);

                if (x1x2y) {
                    //const unsigned int yCount = dist->xyCounts.getClassCount(y);
                    //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
                    //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);

                    m += (x1x2y / totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y /
                            (static_cast<double> (dist.xyCounts.getCount(x1, v1, y)) *
                            dist.xyCounts.getCount(x2, v2, y)));
                    //                    m += (x1x2y / dist.xyCounts.getClassCount(y)) * log2(dist.xyCounts.getClassCount(y) * x1x2y /
                    //                            (static_cast<double> (dist.xyCounts.getCount(x1, v1, y)) *
                    //                            dist.xyCounts.getCount(x2, v2, y)));
                }
            }



            //assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

            cmi[x1][x2] = m;
            cmi[x2][x1] = m;
        }
    }
}

void getCondMutualInflocloc(xxyDist &dist, CategoricalAttribute y, crosstab<float> &cmi, const instance & inst) {
    // dist.xyCounts.update(inst);
    const double totalCount = dist.xyCounts.count;

    for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
        //for (CategoricalAttribute x1 = 1; x1 < 5; x1++) {
        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
            float m = 0.0;
            CatValue v1 = inst.getCatVal(x1);
            CatValue v2 = inst.getCatVal(x2);
            const double x1x2y = dist.getCount(x1, v1, x2, v2, y);

            if (x1x2y) {
                m +=  log2(dist.xyCounts.getClassCount(y) * x1x2y /
                        (static_cast<double> (dist.xyCounts.getCount(x1, v1, y)) *
                        dist.xyCounts.getCount(x2, v2, y)));
            }
            if(m < -0.00000001)
                m = 0;
            cmi[x1][x2] = m;
            cmi[x2][x1] = m;
        }
    }
}

/*
 *                 __
 *                 \                    P(x1,x2|y)
 * CMI(X1,X2|Y)= = /_   P(x1,x2,y) log-------------
 *               x1,x2,y              P(x1|y)P(x2|y)
 *
 */
void getCMI_Ratio(xxyDist &dist, crosstab<float> &cmi) {
    const double totalCount = dist.xyCounts.count;

    for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
            float m = 0.0;
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                    for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                        const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                        if (x1x2y) {
                            //const unsigned int yCount = dist->xyCounts.getClassCount(y);
                            //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
                            //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);

                            m += (x1x2y / totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y /
                                    (static_cast<double> (dist.xyCounts.getCount(x1, v1, y)) *
                                    dist.xyCounts.getCount(x2, v2, y)));
                        }
                    }
                }
            }
            double hx1x2 = 0.0;
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                    hx1x2 -= dist.getCount(x1, v1, x2, v2) / totalCount;
                }
            }

            assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

            cmi[x1][x2] = m / hx1x2;
            cmi[x2][x1] = m / hx1x2;
        }
    }
}

void getH(xxyDist &dist, crosstab<float> &cmi) {
    const double totalCount = dist.xyCounts.count;

    for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
            float m = 0.0;
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                    for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                        const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                        if (x1x2y) {
                            m -= (x1x2y / totalCount) * log2(x1x2y / dist.xyCounts.getClassCount(y));
                        }
                    }
                }
            }



            assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

            cmi[x1][x2] = -m; //乘了负号！！
            cmi[x2][x1] = -m;
        }
    }
}

/*
 *
 *                      __
 *                      \                       P(x1,x2|X3=x3,y)
 * XCMI(X1,X2|X3=x3,Y)= /_ P(x1,x2,X3=x3,y)log------------------------
 *                   x1,x2,X3=x3,y            P(x1|X3=x3,y)P(x2|X3=x3,y)
 *
 *
 */
void getXCMI(xxxyDist &dist, crosstab<float> &xcmi, const CategoricalAttribute & attno, const CategoricalAttribute & attvalno) {
    const double totalCount = dist.xxyCounts.xyCounts.count;

    for (CategoricalAttribute x1 = 2; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {
            CategoricalAttribute x3 = attno;
            float m1 = 0.0;
            if (x1 != x3 && x2 != x3) {
                for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                    for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                        CatValue v3 = attvalno;
                        for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                            const double x1x2x3y = dist.getCount(x1, v1, x2, v2, x3, v3, y);
                            if (x1x2x3y) {
                                const unsigned int x1x3y = dist.xxyCounts.getCount(x1, v1, x3, v3, y);
                                const unsigned int x2x3y = dist.xxyCounts.getCount(x2, v2, x3, v3, y);
                                m1 += (x1x2x3y / totalCount) * log2(dist.xxyCounts.xyCounts.getCount(x3, v3, y) * x1x2x3y /
                                        (static_cast<double> (x1x3y) * x2x3y));
                            }
                        }

                    }
                }
            } else {

                m1 = 0.0;
            }
            //assert(m1 >= -0.00000001);

            xcmi[x1][x2] = m1;
            xcmi[x2][x1] = m1;
        }
    }
}

void getXCMI_loc(xxxyDist &dist, crosstab<float> &xcmi, const CategoricalAttribute & attno, const CategoricalAttribute & attvalno, const instance & inst) {
    const double totalCount = dist.xxyCounts.xyCounts.count;

    for (CategoricalAttribute x1 = 2; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {
            CategoricalAttribute x3 = attno;
            float m1 = 0.0;
            if (x1 != x3 && x2 != x3) {
                CatValue v1 = inst.getCatVal(x1);
                CatValue v2 = inst.getCatVal(x2);
                CatValue v3 = attvalno;
                for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                    const double x1x2x3y = dist.getCount(x1, v1, x2, v2, x3, v3, y);
                    if (x1x2x3y) {
                        const unsigned int x1x3y = dist.xxyCounts.getCount(x1, v1, x3, v3, y);
                        const unsigned int x2x3y = dist.xxyCounts.getCount(x2, v2, x3, v3, y);
                        m1 += (x1x2x3y / totalCount) * log2(dist.xxyCounts.xyCounts.getCount(x3, v3, y) * x1x2x3y /
                                (static_cast<double> (x1x3y) * x2x3y));
                    }
                }
            } else {

                m1 = 0.0;
            }
            //assert(m1 >= -0.00000001);

            xcmi[x1][x2] = m1;
            xcmi[x2][x1] = m1;
        }
    }
}


/*
 *                      __                                      __
 *                      \                    P(x2,y|x1)         \                    P(x1,y,x2)P(x1)
 * DoubleMI(Y;X2|X1)=   /_   P(x1,x2,y) log-------------     =  /_   P(x1,x2,y) log-------------    
 *                     x1,x2,y              P(x2|x1)P(y|x1)    x1,x2,y              P(x1,x2)P(y,x1)
 *
 */

/*
 *                      __                                     
 *                      \                    P(x1,x2,y)        
 * DoubleMI(Y;X2,X1)=   /_   P(x1,x2,y) log-------------     
 *                     x1,x2,y              P(x2,x1)P(y)      
 *
 */
void getDoubleMutualInf(xxyDist &dist, crosstab<float> &DoubleMI) {
    for (CategoricalAttribute x1 = 0; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < dist.getNoCatAtts(); x2++) {
            if (x1 != x2) {
                DoubleMI[x1][x2] = 0.0;
                for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                    for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                        for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                            const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                            if (x1x2y) {

                                DoubleMI[x1][x2] += dist.jointP(x1, v1, x2, v2, y) * log2(dist.jointP(x1, v1, x2, v2, y) /
                                        (static_cast<double> (dist.jointP(x1, v1, x2, v2) * dist.xyCounts.p(y))));
                            }
                        }
                    }
                }
            }

        }
    }
}

void getErrorDiff(xxyDist &dist, crosstab<float> &cm) {
    for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
            float m = 0.0;
            std::vector<double> classDistIndep(dist.getNoClasses());
            std::vector<double> classDist(dist.getNoClasses());

            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                    for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                        const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                        if (x1x2y) {
                            classDistIndep[y] = dist.xyCounts.p(y) * dist.xyCounts.p(x1, v1, y)
                                    * dist.xyCounts.p(x2, v2, y);
                            classDist[y] = dist.xyCounts.p(y) * dist.xyCounts.p(x1, v1, y)
                                    * dist.p(x2, v2, x1, v1, y);
                        } else {
                            classDistIndep[y] = 0.0;
                            classDist[y] = 0.0;
                        }
                    }
                    normalise(classDistIndep);
                    normalise(classDist);
                    for (CatValue y = 0; y < dist.getNoClasses(); y++) {

                        m += fabs(classDistIndep[y] - classDist[y]);
                    }
                }
            }

            cm[x1][x2] = m;
            cm[x2][x1] = m;
        }
    }
}

double getSymmetricalUncert(const xxyDist &dist, CategoricalAttribute x1, CategoricalAttribute x2) {

    const double totalCount = dist.xyCounts.count;

    double x1Ent = 0.0;
    double x2Ent = 0.0;

    double m = 0.0;

    for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
        double px2 = dist.xyCounts.getCount(x2, v2) / totalCount;
        if (px2)
            x2Ent += px2 * log2(px2);
    }
    for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {

        double px1 = dist.xyCounts.getCount(x1, v1) / totalCount;
        if (px1)
            x1Ent += px1 * log2(px1);

        for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
            const InstanceCount x2Count = dist.xyCounts.getCount(x2, v2);
            const InstanceCount x1x2Count = dist.getCount(x1, v1, x2, v2);

            if (x1x2Count) {

                m += (x1x2Count / totalCount)
                        * log2(x1x2Count / (px1 * x2Count));
            }

        }

    }
    return 2 * (m / (-x1Ent - x2Ent));

}

void getCondSymmUncert(xxyDist &dist, crosstab<float> &csu) {
    const double totalCount = dist.xyCounts.count;


    for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
            float m = 0.0;
            double x1yEnt = 0.0;
            double x2yEnt = 0.0;
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                    const double x1y = dist.xyCounts.getCount(x1, v1, y);
                    if (x1y)
                        x1yEnt += (x1y / totalCount) *
                        log2(x1y / dist.xyCounts.getClassCount(y));
                    for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                        if (v1 == 0) {
                            const double x2y = dist.xyCounts.getCount(x2, v2, y);
                            if (x2y)
                                x2yEnt += (x2y / totalCount) *
                                log2(x2y / dist.xyCounts.getClassCount(y));
                        }
                        const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                        if (x1x2y) {
                            //const unsigned int yCount = dist->xyCounts.getClassCount(y);
                            //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
                            //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);
                            m += (x1x2y / totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y /
                                    (static_cast<double> (dist.xyCounts.getCount(x1, v1, y)) *
                                    dist.xyCounts.getCount(x2, v2, y)));
                        }
                    }
                }
            }
            assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

            //ocurring when x1 and x2 are useless attributes, it is replaced by -1.

            if ((-x1yEnt - x2yEnt) == 0)
                m = -1;
            m = 2 * (m / (-x1yEnt - x2yEnt));

            csu[x1][x2] = m;
            csu[x2][x1] = m;
        }
    }
}

double chiSquare(const InstanceCount *cells, const unsigned int rows,
        const unsigned int cols) {
    std::vector<unsigned int> rowSums(rows, 0);
    std::vector<unsigned int> colSums(cols, 0);
    unsigned int n = 0;
    int degreesOfFreedom = (rows - 1) * (cols - 1);

    if (degreesOfFreedom == 0) return 1.0;

    for (unsigned int r = 0; r < rows; r++) {
        for (unsigned int c = 0; c < cols; c++) {
            rowSums[r] += cells[r * cols + c];
            colSums[c] += cells[r * cols + c];
            n += cells[r * cols + c];
        }
    }

    double chisq = 0.0;

    for (unsigned int r = 0; r < rows; r++) {
        if (rowSums[r] != 0) {
            for (unsigned int c = 0; c < cols; c++) {
                if (colSums[c] != 0) {

                    double expect = rowSums[r]*(colSums[c] / static_cast<double> (n));
                    const double diff = cells[r * cols + c] - expect;
                    chisq += (diff * diff) / expect;
                }
            }
        }
    }

    return alglib::chisquarecdistribution(degreesOfFreedom, chisq);
}

/*
 *
 *                __
 *                \                     P(x1,y|x2)
    CMI(X1,Y|X2)= /_   P(x1,y,x2) log ---------------
 *               x1,x2,y               P(x1|x2)P(y|x2)
 *
 */

void getAttClassCondMutualInf(xxyDist &dist, crosstab<float> &acmi) {
    const double totalCount = dist.xyCounts.count;

    for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
            float m1 = 0.0, m2 = 0.0;
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                    for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                        const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                        if (x1x2y) {

                            m1 += (x1x2y / totalCount) * log2(dist.xyCounts.getCount(x2, v2) * x1x2y /
                                    (static_cast<double> (dist.getCount(x1, v1, x2, v2)) *
                                    dist.xyCounts.getCount(x2, v2, y)));
                            m2 += (x1x2y / totalCount) * log2(dist.xyCounts.getCount(x1, v1) * x1x2y /
                                    (static_cast<double> (dist.getCount(x2, v2, x1, v1)) *
                                    dist.xyCounts.getCount(x1, v1, y)));
                        }
                    }
                }
            }

            assert(m1 >= -0.00000001);
            assert(m2 >= -0.00000001);
            //MI(x1;c|x2)
            acmi[x1][x2] = m1;
            //MI(x2;c|x1)
            acmi[x2][x1] = m2;
        }
    }
}

/*
 *
 *                   __
 *                   \                       P(x1,x2|x3,y)
 * MCMI(X1,X2|X3,Y)= /_ P(x1,x2,x3,y)log------------------------
 *                x1,x2,x3,y               P(x1|x3,y)P(x2|x3,y)
 *
 *
 */
void getMultCondMutualInf(xxxyDist &dist, std::vector<crosstab<float> > &mcmi) {
    const double totalCount = dist.xxyCounts.xyCounts.count;

    for (CategoricalAttribute x1 = 2; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {
            for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {
                float m1 = 0.0, m2 = 0.0, m3 = 0.0;
                for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                    for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                        for (CatValue v3 = 0; v3 < dist.getNoValues(x3); v3++) {
                            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                                const double x1x2x3y = dist.getCount(x1, v1, x2, v2, x3, v3, y);
                                if (x1x2x3y) {

                                    const unsigned int x1x2y = dist.xxyCounts.getCount(x1, v1, x2, v2, y);
                                    const unsigned int x1x3y = dist.xxyCounts.getCount(x1, v1, x3, v3, y);
                                    const unsigned int x2x3y = dist.xxyCounts.getCount(x2, v2, x3, v3, y);
                                    m1 += (x1x2x3y / totalCount) * log2(dist.xxyCounts.xyCounts.getCount(x3, v3, y) * x1x2x3y /
                                            (static_cast<double> (x1x3y) * x2x3y));
                                    m2 += (x1x2x3y / totalCount) * log2(dist.xxyCounts.xyCounts.getCount(x2, v2, y) * x1x2x3y /
                                            (static_cast<double> (x1x2y) * x2x3y));
                                    m3 += (x1x2x3y / totalCount) * log2(dist.xxyCounts.xyCounts.getCount(x1, v1, y) * x1x2x3y /
                                            (static_cast<double> (x1x3y) * x1x2y));
                                }
                            }
                        }
                    }
                }

                assert(m1 >= -0.00000001);
                assert(m2 >= -0.00000001);
                assert(m3 >= -0.00000001);

                mcmi[x1][x2][x3] = m1;
                mcmi[x2][x1][x3] = m1;
                mcmi[x1][x3][x2] = m2;
                mcmi[x3][x1][x2] = m2;
                mcmi[x2][x3][x1] = m3;
                mcmi[x3][x2][x1] = m3;
            }
        }
    }
}

/*                  __
 *                  \                   P(x1,x2,y)
 * PMI(<X1,X2>,Y)=  /_   P(x1,x2,y)log---------------
 *                x1,x2,y              P(x1,x2)P(y)
 *
 */
void getPairMutualInf(xxyDist &dist, crosstab<float> &pmi) {

    const double totalCount = dist.xyCounts.count;

    for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
            float m = 0.0;
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                    for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                        const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                        if (x1x2y) {
                            //const unsigned int yCount = dist->xyCounts.getClassCount(y);
                            //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
                            //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);

                            m += (x1x2y / totalCount) * log2(totalCount * x1x2y /
                                    (static_cast<double> (dist.getCount(x1, v1, x2, v2)) *
                                    dist.xyCounts.getClassCount(y)));
                        }
                    }
                }
            }

            assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

            pmi[x1][x2] = m;
            pmi[x2][x1] = m;
        }
    }
}

void getBothCondMutualInf(xxyDist &dist, crosstab<float> &cmi,
        crosstab<float> &acmi) {
    const double totalCount = dist.xyCounts.count;

    for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
        for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
            float acmi_m1 = 0.0, acmi_m2 = 0.0, m = 0.0;
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
                for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                    for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                        const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
                        if (x1x2y) {

                            acmi_m1 += (x1x2y / totalCount) * log2(dist.xyCounts.getCount(x2, v2) * x1x2y /
                                    (static_cast<double> (dist.getCount(x1, v1, x2, v2)) *
                                    dist.xyCounts.getCount(x2, v2, y)));
                            acmi_m2 += (x1x2y / totalCount) * log2(dist.xyCounts.getCount(x1, v1) * x1x2y /
                                    (static_cast<double> (dist.getCount(x2, v2, x1, v1)) *
                                    dist.xyCounts.getCount(x1, v1, y)));
                            m += (x1x2y / totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y /
                                    (static_cast<double> (dist.xyCounts.getCount(x1, v1, y)) *
                                    dist.xyCounts.getCount(x2, v2, y)));
                        }
                    }
                }
            }

            assert(m >= -0.00000001);
            assert(acmi_m1 >= -0.00000001);
            assert(acmi_m2 >= -0.00000001);
            cmi[x1][x2] = m;
            cmi[x2][x1] = m;
            //MI(x1;c|x2)
            acmi[x1][x2] = acmi_m1;
            //MI(x2;c|x1)
            acmi[x2][x1] = acmi_m2;
        }
    }

}

void getrow(crosstab<InstanceCount> &xtab, unsigned int noClasses, unsigned int trow, std::vector<InstanceCount> &Crow) {
    for (unsigned int k = 0; k < noClasses; k++) {

        Crow[k] = xtab[trow][k];
    }
}

void getcol(crosstab<InstanceCount> &xtab, unsigned int noClasses, unsigned int tcol, std::vector<InstanceCount> &Ccol) {
    for (unsigned int k = 0; k < noClasses; k++) {

        Ccol[k] = xtab[k][tcol];
    }
}

unsigned long long int dotproduct(std::vector<InstanceCount> &Crow, std::vector<InstanceCount> &Ccol, unsigned int noClasses) {
    unsigned long long int val = 0;
    fflush(stdout);
    for (unsigned int k = 0; k < noClasses; k++) {

        val += static_cast<unsigned long long> (Crow[k]) * static_cast<unsigned long long> (Ccol[k]);
    }
    return val;
}

double calcMCC(crosstab<InstanceCount> &xtab) {
    // Compute MCC for multi-class problems as in http://rk.kvl.dk/
    unsigned int noClasses = xtab[0].size();
    double MCC = 0.0;

    //Compute N, sum of all values
    double N = 0.0;
    for (unsigned int k = 0; k < noClasses; k++) {
        for (unsigned int l = 0; l < noClasses; l++) {
            N += xtab[k][l];
        }
    }

    //compute correlation coefficient
    double trace = 0.0;
    for (unsigned int k = 0; k < noClasses; k++) {
        trace += xtab[k][k];
    }
    //sum row col dot product
    unsigned long long int rowcol_sumprod = 0;
    std::vector<InstanceCount> Crow(noClasses);
    std::vector<InstanceCount> Ccol(noClasses);
    for (unsigned int k = 0; k < noClasses; k++) {
        for (unsigned int l = 0; l < noClasses; l++) {
            getrow(xtab, noClasses, k, Crow);
            getcol(xtab, noClasses, l, Ccol);
            rowcol_sumprod += dotproduct(Crow, Ccol, noClasses);
        }
    }

    //sum over row dot products
    unsigned long long int rowrow_sumprod = 0;
    std::vector<InstanceCount> Crowk(noClasses);
    std::vector<InstanceCount> Crowl(noClasses);
    for (unsigned int k = 0; k < noClasses; k++) {
        for (unsigned int l = 0; l < noClasses; l++) {
            getrow(xtab, noClasses, k, Crowk);
            getrow(xtab, noClasses, l, Crowl);
            rowrow_sumprod += dotproduct(Crowk, Crowl, noClasses);
        }
    }

    //sum over col dot products
    unsigned long long int colcol_sumprod = 0;
    std::vector<InstanceCount> Ccolk(noClasses);
    std::vector<InstanceCount> Ccoll(noClasses);
    for (unsigned int k = 0; k < noClasses; k++) {
        for (unsigned int l = 0; l < noClasses; l++) {
            getcol(xtab, noClasses, k, Ccolk);
            getcol(xtab, noClasses, l, Ccoll);
            colcol_sumprod += dotproduct(Ccolk, Ccoll, noClasses);
        }
    }

    double cov_XY = N * trace - rowcol_sumprod;
    double cov_XX = N * N - rowrow_sumprod;
    double cov_YY = N * N - colcol_sumprod;
    double denominator = sqrt(cov_XX * cov_YY);

    if (denominator > 0) {
        MCC = cov_XY / denominator;
    } else if (denominator == 0) {
        MCC = 0;
    } else {
        printf("Error when calculating MCC2");
    }
    return MCC;
}
