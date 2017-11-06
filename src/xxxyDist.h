#pragma once

#include "xxyDist.h"
#include "assert.h"


class constXXYSubDist {
public:
  constXXYSubDist(const std::vector<std::vector<InstanceCount> >* subDist, const unsigned int noOfClasses) : subDist_(subDist), noOfClasses_(noOfClasses) {}

  inline const std::vector<InstanceCount>* getXYSubDist(const CategoricalAttribute x, const CatValue v) const { return &subDist_[x][v*x]; }
  inline const InstanceCount getCount(const CategoricalAttribute x1, const CatValue v1, const CategoricalAttribute x2, const CatValue v2, const CatValue y) const { return subDist_[x1][v1*x1+x2][v2*noOfClasses_+y]; }

private:
  const std::vector<std::vector<InstanceCount> >* subDist_;
  const unsigned int noOfClasses_;
};

class XXYSubDist {
public:
  XXYSubDist(std::vector<std::vector<InstanceCount> >* subDist, const unsigned int noOfClasses) : subDist_(subDist), noOfClasses_(noOfClasses) {}

  inline std::vector<InstanceCount>* getXYSubDist(const CategoricalAttribute x, const CatValue v) { return &subDist_[x][v*x]; }
  inline std::vector<InstanceCount>* getXYSubDist(const CategoricalAttribute x, const CatValue v,const CatValue length) { return &subDist_[x][v*length]; }

  inline InstanceCount getCount(const CategoricalAttribute x1, const CatValue v1, const CategoricalAttribute x2, const CatValue v2, const CatValue y) const { return subDist_[x1][v1*x1+x2][v2*noOfClasses_+y]; }
  inline InstanceCount getCount(const CategoricalAttribute x1, const CatValue v1, const CategoricalAttribute x2, const CatValue v2, const CatValue y,const CatValue length) const { return subDist_[x1][v1*length+x2][v2*noOfClasses_+y]; }

  inline void incCount(const CategoricalAttribute x1, const CatValue v1, const CategoricalAttribute x2, const CatValue v2, const CatValue y) { ++subDist_[x1][v1*x1+x2][v2*noOfClasses_+y]; }

  inline void incCount(const CategoricalAttribute x1, const CatValue v1, const CategoricalAttribute x2, const CatValue v2, const CatValue y,const CatValue length) { ++subDist_[x1][v1*length+x2][v2*noOfClasses_+y]; }

  std::vector<std::vector<InstanceCount> >* subDist_;
  const unsigned int noOfClasses_;
};

class xxxyDist {
public:
	xxxyDist();
	xxxyDist(InstanceStream& stream);
	~xxxyDist(void);

	void reset(InstanceStream& stream);

	void update(const instance& i);

	void clear();

	// p(x1=v1, x2=v2, x3=v3, Y=y) unsmoothed
	inline double rawJointP(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
			CatValue v3, CatValue y) const {
		return (*constRef(x1, v1, x2, v2, x3, v3, y))
				/ (xxyCounts.xyCounts.count);
	}

	// p(x1=v1, x2=v2, x3=v3, Y=y) using M-estimate
	inline double jointP(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
			CatValue v3, CatValue y) const {
		return (*constRef(x1, v1, x2, v2, x3, v3, y)
				+ M
						/ (getNoValues(x1) * getNoValues(x2) * getNoValues(x3)
								* noClasses_)) / (xxyCounts.xyCounts.count + M);
	}
	// p(x1=v1, x2=v2, x3=v3) using M-estimate
	inline double jointP(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
			CatValue v3) const {
		return (getCount(x1, v1, x2, v2, x3, v3)
				+ M / (metaData_->getNoValues(x1) * metaData_->getNoValues(x2)* metaData_->getNoValues(x3)))
				/ (xxyCounts.xyCounts.count + M);

	}
// p(x1=v1|Y=y, x2=v2, x3=v3) using M-estimate
	inline double p(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
			CatValue v3, CatValue y) const {

		return (*constRef(x1, v1, x2, v2, x3, v3, y) + M / getNoValues(x1))
				/ (xxyCounts.getCount(x2, v2, x3, v3, y) + M);
	}

        // p(x1=v1|Y=y, x2=v2, x3=v3) using M-estimate
	inline double unorderedP(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
			CatValue v3, CatValue y) const {
                CategoricalAttribute a1;
                CategoricalAttribute a2;
                CategoricalAttribute a3;
                CatValue V1;
                CatValue V2;
                CatValue V3;

                if (x2 < x3) {
                  CategoricalAttribute t = x2;
                  x2 = x3;
                  x3 = t;
                  CatValue tv = v2;
                  v2 = v3;
                  v3 = tv;
                }
                
                if (x1 < x3) {
                  a1 = x2;
                  a2 = x3;
                  a3 = x1;
                  V1 = v2;
                  V2 = v3;
                  V3 = v1;
                }
                else if (x1 < x2) {
                  a1 = x2;
                  a2 = x1;
                  a3 = x3;
                  V1 = v2;
                  V2 = v1;
                  V3 = v3;
                }
                else {
                  a1 = x1;
                  a2 = x2;
                  a3 = x3;
                  V1 = v1;
                  V2 = v2;
                  V3 = v3;

                }

		return (*constRef(a1, V1, a2, V2, a3, V3, y) + M / getNoValues(x1))
				/ (xxyCounts.getCount(x2, v2, x3, v3, y) + M);
	}

	// p(x1=v1, x2=v2, x3=v3, Y=y)
	inline InstanceCount getCount(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
			CatValue v3, CatValue y) const {
		return *constRef(x1, v1, x2, v2, x3, v3, y);
	}

	// count for instances x1=v1, x2=v2,x3=v3
	inline InstanceCount getCount(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
			CatValue v3) const {
		InstanceCount c = 0;

		for (CatValue y = 0; y < noClasses_; y++) {
			c += getCount(x1, v1, x2, v2, x3, v3, y);
		}
		return c;
	}

	inline unsigned int getNoCatAtts() const {
		return noCatAtts_;
	}

	inline unsigned int getNoValues(const CategoricalAttribute a) const {
		return metaData_->getNoValues(a);
	}

	inline unsigned int getNoClasses() const {
		return noClasses_;
	}

	  inline std::vector<std::vector<InstanceCount > >* getXXYSubDist(CategoricalAttribute x1, CatValue v1) {
	    return &count[x1][v1*x1];
	  }

	  inline std::vector<InstanceCount >* getXYSubDist(CategoricalAttribute x1, CatValue v1,CategoricalAttribute x2, CatValue v2) {
	    return &count[x1][v1*x1+x2][v2*x2];
	  }

private:
	// count[X1=x1][X2=x2][X3=x3][Y=y]
	inline InstanceCount *ref(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
			CatValue v3, CatValue y) {
		if (x2 > x1) {
			CatValue t = x1;
			x1 = x2;
			x2 = t;
			t = v1;
			v1 = v2;
			v2 = t;
		}
		if (x3 > x2) {
			CatValue t = x2;
			x2 = x3;
			x3 = t;
			t = v2;
			v2 = v3;
			v3 = t;
		}
		if (x2 > x1) {
			CatValue t = x1;
			x1 = x2;
			x2 = t;
			t = v1;
			v1 = v2;
			v2 = t;
		}
		assert(x1 > x2 && x2 >x3);
		return &count[x1][v1 * x1 + x2][v2 * x2 + x3][v3 * noClasses_ + y];
	}

//	// count[X1=x1][X2=x2][X3=x3]
//	inline std::vector<instanceCount> *xxxref(CategoricalAttribute x1, CatValue v1,
//			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3, CatValue v3) {
//		assert(x1 < x2 && x2 < x3);
//
//		return &count[x1][v1][x2][v2][x3][v3];
//	}

// count[X1=x1][X2=x2][X3=x3][Y=y]
	inline const InstanceCount *constRef(CategoricalAttribute x1, CatValue v1,
			CategoricalAttribute x2, CatValue v2, CategoricalAttribute x3,
			CatValue v3, CatValue y) const {
		if (x2 > x1) {
			CatValue t = x1;
			x1 = x2;
			x2 = t;
			t = v1;
			v1 = v2;
			v2 = t;
		}
		if (x3 > x2) {
			CatValue t = x2;
			x2 = x3;
			x3 = t;
			t = v2;
			v2 = v3;
			v3 = t;
		}
		if (x2 > x1) {
			CatValue t = x1;
			x1 = x2;
			x2 = t;
			t = v1;
			v1 = v2;
			v2 = t;
		}
		assert(x1 > x2 && x2 > x3);
		return &count[x1][v1 * x1 + x2][v2 * x2 + x3][v3 * noClasses_ + y];
	}

public:
	xxyDist xxyCounts;

private:
	unsigned int noCatAtts_; ///< the number of categorical CategoricalAttributes.
	unsigned int noClasses_;  ///< the number of classes

	InstanceStream::MetaData* metaData_;

	// a four-dimensional semiflattened representation of count[X1=x1][X2=x2][X3=x3][Y=y] storing only X1 > X2 > X3
	// outer vector is indexed by X1
	// second vector is indexed by x1*X2
	// third vector is indexed by x2*X3
	// inner vector is indexed by x3*y
	std::vector<std::vector<std::vector<std::vector<InstanceCount> > > > count;

};
