#include "xxxxyDist3.h"

#include "utils.h"


xxxxyDist3::xxxxyDist3() {

}
xxxxyDist3::xxxxyDist3(InstanceStream& stream) {
	reset(stream);

	// pass through the stream updating the counts incrementally
	stream.rewind();

	instance i;

	while (stream.advance(i)) {
		update(i);
	}

}
void xxxxyDist3::setOrder(std::vector<CategoricalAttribute> &order) {
	order_ = order;
}

void xxxxyDist3::setNoSelectedCatAtts(unsigned int noSelectedCatAtts) {
	noSelectedCatAtts_ = noSelectedCatAtts;
}
void xxxxyDist3::reset(InstanceStream& stream) {

	metaData_ = stream.getMetaData();

	instanceStream_ = &stream;

	xxxyCounts.reset(stream);

	noCatAtts_ = instanceStream_->getNoCatAtts();
	noClasses_ = instanceStream_->getNoClasses();

	noUnSelectedCatAtts_ = noCatAtts_ - noSelectedCatAtts_;

	//out vector
	countSelected.resize(noSelectedCatAtts_);

	countUnSelected.resize(noSelectedCatAtts_);

	for (CategoricalAttribute x1 = 2; x1 < noSelectedCatAtts_; x1++) {

		//second vector
		unsigned int noValueX1 = getNoValues(x1);
		countSelected[x1].resize(noValueX1 * x1);
		countUnSelected[x1].resize(noValueX1 * x1);

		for (CatValue v1 = 0; v1 < noValueX1; v1++) {
			for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {

				unsigned int noValueX2 = getNoValues(x2);
				//third vector
				countSelected[x1][v1 * x1 + x2].resize(noValueX2 * x2);
				countUnSelected[x1][v1 * x1 + x2].resize(noValueX2 * x2);

				for (CatValue v2 = 0; v2 < noValueX2; v2++) {
					for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {

						//fourth vector
						unsigned int noValueX3 = getNoValues(x3);
						countSelected[x1][v1 * x1 + x2][v2 * x2 + x3].resize(
								noValueX3 * x3);
						countUnSelected[x1][v1 * x1 + x2][v2 * x2 + x3].resize(
								noValueX3 * noUnSelectedCatAtts_);

						for (CatValue v3 = 0; v3 < noValueX3; v3++) {

							for (CategoricalAttribute x4 = 0; x4 < x3; x4++)
							{
								//inner vector
								countSelected[x1][v1 * x1 + x2][v2 * x2 + x3][v3
										* x3 + x4].assign(
												getNoValues(x4) * noClasses_, 0);
							}
							for (CategoricalAttribute x4 = 0;
									x4 < noUnSelectedCatAtts_; x4++)

								//inner vector
								countUnSelected[x1][v1 * x1 + x2][v2 * x2 + x3][v3
										* noUnSelectedCatAtts_ + x4].assign(
										getNoValues(x4 + noSelectedCatAtts_)
												* noClasses_, 0);

						}
					}
				}

			}
		}
	}
}

xxxxyDist3::~xxxxyDist3(void) {

}

void xxxxyDist3::update(const instance &i) {
	xxxyCounts.update(i);


//	for (CategoricalAttribute x1 = 0; x1 < noCatAtts_; x1++) {
//		printf("%u,",order_[x1]);
//					}
	CatValue theClass = i.getClass();

	for (CategoricalAttribute x1 = 2; x1 < noSelectedCatAtts_; x1++) {
		CatValue v1 = i.getCatVal(order_[x1]);

		for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {
			CatValue v2 = i.getCatVal(order_[x2]);

			XXYSubDist xxySubDist(getXXYSubDist(x1, v1,x2,v2), noClasses_);

			XXYSubDist xxySubDistRest(getXXYSubDistRest(x1, v1,x2,v2), noClasses_);

			for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {
				CatValue v3 = i.getCatVal(order_[x3]);

				XYSubDist xySubDist(xxySubDist.getXYSubDist(x3, v3), noClasses_);

				XYSubDist xySubDistRest(
						xxySubDistRest.getXYSubDist(x3, v3, noUnSelectedCatAtts_),
						noClasses_);


				for (CategoricalAttribute x4 = 0; x4 < x3; x4++) {
					CatValue v4 = i.getCatVal(order_[x4]);

					xySubDist.incCount(x4, v4, theClass);
					assert(
							*ref(x1,v1,x2,v2,x3,v3,x4,v4,theClass) <= xxxyCounts.xxyCounts.xyCounts.count);

				}


				for (CategoricalAttribute x4 = 0; x4 < noUnSelectedCatAtts_; x4++) {
					CatValue v4 = i.getCatVal(order_[x4 + noSelectedCatAtts_]);

					xySubDistRest.incCount(x4, v4, theClass);
					assert(
							*refRest(x1,v1,x2,v2,x3,v3,x4,v4,theClass) <= xxxyCounts.xxyCounts.xyCounts.count);

				}

			}
		}

	}
}

void xxxxyDist3::clear() {
//count.swap(std::vector<int>());
//std::vector<std::vector<std::vector<std::vector<InstanceCount> > > >( count.begin(), count.end() ).swap ( count );
	countSelected.clear();
	countUnSelected.clear();
	xxxyCounts.clear();
}
