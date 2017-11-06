#include "xxxxyDist.h"

xxxxyDist::xxxxyDist() {

}
xxxxyDist::xxxxyDist(InstanceStream& stream) {
	reset(stream);

	// pass through the stream updating the counts incrementally
	stream.rewind();

	instance i;

	while (stream.advance(i)) {
		update(i);
	}

}

void xxxxyDist::reset(InstanceStream& stream) {

	instanceStream_ = &stream;

	xxxyCounts.reset(stream);

	noCatAtts_ = instanceStream_->getNoCatAtts();
	noClasses_ = instanceStream_->getNoClasses();

	//out vector
	count.resize(noCatAtts_);
	for (CategoricalAttribute x1 = 3; x1 < noCatAtts_; x1++) {

		//second vector
		count[x1].resize(instanceStream_->getNoValues(x1) * x1);
		for (CatValue v1 = 0; v1 < instanceStream_->getNoValues(x1); v1++) {
			for (CategoricalAttribute x2 = 2; x2 < x1; x2++) {

				//third vector
				count[x1][v1 * x1 + x2].resize(
						instanceStream_->getNoValues(x2) * x2);
				for (CatValue v2 = 0; v2 < instanceStream_->getNoValues(x2);
						v2++) {
					for (CategoricalAttribute x3 = 1; x3 < x2; x3++) {

						//fourth vector
						count[x1][v1 * x1 + x2][v2 * x2 + x3].resize(
								instanceStream_->getNoValues(x3) * x3);
						for (CatValue v3 = 0;
								v3 < instanceStream_->getNoValues(x3); v3++)
							for (CategoricalAttribute x4 = 0; x4 < x3; x4++)

								//inner vector
								count[x1][v1 * x1 + x2][v2 * x2 + x3][v3 * x3
										+ x4].assign(
										instanceStream_->getNoValues(x4)
												* noClasses_, 0);
					}
				}
			}
		}
	}
}

xxxxyDist::~xxxxyDist(void) {

}

void xxxxyDist::update(const instance &i) {
	xxxyCounts.update(i);

	CatValue theClass = i.getClass();

	for (CategoricalAttribute x1 = 3; x1 < noCatAtts_; x1++) {
		CatValue v1 = i.getCatVal(x1);

		for (CategoricalAttribute x2 = 2; x2 < x1; x2++) {
			CatValue v2 = i.getCatVal(x2);

			for (CategoricalAttribute x3 = 1; x3 < x2; x3++) {
				CatValue v3 = i.getCatVal(x3);

				for (CategoricalAttribute x4 = 0; x4 < x3; x4++) {
					CatValue v4 = i.getCatVal(x4);

					(*ref(x1, v1, x2, v2, x3, v3, x4, v4, theClass))++;
				}
			}
		}

	}
}

void xxxxyDist::clear() {
	//count.swap(std::vector<int>());
	//std::vector<std::vector<std::vector<std::vector<InstanceCount> > > >( count.begin(), count.end() ).swap ( count );
	count.clear();
	xxxyCounts.clear();
}
