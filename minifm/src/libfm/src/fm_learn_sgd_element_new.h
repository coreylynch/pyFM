/*
	Stochastic Gradient Descent based learning for classification and regression

	Based on the publication(s):
	Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th IEEE International Conference on Data Mining (ICDM 2010), Sydney, Australia.

	Author:   Steffen Rendle, http://www.libfm.org/
	modified: 2012-01-04

	Copyright 2010-2012 Steffen Rendle, see license.txt for more information
*/

#ifndef FM_LEARN_SGD_ELEMENT_NEW_H_
#define FM_LEARN_SGD_ELEMENT_NEW_H_

class fm_learn_sgd_element_new{
	protected:
		//fm_learn_sgd stuff
		DVector<double> sum, sum_sqr;
	public:

		//fm_learn stuff
		fm_model* fm;
		double min_target;
		double max_target;
		int task;
		const static int TASK_REGRESSION = 0;
		const static int TASK_CLASSIFICATION = 1;
		
		//fm_learn_sgd stuff
		int num_iter;
		double learn_rate;

		void init() {
			//fm_learn_init is only logging

			//fm_learn_sgd_stuff
			sum.setSize(fm->num_factor);
			sum_sqr.setSize(fm->num_factor);

		}


		void learn(Data& train) {
			//std::cout << "SGD: DON'T FORGET TO SHUFFLE THE ROWS IN TRAINING DATA TO GET THE BEST RESULTS." << std::endl; 
			// SGD
			for (int i = 0; i < num_iter; i++) {
			
				for (train.data->begin(); !train.data->end(); train.data->next()) {
					
					double p = fm->predict(train.data->getRow(), sum, sum_sqr);
					double mult = 0;
					if (task == 0) {
						p = std::min(max_target, p);
						p = std::max(min_target, p);
						mult = -(train.target(train.data->getRowIndex())-p);
					} else if (task == 1) {
						mult = -train.target(train.data->getRowIndex())*(1.0-1.0/(1.0+exp(-train.target(train.data->getRowIndex())*p)));
					}				
					fm_SGD(fm, learn_rate, train.data->getRow(), mult, sum);					
				}				
			}		
		}

		double predict_case(Data& data) {
			return fm->predict(data.data->getRow());
		}

		void predict(Data& data, DVector<double>& out) {
			assert(data.data->getNumRows() == out.dim);
			for (data.data->begin(); !data.data->end(); data.data->next()) {
				double p = predict_case(data);
				if (task == TASK_REGRESSION ) {
					p = std::min(max_target, p);
					p = std::max(min_target, p);
				} else if (task == TASK_CLASSIFICATION) {
				p = 1.0/(1.0 + exp(-p));
				} else {
					throw "task not supported";
				}

				out(data.data->getRowIndex()) = p;				
		}
	}
};

#endif /*FM_LEARN_SGD_ELEMENT_H_*/
