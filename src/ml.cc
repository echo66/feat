/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "ml.h"

using namespace shogun;

namespace FT{
	
    ML::ML(const Parameters& params, bool norm)
    {
        /*!
         * use string to specify a desired ML algorithm from shogun.
         */
        
        ml_type = params.ml;
        prob_type = PT_REGRESSION;
        max_train_time=30; 
        normalize = norm;
        if (params.classification)
        { 
            if (params.n_classes==2)
                prob_type = PT_BINARY;
            else
                prob_type = PT_MULTICLASS;               
        }
    }
    
    void ML::init()
    {
        // set up ML based on type
        if (!ml_type.compare("LeastAngleRegression"))
            p_est = make_shared<sh::CLeastAngleRegression>();
        else if (!ml_type.compare("LinearRidgeRegression"))
            p_est = make_shared<sh::CLinearRidgeRegression>();
        else if (!ml_type.compare("RandomForest"))
        {
            p_est = make_shared<sh::CRandomForest>();
            dynamic_pointer_cast<sh::CRandomForest>(p_est)->
                                                       set_machine_problem_type(prob_type);
            dynamic_pointer_cast<sh::CRandomForest>(p_est)->set_num_bags(100);
                               
            if (prob_type != PT_REGRESSION)
            {
                auto CR = some<sh::CMajorityVote>();                        
                dynamic_pointer_cast<sh::CRandomForest>(p_est)->set_combination_rule(CR);
            }
            else
            {
                auto CR = some<sh::CMeanRule>();
                dynamic_pointer_cast<sh::CRandomForest>(p_est)->set_combination_rule(CR);
            }
            
        }
        else if (!ml_type.compare("CART"))
        {
            p_est = make_shared<sh::CMyCARTree>();
            dynamic_pointer_cast<sh::CMyCARTree>(p_est)->
                                                       set_machine_problem_type(prob_type);
            dynamic_pointer_cast<sh::CMyCARTree>(p_est)->
                                                       set_max_depth(6);                
        }
                       
        else if (!ml_type.compare("SVM"))
        {               
        	if(prob_type==PT_BINARY)
                p_est = make_shared<sh::CMyLibLinear>(sh::L2R_L2LOSS_SVC_DUAL);       
            else if (prob_type==PT_MULTICLASS){
                p_est = make_shared<CMyMulticlassLibLinear>();
                dynamic_pointer_cast<CMyMulticlassLibLinear>(p_est)->
                                                             set_prob_heuris(sh::OVA_NORM);

            }
            else                // SVR
            	p_est = make_shared<sh::CLibLinearRegression>(); 
            
        }
        else if (!ml_type.compare("LR"))
        {
            assert(prob_type!=PT_REGRESSION && "LR only works with classification.");
            if (prob_type == PT_BINARY){
        	    p_est = make_shared<sh::CMyLibLinear>(sh::L2R_LR);
                // setting parameters to match sklearn defaults
                dynamic_pointer_cast<sh::CMyLibLinear>(p_est)->set_compute_bias(false);
                dynamic_pointer_cast<sh::CMyLibLinear>(p_est)->set_epsilon(0.0001);
                /* dynamic_pointer_cast<sh::CMyLibLinear>(p_est)->set_C(1.0,1.0); */
                dynamic_pointer_cast<sh::CMyLibLinear>(p_est)->set_max_iterations(1000);
                //cout << "set ml type to CMyLibLinear\n";
            }
            else    // multiclass  
            {
                p_est = make_shared<sh::CMulticlassLogisticRegression>();
                dynamic_pointer_cast<sh::CMulticlassLogisticRegression>(p_est)->
                                                             set_prob_heuris(sh::OVA_NORM);
            }
	
        
        }
        else
            HANDLE_ERROR_NO_THROW("'" + ml_type + "' is not a valid ml choice\n");
        
        p_est->set_max_train_time(max_train_time);  // set maximum training time per model
    }

    ML::~ML(){}
    
    void ML::set_dtypes(const vector<char>& dtypes)
    {
        if (!ml_type.compare("CART") || !ml_type.compare("RandomForest"))
        {
            // set attribute types True if boolean, False if continuous/ordinal
            sh::SGVector<bool> dt(dtypes.size());
            for (unsigned i = 0; i< dtypes.size(); ++i)
                dt[i] = dtypes[i] == 'b';
            if (!ml_type.compare("CART"))
                dynamic_pointer_cast<sh::CMyCARTree>(p_est)->set_feature_types(dt);
            else if (!ml_type.compare("RandomForest"))
                dynamic_pointer_cast<sh::CRandomForest>(p_est)->set_feature_types(dt);
        }
    }
    
    vector<double> ML::get_weights()
    {    
        /*!
         * return weight vector from model.
         */
        vector<double> w;
        
        if (!ml_type.compare("LeastAngleRegression") || !ml_type.compare("LinearRidgeRegression")||
        	!ml_type.compare("SVM") || (!ml_type.compare("LR")))
        {
            if(prob_type == PT_MULTICLASS && ( !ml_type.compare("LR") || !ml_type.compare("SVM") ) ) 
            {
                /* cout << "in get_weights(), multiclass LR\n"; */
                vector<SGVector<double>> weights;

                if( !ml_type.compare("LR"))
                    weights = dynamic_pointer_cast<sh::CMulticlassLogisticRegression>(p_est)
                                                                                        ->get_w();
                else //SVM
                    weights = dynamic_pointer_cast<sh::CMyMulticlassLibLinear>(p_est)->get_w();
           
                /* cout << "set weights from get_w()\n"; */
                    
                /* for( int j = 0;j<weights.at(0).size(); j++) */ 
                /*     w.push_back(0); */
                w = vector<double>(weights[0].size());
                /* cout << "weights.size(): " << weights.size() << "\n"; */
                /* cout << "w size: " << w.size() << "\n"; */
                /* cout << "getting abs weights\n"; */
                
                for( int i = 0 ; i < weights.size(); ++i )
                {
                    /* cout << "weights:\n"; */
                    /* weights.at(i).display_vector(); */

                    for( int j = 0;j<weights.at(i).size(); ++j) 
                    {
                        w.at(j) += fabs(weights.at(i)[j]);
                        w.at(j) += weights.at(i)[j];
                    }
                }
                /* cout << "normalizing weights\n"; */ 
                for( int i = 0; i < w.size() ; i++) 
                    w[i] = w[i]/weights.size(); 
                
                /* cout << "get_weights(): w: " << w.size() << ":"; */
                /* for (auto tmp : w) cout << tmp << " " ; */
                /* cout << "\n"; */                 
                /* cout << "returning weights\n"; */
                /* cout << "freeing SGVector weights\n"; */
                /* weights.clear(); */
                /* for (unsigned i =0; i<weights.size(); ++i) */
                /*     weights[i].unref(); */

	            return w;		
            }
	        
            auto tmp = dynamic_pointer_cast<sh::CLinearMachine>(p_est)->get_w();
            
            w.assign(tmp.data(), tmp.data()+tmp.size());          
            /* for (unsigned i =0; i<w.size(); ++i)    // take absolute value of weights */
            /*     w[i] = fabs(w[i]); */
	    }
        else if (!ml_type.compare("CART"))           
            w = dynamic_pointer_cast<sh::CMyCARTree>(p_est)->feature_importances();
        else
            HANDLE_ERROR_NO_THROW("ERROR: ML::get_weights not implemented for " + ml_type + "\n");
            
        return w;
    }

    shared_ptr<CLabels> ML::fit(MatrixXd& X, VectorXd& y, const Parameters& params, bool& pass,
                     const vector<char>& dtypes)
    { 
    	/*!
         * Trains ml on X, y to generate output yhat = f(X). 
         *
         *  Input: 
         
         *       X: n_features x n_samples matrix
         *       y: n_samples vector of training labels
         *       params: feat parameters
         *       ml: the ML model to be trained on X
         
         *  Output:
         
         *       yhat: n_samples vector of outputs
        */ 
        
                // for random forest we need to set the number of features per bag
        init();
        if (!ml_type.compare("RandomForest"))
        {
            //std::cout << "setting max_feates\n";
            // set max features to sqrt(n_features)
            int max_feats = std::sqrt(X.rows());
            dynamic_pointer_cast<sh::CRandomForest>(p_est)->set_num_random_features(max_feats);
        }
        // for tree-based methods we need to specify data types 
        if (!ml_type.compare("RandomForest") || !ml_type.compare("CART"))
        {            
            //std::cout << "setting dtypes\n";
            if (dtypes.empty())
                set_dtypes(params.dtypes);
            else
                set_dtypes(dtypes);
        }
       
        if (normalize)
        {
            if (dtypes.empty())
                N.fit_normalize(X, params.dtypes);  
            else 
                N.fit_normalize(X, dtypes);
        }
        /* else */
            /* cout << "normlize is false\n"; */

        auto features = some<CDenseFeatures<float64_t>>(SGMatrix<float64_t>(X));
        /* cout << "Phi:\n"; */
        /* for (int i = 0; i < 10 ; ++i) */
        /* { */
        /*     cout << X.col(i) << (i < 10 ? " " : "\n"); */ 
        /* } */
        //std::cout << "setting labels (n_classes = " << params.n_classes << ")\n"; 
        /* cout << "y is " << y.transpose() << "\n"; */
         
        if(prob_type==PT_BINARY && 
                (!ml_type.compare("LR") || !ml_type.compare("SVM")))  // binary classification           	
        {
            p_est->set_labels(some<CBinaryLabels>(SGVector<float64_t>(y), 0.5));       	
            
        }
        else if (prob_type!=PT_REGRESSION)                         // multiclass classification       
        {
            p_est->set_labels(some<CMulticlassLabels>(SGVector<float64_t>(y)));
            /* auto labels_train = (CMulticlassLabels *)p_est->get_labels(); */
            /* SGVector<double> labs = labels_train->get_unique_labels(); */
            /* std::cout << "unique labels: \n"; */ 
            /* for (int i = 0; i < labs.size(); ++i) std::cout << labs[i] << " " ; std::cout << "\n"; */

            /* int nclasses = labels_train->get_num_classes(); */
            /* std::cout << "nclasses: " << nclasses << "\n"; */
        }
        else                                                    // regression
            p_est->set_labels(some<CRegressionLabels>(SGVector<float64_t>(y)));
        
        // train ml
        params.msg("ML training on thread" + std::to_string(omp_get_thread_num()) + "...",3," ");       
        // *** Train the model ***  
        p_est->train(features);
        // *** Train the model ***
        params.msg("done.",3);
       
        //get output
        SGVector<double> y_pred; 
        shared_ptr<CLabels> labels;

        if (prob_type==PT_BINARY && 
             (!ml_type.compare("LR") || !ml_type.compare("SVM")))     // binary classification
        {
            bool proba = params.scorer.compare("log")==0;

            labels = shared_ptr<CLabels>(p_est->apply_binary(features));
            
            if (proba)
                dynamic_pointer_cast<sh::CMyLibLinear>(p_est)->set_probabilities(labels.get(), 
                                                                                 features);

            y_pred = dynamic_pointer_cast<sh::CBinaryLabels>(labels)->get_labels();
			/* cout << "y_pred: "; */
			/* y_pred.display_vector(); */
			
        	/* SGVector<double> tmp = dynamic_pointer_cast<sh::CBinaryLabels>(labels)->get_values(); */
			/* cout << "y_proba:\n"; */
			/* tmp.display_vector(); */
           
        }
        else if (params.classification)                         // multiclass classification
        {
            labels = shared_ptr<CLabels>(p_est->apply_multiclass(features));
            y_pred = dynamic_pointer_cast<sh::CMulticlassLabels>(labels)->get_labels();
        }
        else                                                    // regression
        {
            labels = shared_ptr<CLabels>(p_est->apply_regression(features));
            y_pred = dynamic_pointer_cast<sh::CRegressionLabels>(labels)->get_labels();
            /* delete reg; */
        }
        //y_pred.display_vector();
        // map to Eigen vector
        Map<VectorXd> yhat(y_pred.data(),y_pred.size());
       
        /* std::cout << "yhat: " << yhat.transpose() << "\n"; */ 

        if (isinf(yhat.array()).any() || isnan(yhat.array()).any() || yhat.size()==0)
        {
            pass = false;
        }
        //std::cout << "Returning from fit() from the ml.h" << std::endl;
        return labels;
    }

    VectorXd ML::fit_vector(MatrixXd& X, VectorXd& y, const Parameters& params, bool& pass,
                     const vector<char>& dtypes)
    {
        shared_ptr<CLabels> labels = fit(X, y, params, pass, dtypes); 
        
        return labels_to_vector(labels);     
    }

    shared_ptr<CLabels> ML::predict(MatrixXd& X)
    {

        if (normalize)
            N.normalize(X);
        auto features = some<CDenseFeatures<float64_t>>(SGMatrix<float64_t>(X));
        
        shared_ptr<CLabels> labels;
       
        if (prob_type==PT_BINARY && 
                (!ml_type.compare("SVM") || !ml_type.compare("LR"))){
            labels = std::shared_ptr<CLabels>(p_est->apply_binary(features));
            dynamic_pointer_cast<sh::CMyLibLinear>(p_est)->set_probabilities(labels.get(), 
                                                                             features);
        }
        else if (prob_type != PT_REGRESSION)
            labels = std::shared_ptr<CLabels>(p_est->apply_multiclass(features));
        else
            labels = std::shared_ptr<CLabels>(p_est->apply_regression(features));
        
        return labels ;
    }

    VectorXd ML::predict_vector(MatrixXd& X)
    {
        shared_ptr<CLabels> labels = predict(X);
        return labels_to_vector(labels);     
        
    }

    ArrayXXd ML::predict_proba(MatrixXd& X)
    {
        shared_ptr<CLabels> labels = shared_ptr<CLabels>(predict(X));
           
        if (prob_type==PT_BINARY && 
                (!ml_type.compare("SVM") || !ml_type.compare("LR")))
        {
            shared_ptr<CBinaryLabels> BLabels = dynamic_pointer_cast<CBinaryLabels>(labels);
            /* BLabels->scores_to_probabilities(); */
            SGVector<double> tmp= BLabels->get_values();
            ArrayXXd confidences(1,tmp.size());
            confidences.row(0) = Map<ArrayXd>(tmp.data(),tmp.size()); 
            return confidences;
        }
        else if (prob_type == PT_MULTICLASS)
        {
            shared_ptr<CMulticlassLabels> MLabels = dynamic_pointer_cast<CMulticlassLabels>(labels);
            MatrixXd confidences(MLabels->get_num_classes(), MLabels->get_num_labels()) ; 
            for (unsigned i =0; i<confidences.rows(); ++i)
            {
                SGVector<double> tmp = MLabels->get_multiclass_confidences(int(i));
                confidences.row(i) = Map<ArrayXd>(tmp.data(),tmp.size());
                /* std::cout << confidences.row(i) << "\n"; */
            }
            return confidences;
        }
        else
            HANDLE_ERROR_THROW("Error: predict_proba not defined for problem type or ML method");
    }

    VectorXd ML::labels_to_vector(shared_ptr<CLabels>& labels)
    {
        SGVector<double> y_pred;
        if (prob_type==PT_BINARY && 
                (!ml_type.compare("SVM") || !ml_type.compare("LR")))
            y_pred = dynamic_pointer_cast<sh::CBinaryLabels>(labels)->get_labels();
        else if (prob_type != PT_REGRESSION)
            y_pred = dynamic_pointer_cast<sh::CMulticlassLabels>(labels)->get_labels();
        else
            y_pred = dynamic_pointer_cast<sh::CRegressionLabels>(labels)->get_labels();
       
        Map<VectorXd> yhat(y_pred.data(),y_pred.size());
        
        if (prob_type==PT_BINARY && (!ml_type.compare("LR") || !ml_type.compare("SVM")))
            // convert -1 to 0
            yhat = (yhat.cast<int>().array() == -1).select(0,yhat);

        return yhat;
    }
}
