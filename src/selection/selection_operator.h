/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

namespace FT{

    /*!
     * @class SelectionOperator
     */ 
    struct SelectionOperator 
    {
        /*!
         * @brief base class for selection operators.
         */

        bool survival; 

        //SelectionOperator(){}

        virtual ~SelectionOperator(){}
        
        virtual vector<size_t> select(const MatrixXd& F, const Parameters& p, Rnd& r) 
        {   
            std::cerr << "Undefined select() operation\n";
            throw;
        }
        virtual vector<size_t> select(Population& pop, const Parameters& p, Rnd& r)
        {
            std::cerr << "Undefined select() operation\n";
            throw;
        };

    };
	
	
	
	
}