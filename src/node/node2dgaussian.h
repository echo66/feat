/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_2DGAUSSIAN
#define NODE_2DGAUSSIAN

#include <numeric>

#include "nodeDx.h"

namespace FT{
	class Node2dGaussian : public NodeDx
    {
    	public:
    	
    		Node2dGaussian(vector<double> W0 = vector<double>())
            {
                name = "gaussian2d";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 4;

                if (W0.empty())
                {
                    for (int i = 0; i < arity['f']; i++) {
                        W.push_back(r.rnd_dbl());
                    }
                }
                else
                    W = W0;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
        		string x1 = stack.fs.pop();
                string x2 = stack.fs.pop();
                stack.fs.push("gauss2d(" + x1 + "," + x2 + ")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                // TODO 
                
                switch (loc) {
                    case 1: // d/dw0
                        return -2 * W[0] * pow(stack_f[stack_f.size() - 1], 2) * exp(-pow(W[0] * stack_f[stack_f.size() - 1], 2));
                    case 0: // d/dx0
                    default:
                        return -2 * pow(W[0], 2) * stack_f[stack_f.size() - 1] * exp(-pow(W[0] * stack_f[stack_f.size() - 1], 2));
                } 
            }
            
        protected:
                Node2dGaussian* clone_impl() const override { return new Node2dGaussian(*this); };  
    };
#ifndef USE_CUDA
void Node2dGaussian::evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
    {
        ArrayXd x1 = stack.f.pop();
        ArrayXd x2 = stack.f.pop();
        
        stack.f.push(limited(exp(-1*(pow(W[0]*(x1-x1.mean()), 2)/(2*variance(x1)) 
                          + pow(W[1]*(x2 - x2.mean()), 2)/variance(x2)))));
    }
#endif
}	

#endif
