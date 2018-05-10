/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LOG
#define NODE_LOG

#include "nodeDx.h"

namespace FT{
	class NodeLog : public NodeDx
    {
    	public:
    	
    		NodeLog(vector<double> W0 = vector<double>())
       		{
    			name = "log";
    			otype = 'f';
    			arity['f'] = 1;
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

            /// Safe log: pushes log(abs(x)) or MIN_DBL if x is near zero. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
			{
           		ArrayXd x = stack.f.pop();
                stack.f.push( (abs(x) > NEAR_ZERO).select(log(abs(W[0] * x)),MIN_DBL));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("log(" + stack.fs.pop() + ")");
            }

            ArrayXd getDerivative(vector<ArrayXd>& stack_f, int loc) {
                switch (loc) {
                    case 1: // d/dw0
                        return limited(1/(W[0] * ArrayXd::Ones(stack_f[stack_f.size()-1].size())));
                    case 0: // d/dx0
                    default:
                        return limited(1/stack_f[stack_f.size()-1]);
                } 
            }
        protected:
            NodeLog* clone_impl() const override { return new NodeLog(*this); };  
    };
}	

#endif
