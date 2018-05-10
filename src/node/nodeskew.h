/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SKEW
#define NODE_SKEW

#include "node.h"

namespace FT{
	class NodeSkew : public Node
    {
    	public:
    	
    		NodeSkew()
            {
                name = "skew";
    			otype = 'f';
    			arity['f'] = 0;
    			arity['b'] = 0;
    			arity['z'] = 1;
    			complexity = 3;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
                ArrayXd tmp(stack.z.top().first.size());
                
                int x;
                
                for(x = 0; x < stack.z.top().first.size(); x++)
                    tmp(x) = skew(limited(stack.z.top().first[x]));
                    
                stack.z.pop();

                stack.f.push(tmp);
                
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("skew(" + stack.zs.pop() + ")");
            }
        protected:
            NodeSkew* clone_impl() const override { return new NodeSkew(*this); }; 
    };
}	

#endif
