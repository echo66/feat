/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_FLOAT
#define NODE_FLOAT

#include "node.h"

namespace FT{
	class NodeFloat : public Node
    {
    	public:
    	
    		NodeFloat();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeFloat* clone_impl() const override;

            NodeFloat* rnd_clone_impl() const override;
    };
    
}	

#endif
