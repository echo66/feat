/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_TIME
#define NODE_TIME

#include "node.h"

namespace FT{
	class NodeTime : public Node
    {
    	public:
    	
    		NodeTime();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeTime* clone_impl() const override; 
            NodeTime* rnd_clone_impl() const override; 
    };
}	

#endif
