/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MODE
#define NODE_MODE

#include "node.h"

namespace FT{
	class NodeMode : public Node
    {
    	public:
    	
    		NodeMode();
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(Data& data, Stacks& stack);

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack);
            
        protected:
            NodeMode* clone_impl() const override;

            NodeMode* rnd_clone_impl() const override;
    };
}	

#endif
