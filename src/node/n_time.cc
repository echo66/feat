/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_time.h"
    
namespace FT{
	
    NodeTime::NodeTime()
    {
        name = "time";
	    otype = 'f';
	    arity['f'] = 0;
	    arity['b'] = 0;
	    arity['z'] = 1;
	    complexity = 1;
    }

    /// Evaluates the node and updates the stack states. 
    void NodeTime::evaluate(Data& data, Stacks& stack)
    {
        ArrayXd tmp(stack.z.top().first.size());
        
        int x;
        
        for(x = 0; x < stack.z.top().first.size(); x++)
            tmp(x) = limited(stack.z.top().first[x])[0];
            
        stack.z.pop();

        stack.f.push(tmp);
        
    }

    /// Evaluates the node symbolically
    void NodeTime::eval_eqn(Stacks& stack)
    {
        stack.fs.push("time(" + stack.zs.pop() + ")");
    }
    
    NodeTime* NodeTime::clone_impl() const { return new NodeTime(*this); }

    NodeTime* NodeTime::rnd_clone_impl() const { return new NodeTime(); }
}

