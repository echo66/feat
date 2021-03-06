/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "n_sqrt.h"

namespace FT{
    	
    NodeSqrt::NodeSqrt(vector<double> W0)
    {
        name = "sqrt";
	    otype = 'f';
	    arity['f'] = 1;
	    arity['b'] = 0;
	    complexity = 2;

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
    void NodeSqrt::evaluate(Data& data, Stacks& stack)
    {
        stack.f.push(sqrt(W[0]*stack.f.pop().abs()));
    }

    /// Evaluates the node symbolically
    void NodeSqrt::eval_eqn(Stacks& stack)
    {
        stack.fs.push("sqrt(|" + stack.fs.pop() + "|)");
    }

    ArrayXd NodeSqrt::getDerivative(Trace& stack, int loc) {
        switch (loc) {
            case 1: // d/dw0
                return stack.f[stack.f.size()-1] / (2 * sqrt(this->W[0] * stack.f[stack.f.size()-1]));
            case 0: // d/dx0
            default:
                return this->W[0] / (2 * sqrt(this->W[0] * stack.f[stack.f.size()-1]));
        } 
    }

    NodeSqrt* NodeSqrt::clone_impl() const { return new NodeSqrt(*this); }

    NodeSqrt* NodeSqrt::rnd_clone_impl() const { return new NodeSqrt(); }  
}
