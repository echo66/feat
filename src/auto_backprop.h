#ifndef AUTO_BACKPROP_H
#define AUTO_BACKPROP_H

#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <vector>
#include "nodevector.h"
#include "stack.h"
#include "node/node.h"
#include "node/nodeDx.h"
#include "nodevector.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;

/*
NOTE Currently the algorithm doesn't seem to correctly computing weights past the first iteration. Noted a discrepancy in the cost function for the second iteration even though the weights were the same (this was holding the weight of the node to 1)

Potential Optimization: 
Check for non differentiable root at start
Remove from consideration non-differentiable parts of tree
*/

namespace FT {
	class AutoBackProp 
    {
        /* @class AutoBackProp
         * @brief performs back propagation on programs to adapt weights.
         */
	public:
	
        typedef VectorXd (*callback)(const VectorXd&, const VectorXd&);
        
        AutoBackProp(callback d_cost_func, int iters=1000, double n=0.1, int store_optimal=0, callback cost_func=NULL) 
        {
			this->d_cost_func = d_cost_func;
			this->iters = iters;
			this->n = n;
			this->STORE = store_optimal;
			this->cost_func = cost_func;
		}
        /// adapt weights
		void run(NodeVector& program, MatrixXd& X, VectorXd& y,
                 std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z);

        /* ~AutoBackProp() */
        /* { */
        /*     /1* for (const auto& p: program) *1/ */
        /*         /1* p = nullptr; *1/ */
        /* } */


	private:
		double n;                   //< learning rate
		callback d_cost_func;       //< derivative of cost function pointer
        callback cost_func;         //< cost function pointer
        int iters;                  //< iterations
        int STORE;					//< flag for storing best weights

        struct WEIGHT_BUNDLE {
        	vector<vector<double>> weights;
        	double cost;
        };

        WEIGHT_BUNDLE BEST_W;

		struct BP_NODE
		{
			NodeDx* n;
			vector<ArrayXd> deriv_list;
		};

		void print_weights(NodeVector& program) {
			for (const auto& p : program) {
				if (isNodeDx(p)) {
					NodeDx* dNode = dynamic_cast<NodeDx*>(p.get());
                    dNode->print_weight();
                    dNode = nullptr;
				} else {
					cout << "Node: " << p->name;
				}
				cout << "\n";
			}
			std::cout << "*******************************\n";
		}

		/// Return the f_stack
		vector<ArrayXd> forward_prop(NodeVector& program, int start, int end, 
                                     MatrixXd& X, VectorXd& y, 
                               std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z);

		/// Updates stacks to have proper value on top
		void next_branch(vector<BP_NODE>& executing, vector<Node*>& bp_program, 
                         vector<ArrayXd>& derivatives);

		/// Removes the substree associated with a non-differentiable node from the tree
		void cleanNonDif(vector<Node*>& bp_program);

        /// Compute gradients and update weights 
		void backprop(vector<ArrayXd>& f_stack, NodeVector& program, int start, int end,
                      MatrixXd& X, VectorXd& y, 
                               std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z);

		// Updates the current "best program" found
		void update_best(double cost, NodeVector& program, int start, int end);

        /// check if differentiable node    
        bool isNodeDx(Node* n){ return NULL != dynamic_cast<NodeDx*>(n); }

		bool isNodeDx(const std::unique_ptr<Node>& n) {
            Node * tmp = n.get();
			bool answer = isNodeDx(tmp); 
            tmp = nullptr;
            return answer;
		}

		template <class T>
		T pop(vector<T>* v) {
			T value = v->back();
			v->pop_back();
			return value;
		}

		template <class T>
		T pop_front(vector<T>* v) {
			T value = v->front();
			v->erase(v->begin());
			return value;
		}
	};

/////////////////////////////////////////////////////////////////////////////////////// Definitions
    // adapt weights 
    void AutoBackProp::run(NodeVector& program, MatrixXd& X, VectorXd& y, 
                            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z)
    {
    	/*!
        * Finds weights for an input program that increase the performance of the model
            
        * Input:
          
        *		program, a NodeVector of the nodes that will be evaluated
        *		X, the input data to train on
        *		y, the labels to use when evaluating error
        *		Z, TODO
            
        * Output:
           
        *		The method operates in place, and thus the "output" is the input program with updated weights
          
        * note that the function only updates weights for differentable nodes and will completelt skip subtrees of the program if the root node is not differentiable
        */
        // Initialialize struct for tracking best weights if using that option
        if (this->STORE) {
        	this->BEST_W.weights = vector<vector<double>>(program.size());
        	for (int i = 0; i < program.size(); i++) {
        		Node* n = program[i].get();
	    		if (isNodeDx(n)) {
	    			NodeDx* dNode = dynamic_cast<NodeDx*>(n);
	    			this->BEST_W.weights[i] = vector<double>(dNode->arity['f']);
	    			for (int j = 0; j < dNode->arity['f']; j++)
	    				this->BEST_W.weights[i][j] = dNode->W[j];
	    		}
        	}
        	this->BEST_W.cost = 0;
        }

        cout << "Starting up AutoBackProp with " << this->iters << " iterations.";
        // Computes weights via backprop
        // grab subtrees to backprop over
        for (int s : program.roots())
        {
            if (isNodeDx(program[s]))
            {
            	// Could save on computation by storing the output of program.substree()
                cout << "\ntraining sub-program " << program.subtree(s) << " to " << s << "\n";
                cout << "\nIteration\tLoss\tGrad\t\n";
                for (int x = 0; x < this->iters; x++) {
                    // Evaluate forward pass 
                    vector<ArrayXd> stack_f = forward_prop(program, program.subtree(s), s, X, y, Z);
                    ArrayXd output = stack_f[stack_f.size() - 1];
                    // if ((x % round(this->iters/4)) == 0 or x == this->iters - 1) {
                    // }
                    cout << x << "\t" 
                         << (y.array()-stack_f[stack_f.size() - 1]).array().pow(2).mean() << "\t"
                         << this->d_cost_func(y, stack_f[stack_f.size() - 1]).mean() << "\n"; 
                    
                    // Evaluate backward pass
                    backprop(stack_f, program, program.subtree(s), s, X, y, Z);  
                }
            }
        }
        cout << "Finished backprop. returning program:\n";
        print_weights(program);    
        /* return this->program; */
    }
    
    // Return the f_stack
    vector<ArrayXd> AutoBackProp::forward_prop(NodeVector& program, int start, int end, 
                                                MatrixXd& X, VectorXd& y, 
                               std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z) 
    {
        /*!
        * Computes and stores the results of each node in the program in a manner convenient for backprop
            
        * Input:
          
        *		program, a NodeVector of the nodes that will be evaluated
        *		start, the index of the first node to consdier
        *		end, the index of the last node to consider
        *		X, the input data to train on
        *		y, the labels to use when evaluating error
        *		Z, TODO
            
        * Output:
           
        *		stack_f, an ordered list of output values
          
        * note that the returned list groups inputs so that the tree structure of the program can be avoided in the backprop step.
        * The returned list groups values in the form [[input1, input2], [input1, input2], ...] where each [...] represents the inputs needed to evaluate a node
        * For each node there will be arity elements on the returned stack
        * The top element of the stack should correspond to the input to the cost function
        */

        // Iterate through all the nodes evaluating and tracking inputs and corresponding ouputs
        vector<ArrayXd> stack_f; // Tracks output values
        FT::Stacks stack;

        // Use stack_f and execution stack to avoid issue of branches affecting what elements 
        // appear before a node 
        for (int s = start; s <= end; ++s) 
        {
        	// Place the inputs on the stack to return
            for (int i = 0; i < program[s]->arity['f']; i++) {
                stack_f.push_back(stack.f.at(stack.f.size() - (program[s]->arity['f'] - i)));
            }

            program[s]->evaluate(X, y, Z, stack);
            program[s]->visits = 0;
        }
        stack_f.push_back(stack.f.pop());

        if (this->STORE) {
        	double cost = this->cost_func(y, stack_f[stack_f.size() - 1]).sum();
        	if (this->BEST_W.cost < cost) {
        		update_best(cost, program, start, end);
        	}
        	
        }

        return stack_f;
    }
    
    // Updates stacks to have proper value on top
    void AutoBackProp::next_branch(vector<BP_NODE>& executing, vector<Node*>& bp_program, 
                                   vector<ArrayXd>& derivatives) 
    {
    	/*!
        * Find the next node to continue computation from after reaching the bottom of a program branch
            
        * Input:
          
        *		executing, the list of nodes that have unexplored branches grouped with their corresponding derivatives
        *		bp_program, the list of nodes that will be evaluated by backprop
        *		derivatives, the current list of derivatives contributing to the chain rule used in backprop
            
        * Output:
           
        *		The function operates in place on the three input arguments, returning them in a state that allows backprop to continue running
        */

        // While there are still nodes with branches to explore
        if(!executing.empty()) {
            // Declare variable to hold node and its associated derivatives
            BP_NODE bp_node = pop<BP_NODE>(&executing); // Check first element
            // Loop until branch to explore is found
            while (bp_node.deriv_list.empty() && !executing.empty()) {
                bp_node = pop<BP_NODE>(&executing); // Get node and its derivatves

                pop<ArrayXd>(&derivatives); // Remove associated gradients from stack
                if (executing.empty()) {
                    return;
                }
            }
            
            // Should now have the next parent node and derivatves (stored in bp_node)
            if (!bp_node.deriv_list.empty()) 
            {
                bp_program.push_back(bp_node.n);
                // Pull derivative from front of list due to how we stored them earlier
                derivatives.push_back(pop_front<ArrayXd>(&(bp_node.deriv_list)));                 
                // Push it back on the stack in order to sync all the stacks
                executing.push_back(bp_node);             
            }
        }
    }

    void AutoBackProp::cleanNonDif(vector<Node*>& bp_program) 
    {
    	/*!
        * Removes program tree rooted by current node at top of list
            
        * Input:
          
        *		bp_program, list of nodes currently being consdiered by backprop with a non-differentiable node at the head of the stack
            
        * Output:
           
        *		The method operates in place, and thus the "output" is the input program with updated structure so that the tree rooted at the non-differentiable node is removed
          
        * note that the current version of the method fails to account for input nodes that are not specified in the 'f' arity. 
        */

		// Starting from node, remove arguments from the queue that would have been in the non-differentiable branch
		// While there are still nodes with branches to explore
		vector<Node*> executing;
		while (bp_program.size() > 0) {
			Node* node = pop<Node*>(&bp_program);
			if (node->arity['f'] == 0) { // Might need to add code to check for bool arity
				if (executing.empty())
					return;
				bp_program.push_back(pop<Node*>(&executing));
			} else {
				node->visits++;
				if (node->visits > node->arity['f']) {
					if (executing.empty())
						return;
					bp_program.push_back(pop<Node*>(&executing));
				} else {
					executing.push_back(node);
				}
			}
		}
	}

    // Compute gradients and update weights 
    void AutoBackProp::backprop(vector<ArrayXd>& f_stack, NodeVector& program, int start, int end, 
                                MatrixXd& X, VectorXd& y, 
                                std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >& Z)    
    {
    	/*!
        * Iterates over the nodes in the program, updating their values in accordance with the results of the forward pass specified by f_stack
            
        * Input:
          
        *		f_stack, the list of values that resulted from the forward pass
        *		program, the list of nodes describing the program
        *		start, the index of the first node to consider
        *		end, the index of the last node to consider
        *		X, input data used for training
        *		y, labels which the program will try to converge to
        *		Z, TODO
            
        * Output:
           
        *		The method operates in place, and thus the "output" is the input program with updated weights
          
        * note that the function only updates weights for differentable nodes and will completelt skip subtrees of the program if the root node is not differentiable
        */

        vector<ArrayXd> derivatives;
        derivatives.push_back(this->d_cost_func(y, f_stack[f_stack.size() - 1])); 
        // Might need a cost function node (still need to address this)
        // Working according to test program 
        pop<ArrayXd>(&f_stack); // Get rid of input to cost function
        vector<BP_NODE> executing; // Stores node and its associated derivatves
        // Currently I don't think updates will be saved, might want a pointer of nodes so don't 
        // have to restock the list
        // Program we loop through and edit during algorithm (is this a shallow or deep copy?)
        vector<Node*> bp_program = program.get_data(start, end);         
        while (bp_program.size() > 0) {
            Node* node = pop<Node*>(&bp_program);
            vector<ArrayXd> n_derivatives;

            if (isNodeDx(node) && node->visits == 0 && node->arity['f'] > 0) {
                NodeDx* dNode = dynamic_cast<NodeDx*>(node); // Could probably put this up one and have the if condition check if null
                // Calculate all the derivatives and store them, then update all the weights and throw away the node
                for (int i = 0; i < node->arity['f']; i++) {
                    dNode->derivative(n_derivatives, f_stack, i);
                }
                dNode->update(derivatives, f_stack, this->n);

                // Get rid of the input arguments for the node
                for (int i = 0; i < dNode->arity['f']; i++) {
                    pop<ArrayXd>(&f_stack);
                }

                if (!n_derivatives.empty()) {
                    derivatives.push_back(pop_front<ArrayXd>(&n_derivatives));
                }

                executing.push_back({dNode, n_derivatives});
            }

            // Choosing how to move through tree
			if (node->arity['f'] == 0) {
				// Clean up gradients and find the parent node
				pop<ArrayXd>(&derivatives);
				next_branch(executing, bp_program, derivatives);
			} else if (!isNodeDx(node)) {
				// Have to remove the elements associated with the non differentiable nodes from the stack
				cleanNonDif(bp_program);
				next_branch(executing, bp_program, derivatives);
			} else {
				node->visits += 1;
				if (node->visits > node->arity['f']) {
					next_branch(executing, bp_program, derivatives);
				}
			}
        }

        // point bp_program to null
        for (unsigned i = 0; i < bp_program.size(); ++i)
            bp_program[i] = nullptr;
    }

    void AutoBackProp::update_best(double cost, NodeVector& program, int start, int end) {
    	if (cost > this->BEST_W.cost)
    		return;

    	this->BEST_W.cost = cost;

    	for (int i = 0; i < end - start; i++) {
    		Node* n = program[start + i].get();
    		if (isNodeDx(n)) {
    			NodeDx* dNode = dynamic_cast<NodeDx*>(n);
    			for (int j = 0; j < dNode->arity['f']; j++)
    				this->BEST_W.weights[start + i][j] = dNode->W[j];
    		}
    		
    	}
    	
    }
}

#endif
