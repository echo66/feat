/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODEVECTOR_H
#define NODEVECTOR_H
#include <memory>
#include "node/node.h"

namespace FT{
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations

    /*!
     * @class NodeVector
     * @brief an extension of a vector of unique pointers to nodes 
     */
    struct NodeVector : public std::vector<std::unique_ptr<Node>> {
        
        NodeVector() = default;
        ~NodeVector() = default; 
        NodeVector(const NodeVector& other)
        {
            /* std::cout<<"in NodeVector(const NodeVector& other)\n"; */
            this->resize(0);
            for (const auto& p : other)
                this->push_back(p->clone());
        }
        NodeVector(NodeVector && other) = default;
        /* { */
        /*     std::cout<<"in NodeVector(NodeVector&& other)\n"; */
        /*     for (const auto& p : other) */
        /*         this->push_back(p->clone()); */
        /* } */
        NodeVector& operator=(NodeVector const& other)
        { 

            /* std::cout << "in NodeVector& operator=(NodeVector const& other)\n"; */
            this->resize(0);
            for (const auto& p : other)
                this->push_back(p->clone());
            return *this; 
        }        
        NodeVector& operator=(NodeVector && other) = default;
        
        /// returns vector of raw pointers to nodes in [start,end], or all if both are zero
        vector<Node*> get_data(int start=0,int end=0)
        {
            vector<Node*> v;
            if (end == 0)
                end = this->size();
            for (unsigned i = start; i<=end; ++i)
                v.push_back(this->at(i).get());

            return v;
        }

        /// returns indices of root nodes 
        vector<size_t> roots()
        {
            // find "root" nodes of program, where roots are final values that output 
            // something directly to the stack
            // assumes a program's subtrees to be contiguous
             
            vector<size_t> indices;     // returned root indices
            int total_arity = -1;       //end node is always a root
            for (size_t i = this->size(); i>0; --i)   // reverse loop thru program
            {    
                if (total_arity <= 0 ){ // root node
                    indices.push_back(i-1);
                    total_arity=0;
                }
                else
                    --total_arity;
               
                total_arity += this->at(i-1)->total_arity(); 
               
            }
           
            return indices; 
        }

        size_t subtree(size_t i, char otype='0') const 
        {

           /*!
            * finds index of the end of subtree in program with root i.
            
            * Input:
            
            *		i, root index of subtree
            
            * Output:
            
            *		last index in subtree, <= i
            
            * note that this function assumes a subtree's arguments to be contiguous in the program.
            */
           
           size_t tmp = i;
           assert(i>=0 && "attempting to grab subtree with index < 0");
                  
           if (this->at(i)->total_arity()==0)    // return this index if it is a terminal
               return i;
           
           std::map<char, unsigned int> arity = this->at(i)->arity;

           if (otype!='0')  // if we are recursing (otype!='0'), we need to find 
                            // where the nodes to recurse are.  
           {
               while (i>0 && this->at(i)->otype != otype) --i;    
               assert(this->at(i)->otype == otype && "invalid subtree arguments");
           }
                  
           for (unsigned int j = 0; j<arity['f']; ++j)  
               i = subtree(--i,'f');                   // recurse for floating arguments      
           size_t i2 = i;                              // index for second recursion
           for (unsigned int j = 0; j<arity['b']; ++j)
               i2 = subtree(--i2,'b');
           size_t i3 = i2;                 // recurse for boolean arguments
           for (unsigned int j = 0; j<arity['z']; ++j)
               i3 = subtree(--i3,'z'); 
           return std::min(i,i3);
        }
    }; //NodeVector
} // FT
#endif
