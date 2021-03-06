/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef RND_H
#define RND_H
//external includes
#include <random>
#include <limits>
#include <vector>

#include "init.h"

using namespace std;
using std::swap;

namespace FT {
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    
    /*!
     * @class Rnd
     * @brief Defines a multi-core random number generator and its operators.
     */
    // forward declaration of Node class
    class Node;

    class Rnd
    {
        public:
            
            Rnd();

            void set_seed(int seed);

            
            int rnd_int( int lowerLimit, int upperLimit );

            float rnd_flt(float min=0.0, float max=1.0);

            double rnd_dbl(double min=0.0, double max=1.0);
            
            double operator()(unsigned i);
            
            float operator()();

			template <class RandomAccessIterator>
			void shuffle (RandomAccessIterator first, RandomAccessIterator last)
			{
	            for (auto i=(last-first)-1; i>0; --i) 
                {
	                std::uniform_int_distribution<decltype(i)> d(0,i);
		            swap (first[i], first[d(rg[omp_get_thread_num()])]);
	            }
	        }    
            
            template<typename Iter>                                    
            Iter select_randomly(Iter start, Iter end)
            {
                std::uniform_int_distribution<> dis(0, distance(start, end) - 1);
                advance(start, dis(rg[omp_get_thread_num()]));
                return start;
            }
           
            template<typename T>
            T random_choice(const vector<T>& v)
            {
               /*!
                * return a random element of a vector.
                */          
                assert(v.size()>0 && " attemping to return random choice from empty vector");
                return *select_randomly(v.begin(),v.end());
            }
 
           
            template<typename T, typename D>
            T random_choice(const vector<T>& v, const vector<D>& w )
            {
                /*!
                 * return a weighted random element of a vector
                 */
                 
                if(w.size() == 0)
                {   
                    cout<<"random_choice() w.size() = 0 Calling random_choice(v)\n";
                    return random_choice(v);
                }
                if(w.size() != v.size())
                {   
                    cout<<"WARN! random_choice() w.size() " << w.size() << "!= v.size() " 
                        << v.size() << ", Calling random_choice(v)\n";
                    return random_choice(v);
                }
                else
                {
                    assert(v.size() == w.size());
                    std::discrete_distribution<size_t> dis(w.begin(), w.end());

                    return v[dis(rg[omp_get_thread_num()])]; 
                }
            }
            
            float gasdev();
            
            ~Rnd();

        private:
            vector<std::mt19937> rg;
     
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
    static Rnd r;   // random number generator     
}
#endif
