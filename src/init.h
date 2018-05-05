/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef INIT_H
#define INIT_H

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
    #define omp_get_max_threads() 1
#endif

namespace FT{

    double NEAR_ZERO = 0.0000001;
    static double MAX_DBL = std::numeric_limits<double>::max();
    static double MIN_DBL = std::numeric_limits<double>::lowest();

}
#endif
