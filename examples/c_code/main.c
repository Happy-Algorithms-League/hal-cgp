#include "individual.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


double target(const double x_0, const double x_1) {
    return x_0 * x_1 + 1.0;
}

/* generate a random floating point number from min to max */
double rand_from_to(double min, double max)
{
    const double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}


double loss() {
    int n_samples = 100;
    srand(1234); // fix seed

    double sum_l2_difference = 0.0;

    const double min = -1.0;
    const double max = 1.0;

    for(int i=0;i<n_samples;i++){
        /* generate two random values for x_0, x_1 */
        const double x_0_rand=rand_from_to(min, max);
        const double x_1_rand=rand_from_to(min, max);

        const double target_value=target(x_0_rand, x_1_rand);
        const double rule_output=rule(x_0_rand, x_1_rand);
        
        sum_l2_difference += pow(target_value-rule_output, 2);
    }
    return sum_l2_difference/(double)n_samples;
}

int main(){
    printf("%f", loss());
    return 0;
}
