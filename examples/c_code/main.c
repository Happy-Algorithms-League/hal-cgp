#include "individual.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


double target(double x_0, double x_1) {
    double target;
    target = x_0 * x_1 + 1.0;
    return target;
}

/* generate a random floating point number from min to max */
double rand_from(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}


double l2_norm_rule_target() {
    int sz = 100;
    srand(1234); // fix seed
    double x_0_rand;
    double x_1_rand;

    double target_value;
    double rule_output;
    double sum_l2_difference = 0.0;

    for(int i=0;i<sz;i++){
        /* generate two random values for x_0, x_1 */
        double min = -1.0;
        double max = 1.0;
        x_0_rand=rand_from(min, max);
        x_1_rand=rand_from(min, max);

        target_value=target(x_0_rand, x_1_rand);
        rule_output=rule(x_0_rand, x_1_rand);
        
        sum_l2_difference += pow(target_value-rule_output, 2);
    }
    return sum_l2_difference/sz;
}

int main(){
    printf("%f", l2_norm_rule_target());
    return 0;
}
