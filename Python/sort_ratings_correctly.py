#!/usr/bin/env python3
import scipy.stats as stats
import math

def ci_lower_bound(num_pos_ratings, num_total_ratings, confidence):
    f num_total_ratings == 0:
        return 0
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * num_pos_ratings / num_total_ratings

    return (phat + z * z / (2 * num_total_ratings) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * num_total_ratings)) / num_total_ratings)) / (1 + z * z / num_total_ratings)


