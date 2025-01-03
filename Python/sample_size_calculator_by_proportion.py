def get_sample_size(pool_size, z_score, pop_proportion, error_margin, pop_size=None):
    if not pop_size:
        pop_size = pool_size
    return (pop_size / (1 + (((z_score ** 2) * (pop_proportion * (1 - pop_proportion))) / ((error_margin ** 2) * pop_size))))

print(get_sample_size(100000, 2.575, 0.38, 0.005, 100000))
print(get_sample_size(100000, 2.575, 0.38, 0.005))
print(get_sample_size(100000, 2.575, 0.38, 0.005, 1000000))
