import numpy as np
from scipy.stats import ttest_ind

conv = [54.2, 52.2, 51.4, 51.8, 55.4, 54.1, 52.5, 54.5, 56.8, 50.4, 50.9, 53.2]
non_conv = [40.6, 33.1, 36.6, 40.5, 39.0, 40.1, 37.6, 37.6, 37.2, 36.9]

# Perform a one-sided t-test: H1 is that ConvAdapter > non-ConvAdapter
t_stat, p_value_two_sided = ttest_ind(conv, non_conv, equal_var=False)  # Welch's t-test

# Convert to one-sided p-value
p_value_one_sided = p_value_two_sided / 2

print("t-statistic:", t_stat)
print("one-sided p-value:", p_value_one_sided)
print(np.mean(conv), np.std(conv))
print(np.mean(non_conv), np.std(non_conv))1.8
