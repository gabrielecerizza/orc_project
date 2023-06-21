import numpy as np

np.random.seed(42)


def generate_problem(m, n, density=0.5):
    """Generate a covering problem with m rows,
    n columns and a percentage of non-zero columns in 
    each row equal to density.
    """

    A = []
    b = []
    for _ in range(m):
        b_i = np.random.randint(n, n ** 2)
        b.append(b_i)

        # Determine the sum of values to be place on the
        # left hand side of each constraint
        lhs_sum = np.random.randint(
            int(b_i * 2), int(b_i * 5))
        
        num_nonzero_cols = int(n * density)
        
        # Determine the columns that will have a non-zero
        # entry
        indices = np.random.choice(
            n, num_nonzero_cols, replace=False)
        
        a_i = np.zeros((n, )) 
        
        # Initialize with ones to make sure
        # that each entry is greater than zero
        sub_a_i = np.ones((num_nonzero_cols, ))
        
        sum_left = lhs_sum - len(sub_a_i)
        for _ in range(sum_left):
            j = np.random.randint(0, sub_a_i.shape[0])
            sub_a_i[j] += 1
        a_i[indices] = sub_a_i
        A.append(a_i)
    
    A = np.array(A)
    b = np.array(b)
    assert np.all(np.sum(A, axis=-1) >= b), (A, b)
    return A, b
