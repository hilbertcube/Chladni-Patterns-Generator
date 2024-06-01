from scipy.special import jnp_zeros

n = 0   # Order of the Bessel function
m = 7   # Zero number

def bessel_derivative_zero(n, m):
    if n == 0:
        zeros = list(jnp_zeros(n, m + 1))
        zeros.insert(0, 0)
        return zeros[m]
    else:
        return jnp_zeros(n, m + 1)[-1]  # Get the m-th zero, treating first zero as the 0th zero


for i in range (m):
    zero = bessel_derivative_zero(n, i)
    print(round(zero, 7) )
    #print(f"The {i}-th zero of the derivative of the {n}-th order Bessel function is approximately: {zero}")