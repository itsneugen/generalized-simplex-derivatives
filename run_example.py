from testgen import run_simplex_test

# Set up your test parameters here
f_expr = "x0**6 + 2*x0**2*x1**2 + x1**4"
x0 = (1, 1)
P = 1
h_list = [1e-5] * P

# Choose which layers to show (None for just the highest order)
custom_layers = None # or [1, 2] to show both gradient and Hessian

# Run the test
run_simplex_test(
    f_expr=f_expr,
    x0=x0,
    P=P,
    mode="auto",
    h=h_list[0],
    h_list=h_list,
    show_all=False,
    custom_layers=custom_layers
)