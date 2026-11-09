"""Functions extracted from the revised lecture notes; examples are not executed."""
import math
import numpy as np
import pandas as pd
from scipy.stats import norm, binom
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.sparse import diags
from scipy.sparse.linalg import splu



# Source: section05_numpy_scipy_functions.tex / black_scholes_call_price
def black_scholes_call_price(
    spot,
    strike,
    maturity,
    rate,
    volatility,
    dividend_yield=0.0,
):
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    maturity = np.asarray(maturity, dtype=float)
    rate = np.asarray(rate, dtype=float)
    volatility = np.asarray(volatility, dtype=float)
    dividend_yield = np.asarray(dividend_yield, dtype=float)

    if np.any(spot <= 0.0):
        raise ValueError("spot must be positive")
    if np.any(strike <= 0.0):
        raise ValueError("strike must be positive")
    if np.any(maturity <= 0.0):
        raise ValueError("maturity must be positive")
    if np.any(volatility <= 0.0):
        raise ValueError("volatility must be positive")

    sqrt_maturity = np.sqrt(maturity)
    log_moneyness = np.log(spot / strike)
    drift = rate - dividend_yield + 0.5 * volatility**2
    d1 = (log_moneyness + drift * maturity) / (volatility * sqrt_maturity)
    d2 = d1 - volatility * sqrt_maturity

    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    price = discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)

    return price

# Source: section07_monte_carlo_option_pricing.tex / option_payoff
def option_payoff(terminal_stock, strike, option_type):
    terminal_stock = np.asarray(terminal_stock, dtype=float)
    strike = float(strike)
    option_type = str(option_type).lower()

    if option_type == "call":
        return np.maximum(terminal_stock - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - terminal_stock, 0.0)

    raise ValueError("option_type must be call or put")

# Source: section07_monte_carlo_option_pricing.tex / simulate_gbm_paths
def simulate_gbm_paths(
    spot,
    maturity,
    rate,
    volatility,
    n_paths,
    n_steps,
    seed,
    dividend_yield=0.0,
):
    spot = float(spot)
    maturity = float(maturity)
    rate = float(rate)
    volatility = float(volatility)
    dividend_yield = float(dividend_yield)
    counts = (("n_paths", n_paths, 1), ("n_steps", n_steps, 1), ("seed", seed, 0))
    for name, value, minimum in counts:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must not be bool")
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    n_paths = int(n_paths)
    n_steps = int(n_steps)
    seed = int(seed)

    parameters = [spot, maturity, rate, volatility, dividend_yield]
    if not np.isfinite(parameters).all():
        raise ValueError("model parameters must be finite")

    if spot <= 0.0 or maturity <= 0.0:
        raise ValueError("spot and maturity must be positive")
    if volatility < 0.0:
        raise ValueError("volatility must be nonnegative")
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")

    dt = maturity / n_steps
    sqrt_dt = np.sqrt(dt)
    time_grid = np.linspace(0.0, maturity, num=n_steps + 1)

    rng = np.random.default_rng(seed)
    paths = np.empty((n_steps + 1, n_paths), dtype=float)
    paths[0, :] = spot

    for step in range(n_steps):
        current_stock = paths[step, :]
        z = rng.standard_normal(n_paths)
        drift = (rate - dividend_yield) * current_stock * dt
        diffusion = volatility * current_stock * sqrt_dt * z
        next_stock = current_stock + drift + diffusion

        if np.any(next_stock <= 0.0):
            raise RuntimeError("Euler scheme produced a nonpositive stock price")

        paths[step + 1, :] = next_stock

    return time_grid, paths

# Source: section07_monte_carlo_option_pricing.tex / simulate_gbm_terminal
def simulate_gbm_terminal(
    spot,
    maturity,
    rate,
    volatility,
    n_paths,
    n_steps,
    seed,
    dividend_yield=0.0,
):
    spot = float(spot)
    maturity = float(maturity)
    rate = float(rate)
    volatility = float(volatility)
    dividend_yield = float(dividend_yield)
    counts = (("n_paths", n_paths, 1), ("n_steps", n_steps, 1), ("seed", seed, 0))
    for name, value, minimum in counts:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must not be bool")
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    n_paths = int(n_paths)
    n_steps = int(n_steps)
    seed = int(seed)

    parameters = [spot, maturity, rate, volatility, dividend_yield]
    if not np.isfinite(parameters).all():
        raise ValueError("model parameters must be finite")

    if spot <= 0.0 or maturity <= 0.0:
        raise ValueError("spot and maturity must be positive")
    if volatility < 0.0:
        raise ValueError("volatility must be nonnegative")
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")

    dt = maturity / n_steps
    sqrt_dt = np.sqrt(dt)
    rng = np.random.default_rng(seed)
    stock = np.full(n_paths, spot, dtype=float)

    for _ in range(n_steps):
        z = rng.standard_normal(n_paths)
        drift = (rate - dividend_yield) * stock * dt
        diffusion = volatility * stock * sqrt_dt * z
        stock = stock + drift + diffusion

        if np.any(stock <= 0.0):
            raise RuntimeError("Euler scheme produced a nonpositive stock price")

    return stock

# Source: section07_monte_carlo_option_pricing.tex / monte_carlo_european_option
def monte_carlo_european_option(
    spot,
    strike,
    maturity,
    rate,
    volatility,
    n_paths,
    n_steps,
    seed,
    option_type="call",
    dividend_yield=0.0,
):
    strike = float(strike)
    maturity = float(maturity)
    rate = float(rate)

    parameters = [strike, maturity, rate]
    if not np.isfinite(parameters).all():
        raise ValueError("model parameters must be finite")
    if isinstance(n_paths, (bool, np.bool_)):
        raise TypeError("n_paths must not be bool")
    if not isinstance(n_paths, (int, np.integer)):
        raise TypeError("n_paths must be an integer")
    if n_paths < 2:
        raise ValueError("at least two paths are required for a standard error")
    n_paths = int(n_paths)

    terminal_stock = simulate_gbm_terminal(
        spot=spot,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        dividend_yield=dividend_yield,
    )

    payoff = option_payoff(terminal_stock, strike, option_type)
    discounted_payoff = np.exp(-rate * maturity) * payoff

    price = float(discounted_payoff.mean())
    standard_error = float(discounted_payoff.std(ddof=1) / np.sqrt(n_paths))
    lower = price - 1.96 * standard_error
    upper = price + 1.96 * standard_error

    return {
        "price": price,
        "standard_error": standard_error,
        "lower": lower,
        "upper": upper,
    }

# Source: section07_monte_carlo_option_pricing.tex / monte_carlo_asian_call
def monte_carlo_asian_call(
    spot,
    strike,
    maturity,
    rate,
    volatility,
    n_paths,
    n_steps,
    seed,
    dividend_yield=0.0,
):
    spot = float(spot)
    strike = float(strike)
    maturity = float(maturity)
    rate = float(rate)
    volatility = float(volatility)
    dividend_yield = float(dividend_yield)
    counts = (("n_paths", n_paths, 2), ("n_steps", n_steps, 1), ("seed", seed, 0))
    for name, value, minimum in counts:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must not be bool")
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    n_paths = int(n_paths)
    n_steps = int(n_steps)
    seed = int(seed)

    parameters = [spot, strike, maturity, rate, volatility, dividend_yield]
    if not np.isfinite(parameters).all():
        raise ValueError("model parameters must be finite")

    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike and maturity must be positive")
    if volatility < 0.0 or n_paths <= 0 or n_steps <= 0:
        raise ValueError("volatility, n_paths and n_steps are invalid")

    dt = maturity / n_steps
    sqrt_dt = np.sqrt(dt)
    rng = np.random.default_rng(seed)

    stock = np.full(n_paths, spot, dtype=float)
    running_sum = np.zeros(n_paths, dtype=float)

    for _ in range(n_steps):
        z = rng.standard_normal(n_paths)
        drift = (rate - dividend_yield) * stock * dt
        diffusion = volatility * stock * sqrt_dt * z
        stock = stock + drift + diffusion

        if np.any(stock <= 0.0):
            raise RuntimeError("Euler scheme produced a nonpositive stock price")

        running_sum += stock

    average_stock = running_sum / n_steps
    payoff = np.maximum(average_stock - strike, 0.0)
    discounted_payoff = np.exp(-rate * maturity) * payoff

    price = float(discounted_payoff.mean())
    standard_error = float(discounted_payoff.std(ddof=1) / np.sqrt(n_paths))
    return price, standard_error

# Source: section07_monte_carlo_option_pricing.tex / monte_carlo_cev_option
def monte_carlo_cev_option(
    spot,
    strike,
    maturity,
    rate,
    sigma0,
    beta,
    n_paths,
    n_steps,
    seed,
    option_type="call",
    dividend_yield=0.0,
):
    spot = float(spot)
    strike = float(strike)
    maturity = float(maturity)
    rate = float(rate)
    sigma0 = float(sigma0)
    beta = float(beta)
    dividend_yield = float(dividend_yield)
    counts = (("n_paths", n_paths, 2), ("n_steps", n_steps, 1), ("seed", seed, 0))
    for name, value, minimum in counts:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must not be bool")
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    n_paths = int(n_paths)
    n_steps = int(n_steps)
    seed = int(seed)

    parameters = [spot, strike, maturity, rate, sigma0, beta, dividend_yield]
    if not np.isfinite(parameters).all():
        raise ValueError("model parameters must be finite")

    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike and maturity must be positive")
    if sigma0 <= 0.0 or not 0.0 < beta <= 1.0:
        raise ValueError("require sigma0 > 0 and 0 < beta <= 1")
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")

    dt = maturity / n_steps
    sqrt_dt = np.sqrt(dt)
    diffusion_scale = sigma0 * spot ** (1.0 - beta)

    rng = np.random.default_rng(seed)
    stock = np.full(n_paths, spot, dtype=float)

    for _ in range(n_steps):
        z = rng.standard_normal(n_paths)
        positive_stock = np.maximum(stock, 0.0)
        drift = (rate - dividend_yield) * positive_stock * dt
        diffusion = diffusion_scale * positive_stock**beta * sqrt_dt * z
        stock = np.maximum(positive_stock + drift + diffusion, 0.0)

    payoff = option_payoff(stock, strike, option_type)
    discounted_payoff = np.exp(-rate * maturity) * payoff
    price = float(discounted_payoff.mean())
    standard_error = float(discounted_payoff.std(ddof=1) / np.sqrt(n_paths))
    return price, standard_error

# Source: section07_monte_carlo_option_pricing.tex / monte_carlo_heston_option
def monte_carlo_heston_option(
    spot,
    strike,
    maturity,
    rate,
    initial_variance,
    kappa,
    theta,
    vol_of_variance,
    correlation,
    n_paths,
    n_steps,
    seed,
    option_type="call",
    dividend_yield=0.0,
):
    spot = float(spot)
    strike = float(strike)
    maturity = float(maturity)
    rate = float(rate)
    initial_variance = float(initial_variance)
    kappa = float(kappa)
    theta = float(theta)
    vol_of_variance = float(vol_of_variance)
    correlation = float(correlation)
    dividend_yield = float(dividend_yield)
    counts = (("n_paths", n_paths, 2), ("n_steps", n_steps, 1), ("seed", seed, 0))
    for name, value, minimum in counts:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must not be bool")
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    n_paths = int(n_paths)
    n_steps = int(n_steps)
    seed = int(seed)

    parameters = [spot, strike, maturity, rate, initial_variance, kappa, theta, vol_of_variance, correlation, dividend_yield]
    if not np.isfinite(parameters).all():
        raise ValueError("model parameters must be finite")

    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike and maturity must be positive")
    if initial_variance < 0.0 or theta < 0.0 or vol_of_variance < 0.0:
        raise ValueError("variance parameters must be nonnegative")
    if kappa < 0.0 or n_paths <= 0 or n_steps <= 0:
        raise ValueError("kappa must be nonnegative; counts must be positive")
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must be between -1 and 1")

    dt = maturity / n_steps
    sqrt_dt = np.sqrt(dt)
    independent_scale = np.sqrt(1.0 - correlation**2)

    rng = np.random.default_rng(seed)
    stock = np.full(n_paths, spot, dtype=float)
    variance = np.full(n_paths, initial_variance, dtype=float)

    for _ in range(n_steps):
        z_variance = rng.standard_normal(n_paths)
        z_independent = rng.standard_normal(n_paths)
        z_stock = correlation * z_variance + independent_scale * z_independent

        positive_variance = np.maximum(variance, 0.0)
        sqrt_variance = np.sqrt(positive_variance)

        next_stock = stock + (rate - dividend_yield) * stock * dt
        next_stock += sqrt_variance * stock * sqrt_dt * z_stock

        next_variance = variance + kappa * (theta - positive_variance) * dt
        next_variance += vol_of_variance * sqrt_variance * sqrt_dt * z_variance

        stock = np.maximum(next_stock, 0.0)
        variance = np.maximum(next_variance, 0.0)

    payoff = option_payoff(stock, strike, option_type)
    discounted_payoff = np.exp(-rate * maturity) * payoff
    price = float(discounted_payoff.mean())
    standard_error = float(discounted_payoff.std(ddof=1) / np.sqrt(n_paths))

    return price, standard_error

# Source: section08_pde_option_pricing.tex / option_payoff_grid
def option_payoff_grid(stock_grid, strike, option_type):
    stock_grid = np.asarray(stock_grid, dtype=float)
    strike = float(strike)
    option_type = str(option_type).lower()

    if option_type == "call":
        return np.maximum(stock_grid - strike, 0.0)
    if option_type == "put":
        return np.maximum(strike - stock_grid, 0.0)

    raise ValueError("option_type must be call or put")

# Source: section08_pde_option_pricing.tex / european_boundary_values
def european_boundary_values(
    tau,
    s_max,
    strike,
    rate,
    dividend_yield,
    option_type,
):
    tau = float(tau)
    option_type = str(option_type).lower()

    if option_type == "call":
        left = 0.0
        right = s_max * np.exp(-dividend_yield * tau)
        right -= strike * np.exp(-rate * tau)
        return left, right

    if option_type == "put":
        left = strike * np.exp(-rate * tau)
        return left, 0.0

    raise ValueError("option_type must be call or put")

# Source: section08_pde_option_pricing.tex / black_scholes_pde_price
def black_scholes_pde_price(
    spot,
    strike,
    maturity,
    rate,
    volatility,
    s_max,
    n_space,
    n_time,
    option_type="call",
    dividend_yield=0.0,
):
    spot = float(spot)
    strike = float(strike)
    maturity = float(maturity)
    rate = float(rate)
    volatility = float(volatility)
    s_max = float(s_max)
    dividend_yield = float(dividend_yield)
    for name, value, minimum in (("n_space", n_space, 3), ("n_time", n_time, 1)):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must not be bool")
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    n_space = int(n_space)
    n_time = int(n_time)
    option_type = str(option_type).lower()

    parameters = [spot, strike, maturity, rate, s_max, dividend_yield]
    if not np.isfinite(parameters).all():
        raise ValueError("PDE parameters must be finite")
    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike and maturity must be positive")
    if s_max <= max(spot, strike):
        raise ValueError("s_max must exceed both spot and strike")

    required_log_ratio = max((dividend_yield - rate) * maturity, 0.0)
    if np.log(s_max / strike) <= required_log_ratio:
        raise ValueError("s_max is too small for this boundary condition")

    if not np.isfinite(volatility) or volatility <= 0.0:
        raise ValueError("volatility must be finite and positive")
    if n_space < 3 or n_time <= 0:
        raise ValueError("check n_space and n_time")

    stock_grid = np.linspace(0.0, s_max, num=n_space + 1)
    d_s = s_max / n_space
    d_tau = maturity / n_time
    values = option_payoff_grid(stock_grid, strike, option_type)

    interior_stock = stock_grid[1:-1]
    diffusion = 0.5 * volatility**2 * interior_stock**2
    convection = (rate - dividend_yield) * interior_stock

    lower = diffusion / d_s**2 - convection / (2.0 * d_s)
    center = -2.0 * diffusion / d_s**2 - rate
    upper = diffusion / d_s**2 + convection / (2.0 * d_s)

    matrix = diags(
        diagonals=(
            -d_tau * lower[1:],
            1.0 - d_tau * center,
            -d_tau * upper[:-1],
        ),
        offsets=(-1, 0, 1),
        format="csc",
    )
    solver = splu(matrix)

    for step in range(n_time):
        tau = (step + 1) * d_tau
        left, right = european_boundary_values(
            tau,
            s_max,
            strike,
            rate,
            dividend_yield,
            option_type,
        )

        right_hand_side = values[1:-1].copy()
        right_hand_side[0] += d_tau * lower[0] * left
        right_hand_side[-1] += d_tau * upper[-1] * right

        values[1:-1] = solver.solve(right_hand_side)
        values[0] = left
        values[-1] = right

    price = np.interp(spot, stock_grid, values)
    return float(price), stock_grid, values

# Source: section08_pde_option_pricing.tex / projected_american_pde_price
def projected_american_pde_price(
    spot,
    strike,
    maturity,
    rate,
    volatility,
    s_max,
    n_space,
    n_time,
    option_type="put",
    dividend_yield=0.0,
):
    spot = float(spot)
    strike = float(strike)
    maturity = float(maturity)
    rate = float(rate)
    volatility = float(volatility)
    s_max = float(s_max)
    dividend_yield = float(dividend_yield)
    for name, value, minimum in (("n_space", n_space, 3), ("n_time", n_time, 1)):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must not be bool")
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    n_space = int(n_space)
    n_time = int(n_time)
    option_type = str(option_type).lower()

    parameters = [spot, strike, maturity, rate, s_max, dividend_yield]
    if not np.isfinite(parameters).all():
        raise ValueError("PDE parameters must be finite")
    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike and maturity must be positive")
    if s_max <= max(spot, strike):
        raise ValueError("s_max must exceed both spot and strike")

    required_log_ratio = max((dividend_yield - rate) * maturity, 0.0)
    if np.log(s_max / strike) <= required_log_ratio:
        raise ValueError("s_max is too small for this boundary condition")

    if not np.isfinite(volatility) or volatility <= 0.0:
        raise ValueError("volatility must be finite and positive")

    stock_grid = np.linspace(0.0, s_max, num=n_space + 1)
    d_s = s_max / n_space
    d_tau = maturity / n_time
    payoff = option_payoff_grid(stock_grid, strike, option_type)
    values = payoff.copy()

    interior_stock = stock_grid[1:-1]
    diffusion = 0.5 * volatility**2 * interior_stock**2
    convection = (rate - dividend_yield) * interior_stock

    lower = diffusion / d_s**2 - convection / (2.0 * d_s)
    center = -2.0 * diffusion / d_s**2 - rate
    upper = diffusion / d_s**2 + convection / (2.0 * d_s)

    matrix = diags(
        diagonals=(
            -d_tau * lower[1:],
            1.0 - d_tau * center,
            -d_tau * upper[:-1],
        ),
        offsets=(-1, 0, 1),
        format="csc",
    )
    solver = splu(matrix)

    for step in range(n_time):
        tau = (step + 1) * d_tau

        if option_type == "put":
            left, right = strike, 0.0
        else:
            left = 0.0
            european_right = s_max * np.exp(-dividend_yield * tau)
            european_right -= strike * np.exp(-rate * tau)
            right = max(s_max - strike, european_right)

        right_hand_side = values[1:-1].copy()
        right_hand_side[0] += d_tau * lower[0] * left
        right_hand_side[-1] += d_tau * upper[-1] * right

        continuation = solver.solve(right_hand_side)
        values[1:-1] = np.maximum(continuation, payoff[1:-1])
        values[0] = left
        values[-1] = right

    return float(np.interp(spot, stock_grid, values))

# Source: section08_pde_option_pricing.tex / local_volatility_pde_price
def local_volatility_pde_price(
    spot,
    strike,
    maturity,
    rate,
    s_max,
    n_space,
    n_time,
    local_volatility,
    option_type="call",
    dividend_yield=0.0,
):
    spot = float(spot)
    strike = float(strike)
    maturity = float(maturity)
    rate = float(rate)
    s_max = float(s_max)
    dividend_yield = float(dividend_yield)
    for name, value, minimum in (("n_space", n_space, 3), ("n_time", n_time, 1)):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must not be bool")
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    n_space = int(n_space)
    n_time = int(n_time)
    option_type = str(option_type).lower()

    parameters = [spot, strike, maturity, rate, s_max, dividend_yield]
    if not np.isfinite(parameters).all():
        raise ValueError("PDE parameters must be finite")
    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike and maturity must be positive")
    if s_max <= max(spot, strike):
        raise ValueError("s_max must exceed both spot and strike")

    required_log_ratio = max((dividend_yield - rate) * maturity, 0.0)
    if np.log(s_max / strike) <= required_log_ratio:
        raise ValueError("s_max is too small for this boundary condition")

    stock_grid = np.linspace(0.0, s_max, num=n_space + 1)
    d_s = s_max / n_space
    d_tau = maturity / n_time
    values = option_payoff_grid(stock_grid, strike, option_type)
    interior_stock = stock_grid[1:-1]

    for step in range(n_time):
        tau = (step + 1) * d_tau
        calendar_time = maturity - tau

        sigma = np.asarray(local_volatility(calendar_time, interior_stock), dtype=float)
        sigma = np.broadcast_to(sigma, interior_stock.shape)

        if not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
            raise ValueError("local volatility must be finite and positive")

        diffusion = 0.5 * sigma**2 * interior_stock**2
        convection = (rate - dividend_yield) * interior_stock

        lower = diffusion / d_s**2 - convection / (2.0 * d_s)
        center = -2.0 * diffusion / d_s**2 - rate
        upper = diffusion / d_s**2 + convection / (2.0 * d_s)

        matrix = diags(
            diagonals=(
                -d_tau * lower[1:],
                1.0 - d_tau * center,
                -d_tau * upper[:-1],
            ),
            offsets=(-1, 0, 1),
            format="csc",
        )

        left, right = european_boundary_values(
            tau,
            s_max,
            strike,
            rate,
            dividend_yield,
            option_type,
        )

        right_hand_side = values[1:-1].copy()
        right_hand_side[0] += d_tau * lower[0] * left
        right_hand_side[-1] += d_tau * upper[-1] * right

        values[1:-1] = splu(matrix).solve(right_hand_side)
        values[0] = left
        values[-1] = right

    price = np.interp(spot, stock_grid, values)
    return float(price), stock_grid, values

# Source: section08_pde_option_pricing.tex / heston_spatial_operator
def heston_spatial_operator(
    values,
    stock_grid,
    variance_grid,
    rate,
    dividend_yield,
    kappa,
    theta,
    vol_of_variance,
    correlation,
):
    values = np.asarray(values, dtype=float)
    stock_grid = np.asarray(stock_grid, dtype=float)
    variance_grid = np.asarray(variance_grid, dtype=float)

    d_s = stock_grid[1] - stock_grid[0]
    d_v = variance_grid[1] - variance_grid[0]

    center_values = values[1:-1, 1:-1]

    value_s = (values[1:-1, 2:] - values[1:-1, :-2]) / (2.0 * d_s)
    value_ss = (values[1:-1, :-2] - 2.0 * center_values + values[1:-1, 2:]) / d_s**2

    value_v = (values[2:, 1:-1] - values[:-2, 1:-1]) / (2.0 * d_v)
    value_vv = (values[:-2, 1:-1] - 2.0 * center_values + values[2:, 1:-1]) / d_v**2

    value_sv = (
        values[2:, 2:] - values[2:, :-2] - values[:-2, 2:] + values[:-2, :-2]
    ) / (4.0 * d_s * d_v)

    stock, variance = np.meshgrid(stock_grid[1:-1], variance_grid[1:-1])

    operator = 0.5 * variance * stock**2 * value_ss
    operator += correlation * vol_of_variance * variance * stock * value_sv
    operator += 0.5 * vol_of_variance**2 * variance * value_vv
    operator += (rate - dividend_yield) * stock * value_s
    operator += kappa * (theta - variance) * value_v
    operator -= rate * center_values

    return operator
