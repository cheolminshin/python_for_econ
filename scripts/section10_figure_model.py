"""One-asset ELS illustration; distinct from the two-asset MC example in Section 10.

Contract: daily close knock-in monitoring, semiannual step-down observations.
Numerics: implicit Euler FDM with multiple solver steps per monitoring day.
The functions store each knock-in state separately and return values just after
scheduled observations for surviving contracts. This is an educational model,
not a calibrated trading system. No credit/funding costs or transaction costs.
"""
from pathlib import Path
import json
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

DEFAULTS = {
    "reference": 100.0, "notional": 100.0, "maturity": 3.0,
    "rate": 0.03, "dividend_yield": 0.0, "volatility": 0.25,
    "coupon_rate": 0.08, "knock_in_level": 60.0,
    "observation_days": [126, 252, 378, 504, 630, 756],
    "autocall_levels": [90.0, 90.0, 85.0, 85.0, 80.0, 80.0],
    "days_per_year": 252, "s_max": 250.0, "n_space": 1000,
    "substeps_per_day": 4, "path_seed": 24, "physical_drift": 0.01,
    "curve_day": 121,
}


def solve_els_fdm(settings=None):
    """Return post-observation values for alive contracts at each daily close."""
    cfg = {**DEFAULTS, **(settings or {})}
    if cfg["dividend_yield"] != 0.0:
        raise ValueError("The associated hedge example assumes no dividends")
    n_space = cfg["n_space"]
    substeps = cfg["substeps_per_day"]
    if type(n_space) is not int or n_space < 10:
        raise ValueError("n_space must be an integer >= 10")
    if type(substeps) is not int or substeps < 1:
        raise ValueError("substeps_per_day must be a positive integer")
    n_days = int(round(cfg["maturity"] * cfg["days_per_year"]))
    if n_days != cfg["observation_days"][-1]:
        raise ValueError("Last observation must be the maturity")
    obs = dict(zip(cfg["observation_days"], cfg["autocall_levels"]))
    s = np.linspace(0.0, cfg["s_max"], num=n_space + 1)
    ds = s[1] - s[0]
    dt = 1.0 / (cfg["days_per_year"] * substeps)
    rate, sigma = cfg["rate"], cfg["volatility"]
    interior = s[1:-1]
    a = 0.5 * sigma**2 * interior**2 / ds**2 - rate * interior / (2.0 * ds)
    b = -sigma**2 * interior**2 / ds**2 - rate
    c = 0.5 * sigma**2 * interior**2 / ds**2 + rate * interior / (2.0 * ds)
    if np.min(a) < -1e-12 or np.min(c) < -1e-12:
        raise ValueError("This demonstration requires nonnegative off-diagonal stencil coefficients")
    matrix = diags((-dt*a[1:], 1.0-dt*b, -dt*c[:-1]), (-1,0,1), format="csc")
    lu = splu(matrix)
    final_coupon = cfg["notional"] * (1.0 + cfg["coupon_rate"] * cfg["maturity"])
    ki_values = np.where(s >= cfg["autocall_levels"][-1], final_coupon,
                         cfg["notional"] * s / cfg["reference"])
    no_values = np.full_like(s, final_coupon)
    hit = s <= cfg["knock_in_level"]
    no_values[hit] = ki_values[hit]  # final knock-in observation
    no_post = np.empty((n_days + 1, n_space + 1), dtype=float)
    ki_post = np.empty_like(no_post)
    no_post[-1], ki_post[-1] = no_values, ki_values
    for step in range(n_days*substeps - 1, -1, -1):
        t = step * dt
        # Boundary is evaluated after any event at t: the next event is strictly later.
        next_day = next(d for d in cfg["observation_days"] if d*substeps > step)
        pay_time = next_day / cfg["days_per_year"]
        upper = cfg["notional"] * (1.0 + cfg["coupon_rate"] * pay_time)
        upper *= np.exp(-rate * (pay_time - t))
        rhs = np.column_stack((no_values[1:-1], ki_values[1:-1]))
        rhs[-1, :] += dt * c[-1] * upper
        sol = lu.solve(rhs)
        no_values = np.r_[0.0, sol[:, 0], upper]
        ki_values = np.r_[0.0, sol[:, 1], upper]
        if step % substeps == 0:
            day = step // substeps
            no_post[day], ki_post[day] = no_values, ki_values
            # Backward event update: value just before today's observations.
            if day > 0:
                no_values[hit] = ki_values[hit]
            if day in obs:
                redeem = s >= obs[day]
                amount = cfg["notional"] * (1.0 + cfg["coupon_rate"] * t)
                no_values[redeem] = amount
                ki_values[redeem] = amount
    assert np.isfinite(no_post).all() and np.isfinite(ki_post).all()
    assert np.min(no_post) >= -1e-8 and np.min(ki_post) >= -1e-8
    assert np.min(no_post-ki_post) >= -1e-7
    return cfg, s, no_post, ki_post


def grid_greeks(values, ds):
    """Central differences within a single state, no mixing of state arrays."""
    delta = (values[..., 2:] - values[..., :-2]) / (2.0 * ds)
    gamma = (values[..., 2:] - 2.0*values[..., 1:-1] + values[..., :-2]) / ds**2
    return delta, gamma


def simulate_daily_path(cfg):
    rng = np.random.default_rng(cfg["path_seed"])
    n = int(cfg["maturity"] * cfg["days_per_year"])
    dt = 1.0/cfg["days_per_year"]
    path = np.empty(n+1, dtype=float)
    path[0] = cfg["reference"]
    for k in range(n):
        drift = cfg["physical_drift"] * path[k] * dt
        diffusion = cfg["volatility"] * path[k] * np.sqrt(dt) * rng.standard_normal()
        path[k+1] = path[k] + drift + diffusion
    if np.any(path <= 0.0):
        raise RuntimeError("Nonpositive Euler stock: change the example's time step")
    return path


def hedge_daily(cfg, stock_grid, no_post, ki_post):
    path = simulate_daily_path(cfg)
    inner = stock_grid[1:-1]
    ds = stock_grid[1]-stock_grid[0]
    delta0, _ = grid_greeks(no_post, ds)
    delta1, _ = grid_greeks(ki_post, ds)
    obs = dict(zip(cfg["observation_days"], cfg["autocall_levels"]))
    days_per_year = cfg["days_per_year"]
    price0 = float(np.interp(path[0], stock_grid, no_post[0]))
    shares = float(np.interp(path[0], inner, delta0[0]))
    cash = price0-shares*path[0]
    knocked_in = path[0] <= cfg["knock_in_level"]
    # day, spot, KI, claim value, target delta after close, shares held on arrival,
    # cash after close/settlement, shares after close, mark-to-market hedge error
    records = [[0, path[0], int(knocked_in), price0, shares, shares, cash, shares, 0.0]]
    residuals = []
    final_day = len(path)-1
    for day in range(1,len(path)):
        old_cash, old_shares = cash, shares
        cash *= np.exp(cfg["rate"]/days_per_year)
        spot = path[day]
        if not (inner[0] <= spot <= inner[-1]):
            raise ValueError("Hedge path is outside the Greek interpolation grid")
        knocked_in = knocked_in or spot <= cfg["knock_in_level"]
        redeemed = day in obs and spot >= obs[day]
        if redeemed or day == len(path)-1:
            amount = cfg["notional"] * (1.0+cfg["coupon_rate"]*day/days_per_year)
            if not redeemed and knocked_in:
                amount = cfg["notional"] * spot/cfg["reference"]
            pnl = cash+shares*spot-amount
            cash, shares = pnl, 0.0
            residuals.append(cash-(old_cash*np.exp(cfg["rate"]/days_per_year)+old_shares*spot-amount))
            records.append([day, spot, int(knocked_in), amount, 0.0, old_shares,cash,shares,pnl])
            final_day=day
            break
        values = ki_post if knocked_in else no_post
        deltas = delta1 if knocked_in else delta0
        value = float(np.interp(spot,stock_grid,values[day]))
        target = float(np.interp(spot,inner,deltas[day]))
        cash -= (target-shares)*spot
        shares = target
        residuals.append(cash-(old_cash*np.exp(cfg["rate"]/days_per_year)-(shares-old_shares)*spot))
        pnl = cash+shares*spot-value
        records.append([day,spot,int(knocked_in),value,target,old_shares,cash,shares,pnl])
    assert max(abs(x) for x in residuals) < 1e-10
    return np.asarray(records), {"initial_price":price0,"redemption_time":final_day/days_per_year,
        "final_pnl":float(records[-1][-1]),"max_self_financing_residual":max(abs(x) for x in residuals),
        "knock_in_occurred":bool(knocked_in)}


def calculate_and_save(output_root):
    root=Path(output_root);dest=root/'_validation';dest.mkdir(exist_ok=True)
    cfg,s,v0,v1=solve_els_fdm()
    rows,meta=hedge_daily(cfg,s,v0,v1)
    day=cfg['curve_day'];ds=s[1]-s[0];d0,g0=grid_greeks(v0[day],ds);d1,g1=grid_greeks(v1[day],ds)
    curves=np.column_stack((s[1:-1],v0[day,1:-1],v1[day,1:-1],d0,d1,g0,g1))
    np.savetxt(dest/'els_state_curves.csv',curves,delimiter=',',comments='',
        header='spot,no_knock_in_value,knocked_in_value,no_knock_in_delta,knocked_in_delta,no_knock_in_gamma,knocked_in_gamma')
    np.savetxt(dest/'els_hedge_trace.csv',rows,delimiter=',',comments='',
        header='day,spot,knocked_in,claim_value,target_delta,shares_before_close,cash_after_close,shares_after_close,hedging_error')
    coarse=[]
    for ns,sub in [(500,2),(1000,2),(1000,4)]:
        if (ns,sub)==(1000,4):p=meta['initial_price']
        else:
            _,ss,a,b=solve_els_fdm({'n_space':ns,'substeps_per_day':sub})
            p=float(np.interp(100.,ss,a[0]))
        coarse.append({'n_space':ns,'substeps_per_day':sub,'price':p})
    meta.update({'settings':cfg,'curve_time':day/cfg['days_per_year'],'grid_checks':coarse,
        'scope':'single-asset educational example; daily KI; no full convergence proof'})
    (dest/'els_figure_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    return cfg,curves,rows,meta

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    _,_,_,meta=calculate_and_save(root)
    print(json.dumps(meta,ensure_ascii=False,indent=2))
