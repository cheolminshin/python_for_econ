import numpy as np,json
from pathlib import Path
r=Path(__file__).resolve().parents[1];p=r/'_validation/els_figure_metadata.json';meta=json.loads(p.read_text());c=meta['settings']
N=50000;sub=c['substeps_per_day'];steps=int(c['maturity']*c['days_per_year']*sub);dt=c['maturity']/steps
rng=np.random.default_rng(2026);s=np.full(N,c['reference']);alive=np.ones(N,bool);hit=np.zeros(N,bool);pv=np.zeros(N)
obs={d*sub:b for d,b in zip(c['observation_days'],c['autocall_levels'])}
for k in range(1,steps+1):
 s=s+c['rate']*s*dt+c['volatility']*s*np.sqrt(dt)*rng.standard_normal(N)
 if k%sub==0:hit[alive]|=s[alive]<=c['knock_in_level']
 if k in obs:
  redeemed=alive&(s>=obs[k]);pv[redeemed]=c['notional']*(1+c['coupon_rate']*k*dt)*np.exp(-c['rate']*k*dt);alive[redeemed]=False
pv[alive&~hit]=c['notional']*(1+c['coupon_rate']*c['maturity'])*np.exp(-c['rate']*c['maturity'])
pv[alive&hit]=c['notional']*s[alive&hit]/c['reference']*np.exp(-c['rate']*c['maturity'])
x={'paths':N,'steps':steps,'seed':2026,'monitoring':'daily close only','price':float(pv.mean()),'standard_error':float(pv.std(ddof=1)/np.sqrt(N)),'fdm_price':meta['initial_price']}
x['difference']=x['fdm_price']-x['price'];x['difference_in_se']=x['difference']/x['standard_error'];x['status']='PASS' if abs(x['difference_in_se'])<3 else 'CHECK'
meta['one_asset_mc_check']=x;p.write_text(json.dumps(meta,ensure_ascii=False,indent=2));print(x)
