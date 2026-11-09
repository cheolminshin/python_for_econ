from pathlib import Path
import sys,importlib.util,ast,re,json,time,platform,traceback
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / '_validation' / 'extracted_models'
RESULT = ROOT / '_validation' / 'test_results.json'
import runpy
runpy.run_path(str(ROOT / '_validation' / 'extract_models.py'), run_name='__main__')

def load(name):
 p=MOD/(name+'.py');spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
s5=load('section05_numpy_scipy_functions');s6=load('section06_binomial_option_pricing');s7=load('section07_monte_carlo_option_pricing');s8=load('section08_pde_option_pricing');s9=load('section09_option_greeks');s10=load('section10_els_pricing_hedging');cuda=load('cuda_monte_carlo_chapter')
s9.simulate_gbm_paths=s7.simulate_gbm_paths;s9.local_volatility_pde_price=s8.local_volatility_pde_price;s9.monte_carlo_heston_option=s7.monte_carlo_heston_option
results=[]
def jsonval(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:jsonval(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jsonval(v) for v in x]
 return x

def test(name,func):
 t=time.perf_counter()
 try:v=func();r={'name':name,'status':'PASS','result':jsonval(v)}
 except Exception as e:r={'name':name,'status':'FAIL','error':repr(e),'traceback':traceback.format_exc()}
 r['elapsed_sec']=time.perf_counter()-t;results.append(r)
 RESULT.write_text(json.dumps({'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'platform':platform.platform(),'gpu':'not executed'},'tests':results},ensure_ascii=False,indent=2))
 print(name,r['status'],r.get('result',r.get('error')),flush=True)

def near(x,y,tol=1e-8):
 assert np.isfinite(x).all() if isinstance(x,np.ndarray) else np.isfinite(x)
 assert np.allclose(x,y,rtol=0.,atol=tol),(x,y);return x

def raises(fn,err=(ValueError,TypeError)):
 try:fn()
 except err as e:return type(e).__name__
 raise AssertionError('invalid input not rejected')
BS=dict(spot=100.,strike=100.,maturity=1.,rate=.05,volatility=.2)
MC=dict(spot=100.,strike=100.,maturity=1.,rate=.03,volatility=.2,n_paths=100_000,n_steps=252,seed=2026)
PDE=dict(**BS,s_max=300.,n_space=400,n_time=1000)
H=dict(spot=100.,strike=100.,maturity=1.,rate=.03,initial_variance=.04,kappa=2.,theta=.04,vol_of_variance=.30,correlation=-.70)
terms={'notional':100.,'reference_spots':np.array([100.,100.]),'observation_times':np.arange(1,7)*.5,'redemption_barriers':np.array([.90,.90,.85,.85,.80,.80]),'coupon_rate':.08,'knock_in_barrier':.6}
market={'spots':np.array([100.,100.]),'rate':.03,'dividend_yields':np.array([.01,.015]),'volatilities':np.array([.20,.25]),'correlation':.5}

def csv_check():
 df=pd.read_csv(ROOT/'data/european_option_market_data.csv');ts=pd.read_csv(ROOT/'data/synthetic_market_time_series.csv')
 assert df.shape==(42,18) and ts.shape==(252,8)
 assert df.isna().sum().sum()==0 and df.contract_id.nunique()==42
 assert ((df.bid_price<=df.mid_price)&(df.mid_price<=df.ask_price)).all()
 T=(pd.to_datetime(df.maturity_date)-pd.to_datetime(df.quote_date)).dt.days/365
 assert np.max(abs(T-df.time_to_maturity))<1e-8
 return {'quotes_shape':df.shape,'timeseries_shape':ts.shape,'max_day_count_error':float(np.max(abs(T-df.time_to_maturity)))}
test('CSV 행수·날짜·호가·결측 검증',csv_check)
test('정수/float 산술과 음수 floor division',lambda:{'division_type':type(1/1000).__name__,'floor':-5//2,'truncate':int(-2.5),'sum_0_1':sum([.1]*10)})
test('할인계수 함수의 float 반환',lambda:({'type':type(s5.discount_factor(3,2)).__name__,'value':s5.discount_factor(3,2)}))
test('BS Section5 스칼라가격',lambda:near(float(s5.black_scholes_call_price(100.,100.,1.,.03,.2)),9.413403383853016))
test('CRR 유럽형 콜 100단계',lambda:near(s6.binomial_option_price(**BS,steps=100),10.430611662249113))
test('CRR 유럽형 풋 100단계',lambda:near(s6.binomial_option_price(**BS,steps=100,option_type='put'),5.553554112321355))
test('CRR 미국형 풋 100단계',lambda:near(s6.binomial_option_price(**BS,steps=100,option_type='put',exercise_style='american'),6.082354409142444))
test('직접 이항 기대값',lambda:near(s6.european_binomial_price_terminal(**BS,steps=100),10.430611662249083))
test('CRR 소수 steps 거절',lambda:raises(lambda:s6.binomial_option_price(**BS,steps=2.9)))
test('CRR bool steps 거절',lambda:raises(lambda:s6.crr_parameters(1.,.05,.2,True)))
test('직접 이항계산의 무차익 실패 거절',lambda:raises(lambda:s6.european_binomial_price_terminal(100.,100.,1.,2.,.2,1)))
test('CEV beta=1 트리-CRR 일치',lambda:near(s6.cev_tree_price(100.,100.,1.,.03,.2,1.,8),s6.binomial_option_price(100.,100.,1.,.03,.2,8)))
test('CEV beta>1 교육범위 거절',lambda:raises(lambda:s6.cev_tree_price(100.,100.,1.,.03,.2,1.2,8)))
test('Heston 4분기 트리',lambda:near(s6.heston_tree_price(**H,steps=6),9.276775,tol=1e-6))
test('MC 유럽형 콜',lambda:s7.monte_carlo_european_option(**MC))
test('MC 경로 하나 생성 허용',lambda:s7.simulate_gbm_paths(100.,1.,.03,.2,1,252,2026)[1].shape)
test('MC 경로 하나로 표준오차 계산 거절',lambda:raises(lambda:s7.monte_carlo_european_option(**{**MC,'n_paths':1})))
test('MC 소수 단계수 거절',lambda:raises(lambda:s7.simulate_gbm_paths(100.,1.,.03,.2,2,252.5,2026)))
test('MC Asian 콜',lambda:s7.monte_carlo_asian_call(**MC))
test('Heston Euler MC',lambda:s7.monte_carlo_heston_option(**H,n_paths=100000,n_steps=252,seed=2026))
test('FDM 콜 기본값',lambda:near(s8.black_scholes_pde_price(**PDE)[0],10.4504878684))
test('FDM 풋 기본값',lambda:near(s8.black_scholes_pde_price(**PDE,option_type='put')[0],5.5735492183))
test('Projected American 풋',lambda:near(s8.projected_american_pde_price(**PDE),6.0884302604))
test('FDM 부적절한 Smax 거절',lambda:raises(lambda:s8.black_scholes_pde_price(**{**PDE,'spot':40.,'s_max':50.})))
test('FDM 소수격자 거절',lambda:raises(lambda:s8.black_scholes_pde_price(**{**PDE,'n_space':400.7})))
LV={k:v for k,v in PDE.items() if k!='volatility'}
test('Local vol FDM NaN 거절',lambda:raises(lambda:s8.local_volatility_pde_price(**LV,local_volatility=lambda t,s:np.full_like(s,np.nan))))
test('Local vol FDM 상수 sigma 동등성',lambda:near(s8.local_volatility_pde_price(**LV,local_volatility=lambda t,s:.2)[0],s8.black_scholes_pde_price(**PDE)[0]))
def cevgreeks():
 p,g,v=s9.cev_pde_solution(**BS); gg=np.array([s9.pde_grid_delta_gamma(g,v,j) for j in range(1,g.size-1)])
 gamma=float(np.interp(100.,g[1:-1],gg[:,1]));near(gamma,.0189609562)
 return {'price':p,'delta':float(np.interp(100.,g[1:-1],gg[:,0])),'gamma':gamma}
test('CEV 격자 Gamma',cevgreeks)
def greekcompare():
 a=s9.black_scholes_greeks(**BS);b=s9.finite_difference_greeks(s9.black_scholes_price,**BS)
 assert all(abs(a[k]-b[k])<1e-5 for k in a)
 return {'analytic':a,'numerical':b}
test('BS 해석적/수치 Greeks 비교',greekcompare)
test('zero bump 거절',lambda:raises(lambda:s9.finite_difference_greeks(s9.black_scholes_price,**BS,spot_bump=0.)))
def hedge():
 _,x=s7.simulate_gbm_paths(100.,1.,.05,.2,1,1008,2026);out={}
 for count in [12,48,252,1008]:
  path=x[:,0][::1008//count];assert path[0]==100. and path[-1]==x[-1,0]
  out[count]=s9.delta_hedge_short_call(path,100.,1.,.05,.2)['hedging_error']
 return out
test('동일 원경로 hedge frequency',hedge)
def els():
 r=s10.price_stepdown_els(terms,market,30000,360,2026);near(r['price'],96.0933921036);near(r['maturity_probability'],.1775);near(sum(r['payment_probabilities']),1.)
 (ROOT/'_validation').mkdir(exist_ok=True)
 (ROOT/'_validation/els_results.json').write_text(json.dumps(jsonval(r),ensure_ascii=False,indent=2))
 return r
test('ELS 가격·만기상환확률·총확률',els)
def els_seed():
 return {seed:s10.stepdown_els_greeks(terms,market,10000,360,seed) for seed in [2026,2027,2028]}
test('ELS Gamma 시드별 불안정성',els_seed)
test('ELS fractional n_paths 거절',lambda:raises(lambda:s10.price_stepdown_els(terms,market,100.5,360,2026)))
test('ELS zero bump 거절',lambda:raises(lambda:s10.stepdown_els_greeks(terms,market,100,360,2026,spot_bump=0.)))
def copytest():
 old={**market,'spots':[100,100]};c=s10.copy_market(old);c['spots'][0]+=.5;assert old['spots']==[100,100];return c['spots']
test('시장입력 복사와 실수형 bump',copytest)
params=(.03,1.5,.04,.02,1.)
test('OU CPU MC 및 해석식',lambda:{'cpu':cuda.bond_mc_cpu(params,10000,252,2026),'exact':cuda.bond_exact(params)})
test('OU n_paths=1 거절',lambda:raises(lambda:cuda.bond_mc_cpu(params,1,252,2026)))
test('OU kappa=0 거절',lambda:raises(lambda:cuda.bond_exact((.03,0.,.04,.02,1.))))
# New-kernel dependency test: no module-level examples are imported.
sys.path.insert(0,str(ROOT));import pricing_models as shared
test('새 커널 공통 모듈 로딩',lambda:{'paths_shape':shared.simulate_gbm_paths(100.,1.,.05,.2,1,4,2026)[1].shape,'fdm':shared.black_scholes_pde_price(**PDE)[0]})
print('PASSED',sum(r['status']=='PASS' for r in results),'of',len(results),flush=True)

if any(item['status'] != 'PASS' for item in results):
    raise SystemExit(1)
