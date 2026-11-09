from pathlib import Path
import re,ast,json
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '_validation' / 'extracted_models'
OUT.mkdir(parents=True, exist_ok=True)
HEADER='''"""Functions extracted from the revised lecture notes; examples are not executed."""\nimport math\nimport numpy as np\nimport pandas as pd\nfrom scipy.stats import norm, binom\nfrom scipy.integrate import quad\nfrom scipy.interpolate import PchipInterpolator\nfrom scipy.optimize import brentq\nfrom scipy.sparse import diags\nfrom scipy.sparse.linalg import splu\n\n'''
summary=[];syntax=[];functions={}
for p in sorted(ROOT.glob('*.tex')):
 text=p.read_text(); defs=[];items=[];num=0
 for m in re.finditer(r'\\begin\{lstlisting\}(?:\[([^\n]*)\])?\n(.*?)\\end\{lstlisting\}',text,re.S):
  if 'style=pythonstyle' not in (m[1] or ''):continue
  num+=1; code=m[2];line=text.count('\n',0,m.start())+2;style=m[1]
  if code.lstrip().startswith('%'):
   syntax.append({'file':p.name,'line':line,'status':'Jupyter magic','caption':style});continue
  try:tree=ast.parse(code)
  except SyntaxError as e:
   syntax.append({'file':p.name,'line':line,'status':'ERROR','error':str(e)});continue
  syntax.append({'file':p.name,'line':line,'status':'PASS','caption':style})
  for node in tree.body:
   if isinstance(node,ast.FunctionDef):
    source=ast.get_source_segment(code,node);defs.append(source);items.append(node.name);functions[(p.name,node.name)]=source
 if defs:
  (OUT/(p.stem+'.py')).write_text(HEADER+'\n\n\n'.join(defs)+'\n')
 summary.append({'file':p.name,'code_blocks':num,'functions':items})
# Reusable helper module: only source functions and their existing dependencies.
exports=[('section05_numpy_scipy_functions.tex',['black_scholes_call_price']),('section07_monte_carlo_option_pricing.tex',['option_payoff','simulate_gbm_paths','simulate_gbm_terminal','monte_carlo_european_option','monte_carlo_asian_call','monte_carlo_cev_option','monte_carlo_heston_option']),('section08_pde_option_pricing.tex',['option_payoff_grid','european_boundary_values','black_scholes_pde_price','projected_american_pde_price','local_volatility_pde_price','heston_spatial_operator'])]
parts=[HEADER]
for fname,names in exports:
 for name in names:parts.append('# Source: '+fname+' / '+name+'\n'+functions[(fname,name)])
(ROOT/'pricing_models.py').write_text('\n\n'.join(parts)+'\n')
ast.parse((ROOT/'pricing_models.py').read_text())
(ROOT / '_validation' / 'syntax_results.json').write_text(json.dumps({'files':summary,'blocks':syntax},ensure_ascii=False,indent=2))
print('blocks',len(syntax),'functions',len(functions),'errors',sum(x['status']=='ERROR' for x in syntax))
