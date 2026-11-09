"""Regenerate the two one-asset figures using the explicitly documented model.
Run from the project root: python scripts/regenerate_section10_figures.py
The separate two-asset payment-probability chart is already updated in the archive.
"""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from section10_figure_model import calculate_and_save


def setup_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ['NanumSquare','Malgun Gothic','AppleGothic','Noto Sans CJK KR']:
        if name in installed:
            plt.rcParams['font.family']=name
            break
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['pdf.fonttype']=42


def stack_pngs(paths,target):
    images=[Image.open(p).convert('RGB') for p in paths]
    width=max(im.width for im in images)
    resized=[im.resize((width,round(im.height*width/im.width))) for im in images]
    gap=24;height=sum(im.height for im in resized)+gap*(len(resized)-1)
    canvas=Image.new('RGB',(width,height),'white')
    y=0
    for im in resized:canvas.paste(im,(0,y));y+=im.height+gap
    canvas.save(target.with_suffix('.png'),dpi=(300,300))
    canvas.save(target.with_suffix('.pdf'),'PDF',resolution=300.)


def main(recalculate=True):
    setup_font()
    dst=ROOT/'figures/section10';dst.mkdir(parents=True,exist_ok=True)
    if recalculate:
        cfg,curves,rows,meta=calculate_and_save(ROOT)
    else:
        meta=json.loads((ROOT/'_validation/els_figure_metadata.json').read_text())
        cfg=meta['settings']
        curves=np.loadtxt(ROOT/'_validation/els_state_curves.csv',delimiter=',',skiprows=1)
        rows=np.loadtxt(ROOT/'_validation/els_hedge_trace.csv',delimiter=',',skiprows=1)
    individual=[]
    def finish(fig,ax,filename,title,xlabel,ylabel,legend=False):
        ax.set_title(title,pad=9);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
        ax.set_axisbelow(True);ax.grid(True,alpha=.3)
        if legend:ax.legend(fontsize=9)
        path=dst/filename
        fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight')
        fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
        plt.close(fig)
        individual.append(path.with_suffix('.png'))
    time=rows[:,0]/cfg['days_per_year']
    fig,ax=plt.subplots(figsize=(8.2,3.2),layout='constrained')
    ax.plot(time,rows[:,1]/cfg['reference'],label='모의 기초자산 성과')
    ax.axhline(cfg['knock_in_level']/cfg['reference'],linestyle='--',label='일별 knock-in 기준')
    od=np.asarray(cfg['observation_days']);levels=np.asarray(cfg['autocall_levels'])
    m=od<=rows[-1,0]
    ax.scatter(od[m]/cfg['days_per_year'],levels[m]/cfg['reference'],marker='D',label='조기상환 기준',zorder=3)
    finish(fig,ax,'els_sample_path','별도의 1자산 ELS: 모의 주가경로','경과시간 (년)',r'$S_t/S_{\mathrm{ref}}$',True)
    fig,ax=plt.subplots(figsize=(8.2,3.2),layout='constrained')
    ax.plot(time,rows[:,4],label='종가 후 목표 Delta (상환 후 0)')
    ax.step(time,rows[:,5],where='pre',linestyle='--',label='직전 종가에서 정한 보유량')
    finish(fig,ax,'els_delta_and_hedge_holdings','일별 Delta와 재조정 전 주식 보유량','경과시간 (년)','주식 수량',True)
    fig,ax=plt.subplots(figsize=(8.2,3.2),layout='constrained')
    ax.plot(time,rows[:,8]);ax.axhline(0.,linestyle=':',linewidth=1.)
    finish(fig,ax,'els_cumulative_hedge_pnl','자기금융 헤지 평가손익 (상환일에는 최종 청산손익)','경과시간 (년)','헤지 자산 - ELS 부채')
    stack_pngs(individual,dst/'els_delta_hedge_path_and_pnl')
    individual=[]
    stock=curves[:,0];valid=(stock>=40.)&(stock<=160.)
    no_ki=valid&(stock>cfg['knock_in_level']+2.*cfg['s_max']/cfg['n_space'])
    for no_col,ki_col,quantity,filename in [(1,2,'가격','els_fdm_price_curve'),(3,4,'Delta','els_fdm_delta_curve'),(5,6,'Gamma','els_fdm_gamma_curve')]:
        fig,ax=plt.subplots(figsize=(8.2,3.2),layout='constrained')
        ax.plot(stock[no_ki],curves[no_ki,no_col],label='knock-in 미발생 상태')
        ax.plot(stock[valid],curves[valid,ki_col],linestyle='--',label='knock-in 발생 상태')
        ax.axvline(60.,linestyle=':',linewidth=1.)
        ax.axvline(90.,linestyle='-.',linewidth=1.)
        if quantity!='가격':ax.axhline(0.,linestyle=':',linewidth=.8)
        ax.set_xlim(40.,160.)
        title=f'상태별 ELS {quantity} (t = {meta["curve_time"]:.3f}년, 당일 관찰 직후)'
        finish(fig,ax,filename,title,'현재 기초자산 가격',quantity,True)
    stack_pngs(individual,dst/'els_fdm_price_delta_gamma')
    print(f'1자산 초기 모형가격: {meta["initial_price"]:.6f}')
    print(f'선택 경로의 상환시점: {meta["redemption_time"]:.2f}년')
    print(f'선택 경로의 최종 자기금융 헤지손익: {meta["final_pnl"]:.6f}')
    print('그림 저장:',dst)
    return meta

if __name__=='__main__':main()
