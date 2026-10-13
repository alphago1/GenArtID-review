"""Render the measured Fig.3 coordinates; no model or dataset access."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
points=json.loads((ROOT/'fig3_points.json').read_text(encoding='utf-8'))['points']
models=[('command_r7b','D','Command R7B (API)'),('qwen25_7b_instruct','s','Qwen2.5-7B (API)'),
        ('glm_z1_9b_0414','^','GLM-Z1-9B (API)$^{\\dagger}$'),('glm0414_base','o','GLM Base'),
        ('glm0414_ssl','*','GLM SSL (mean)')]
colors={'civil':'#2468A0','criminal_law':'#C46A21'}
plt.rcParams.update({'font.family':'serif','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(4.2,4.7));ax=fig.add_axes([.16,.39,.81,.59])
ax.set(xlim=(-2,100),ylim=(0,.85),xlabel='Effective memory rate (%)',ylabel='Macro Recall@5')
ax.set_xticks(range(0,101,20));ax.set_yticks([0,.2,.4,.6,.8]);ax.grid(color='#e9e9e9');ax.set_axisbelow(True)
for d,color in colors.items():
    ps={p['model_run_key']:p for p in points if p['domain']==d}
    b,s=ps['glm0414_base'],ps['glm0414_ssl']
    ax.annotate('',xy=(s['m_mean']*100,s['r_mean']),xytext=(b['m_mean']*100,b['r_mean']),arrowprops={'arrowstyle':'->','color':color,'linestyle':'--'})
    for key,mark,label in models:
        p=ps[key];ax.plot(p['m_mean']*100,p['r_mean'],linestyle='none',marker=mark,color=color,
            markersize=11 if key=='glm0414_ssl' else 6.8,markerfacecolor='white' if key=='glm0414_base' else color)
handles=[Line2D([],[],linestyle='none',marker=m,color='#333333',label=l,markersize=8 if m=='*' else 5.7,
                markerfacecolor='white' if m=='o' else '#333333') for k,m,l in models]
handles += [Line2D([],[],color=colors[d],label=l) for d,l in [('civil','Civil Code'),('criminal_law','Criminal Law')]]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.285),ncol=2,frameon=False,fontsize=8.7)
fig.text(.54,.027,'$^{\\dagger}$Native reasoning; budget 1,024 tokens.\nOnly final answers are scored.',ha='center',va='bottom',fontsize=8.5)
fig.savefig(ROOT/'fig3.pdf',bbox_inches='tight',pad_inches=.04);fig.savefig(ROOT/'fig3.svg',bbox_inches='tight',pad_inches=.04)
print('Rendered 10 measured points; no experiment rerun')
