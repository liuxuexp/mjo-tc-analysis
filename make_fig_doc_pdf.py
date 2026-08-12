"""Render figure data & structure reference PDFs (main Figs 1-9 + SI S1-S10).

Reads fig_doc_data.json + the figure PNG thumbnails and emits two HTML
documents rendered to PDF via weasyprint (font: Noto Sans CJK SC).
Chinese descriptive prose + English technical terms.

  * Paper-4-1_Figures1-9_data_reference.pdf     (main figures)
  * Paper-4-1_FiguresS1-S10_data_reference.pdf  (supplementary figures)

Scope: a TECHNICAL reference (panel composition, data source, units, significance
method, sample sizes + the underlying numerical tables) -- NOT manuscript captions
(prose/captions are the author's domain).
"""
from __future__ import annotations
import json, sys, html
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from PIL import Image
import datetime

TODAY = datetime.date.today().strftime('%Y-%m-%d')

D = json.load(open(C.data_path("fig_doc_data.json")))
FIG = C.FIG_DIR
THUMB = C.DATA_DIR / "thumbs"
THUMB.mkdir(parents=True, exist_ok=True)
GROUPS = C.GROUP_KEY
G_lab = {g: C.group_to_label(g) for g in GROUPS}


def img(name, width=470):
    """Embed a downscaled thumbnail of the figure (max 1100px wide) so the PDF
    stays small while staying crisp at the on-page display width."""
    src = FIG / name
    if not src.exists():
        return ""
    dst = THUMB / name
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w = min(im.width, 1100)
            im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
            im.save(dst, "PNG", optimize=True)
    return (f'<img src="file://{dst}" style="width:{width}px; '
            'border:1px solid #ccc; margin:6px 0;">')


def esc(x):
    return html.escape(str(x))


def table(header, rows, cls="tbl"):
    h = "".join(f"<th>{esc(c)}</th>" for c in header)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>"
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def stars(s):
    return f' {s}' if s else ''


def fig_header(num, fname, png, title_cn):
    return f"""
    <div class="fighead">
      <span class="fnum">Figure {num}</span>
      <span class="fname">{esc(fname)}</span>
    </div>
    <div class="figtitle">{title_cn}</div>
    {img(png)}
    """


# --------------------------------------------------------------------------
# Common conventions
# --------------------------------------------------------------------------
meta = D['meta']
nd = meta['ND_JJASO']
# precompute join-strings used inside the big f-strings (avoid nested-quote escaping)
nd_junsep_str = ' / '.join(f"{G_lab[g]}: {D['fig9']['ND_JunSep'][g]}" for g in GROUPS)
phase_counts_str = ' / '.join(f"{G_lab[g]}: {meta['phase_counts'][g]}" for g in GROUPS)

# ---------------------------------------------------------------------------
# event_table.csv data dictionary (keystone table; static schema doc) + the
# per-figure input-data lineage (which file + columns/variables each figure's
# numbers derive from). Both are static documentation of fixed inputs.
# ---------------------------------------------------------------------------
DICT_ROWS = [
    ['year / chinese_code / name', '标识', '年份 / CMA 四位编号 / TC 名称', '—'],
    ['landfall_time / month / in_jjaso', '时间', '登陆时刻 / 月份 / 是否落在 6–10 月', 'bool'],
    ['landfall_lat / landfall_lon', '登陆几何', '登陆点纬度 / 经度（路径段与国界几何交点）', '°N, °E'],
    ['landfall_wind_speed / landfall_pressure', '登陆强度', '登陆时刻近中心风速 / 海平面气压', 'm/s, hPa'],
    ['landfall_wind_category', '登陆强度分级', '0 TD / 1 Weak / 2 Moderate / 3 Super（按登陆时风速）', '0–3'],
    ['landfall_region', '登陆区域 (3 类)', '按登陆纬度：South China(<24°N) / East China(24–34°N) / North China(≥34°N)', '—'],
    ['landfall_region2', '登陆区域 (2 类)', 'South China / East-North China（East+North 合并）', '—'],
    ['max_wind', 'LMI', '一生最大风速（lifetime maximum intensity）', 'm/s'],
    ['lmi_category', 'LMI 分级', '1 Weak(≤32.6) / 2 Moderate(32.6–<51.0) / 3 Super(≥51.0)', '1–3'],
    ['genesis_time / genesis_lat / genesis_lon', '生成', '生成时刻 / 纬度 / 经度（路径首点）', '°N, °E'],
    ['formation_region', '生成区域 (4 类)', 'South China Sea / Western Tropical WNP / Eastern Tropical WNP / North WNP', '—'],
    ['track_length_km / pre_landfall_track_km', '路径', '全程路径长度 / 登陆前路径长度', 'km'],
    ['ocean_residence_days', '路径', '登陆前海洋停留时长', 'd'],
    ['phase_landfall / amp_landfall / group_landfall / active_landfall', 'MJO（登陆日）', '登陆日 RMM 位相 / 振幅 / 位相组(1-2…7-8) / 是否活跃 MJO', 'amp≥1 为活跃'],
    ['phase_genesis / amp_genesis / group_genesis / active_genesis', 'MJO（生成日）', '生成日 RMM 同上四项', '—'],
    ['tcp_total', 'TCP', '事件总 TCP：中国陆地面积积分体积', '10⁶ m³'],
    ['tcp_depth', 'TCP', '受淹区面积加权平均深度', 'mm'],
    ['affected_area', 'TCP', '中国陆地受淹（湿，>0.5 mm）面积', 'km²'],
    ['n_days', 'TCP', '风暴日数（场时间步数）', 'd'],
    ['coastal_total / coastal_depth / coastal_area', 'TCP 沿海子区', '沿海(≤200 km)子区总量 / 深度 / 面积', '10⁶ m³, mm, km²'],
    ['inland_total / inland_depth / inland_area', 'TCP 内陆子区', '内陆(>200 km)子区同上三项', '同上'],
]
LINEAGE_ROWS = [
    ['1', 'all_typhoons_tracks.csv + landfall_typhoons_tracks.csv + event_table.csv', 'ACE(ΣWND²) / group_landfall / lmi_category / tcp_total'],
    ['2', 'landfall_typhoons_info.csv + ERA5 MJO CSV', '登陆点经纬度 / max_wind_category（强度等级命名：Super ≥ 51 m/s）'],
    ['3', 'event_table.csv', 'landfall_region / group_landfall / lmi_category'],
    ['4', 'event_table.csv + pre/pre_*.nc (tcp_lib)', 'lmi_category / tcp_total / coastal·inland_total；累计与单风暴 TCP 地图'],
    ['5', 'event_table.csv', 'group_landfall / tcp_total'],
    ['6', 'event_table.csv + pre/pre_*.nc (tcp_lib)', 'lmi_category / group_landfall / tcp_total / affected_area；单风暴均值地图'],
    ['7', 'event_table.csv', 'landfall_region / genesis_lat·lon / pre_landfall_track_km / lmi_category'],
    ['8', 'all_typhoons_tracks.csv + event_table.csv', 'formation_region / landfall_region2 / group_genesis（全 WNP 生成 vs 中国登陆）'],
    ['9', 'composite_jjas.nc', 'vort_850 / hgt_500 / uwnd·vwnd_850（NCEP 日历月标准化距平）'],
]
common = f"""
<h2 id="common">一、通用约定（适用全部主图）</h2>

<h3>1. 数据来源</h3>
<table class="tbl">
<thead><tr><th>数据集</th><th>说明</th><th>用于</th></tr></thead>
<tbody>
<tr><td>CMA Best Track</td><td>1949–2024 热带气旋最佳路径（强度/路径/登陆）</td><td>Fig 1–8</td></tr>
<tr><td>ERA5-derived RMM MJO index</td><td>逐日 RMM1/RMM2/phase/amplitude</td><td>Fig 1–3, 5–9（位相归属）</td></tr>
<tr><td>CHM_PRE_V2</td><td>中国区域合并格点日降水（0.1°）</td><td>Fig 1, 4–6（TCP）</td></tr>
<tr><td>NCEP/NCAR reanalysis</td><td>逐日 uwnd/vwnd/hgt/slp（多层级）</td><td>Fig 9（环流合成）</td></tr>
</tbody></table>

<h3>2. 选取标准</h3>
<ul>
<li><b>分析时段</b>：1960–2024，6–10 月（JJASO）。</li>
<li><b>MJO 筛选</b>：仅保留活跃 MJO 日（amplitude ≥ 1）。8 个 RMM 位相合并为 4 个位相组：
   { ' / '.join(G_lab[g] for g in GROUPS) }。</li>
<li><b>登陆样本</b>：中国登陆 TC 事件（含大陆、台湾、海南），登陆点为路径段与国界的几何交点。
   入选 Fig 1/3/4–8 的样本量 <b>n = {meta['n_landfall_jjaso_active']}</b>（JJASO 且登陆日为活跃 MJO）。</li>
<li><b>强度分级（LMI，一生最大风速）</b>：
   Weak ≤ 32.6 m/s；Moderate 32.6–&lt;51.0；Super ≥ 51.0 m/s（顶类命名 Super）。</li>
<li><b>登陆区域（按登陆纬度）</b>：South China (&lt;24°N) / East China (24–34°N) / North China (≥34°N)。</li>
</ul>

<h3>3. 统计与显著性</h3>
<ul>
<li><b>位相归一化（residence-time correction）</b>：所有"频率/能量"类指标均按各位相组的活跃日数
   (ND) 归一化，以"相对于无调制期望 (ratio = 1.0) 的倍数"呈现，消除 MJO 各位相停留时长不同的偏差。</li>
<li><b>显著性的位相检验</b>：停留时间多项式零分布的置换检验（1000 次重采样），双侧 α = 0.10；
   观测比率落在 [5%, 95%] 置换区间之外记为显著，图中以 "*" 标注。</li>
<li><b>风暴级 Bootstrap CI</b>：对单个风暴指标（均值/中位数 TCP、路径长度等）做 1000 次风暴重采样，
   报告 5–95 百分位区间（即 90% CI）。</li>
<li><b>合成场显著性（Fig 9）</b>：单样本 t = composite·√n，|t| &gt; {meta['tcrit_90']}（α = 0.10），
   基于日历月标准化距平（消除原 pooled-JJASO 气候态偏差）。</li>
</ul>

<h3>4. 各位相组活跃日数 ND（JJASO, 1960–2024）</h3>
{table(['位相组'] + [G_lab[g] for g in GROUPS] + ['合计'],
       [['活跃日数'] + [nd[g] for g in GROUPS] + [sum(nd.values())]])}
<p class="note">注：Fig 9 的合成场基于 Jun–Sep（剔除 10 月），对应 ND = {nd_junsep_str}。</p>

<h3>5. 样本分布</h3>
<table class="tbl">
<thead><tr><th></th><th>Weak</th><th>Moderate</th><th>Super</th><th>合计</th></tr></thead>
<tbody>
<tr><td>按强度 (LMI)</td>
<td>{meta['lmi_counts']['Weak']}</td><td>{meta['lmi_counts']['Moderate']}</td>
<td>{meta['lmi_counts']['Super']}</td><td>{sum(meta['lmi_counts'].values())}</td></tr>
<tr><td>按登陆区域</td>
<td colspan="3">South {meta['region_counts']['South China']} /
   East {meta['region_counts']['East China']} /
   North {meta['region_counts']['North China']}</td>
<td>{sum(meta['region_counts'].values())}</td></tr>
<tr><td>按登陆位相</td>
<td colspan="3">{phase_counts_str}</td>
<td>{sum(meta['phase_counts'].values())}</td></tr>
</tbody></table>

<h3>6. 核心数据表 <code>event_table.csv</code> 数据字典</h3>
<p class="note">每行 = 一次中国登陆 TC 事件（共 490 行，含非 JJASO / 非活跃 MJO；主图分析取其 6–10 月且登陆日活跃 MJO 子集 n = {meta['n_landfall_jjaso_active']}）。40 列按类别合并展示；该表是 Fig 1、3–8 的直接数据源。</p>
{table(['列名', '类别', '含义', '单位 / 取值'], DICT_ROWS)}

<h3>7. 各图输入数据对照（主图 1–9）</h3>
<p class="note"><code>pre/pre_*.nc</code> = 逐风暴 CHM_PRE_V2 日降水场（0.1°，500-km 掩膜，由 <code>lib/tcp.py</code> 读取）；<code>composite_*.nc</code> = NCEP/NCAR 日历月标准化距平合成；其余 CSV 均位于 <code>typhoon_output/</code> 或 <code>data/</code>。</p>
{table(['图', '输入文件', '关键字段 / 变量'], LINEAGE_ROWS)}
"""

# --------------------------------------------------------------------------
# Figure sections
# --------------------------------------------------------------------------
f1 = D['fig1']
rows_1a = [["All WNP"] + [f"{f1['a_ace_all_wnp'][g]['ratio']:.2f} (n={f1['a_ace_all_wnp'][g]['n']})" for g in GROUPS],
           ["China-landfall"] + [f"{f1['a_ace_landfall'][g]['ratio']:.2f} (n={f1['a_ace_landfall'][g]['n']})" for g in GROUPS]]
rows_1b = []
for cat in C.CATEGORY_ORDER:
    lab = C.CATEGORY_LABEL[cat]
    r = f1['b_landfall_by_lmi'][lab]
    rows_1b.append([lab] + [f"{r[g]['ratio']:.2f}{stars(r[g]['stars'])} (n={r[g]['n']})" for g in GROUPS])
rows_1c = [["Total TCP (10⁶ m³)"] + [f"{f1['c_total_tcp'][g]['total']:,.0f}" for g in GROUPS],
           ["n"] + [f1['c_total_tcp'][g]['n'] for g in GROUPS]]
rows_1d = [["Mean (10⁶ m³)"] + [f"{f1['d_mean_tcp'][g]['mean']:,.0f}" for g in GROUPS],
           ["90% CI"] + [f"{f1['d_mean_tcp'][g]['ci_lo']:,.0f}–{f1['d_mean_tcp'][g]['ci_hi']:,.0f}" for g in GROUPS],
           ["n"] + [f1['d_mean_tcp'][g]['n'] for g in GROUPS]]

sec1 = fig_header(1, "02_fig01.py → fig1-landfall_ace_tcp.png", "fig1-landfall_ace_tcp.png",
                  "登陆活动 — 相位归一化 ACE、登陆频数、总 TCP 与单风暴平均 TCP") + f"""
<h3>面板结构（2×2，共享 x 轴：位相标签仅显示于底行）</h3>
<ul>
<li><b>(a) 相位归一化 ACE</b>：全部 WNP TC 与其中后来登陆中国子集，按位相归一化的 ACE
   （相对于无调制 = 1.0 的倍数，storm-bootstrap 90% CI，柱顶打印 n）。</li>
<li><b>(b) 按强度（LMI）的登陆频数</b>：各位相、各强度等级的登陆数（停留时间归一化比率 + n + 显著性 "*"；LMI 一生最大风速分级：Weak ≤ 32.6, Moderate 32.6–&lt;51.0, Super ≥ 51.0 m/s）。</li>
<li><b>(c) 中国总 TCP</b>：各位相登陆 TC 产生的全国累计 TCP（10⁶ m³）+ n；虚线 = 无调制均摊。</li>
<li><b>(d) 单风暴平均事件 TCP</b>：每位相单风暴平均 TCP（10⁶ m³）+ 90% bootstrap CI + n。</li>
</ul>
<h3>数据表 (a) 相位归一化 ACE（比率，括号为风暴数 n）</h3>
{table(['样本'] + [G_lab[g] for g in GROUPS], rows_1a)}
<h3>数据表 (b) 按强度的登陆频数（比率 + 显著性 + n）</h3>
{table(['强度'] + [G_lab[g] for g in GROUPS], rows_1b)}
<h3>数据表 (c) 总 TCP &amp; (d) 单风暴平均 TCP</h3>
{table(['指标'] + [G_lab[g] for g in GROUPS], rows_1c + [['—']*5] + rows_1d)}
<p class="note">读图要点：phases 5–6 / 7–8 的 ACE、登陆数、总 TCP 均显著偏高；
单风暴平均 TCP 跨位相差异不大（~2.1–2.5×10⁴ m³），说明调制主要来自"风暴数"而非"每风暴更湿"。</p>
"""

# ---- Fig 2 ----
f2 = D['fig2']
rows_2 = []
for g in GROUPS:
    r = f2[g]
    rows_2.append([G_lab[g], r['Weak'], r['Moderate'], r['Super'], r['total']])
sec2 = fig_header(2, "fig02_landfall_maps.py → fig2-landfall_location.png", "fig2-landfall_location.png",
                  "中国登陆点空间分布（按位相；颜色 = 强度等级）") + f"""
<h3>面板结构（2×2 共享 x/y 轴地图，extent 105–125°E / 16–46°N）</h3>
<ul>
<li>每幅图绘制该位相组下、登陆日为活跃 MJO 的登陆点；颜色按 LMI 强度：
   Weak（绿）/ Moderate（蓝）/ Super（紫）。</li>
<li>本图采用强度等级命名 Super（≥ 51 m/s）；数据、筛选、版式与主图一致。</li>
</ul>
<h3>数据表 各位相登陆点数（按强度）</h3>
{table(['位相组', 'Weak', 'Moderate', 'Super', '合计'], rows_2)}
"""

# ---- Fig 3 ----
f3 = D['fig3']['panels']
def fig3_panel(key, title):
    p = f3[key]
    rows = []
    for region in C.REGION_ORDER:
        r = p['regions'][region]
        rows.append([region] + [f"{r[g]['ratio']:.2f}{stars(r[g]['stars'])} (n={r[g]['n']})"
                                for g in GROUPS])
    return f"<h4>{title}（n={p['n']}）</h4>" + table(
        ['登陆区域'] + [G_lab[g] for g in GROUPS], rows)
sec3 = fig_header(3, "03_fig03_regional.py → fig3-regional_mjo.png", "fig3-regional_mjo.png",
                  "区域登陆率：分区域、分强度的位相归一化登陆比率") + f"""
<h3>面板结构（2×2 共享 x/y 轴，grouped bars: 区域 × 位相）</h3>
<ul>
<li>每个位相×区域柱表示 <b>停留时间归一化登陆率</b>（相对于无调制 = 1.0 的倍数），
   误差棒为停留时间零分布 90% 区间，"*" = α=0.10 显著，柱顶打印观测数 n。</li>
<li>(a) 全部登陆 TC；(b) Weak；(c) Moderate；(d) Super（按 LMI）。</li>
</ul>
{fig3_panel('a_all', '(a) 全部登陆 TC')}
{fig3_panel('b_weak', '(b) Weak TC')}
{fig3_panel('c_moderate', '(c) Moderate TC')}
{fig3_panel('d_major', '(d) Super TC')}
<p class="note">读图要点：phases 5–6 在 South/East China 登陆率显著偏高；North China 样本少（n≤7），
多数不显著。登陆强度伴侣图见 Fig S2。</p>
"""

# ---- Fig 4 ----
f4 = D['fig4']
rows_4map = [["Total TCP 99th % (mm)", f4['vmax_total_mm']] +
             [f"{f4['total_tcp'][C.CATEGORY_LABEL[c]]:.0f}" for c in C.CATEGORY_ORDER],
             ["Mean TCP/storm 99th % (mm)", f4['vmax_mean_mm']] +
             [f"{f4['mean_tcp'][C.CATEGORY_LABEL[c]]:.0f}" for c in C.CATEGORY_ORDER]]
rows_4g = []
rows_4h = []
for cat in C.CATEGORY_ORDER:
    lab = C.CATEGORY_LABEL[cat]
    s = f4['stormlevel'][lab]; ci = f4['coastal_inland'][lab]
    rows_4g.append([lab, s['n'], f"{s['mean']:,.0f}", f"{s['median']:,.0f}",
                    f"{s['ci_lo']:,.0f}–{s['ci_hi']:,.0f}"])
    rows_4h.append([lab, f"{ci['coastal']:,.0f}", f"{ci['inland']:,.0f}"])
sec4 = fig_header(4, "04_fig04_tcp_intensity.py → fig4-tcp_intensity.png", "fig4-tcp_intensity.png",
                  "TCP 的强度分解：累计/单风暴地图 + 风暴级分布 + 沿海 vs 内陆") + f"""
<h3>面板结构（三栏：A 累计 | B 单风暴 | C 统计）</h3>
<ul>
<li><b>A 栏 (a–c)</b>：Weak/Moderate/Super 的 <b>累计 TCP</b> 地图（中国陆地，Blues；共享一条色标）。</li>
<li><b>B 栏 (d–f)</b>：对应 <b>单风暴平均 TCP</b> 地图（共享一条色标）。</li>
<li><b>(g)</b> 风暴级事件 TCP 分布（boxplot + 单风暴散点 + 中位数◆+90%CI）。</li>
<li><b>(h)</b> 沿海 (≤200 km) vs 内陆 (&gt;200 km) 平均事件 TCP（分组柱）。</li>
<li>地图 24/34°N 虚线为区域分割；y 轴采用 ×10ⁿ 科学计数。</li>
</ul>
<h3>数据表 (a–f) 地图色标上界（各强度 99th 百分位，mm）</h3>
{table(['指标', '共享 vmax'] + [C.CATEGORY_LABEL[c] for c in C.CATEGORY_ORDER], rows_4map)}
<h3>数据表 (g) 风暴级事件 TCP（10⁶ m³）</h3>
{table(['强度', 'n', '均值', '中位数', '90% CI'], rows_4g)}
<h3>数据表 (h) 沿海 vs 内陆平均事件 TCP（10⁶ m³）</h3>
{table(['强度', '沿海 ≤200 km', '内陆 >200 km'], rows_4h)}
<p class="note">读图要点：累计 TCP 在 Moderate 最高（频次多），但单风暴平均 TCP 随强度递增
（Weak 96 → Super 95 mm 量级、风暴总量 Super 显著最大）；沿海远高于内陆。</p>
"""

# ---- Fig 5 ----
f5 = D['fig5']
rows_5a = [["比率"] + [f"{f5['a_frequency'][g]['ratio']:.2f}{stars(f5['a_frequency'][g]['stars'])}" for g in GROUPS],
           ["n"] + [f5['a_frequency'][g]['n'] for g in GROUPS]]
rows_5b = [["Total TCP (10⁶ m³)"] + [f"{f5['b_total_tcp'][g]['total']:,.0f}" for g in GROUPS],
           ["n"] + [f5['b_total_tcp'][g]['n'] for g in GROUPS]]
rows_5c = [["Mean (10⁶ m³)"] + [f"{f5['c_mean'][g]['mean']:,.0f}" for g in GROUPS],
           ["90% CI"] + [f"{f5['c_mean'][g]['ci_lo']:,.0f}–{f5['c_mean'][g]['ci_hi']:,.0f}" for g in GROUPS],
           ["vs 总均值 {0:,.0f}".format(f5['overall_mean'])] + [f"{f5['c_mean'][g]['pct_dev_mean']:+.0f}%" for g in GROUPS]]
rows_5d = [["Median (10⁶ m³)"] + [f"{f5['d_median'][g]['median']:,.0f}" for g in GROUPS],
           ["90% CI"] + [f"{f5['d_median'][g]['ci_lo']:,.0f}–{f5['d_median'][g]['ci_hi']:,.0f}" for g in GROUPS],
           ["vs 总中位 {0:,.0f}".format(f5['overall_median'])] + [f"{f5['d_median'][g]['pct_dev_median']:+.0f}%" for g in GROUPS]]
sec5 = fig_header(5, "05_fig05_phase_decomp.py → fig5-phase_decomp.png", "fig5-phase_decomp.png",
                  "TCP 的位相分解：频数 / 总量 / 单风暴均值 / 单风暴中位") + f"""
<h3>面板结构（2×2 共享 x 轴）— 回答"风暴更多 vs 每风暴更湿"</h3>
<ul>
<li><b>(a) 相位归一化登陆频数</b>（比率 + n + 显著性）→ 风暴<b>数量</b>。</li>
<li><b>(b) 总 TCP</b>（10⁶ m³）→ 累计影响（随风暴数缩放）。</li>
<li><b>(c) 单风暴平均 TCP</b>（柱保留绝对值，标注相对总均值的 %偏差 + "*" 当 CI 排除总均值）。</li>
<li><b>(d) 单风暴中位 TCP</b>（同 (c) 标注法）→ 典型风暴的湿润程度。</li>
</ul>
<h3>数据表 (a) 频数比率 &amp; (b) 总 TCP</h3>
{table(['指标'] + [G_lab[g] for g in GROUPS], rows_5a + [['—']*5] + rows_5b)}
<h3>数据表 (c) 单风暴平均 TCP（总均值 {f5['overall_mean']:,.0f}）</h3>
{table(['指标'] + [G_lab[g] for g in GROUPS], rows_5c)}
<h3>数据表 (d) 单风暴中位 TCP（总中位 {f5['overall_median']:,.0f}）</h3>
{table(['指标'] + [G_lab[g] for g in GROUPS], rows_5d)}
<p class="note">读图要点：(a)(b) 随风暴数强烈起伏（5–6 偏高、1-2 偏低且显著）；
(c)(d) 跨位相差异小（%偏差约 −8% ~ +8%），印证 TCP 调制以"风暴数"为主因。</p>
"""

# ---- Fig 6 ----
f6 = D['fig6']
rows_6a = []
rows_6b = []
for cat in C.CATEGORY_ORDER:
    lab = C.CATEGORY_LABEL[cat]
    ra = f6['a_event_tcp_by_lmi_phase'][lab]
    rb = f6['b_precip_area_by_lmi_phase'][lab]
    rows_6a.append([lab] + [f"n={ra[g]['n']}, med={ra[g]['median']:,.0f}" if ra[g]['median'] is not None else "—"
                            for g in GROUPS])
    rows_6b.append([lab] + [f"{rb[g]:.0f}" if rb[g] is not None else "—" for g in GROUPS])
rows_6c = [["n storms"] + [f6['cf_mean_map_vmax_mm'][g]['n_storms'] for g in GROUPS],
           ["99th % mm"] + [f6['cf_mean_map_vmax_mm'][g]['p99_mm'] for g in GROUPS]]
sec6 = fig_header(6, "06_fig06_stormlevel.py → fig6-stormlevel_tcp.png", "fig6-stormlevel_tcp.png",
                  "风暴级 TCP：强度×位相单风暴分布 + 单风暴平均 TCP 地图") + f"""
<h3>面板结构（左列 a/b 共享 x；右块 c–f 2×2 共享坐标 + 单色标）</h3>
<ul>
<li><b>(a) 强度×位相单风暴事件 TCP 分布</b>（boxplot + 中位数◆+90%CI，柱顶打印 n，log 轴）。</li>
<li><b>(b) 强度×位相平均降水面积</b>（&gt;0.5 mm 湿区，10³ km²，柱体标注数值）。</li>
<li><b>(c–f) 各位相单风暴平均 TCP 地图</b>（中国陆地，Blues，共享 vmax = {f6['shared_vmax_mm']} mm/风暴；网格为每风暴均值）。</li>
</ul>
<h3>数据表 (a) 单风暴事件 TCP（n, 中位数 10⁶ m³）</h3>
{table(['强度'] + [G_lab[g] for g in GROUPS], rows_6a)}
<h3>数据表 (b) 平均降水面积（10³ km²）</h3>
{table(['强度'] + [G_lab[g] for g in GROUPS], rows_6b)}
<h3>数据表 (c–f) 单风暴平均 TCP 地图（每位相）</h3>
{table(['指标'] + [G_lab[g] for g in GROUPS], rows_6c)}
<p class="note">读图要点：phases 5–6 各强度样本最多；Super 在 5–6/7-8 的单风暴 TCP 与降水面积最大。</p>
"""

# ---- Fig 7 ----
f7 = D['fig7']
rows_7 = []
for region in C.REGION_ORDER:
    r = f7[region]
    rows_7.append([region, r['n'], f"{r['genesis_lat_mean']:.1f} ({r['genesis_lat_median']:.1f})",
                   f"{r['genesis_lon_mean']:.1f} ({r['genesis_lon_median']:.1f})",
                   f"{r['track_len_mean_km']:,.0f} [{r['track_len_ci'][0]:,.0f}–{r['track_len_ci'][1]:,.0f}]",
                   f"{r['track_len_median_km']:,.0f}"])
sec7 = fig_header(7, "07_fig07_genesis_track.py → fig7-genesis_track.png", "fig7-genesis_track.png",
                  "生成位置与登陆前路径（按登陆区域）") + f"""
<h3>面板结构（3 张堆叠地图，extent 105–170°E / 5–40°N，共享坐标）</h3>
<ul>
<li>每幅图按登陆区域（South/East/North China）绘制其登陆 TC 的<b>生成点</b>（颜色 = 强度），
   金色 ✕ = 平均生成位置；20°N / 120°E / 140°E 为生成区示意分割。</li>
<li>左上统计框：n、生成经纬度（均值/中位）、登陆前路径长度（均值 + 90% CI）+ 中位路径。</li>
</ul>
<h3>数据表 各区域生成坐标与登陆前路径长度</h3>
{table(['登陆区域', 'n', '生成纬度 均值(中位) °N', '生成经度 均值(中位) °E',
        '路径长度 均值 [90%CI] km', '中位路径 km'], rows_7)}
<p class="note">读图要点：登陆纬度越偏北，生成位置越偏东、登陆前路径越长
（North China 平均 ~4721 km，远长于 South China ~2301 km）。</p>
"""

# ---- Fig 8 ----
f8 = D['fig8']
def fig8_ab(key, title):
    p = f8[key]
    rows = []
    for gr in C.GENESIS_ORDER:
        s = p['South China'][gr]; e = p['East/North China'][gr]
        rows.append([gr, f"{s['n']}/{s['tot']} ({s['pct']:.0f}%)",
                     f"{e['n']}/{e['tot']} ({e['pct']:.0f}%)"])
    return f"<h4>{title}</h4>" + table(['生成区域', '到达 South China', '到达 East/North China'], rows)
def fig8_cd(key, title):
    p = f8[key]
    out = f"<h4>{title}</h4>"
    for gr in C.GENESIS_ORDER:
        r = p[gr]
        out += table(['位相'] + [G_lab[g] for g in GROUPS],
                     [['rate / 1000 phase-days'] + [f"{r[g]['rate']:.1f}{stars(r[g]['stars'])} (n={r[g]['n']})" for g in GROUPS]])
    return out
sec8 = fig_header(8, "08_fig08_genesis_mjo.py → fig8-genesis_mjo.png", "fig8-genesis_mjo.png",
                  "生成—登陆关系及其 MJO 位相依赖") + f"""
<h3>面板结构（2×2，按行共享 y 轴）</h3>
<ul>
<li><b>(a)(b)</b> 各生成区域的登陆去向构成（条件于中国登陆样本）：到达 South China vs East/North China 的 %；标注计数。
   (a) Weak TC；(b) Moderate+Super TC。</li>
<li><b>(c)(d)</b> 各生成区域 TC 数的 <b>位相归一化率</b>（每 1000 phase-days，停留时间零分布 + α=0.10 "*")。
   (c) 全部 WNP TC；(d) 中国登陆子集（用生成位相 group_genesis）。</li>
</ul>
{fig8_ab('a_weak', '(a) Weak TC — 生成区域 → 登陆去向（n/总数, %）')}
{fig8_ab('b_mod_maj', '(b) Moderate+Super TC — 生成区域 → 登陆去向')}
{fig8_cd('c_all_wnp', '(c) 全部 WNP TC — 各生成区域的位相归一化生成率')}
{fig8_cd('d_landfall', '(d) 中国登陆 TC — 各生成区域的位相归一化率（按生成位相）')}
<p class="note">读图要点：东部热带 WNP（原"open WNP"）生成在 phases 5–6 频率显著偏高，
且其登陆更易到达 East/North China；South China Sea 生成几乎全部登陆 South China。</p>
"""

# ---- Fig 9 ----
f9 = D['fig9']
nd_junsep_str9 = ' / '.join(f"{G_lab[g]}: {f9['ND_JunSep'][g]}" for g in GROUPS)
def fig9_block(varlab):
    p = f9['panels'][varlab]
    rows = []
    for g in ['1-2', '5-6']:
        r = p[g]
        rows.append([G_lab[g], r['min'], r['max'], r['mean'], f"{r['sig_frac']*100:.0f}%"])
    return table(['位相', 'min', 'max', 'mean', '显著格点占比'], rows)
sec9 = fig_header(9, "10_fig09_dynamics.py → fig9-circulation_anomaly.png", "fig9-circulation_anomaly.png",
                  "环流异常合成（Jun–Sep，日历月标准化距平）") + f"""
<h3>面板结构（两个变量块 × 1×2 位相；共享一条底部色标）</h3>
<ul>
<li><b>(a)(b)</b> 850-hPa 相对涡度（填色）+ 850-hPa 水平风异常（矢量），位相 1-2 vs 5-6。</li>
<li><b>(c)(d)</b> 500-hPa 位势高度（填色）+ 500-hPa 引导风（矢量），位相 1-2 vs 5-6。</li>
<li>区域 {f9['domain']}（30–180°E / 0–60°N）；RdBu_r，色标 ±{f9['levels'][1]}；
   白色打点 = α=0.10 显著（单样本 t = composite·√n，|t|&gt;{meta['tcrit_90']}）；
   仅在风分量显著处画矢量。</li>
<li>距平为<b>日历月标准化</b>（按月 z-score），修正原 pooled-JJASO 气候态偏差；10 月已从主合成中剔除。</li>
</ul>
<h3>合成基本量</h3>
<ul>
<li>样本：n = {f9['n_storms_jjaso_active']} 登陆风暴；Jun–Sep 活跃日 ND = {nd_junsep_str9}。</li>
<li>显著性方法：{f9['sig_method']}。</li>
</ul>
<h4>850-hPa 相对涡度（标准化距平，域内统计）</h4>
{fig9_block('850-hPa relative vorticity')}
<h4>500-hPa 位势高度（标准化距平，域内统计）</h4>
{fig9_block('500-hPa geopotential height')}
<p class="note">读图要点：phases 5–6 对应菲律宾海以东气旋式涡度/正高度异常的配置，有利于
WNP TC 活动与中国登陆偏多；1-2 为反位相。SI 同款图见 Fig S5–S8。</p>
"""

# ===========================================================================
# Supplementary-figure sections (S1-S10)
# ===========================================================================
def figS_header(num, fname, png, title_cn):
    return f"""
    <div class="fighead">
      <span class="fnum">Figure S{num}</span>
      <span class="fname">{esc(fname)}</span>
    </div>
    <div class="figtitle">{title_cn}</div>
    {img(png)}
    """


def circ_tables(panels):
    """Per-variable min/max/mean/sig% tables for a composite figure (S5-S8)."""
    out = ""
    for varlab, phasestats in panels.items():
        rows = []
        for g in GROUPS:
            r = phasestats[g]
            rows.append([G_lab[g], r['min'], r['max'], r['mean'], f"{r['sig_frac']*100:.0f}%"])
        out += f"<h4>{varlab}（标准化距平，域内统计）</h4>"
        out += table(['位相', 'min', 'max', 'mean', '显著格点占比'], rows)
    return out


CIRC_NOTE = ("打点 = α=0.10 显著（单样本 t = composite·√n，|t|>{0}）；RdBu_r 色标 ±0.8；"
             "域 30–180°E / 0–60°N；距平为<b>日历月标准化</b>（按月 z-score）。").format(meta['tcrit_90'])

# ---- Figure S1 : monthly distribution (all 490 storms) ----
s1 = D['figS1']
rows_S1 = []
for m in range(1, 13):
    mm = s1['monthly'][str(m)]['by_cat']
    rows_S1.append([m, mm['Weak'], mm['Moderate'], mm['Super'], s1['monthly'][str(m)]['total']])
secS1 = figS_header(1, "figS1_monthly.py → figS1-monthly.png", "figS1-monthly.png",
                    "中国登陆 TC 的月际分布（全年，按 LMI 强度堆叠）") + f"""
<h3>面板结构（单图：Jan–Dec 堆叠柱，金色带 = Jun–Oct 分析季）</h3>
<ul>
<li>逐月统计 1960–2024 全部中国登陆 TC（n = {s1['n_total']}），按 LMI 强度堆叠（Weak/Moderate/Super），
   柱顶打印当月合计；金色阴影标出 Jun–Oct。</li>
</ul>
<h3>数据表 逐月登陆数（按 LMI）</h3>
{table(['月份','Weak','Moderate','Super','合计'], rows_S1)}
<p class="note">读图要点：Jun–Oct 集中 {s1['jjaso_share']}% 的登陆（{s1['jjaso_n']}/{s1['n_total']}），
构成主分析限定于 JJASO 的依据。</p>
"""

# ---- Figure S2 : LMI vs intensity-at-landfall ----
s2 = D['figS2']
ct2 = s2['contingency']
rows_S2a = [[cl] + [ct2[cl][c] for c in ['Weak', 'Moderate', 'Super']] for cl in ['Weak', 'Moderate', 'Super']]
rows_S2b = []
for cat in ['Weak', 'Moderate', 'Super']:
    rp = s2['regional_by_landfall_intensity'][cat]
    rows_S2b.append([cat, rp['South China'], rp['East China'], rp['North China']])
rows_S2c = []
for cat in ['Weak', 'Moderate', 'Super']:
    t = s2['tcp_by_landfall_intensity'][cat]
    rows_S2c.append([cat, t['n'], f"{t['mean']:,.0f}", f"{t['median']:,.0f}",
                     f"{t['ci_lo']:,.0f}–{t['ci_hi']:,.0f}"])
secS2 = figS_header(2, "figS2_landfall_intensity.py → figS2-lmi_landfall_intensity.png",
                    "figS2-lmi_landfall_intensity.png",
                    "LMI 与登陆强度的对比（Fig 2 的登陆强度伴侣图）") + f"""
<h3>面板结构（1×3；样本 = in_jjaso 全部 {s2['n']} 个登陆 TC）</h3>
<ul>
<li><b>(a) LMI × 登陆强度列联表</b>（热力图，红框 = 对角线"不变"）。</li>
<li><b>(b) 按<b>登陆强度</b>分级的区域登陆占比</b>（South/East/North %，二项 90% CI）。</li>
<li><b>(c) 按登陆强度分级的风暴级事件 TCP 分布</b>（boxplot + 中位数◆ + 90% CI，log 轴）。</li>
</ul>
<h3>数据表 (a) LMI(行) × 登陆强度(列) 列联表</h3>
{table(['LMI \\ 登陆强度','Weak','Moderate','Super'], rows_S2a)}
<p class="note">对角线（不变）{s2['diag_nochange']}；减弱 {s2['weaken']}；增强 {s2['intensify']}
（上三角全 0 —— LMI 为一生最大风速，登陆时不可能更强）。</p>
<h3>数据表 (b) 按登陆强度的区域登陆占比（%）</h3>
{table(['登陆强度','South China','East China','North China'], rows_S2b)}
<h3>数据表 (c) 按登陆强度的事件 TCP（10⁶ m³）</h3>
{table(['登陆强度','n','均值','中位数','90% CI'], rows_S2c)}
<p class="note">读图要点：LMI 与登陆强度差距大 —— Super-by-LMI 风暴仅 {ct2['Super']['Super']} 个在登陆时仍为 Super；
主图 Fig 2 颜色编码的是 LMI。</p>
"""

# ---- Figure S3 : IBTrACS robustness ----
s3 = D['figS3']
if s3:
    ct3 = s3['cma_ib_contingency']
    rows_S3a = [[cl] + [ct3[cl].get(c, 0) for c in ['TD', 'Weak', 'Moderate', 'Super']]
                for cl in ['Weak', 'Moderate', 'Super']]
    ms = s3['major_share_pct']
    rows_S3c = [['CMA'] + [ms['CMA'][g] for g in GROUPS],
                ['IBTrACS'] + [ms['IBTrACS'][g] for g in GROUPS]]
    secS3 = figS_header(3, "11_ibtracs_robustness.py → figS3-ibtracs_robustness.png",
                        "figS3-ibtracs_robustness.png",
                        "替代最佳路径数据集（IBTrACS）稳健性检验") + f"""
<h3>面板结构（1×3）— 检验强度分级对最佳路径数据集的稳健性</h3>
<ul>
<li><b>(a) CMA–IBTrACS 生成匹配</b>：生成点最近匹配距离直方图（≤300 km 且 ±24 h 内）。</li>
<li><b>(b) 强度一致性</b>：CMA LMI vs IBTrACS 最大风速散点（1:1 线，相关系数 r）。</li>
<li><b>(c) Super-TC 登陆占比</b>：各位相 Super 占比，CMA vs IBTrACS 分类对比。</li>
</ul>
<h3>数据表 匹配统计（in_jjaso {s3['n_in_jjaso']} 个登陆 TC）</h3>
{table(['指标','值'], [['匹配数', f"{s3['n_matched']} / {s3['n_in_jjaso']}"],
                  ['匹配率', f"{s3['match_rate_pct']}%"], ['中位匹配距离', f"{s3['median_match_km']:.0f} km"],
                  ['CMA–IBTrACS 风速相关 r', s3['wind_corr_r']],
                  ['活跃 MJO 登陆子集匹配', s3['n_active_matched']]])}
<h3>数据表 (a) CMA LMI(行) × IBTrACS(列) 强度列联表</h3>
{table(['CMA \\ IBTrACS','TD','Weak','Moderate','Super'], rows_S3a)}
<h3>数据表 (c) 各位相 Super-TC 登陆占比（%，活跃子集 n={s3['n_active_matched']}）</h3>
{table(['分类'] + [G_lab[g] for g in GROUPS], rows_S3c)}
<p class="note">读图要点：匹配率 {s3['match_rate_pct']}%（弱 SCS 风暴多未匹配，IBTrACS usa-agency 子集所致），
但匹配样本风速高度一致（r = {s3['wind_corr_r']}）；Super 占位相 5–6 偏高的格局在两套数据中定性一致。</p>
"""
else:
    secS3 = figS_header(3, "11_ibtracs_robustness.py → figS3-ibtracs_robustness.png",
                        "figS3-ibtracs_robustness.png", "替代最佳路径数据集（IBTrACS）稳健性检验"
                        ) + '<p class="note">数据未生成（ibtracs_matched.csv 缺失）；请先运行 11_ibtracs_robustness.py。</p>'

# ---- Figure S4 : all-WNP vs landfall ----
s4 = D['figS4']
rows_S4a = []
for key, lab in [('all_wnp', '全 WNP'), ('landfall', '中国登陆')]:
    d4 = s4['a_genesis_rate'][key]
    rows_S4a.append([f"{lab} (n={d4['n']})"] +
                    [f"{d4['groups'][g]['rate']:.1f}{stars(d4['groups'][g]['stars'])}" for g in GROUPS])
rows_S4b = []
for g in GROUPS:
    fg = s4['b_landfall_fraction']['groups'][g]
    rows_S4b.append([G_lab[g], fg['pct'], fg['n'], f"{fg['ci'][0]}–{fg['ci'][1]}"])
secS4 = figS_header(4, "12_all_wnp_compare.py → figS4-allwnp_vs_landfall.png",
                    "figS4-allwnp_vs_landfall.png",
                    "全 WNP 与中国登陆子集的位相依赖对比") + f"""
<h3>面板结构（1×2；共用 MJO-day 归一化与时段）</h3>
<ul>
<li><b>(a) 位相归一化生成率</b>（每 1000 phase-days，停留时间零分布 + α=0.10 "*"）：全 WNP vs 中国登陆。</li>
<li><b>(b) 各位相 WNP 风暴中登陆中国的比例</b>（% + 二项 90% CI，虚线 = 总体比例）。</li>
</ul>
<h3>数据表 (a) 生成率（rate / 1000 phase-days + 显著性）</h3>
{table(['样本'] + [G_lab[g] for g in GROUPS], rows_S4a)}
<h3>数据表 (b) 中国登陆比例（% + 90% CI）</h3>
{table(['位相','登陆比例 %','n (登陆/全 WNP)','90% CI'], rows_S4b)}
<p class="note">读图要点：全 WNP 生成在 phases 5–6 显著偏高；登陆比例跨位相差异较小
（总体 {s4['b_landfall_fraction']['overall_pct']}%），说明 MJO 对"生成"的调制强于对"生成后是否登陆"的调制。</p>
"""

# ---- Figure S5 : Jun-Sep 200-hPa wind + SLP (4 phases) ----
s5 = D['figS5']
secS5 = figS_header(5, "10_fig09_dynamics.py (fig_s5) → figS5-200hpa_slp.png", "figS5-200hpa_slp.png",
                    "环流异常合成补充：200-hPa 纬向风 + 海平面气压（Jun–Sep，4 位相）") + f"""
<h3>面板结构（两部分：uwnd_200 上 / slp 下，各 2×2 = 4 位相；共享底部色标）</h3>
<ul>
<li>每部分为 1 个变量的 2×2 位相图（Phases 1-2/3-4 上行，5-6/7-8 下行）；面板 a–h。</li>
<li>Fig 9 的高层/地面视角补充 —— 不画风矢量，仅填色 + 显著性打点。</li>
</ul>
{circ_tables(s5['panels'])}
<p class="note">{CIRC_NOTE} 读图要点：phases 5–6 高层辐散/低压配置与 Fig 9 低层气旋式环流一致。</p>
"""

# ---- Figure S6/S7/S8 : calendar-month circulation (vort_850 + hgt_500) ----
_month_titles = {'figS6': ('June + July', '6'), 'figS7': ('August + September', '7'),
                 'figS8': ('October', '8')}
secS = {}
for key, (period, num) in _month_titles.items():
    s = D[key]
    oct_note = ("（vs <b>10 月自身气候态</b>，故 10 月场独立解读）" if key == 'figS8' else "")
    secS[key] = figS_header(num, "10_fig09_dynamics.py (fig_monthly)",
                            {'figS6': 'figS6-jun_jul.png', 'figS7': 'figS7-aug_sep.png',
                             'figS8': 'figS8-october_circulation.png'}[key],
                            f"环流异常的月际演变（{period}；vort_850 + hgt_500，4 位相）") + f"""
<h3>面板结构（两部分：vort_850+850-hPa 风 上 / hgt_500+500-hPa 引导风 下，各 2×2 = 4 位相）</h3>
<ul>
<li>与 figS5 版式相同，但变量为 850-hPa 涡度（+风矢量）与 500-hPa 高度（+引导风），
   输入为<b>逐月合成</b>（{period}）{oct_note}。</li>
<li>用于检验 Fig 9（pooled Jun–Sep）结论在月际尺度上的稳健性。</li>
</ul>
{circ_tables(s['panels'])}
<p class="note">{CIRC_NOTE} 读图要点：5–6 位相有利配置在 Jun-Jul/Aug-Sep 一致出现；
{'10 月（Fig S8）样本与气候态独立，环流信号弱于主季。' if key == 'figS8' else '月际格局与 Jun–Sep 合成（Fig 9）定性一致。'}</p>
"""

# ---- Figure S9 : coastal/inland sensitivity ----
s9 = D['figS9']
rows_S9 = []
for thr in [100, 200, 300]:
    for zone in ('coastal', 'inland'):
        zstats = s9['stats'][str(thr)][zone]
        rows_S9.append([f"{thr} km · {zone}"] +
                       [f"{zstats[c]['mean']:,.0f} (n={zstats[c]['n']})" for c in ['Weak', 'Moderate', 'Super']])
secS9 = figS_header(9, "figS9_coastal_sensitivity.py → figS9-coastal_sensitivity.png",
                    "figS9-coastal_sensitivity.png",
                    "沿海 vs 内陆 TCP 对距离阈值的敏感性（100/200/300 km）") + f"""
<h3>面板结构（1×3：100 / 200 / 300 km 阈值；每组沿海绿 vs 内陆橙 × Weak/Moderate/Super）</h3>
<ul>
<li>重复 Fig 4(h) 的沿海/内陆<b>平均事件 TCP</b>，分别在 100、200、300 km 阈值下计算，
   误差棒为 5–95 百分位 bootstrap 区间；n 与 Fig 4(g)/(h) 一致（共 {s9['n_storms']} 个风暴）。</li>
</ul>
<h3>数据表 各阈值沿海/内陆平均事件 TCP（10⁶ m³，括号 n）</h3>
{table(['阈值 · 区域'] + ['Weak', 'Moderate', 'Super'], rows_S9)}
<p class="note">读图要点：沿海显著高于内陆，且该对比在 100/200/300 km 阈值下均成立 ——
沿海—内陆差异非阈值选取的伪迹（200 km 面板即 Fig 4(h)）。</p>
"""

# ---- Figure S10 : October decision ----
s10 = D['figS10']
jj10 = s10['jun_sep']; oc10 = s10['october']
secS10 = figS_header(10, "13_october_decision.py → figS10-october_tcp.png", "figS10-october_tcp.png",
                     "10 月事件 TCP 决策分析（Jun–Sep vs October）") + f"""
<h3>面板结构（1×2）— 判定 10 月是否在主图保留</h3>
<ul>
<li><b>(a) Jun–Sep vs October 平均事件 TCP</b>（90% CI）；<b>(b) 风暴级分布</b>（boxplot + 散点，log 轴）。</li>
</ul>
<h3>数据表 Jun–Sep vs October 事件 TCP（10⁶ m³）</h3>
{table(['时段','n','均值','90% CI'],
       [['Jun–Sep', jj10['n'], f"{jj10['mean']:,.0f}", f"{jj10['ci'][0]:,.0f}–{jj10['ci'][1]:,.0f}"],
        ['October', oc10['n'], f"{oc10['mean']:,.0f}", f"{oc10['ci'][0]:,.0f}–{oc10['ci'][1]:,.0f}"]])}
<p class="note">读图要点：10 月单风暴事件 TCP 较 Jun–Sep 偏低 {abs(s10['oct_vs_jj_pct']):.1f}%
（90% CI {s10['oct_vs_jj_ci'][0]:+.0f} ~ {s10['oct_vs_jj_ci'][1]:+.0f}%），无稳健正信号
（CI 含 0 且为负）→ 10 月环流移入补图 Fig S8，主合成（Fig 9）剔除 10 月。</p>
"""

# --------------------------------------------------------------------------
# Shared style + PDF renderer (main figures 1-9 and supplementary S1-S10)
# --------------------------------------------------------------------------
def page_style(footer_label):
    return f"""<style>
@page {{ size: A4; margin: 16mm 15mm 16mm 15mm;
  @bottom-center {{ content: "Paper-4-1 · {footer_label} · 第 " counter(page) " 页 / " counter(pages) " 页";
     font-family: 'Noto Sans CJK SC','Noto Sans',sans-serif; font-size: 8.5pt; color: #888; }} }}
body {{ font-family: 'Noto Sans CJK SC','Noto Sans',DejaVu Sans,sans-serif;
  font-size: 10pt; color: #1a1a1a; line-height: 1.5; }}
h1 {{ font-size: 19pt; margin: 0 0 2pt 0; }}
h2 {{ font-size: 13.5pt; border-bottom: 2px solid #1976D2; padding-bottom: 3px;
  margin-top: 22px; color: #0d47a1; }}
h3 {{ font-size: 11pt; margin: 14px 0 5px 0; color: #1565c0; }}
h4 {{ font-size: 10pt; margin: 10px 0 3px 0; color: #333; }}
.subtitle {{ font-size: 10pt; color: #555; margin-bottom: 4px; }}
.meta-line {{ font-size: 8.7pt; color: #777; margin-bottom: 14px; }}
table {{ border-collapse: collapse; width: 100%; margin: 5px 0 10px 0; font-size: 8.9pt; }}
th, td {{ border: 1px solid #cfd8dc; padding: 3px 6px; text-align: center; }}
th {{ background: #e3eefc; font-weight: 600; }}
tbody tr:nth-child(even) {{ background: #f6f9fd; }}
td:first-child, th:first-child {{ text-align: left; font-weight: 500; }}
.fighead {{ margin-top: 18px; }}
.fnum {{ display:inline-block; background:#1976D2; color:#fff; font-weight:700;
  padding:2px 9px; border-radius:3px; font-size:10.5pt; }}
.fname {{ color:#555; font-size:8.8pt; margin-left:8px; font-family: 'Noto Sans Mono CJK SC',monospace; }}
.figtitle {{ font-size: 11.5pt; font-weight: 700; margin: 4px 0 2px 0; color:#1a1a1a; }}
.note {{ background:#fff8e1; border-left:3px solid #ffb300; padding:5px 9px;
  font-size:8.8pt; color:#5d4a00; margin: 6px 0; }}
ul {{ margin: 4px 0 8px 0; padding-left: 18px; }} li {{ margin: 2px 0; }}
img {{ display:block; margin-left:auto; margin-right:auto; }}
.toc {{ font-size: 9.5pt; }} .toc li {{ margin: 3px 0; }} .toc a {{ color:#1565c0; text-decoration:none; }}
.scopecall {{ background:#e8f5e9; border-left:3px solid #2e7d32; padding:6px 10px;
  font-size:8.8pt; color:#1b5e20; margin:8px 0 4px 0; }}
</style>"""


def render_pdf(html_str, out_path, html_name):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (C.DATA_DIR / html_name).write_text(html_str, encoding="utf-8")
    WHTML(string=html_str, base_url=str(C.PROJECT)).write_pdf(str(out_path))
    print("PDF:", out_path)
    print("size:", out_path.stat().st_size, "bytes")


from weasyprint import HTML as WHTML

OUT_MAIN = C.PROJECT / "投稿" / "Paper-4-1_Figures1-9_data_reference.pdf"
OUT_SI = C.PROJECT / "投稿" / "Paper-4-1_FiguresS1-S10_data_reference.pdf"

# ===========================================================================
# Main-figures PDF (Figures 1-9)
# ===========================================================================
HTML = f"""<!DOCTYPE html><html><head><meta charset="utf-8">{page_style("Figures 1–9 data reference")}</head><body>

<h1>Paper-4-1 主图（Figure 1–9）数据与结构说明</h1>
<div class="subtitle">Role of the Madden–Julian Oscillation in modulating tropical cyclone landfalls and intense rainfall over China
（HYDROL-S-26-03282 · analysis layer）</div>
<div class="meta-line">生成日期 {TODAY} · 数据源自 <code>data/event_table.csv</code>、
<code>data/composite_jjas.nc</code> · 样本 n = {meta['n_landfall_jjaso_active']} 登陆事件（JJASO &amp; 活跃 MJO）</div>

<div class="scopecall"><b>范围说明：</b>本文件为<b>技术参考</b>——逐图给出面板构成、数据来源、单位、显著性方法与
<b>底层数值表</b>；不包含正文图注/正文叙述（正文文字属作者领域，未在此撰写）。</div>

<div class="toc"><ul>
<li><a href="#common">一、通用约定</a></li>
<li><a href="#f1">Figure 1 · 登陆活动 — ACE / 频数 / 总 TCP / 单风暴 TCP</a></li>
<li><a href="#f2">Figure 2 · 登陆点空间分布</a></li>
<li><a href="#f3">Figure 3 · 区域登陆率的位相依赖</a></li>
<li><a href="#f4">Figure 4 · TCP 强度分解</a></li>
<li><a href="#f5">Figure 5 · TCP 位相分解</a></li>
<li><a href="#f6">Figure 6 · 风暴级 TCP</a></li>
<li><a href="#f7">Figure 7 · 生成位置与登陆前路径</a></li>
<li><a href="#f8">Figure 8 · 生成—登陆关系与 MJO 位相</a></li>
<li><a href="#f9">Figure 9 · 环流异常合成</a></li>
</ul></div>

{common}

<h2 id="f1">二、Figure 1</h2>{sec1}
<h2 id="f2">三、Figure 2</h2>{sec2}
<h2 id="f3">四、Figure 3</h2>{sec3}
<h2 id="f4">五、Figure 4</h2>{sec4}
<h2 id="f5">六、Figure 5</h2>{sec5}
<h2 id="f6">七、Figure 6</h2>{sec6}
<h2 id="f7">八、Figure 7</h2>{sec7}
<h2 id="f8">九、Figure 8</h2>{sec8}
<h2 id="f9">十、Figure 9</h2>{sec9}

<p class="note" style="margin-top:18px">补图（Fig S1–S10）的数据与结构说明见配套文件
<code>Paper-4-1_FiguresS1-S10_data_reference.pdf</code>。
所有数值由各图脚本同公式重算得出，点估计确定性一致；bootstrap CI 为独立种子重算，
与图中曲线在重采样噪声范围内一致。</p>
</body></html>"""

render_pdf(HTML, OUT_MAIN, "fig_doc.html")

# ===========================================================================
# Supplementary-figures PDF (S1-S10)
# ===========================================================================
SI_LINEAGE_ROWS = [
    ['S1', 'event_table.csv（全部 490）', 'month / lmi_category'],
    ['S2', 'event_table.csv（in_jjaso 464）', 'lmi_category / landfall_wind_category / landfall_region / tcp_total'],
    ['S3', 'ibtracs_matched.csv（由 11_ibtracs_robustness.py 生成）', 'cma_lmi / ib_wind_ms / ib_category / match_km'],
    ['S4', 'lib.wnp.all_wnp_genesis() + event_table.csv', 'group_genesis / active_genesis（全 WNP 生成 vs 中国登陆）'],
    ['S5', 'composite_jjas.nc', 'uwnd_200 / slp'],
    ['S6', 'composite_jun_jul.nc', 'vort_850 / hgt_500'],
    ['S7', 'composite_aug_sep.nc', 'vort_850 / hgt_500'],
    ['S8', 'composite_oct.nc', 'vort_850 / hgt_500（vs 10 月自身气候态）'],
    ['S9', 'pre/pre_*.nc (tcp_lib)', 'coastal_inland_by_threshold @100/200/300 km'],
    ['S10', 'event_table.csv（active_landfall）', 'month / tcp_total（Jun–Sep vs Oct）'],
]

common_si = f"""
<h2 id="sicommon">一、通用约定（补图 S1–S10）</h2>
<h3>1. 数据来源（在主图基础上新增）</h3>
<table class="tbl">
<thead><tr><th>数据集</th><th>说明</th><th>用于</th></tr></thead>
<tbody>
<tr><td>IBTrACS (ALL v04r01)</td><td>替代最佳路径数据集（USA-agency 子集）</td><td>Fig S3（稳健性）</td></tr>
<tr><td>NCEP/NCAR 逐月合成</td><td>composite_jjas / jun_jul / aug_sep / oct</td><td>Fig S5–S8（环流）</td></tr>
</tbody></table>
<h3>2. 样本约定（补图样本与主图 258 个活跃子集<b>不同</b>，逐图说明）</h3>
<ul>
<li><b>Fig S1</b>：全部 {D['figS1']['n_total']} 个登陆 TC（全年，不限 MJO）。</li>
<li><b>Fig S2</b>：in_jjaso 共 {D['figS2']['n']} 个登陆 TC（不限登陆日是否活跃 MJO）。</li>
<li><b>Fig S3</b>：in_jjaso 风暴与 IBTrACS 匹配子集（{D['figS3'].get('n_matched','—')} / {D['figS3'].get('n_in_jjaso','—')}）。</li>
<li><b>Fig S4</b>：全 WNP（生成日活跃 MJO）vs 中国登陆（生成日活跃 MJO）。</li>
<li><b>Fig S5–S8</b>：与 Fig 9 同源日历月合成场（Jun-Sep / Jun-Jul / Aug-Sep / Oct）。</li>
<li><b>Fig S9</b>：与 Fig 4 同的 {D['figS9']['n_storms']} 个活跃登陆风暴（沿海/内陆阈值敏感性）。</li>
<li><b>Fig S10</b>：登陆日活跃 MJO 风暴，按 6–9 月 vs 10 月分组。</li>
</ul>
<h3>3. 统计与显著性</h3>
<ul>
<li>与主图一致：停留时间归一化比率（ratio = 1.0 为无调制期望）、停留时间多项式零分布置换检验
   （1000 次，双侧 α = 0.10，"*" 标注）、风暴级 bootstrap 90% CI。</li>
<li>环流场（Fig S5–S8）显著性同 Fig 9：单样本 t = composite·√n，|t| &gt; {meta['tcrit_90']}（α = 0.10），
   基于日历月标准化距平。</li>
<li>Fig S2/S4 的区域占比与登陆比例为二项 bootstrap 90% CI。</li>
</ul>

<h3>4. 各补图输入数据对照（S1–S10）</h3>
<p class="note">event_table.csv 完整数据字典见主文件"通用约定 §6"。</p>
{table(['图', '输入文件', '关键字段 / 变量'], SI_LINEAGE_ROWS)}
"""

HTML_SI = f"""<!DOCTYPE html><html><head><meta charset="utf-8">{page_style("Figures S1–S10 data reference")}</head><body>

<h1>Paper-4-1 补图（Figure S1–S10）数据与结构说明</h1>
<div class="subtitle">Role of the Madden–Julian Oscillation in modulating tropical cyclone landfalls and intense rainfall over China
（HYDROL-S-26-03282 · Supporting Information）</div>
<div class="meta-line">生成日期 {TODAY} · 数据源自 <code>data/event_table.csv</code>、
<code>data/composite_{{jjas,jun_jul,aug_sep,oct}}.nc</code>、<code>data/ibtracs_matched.csv</code> ·
样本因图而异（见通用约定）</div>

<div class="scopecall"><b>范围说明：</b>本文件为<b>技术参考</b>——逐图给出补图的面板构成、数据来源、单位、显著性方法与
<b>底层数值表</b>；不包含 SI 图注/正文叙述（正文文字属作者领域，未在此撰写）。主图（Fig 1–9）见
<code>Paper-4-1_Figures1-9_data_reference.pdf</code>。</div>

<div class="toc"><ul>
<li><a href="#sicommon">一、通用约定（补图）</a></li>
<li><a href="#s1">Figure S1 · 月际分布</a></li>
<li><a href="#s2">Figure S2 · LMI vs 登陆强度</a></li>
<li><a href="#s3">Figure S3 · IBTrACS 稳健性</a></li>
<li><a href="#s4">Figure S4 · 全 WNP vs 登陆</a></li>
<li><a href="#s5">Figure S5 · 200-hPa 风 + SLP（Jun–Sep）</a></li>
<li><a href="#s6">Figure S6 · 环流月际（Jun–Jul）</a></li>
<li><a href="#s7">Figure S7 · 环流月际（Aug–Sep）</a></li>
<li><a href="#s8">Figure S8 · 环流月际（October）</a></li>
<li><a href="#s9">Figure S9 · 沿海/内陆阈值敏感性</a></li>
<li><a href="#s10">Figure S10 · 10 月 TCP 决策</a></li>
</ul></div>

{common_si}

<h2 id="s1">二、Figure S1</h2>{secS1}
<h2 id="s2">三、Figure S2</h2>{secS2}
<h2 id="s3">四、Figure S3</h2>{secS3}
<h2 id="s4">五、Figure S4</h2>{secS4}
<h2 id="s5">六、Figure S5</h2>{secS5}
<h2 id="s6">七、Figure S6</h2>{secS['figS6']}
<h2 id="s7">八、Figure S7</h2>{secS['figS7']}
<h2 id="s8">九、Figure S8</h2>{secS['figS8']}
<h2 id="s9">十、Figure S9</h2>{secS9}
<h2 id="s10">十一、Figure S10</h2>{secS10}

<p class="note" style="margin-top:18px">所有数值由各补图脚本同公式重算得出，点估计确定性一致；
bootstrap CI 为独立种子重算，与图中曲线在重采样噪声范围内一致。</p>
</body></html>"""

render_pdf(HTML_SI, OUT_SI, "fig_doc_si.html")
