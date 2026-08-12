# CMABSTdata — CMA Best Track (external upstream source)

**Status:** placeholder — this repository does **not** read CMA Best Track files
directly. It consumes the *derived* landfall/track tables already committed in
[`../typhoon_output/`](../typhoon_output/), which were produced from CMA Best Track
by the upstream landfall-detection step (`01_landfall.py`, geometric intersection of
each track with the China boundary — mainland + Taiwan + Hainan). This directory only
records the provenance of that derivation.

**Dataset:** CMA/STI (Shanghai Typhoon Institute) Tropical Cyclone Best Track
Dataset, 1949–present. Yearly text files `CH{year}BST.txt`.

**Download (CMA Tropical Cyclone Data Center):** https://tcdata.typhoon.org.cn/zjljsjj_zlhq.html
  (English: https://tcdata.typhoon.org.cn/en/zjl/zlhq.html)

**Cite:** Ying, M., Zhang, W., Yun, H., Lu, X., Chen, J., Gao, Y., et al. (2014). An
overview of the China Meteorological Administration tropical cyclone database. Bull.
Amer. Meteor. Soc., 95, 703–712. https://doi.org/10.1175/BAMS-D-12-00174.1
  (updated annually; see the data center for the current version note).

**Wind-intensity convention used downstream** (preserved verbatim in `config.py` →
`get_wind_category`): category 0 = TD (≤17.1 m/s), 1 = Weak (≤32.6), 2 = Moderate
(<51.0), 3 = Major (≥51.0). The top tier is labelled "Major" (not "Super") throughout
this revision. Two parallel classifications are carried: **LMI** (lifetime-maximum
intensity, `max_wind_category`) and **intensity at first Chinese landfall**
(`landfall_wind_category`).
