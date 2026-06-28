"""Render the spike's markdown result tables into ONE dark-mode tabbed HTML.

Parses the `### <Name> DGP (...)` sections + their markdown tables from a results file
and emits a self-contained dark page: one tab per DGP, the full metrics table, plus a
short read-the-table narrative. No deps.

Run:
  /opt/homebrew/bin/python3.11 exploration/build_dashboard.py \
     --in exploration/results_logit_cholesky.md \
     --out exploration/results_logit_cholesky.html
"""
import argparse
import html
import re


def parse_sections(text):
    """-> [(title, [header_cells], [[row_cells],...]), ...] for each '### ...' table."""
    sections = []
    blocks = re.split(r"^###\s+", text, flags=re.M)[1:]
    for b in blocks:
        title = b.splitlines()[0].strip()
        rows = [ln for ln in b.splitlines() if ln.strip().startswith("|")]
        if len(rows) < 2:
            continue
        def cells(line):
            return [c.strip() for c in line.strip().strip("|").split("|")]
        header = cells(rows[0])
        body = [cells(r) for r in rows[2:]]  # skip the |---| separator
        sections.append((title, header, body))
    return sections


def highlight(method):
    m = method.lower()
    if "oracle[" in m or "flm[oracle" in m:
        return "ceiling"
    if "cholesky" in m:
        return "fix"
    if m == "oracle":
        return "anchor"
    if "riesznet" in m:
        return "anchor"
    if "naive" in m:
        return "bad"
    return ""


def table_html(header, body):
    th = "".join(f"<th>{html.escape(c)}</th>" for c in header)
    trs = []
    for row in body:
        cls = highlight(row[0])
        tds = "".join(f"<td>{html.escape(c)}</td>" for c in row)
        trs.append(f'<tr class="{cls}">{tds}</tr>')
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--ink:#e6edf3;--mut:#8b949e;
--fix:#1f6feb;--ceil:#2ea043;--bad:#f85149;--accent:#d29922}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"SF Pro Text",Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1000px;margin:0 auto;padding:32px 24px 64px}}
h1{{font-size:22px;font-weight:600;margin:0 0 4px}}
.sub{{color:var(--mut);margin:0 0 24px;font-size:14px}}
.tabs{{display:flex;gap:4px;border-bottom:1px solid var(--line);margin-bottom:20px}}
.tab{{padding:9px 16px;cursor:pointer;color:var(--mut);border-bottom:2px solid transparent;
font-weight:500}}
.tab.on{{color:var(--ink);border-bottom-color:var(--fix)}}
.panel{{display:none}} .panel.on{{display:block}}
.cap{{color:var(--mut);font-size:13px;margin:0 0 14px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}}
th,td{{text-align:right;padding:8px 12px;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left;font-weight:500}}
thead th{{color:var(--mut);font-weight:500;border-bottom:1px solid var(--line)}}
tr.fix td{{background:rgba(31,111,235,.14)}}
tr.ceiling td{{background:rgba(46,160,67,.13)}}
tr.bad td{{color:var(--mut)}}
tr.fix td:first-child{{box-shadow:inset 3px 0 var(--fix)}}
tr.ceiling td:first-child{{box-shadow:inset 3px 0 var(--ceil)}}
.note{{margin-top:22px;padding:16px 18px;background:var(--panel);border:1px solid var(--line);
border-radius:8px;color:var(--ink);font-size:14px}}
.note b{{color:var(--accent)}}
.legend{{margin-top:16px;color:var(--mut);font-size:12.5px}}
.legend span{{display:inline-block;margin-right:16px}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;
margin-right:5px}}
</style></head><body><div class="wrap">
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
<div class="tabs">{tabbtns}</div>
{panels}
<div class="legend">
<span><i class="sw" style="background:var(--fix)"></i>FLM, general PSD Λ̂(x) (the fix)</span>
<span><i class="sw" style="background:var(--ceil)"></i>Oracle-Λ ceiling (true Λ injected)</span>
<span>Anchors: Oracle-MLE, RieszNet. Naive = no correction.</span>
</div>
</div>
<script>
const tabs=[...document.querySelectorAll('.tab')],panels=[...document.querySelectorAll('.panel')];
tabs.forEach((t,i)=>t.onclick=()=>{{tabs.forEach(x=>x.classList.remove('on'));
panels.forEach(x=>x.classList.remove('on'));t.classList.add('on');panels[i].classList.add('on');}});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="exploration/results_logit_cholesky.md")
    ap.add_argument("--out", dest="out", default="exploration/results_logit_cholesky.html")
    ap.add_argument("--title", default="General Λ(x) on logit: FLM vs RieszNet vs Oracle")
    ap.add_argument("--subtitle", default="PSD-by-construction Λ̂(x)=L(x)L(x)ᵀ. ATE on the "
                    "probability scale. Truth from Monte Carlo.")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    with open(args.inp) as f:
        text = f.read()
    sections = parse_sections(text)

    btns, panels = [], []
    for i, (title, header, body) in enumerate(sections):
        on = " on" if i == 0 else ""
        short = title.split(" DGP")[0]
        btns.append(f'<div class="tab{on}">{html.escape(short)}</div>')
        panels.append(f'<div class="panel{on}"><p class="cap">{html.escape(title)}</p>'
                      f'{table_html(header, body)}</div>')

    note = f'<div class="note">{args.note}</div>' if args.note else ""
    page = PAGE.format(title=html.escape(args.title), subtitle=html.escape(args.subtitle),
                       tabbtns="".join(btns), panels="".join(panels) + note)
    with open(args.out, "w") as f:
        f.write(page)
    print(f"wrote {args.out}  ({len(sections)} DGP tabs)")


if __name__ == "__main__":
    main()
