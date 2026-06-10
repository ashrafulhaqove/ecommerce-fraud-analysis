from pathlib import Path
from datetime import date

import duckdb
import jinja2


QUERY_SUMMARY = """
select
    count(*)                                                              as total_orders,
    sum(fraud_flag)                                                       as flagged_orders,
    round(sum(fraud_flag) * 100.0 / count(*), 1)                         as fraud_rate_pct,
    sum(case when risk_tier = 'high'   then 1 else 0 end)                as high_risk,
    sum(case when risk_tier = 'medium' then 1 else 0 end)                as medium_risk,
    sum(case when risk_tier = 'low'    then 1 else 0 end)                as low_risk,
    sum(case when risk_tier = 'none'   then 1 else 0 end)                as no_risk,
    round(sum(case when fraud_flag = 1 then order_amount else 0 end), 0) as flagged_amount,
    round(avg(case when fraud_flag = 1 then risk_score end), 1)          as avg_risk_score
from fct_fraud_signals
"""

QUERY_RULES = """
select rule, hits from (
    select 'High order amount (>=150)'              as rule, sum(case when order_amount >= 150 then 1 else 0 end)                         as hits from fct_fraud_signals
    union all
    select 'Country mismatch (billing != shipping)',          sum(case when country_mismatch then 1 else 0 end)                           from fct_fraud_signals
    union all
    select 'IP mismatch (IP != billing country)',             sum(case when ip_mismatch then 1 else 0 end)                                from fct_fraud_signals
    union all
    select 'High velocity (3+ orders/day)',                   sum(case when high_velocity then 1 else 0 end)                              from fct_fraud_signals
    union all
    select 'New customer + high amount (>=100)',              sum(case when is_new_customer and order_amount >= 100 then 1 else 0 end)    from fct_fraud_signals
    union all
    select 'Gift card purchase',                              sum(case when product_category = 'gift_cards' then 1 else 0 end)           from fct_fraud_signals
    union all
    select 'New customer + crypto payment',                   sum(case when is_new_customer and payment_method = 'crypto' then 1 else 0 end)  from fct_fraud_signals
    union all
    select 'Late-night order (1am-5am)',                      sum(case when order_hour between 1 and 5 then 1 else 0 end)                 from fct_fraud_signals
) order by hits desc
"""

QUERY_ML_METRICS = """
select metric, value from ml_metrics
"""

QUERY_ML_SUMMARY = """
select
    sum(ml_fraud_flag)                                        as ml_flagged,
    round(sum(ml_fraud_flag) * 100.0 / count(*), 1)          as ml_flag_rate,
    sum(case when ml_fraud_flag = 1 and f.fraud_flag = 1
             then 1 else 0 end)                               as both_flagged
from ml_predictions m
join fct_fraud_signals f using (order_id)
"""

QUERY_FLAGGED = """
select
    f.order_id,
    f.customer_id,
    f.order_date,
    f.order_amount,
    f.payment_method,
    f.product_category,
    f.billing_country,
    f.shipping_country,
    f.ip_country,
    f.risk_score,
    f.risk_tier,
    coalesce(m.ml_fraud_prob, 0.0) as ml_fraud_prob,
    coalesce(m.ml_fraud_flag, 0)   as ml_fraud_flag
from fct_fraud_signals f
left join ml_predictions m using (order_id)
where f.fraud_flag = 1
order by ml_fraud_prob desc
limit 50
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ecommerce Fraud Analysis</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f1f5f9; color: #1e293b; font-size: 14px; line-height: 1.5;
}
header {
  background: #0f172a; color: #f8fafc; padding: 20px 40px;
  display: flex; align-items: center; justify-content: space-between;
}
header h1 { font-size: 20px; font-weight: 600; letter-spacing: -0.3px; }
.hdr-sub { font-size: 12px; color: #64748b; margin-top: 3px; }
.hdr-meta { font-size: 12px; color: #94a3b8; text-align: right; }
.hdr-meta span { display: block; }
main { max-width: 1200px; margin: 0 auto; padding: 28px 24px; }
.sec {
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.8px; color: #64748b; margin: 28px 0 12px;
}
.kpi-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; }
.kpi { background: #fff; border-radius: 10px; padding: 18px 20px; border: 1px solid #e2e8f0; }
.kpi .val { font-size: 26px; font-weight: 700; line-height: 1.1; }
.kpi .lbl { font-size: 11px; color: #64748b; margin-top: 5px; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi .sub { font-size: 11px; color: #94a3b8; margin-top: 2px; }
.kpi.danger  .val { color: #dc2626; }
.kpi.warning .val { color: #d97706; }
.kpi.info    .val { color: #2563eb; }
.mid-row { display: grid; grid-template-columns: 380px 1fr; gap: 14px; }
.card { background: #fff; border-radius: 10px; border: 1px solid #e2e8f0; padding: 20px 24px; }
.card-title { font-size: 13px; font-weight: 600; color: #334155; margin-bottom: 16px; }
.donut-wrap { display: flex; align-items: center; gap: 20px; }
.donut {
  width: 136px; height: 136px; border-radius: 50%; flex-shrink: 0; position: relative;
  background: conic-gradient(
    #dc2626    0%    {{ high_pct }}%,
    #d97706 {{ high_pct }}% {{ medium_end }}%,
    #3b82f6 {{ medium_end }}% {{ low_end }}%,
    #22c55e {{ low_end }}% 100%
  );
}
.donut-hole {
  width: 86px; height: 86px; background: #fff; border-radius: 50%;
  position: absolute; top: 25px; left: 25px;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.dn-v { font-size: 18px; font-weight: 700; color: #1e293b; }
.dn-l { font-size: 9px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
.legend .li { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; font-size: 12px; }
.legend .li-l { display: flex; align-items: center; gap: 7px; }
.legend .dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.legend .ln { color: #475569; }
.legend .lc { font-weight: 600; color: #1e293b; }
.legend .lp { color: #94a3b8; font-size: 11px; margin-left: 4px; }
.rr { display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px solid #f1f5f9; }
.rr:last-child { border-bottom: none; }
.rr-name { flex: 1; font-size: 12px; color: #334155; }
.rr-track { width: 130px; height: 6px; background: #f1f5f9; border-radius: 9999px; overflow: hidden; flex-shrink: 0; }
.rr-fill  { height: 6px; background: #3b82f6; border-radius: 9999px; }
.rr-n { font-size: 12px; font-weight: 600; color: #1e293b; min-width: 28px; text-align: right; }
.rr-p { font-size: 11px; color: #94a3b8; min-width: 42px; text-align: right; }
.tbl-wrap { background: #fff; border-radius: 10px; border: 1px solid #e2e8f0; overflow: hidden; }
table { width: 100%; border-collapse: collapse; }
thead th {
  background: #f8fafc; padding: 10px 14px; text-align: left;
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.5px; color: #64748b; border-bottom: 1px solid #e2e8f0; white-space: nowrap;
}
tbody td { padding: 9px 14px; border-bottom: 1px solid #f1f5f9; font-size: 13px; color: #334155; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: #f8fafc; }
.sc { display: flex; align-items: center; gap: 8px; }
.sc-track { width: 56px; height: 5px; background: #f1f5f9; border-radius: 9999px; overflow: hidden; }
.sc-fill { height: 5px; border-radius: 9999px; }
.sc-fill.high   { background: #dc2626; }
.sc-fill.medium { background: #d97706; }
.sc-fill.low    { background: #3b82f6; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 9999px; font-size: 11px; font-weight: 600; text-transform: capitalize; }
.badge.high   { background: #fee2e2; color: #dc2626; }
.badge.medium { background: #fef3c7; color: #d97706; }
.badge.low    { background: #dbeafe; color: #2563eb; }
.filter-bar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; flex-wrap: wrap; gap: 8px; }
.filter-controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.filter-select {
  padding: 6px 10px; border: 1px solid #e2e8f0; border-radius: 6px;
  font-size: 12px; color: #334155; background: #fff; cursor: pointer;
  outline: none;
}
.filter-select:focus { border-color: #3b82f6; }
.filter-count { font-size: 12px; color: #94a3b8; margin-left: 4px; }
.export-btn {
  padding: 6px 14px; background: #0f172a; color: #fff; border: none;
  border-radius: 6px; font-size: 12px; font-weight: 500; cursor: pointer; white-space: nowrap;
}
.export-btn:hover { background: #1e293b; }
footer { text-align: center; color: #94a3b8; font-size: 11px; padding: 24px; border-top: 1px solid #e2e8f0; margin-top: 16px; }
</style>
</head>
<body>
<header>
  <div>
    <h1>Ecommerce Fraud Analysis</h1>
    <div class="hdr-sub">Transaction risk scoring &nbsp;·&nbsp; dbt Core &nbsp;·&nbsp; DuckDB &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; Python</div>
  </div>
  <div class="hdr-meta">
    <span>Generated {{ generated_date }}</span>
    <span>{{ summary.total_orders }} transactions analysed</span>
  </div>
</header>
<main>

<div class="sec">Overview</div>
<div class="kpi-row">
  <div class="kpi danger">
    <div class="val">{{ summary.flagged_orders }}</div>
    <div class="lbl">Flagged Orders</div>
    <div class="sub">{{ summary.fraud_rate_pct }}% of total</div>
  </div>
  <div class="kpi danger">
    <div class="val">{{ summary.high_risk }}</div>
    <div class="lbl">High Risk</div>
    <div class="sub">score &ge; 80</div>
  </div>
  <div class="kpi warning">
    <div class="val">{{ summary.medium_risk }}</div>
    <div class="lbl">Medium Risk</div>
    <div class="sub">score 50&ndash;79</div>
  </div>
  <div class="kpi info">
    <div class="val">{{ summary.flagged_amount }}</div>
    <div class="lbl">Flagged Value</div>
    <div class="sub">combined order total</div>
  </div>
  <div class="kpi">
    <div class="val">{{ summary.avg_risk_score }}</div>
    <div class="lbl">Avg Risk Score</div>
    <div class="sub">flagged orders only</div>
  </div>
</div>

<div class="sec">ML Model Performance &mdash; XGBoost vs Rule-Based Baseline</div>
<div class="kpi-row" style="grid-template-columns:repeat(4,1fr)">
  <div class="kpi info">
    <div class="val">{{ ml.roc_auc }}</div>
    <div class="lbl">ROC-AUC</div>
    <div class="sub">baseline {{ ml.baseline_auc }}</div>
  </div>
  <div class="kpi info">
    <div class="val">{{ ml.avg_precision }}</div>
    <div class="lbl">Avg Precision</div>
    <div class="sub">baseline {{ ml.baseline_ap }}</div>
  </div>
  <div class="kpi info">
    <div class="val">{{ ml.fraud_f1 }}</div>
    <div class="lbl">Fraud F1</div>
    <div class="sub">baseline {{ ml.baseline_f1 }}</div>
  </div>
  <div class="kpi warning">
    <div class="val">{{ ml_summary.ml_flagged }}</div>
    <div class="lbl">ML Flagged</div>
    <div class="sub">{{ ml_summary.ml_flag_rate }}% of total</div>
  </div>
</div>

<div class="sec">Risk Distribution &amp; Signal Breakdown</div>
<div class="mid-row">

  <div class="card">
    <div class="card-title">Risk Tier Distribution</div>
    <div class="donut-wrap">
      <div class="donut">
        <div class="donut-hole">
          <div class="dn-v">{{ summary.flagged_orders }}</div>
          <div class="dn-l">flagged</div>
        </div>
      </div>
      <div class="legend">
        <div class="li"><div class="li-l"><div class="dot" style="background:#dc2626"></div><span class="ln">High</span></div><div><span class="lc">{{ summary.high_risk }}</span><span class="lp">({{ high_tier_pct }}%)</span></div></div>
        <div class="li"><div class="li-l"><div class="dot" style="background:#d97706"></div><span class="ln">Medium</span></div><div><span class="lc">{{ summary.medium_risk }}</span><span class="lp">({{ medium_tier_pct }}%)</span></div></div>
        <div class="li"><div class="li-l"><div class="dot" style="background:#3b82f6"></div><span class="ln">Low</span></div><div><span class="lc">{{ summary.low_risk }}</span><span class="lp">({{ low_tier_pct }}%)</span></div></div>
        <div class="li"><div class="li-l"><div class="dot" style="background:#22c55e"></div><span class="ln">None</span></div><div><span class="lc">{{ summary.no_risk }}</span><span class="lp">({{ none_tier_pct }}%)</span></div></div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Fraud Signal Hits</div>
    {% for r in rules %}
    <div class="rr">
      <span class="rr-name">{{ r.rule }}</span>
      <div class="rr-track"><div class="rr-fill" style="width:{{ r.bar_width }}%"></div></div>
      <span class="rr-n">{{ r.hits }}</span>
      <span class="rr-p">{{ r.pct_total }}%</span>
    </div>
    {% endfor %}
  </div>

</div>

<div class="sec">Rule-Flagged Orders &mdash; Top 50 by ML Probability</div>
<div class="filter-bar">
  <div class="filter-controls">
    <select class="filter-select" id="f-tier" onchange="applyFilters()">
      <option value="">All Tiers</option>
      <option value="high">High</option>
      <option value="medium">Medium</option>
      <option value="low">Low</option>
    </select>
    <select class="filter-select" id="f-payment" onchange="applyFilters()">
      <option value="">All Payments</option>
      <option value="visa">Visa</option>
      <option value="mastercard">Mastercard</option>
      <option value="discover">Discover</option>
      <option value="american express">American Express</option>
    </select>
    <select class="filter-select" id="f-category" onchange="applyFilters()">
      <option value="">All Categories</option>
      <option value="Electronics">Electronics</option>
      <option value="Clothing">Clothing</option>
      <option value="Travel">Travel</option>
      <option value="Home &amp; Garden">Home &amp; Garden</option>
      <option value="Services">Services</option>
    </select>
    <span class="filter-count" id="row-count"></span>
  </div>
  <button class="export-btn" onclick="exportCSV()">&#8595; Export CSV</button>
</div>
<div class="tbl-wrap">
  <table id="flagged-table">
    <thead>
      <tr>
        <th>Order ID</th><th>Customer</th><th>Date</th><th>Amount</th>
        <th>Payment</th><th>Category</th><th>Billing</th><th>Shipping</th>
        <th>IP Country</th><th>ML Prob</th><th>Risk Score</th><th>Tier</th>
      </tr>
    </thead>
    <tbody>
    {% for r in flagged %}
      <tr data-tier="{{ r.risk_tier }}" data-payment="{{ r.payment_method }}" data-category="{{ r.product_category }}">
        <td>{{ r.order_id }}</td>
        <td>{{ r.customer_id }}</td>
        <td>{{ r.order_date }}</td>
        <td>${{ "%.2f"|format(r.order_amount) }}</td>
        <td>{{ r.payment_method }}</td>
        <td>{{ r.product_category }}</td>
        <td>{{ r.billing_country }}</td>
        <td>{{ r.shipping_country }}</td>
        <td>{{ r.ip_country }}</td>
        <td><strong>{{ "%.0f"|format(r.ml_fraud_prob * 100) }}%</strong></td>
        <td>
          <div class="sc">
            <div class="sc-track">
              <div class="sc-fill {{ r.risk_tier }}" style="width:{{ r.score_bar_pct }}%"></div>
            </div>
            <strong>{{ r.risk_score }}</strong>
          </div>
        </td>
        <td><span class="badge {{ r.risk_tier }}">{{ r.risk_tier }}</span></td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>

</main>
<footer>Ecommerce Fraud Analysis &nbsp;·&nbsp; dbt Core &nbsp;·&nbsp; DuckDB &nbsp;·&nbsp; Python</footer>
<script>
function applyFilters() {
  var tier     = document.getElementById('f-tier').value;
  var payment  = document.getElementById('f-payment').value;
  var category = document.getElementById('f-category').value;
  var visible  = 0;
  document.querySelectorAll('#flagged-table tbody tr').forEach(function(row) {
    var show = (!tier     || row.dataset.tier     === tier)
            && (!payment  || row.dataset.payment  === payment)
            && (!category || row.dataset.category === category);
    row.style.display = show ? '' : 'none';
    if (show) visible++;
  });
  document.getElementById('row-count').textContent = visible + ' orders';
}

function exportCSV() {
  var headers = Array.from(document.querySelectorAll('#flagged-table thead th'))
    .map(function(th) { return '"' + th.textContent.trim() + '"'; });
  var rows = Array.from(document.querySelectorAll('#flagged-table tbody tr'))
    .filter(function(r) { return r.style.display !== 'none'; })
    .map(function(r) {
      return Array.from(r.querySelectorAll('td')).map(function(td) {
        return '"' + td.textContent.trim().replace(/"/g, '""') + '"';
      });
    });
  var csv = [headers.join(',')].concat(rows.map(function(r) { return r.join(','); })).join('\\n');
  var a = document.createElement('a');
  a.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
  a.download = 'fraud_signals_{{ generated_date }}.csv';
  a.click();
}

window.addEventListener('load', function() {
  var total = document.querySelectorAll('#flagged-table tbody tr').length;
  document.getElementById('row-count').textContent = total + ' orders';
});
</script>
</body>
</html>
"""


def main() -> None:
    con = duckdb.connect("data/ecommerce_fraud.duckdb")

    raw = con.execute(QUERY_SUMMARY).df().iloc[0].to_dict()
    summary = {
        "total_orders":   int(raw["total_orders"]),
        "flagged_orders": int(raw["flagged_orders"]),
        "fraud_rate_pct": raw["fraud_rate_pct"],
        "high_risk":      int(raw["high_risk"]),
        "medium_risk":    int(raw["medium_risk"]),
        "low_risk":       int(raw["low_risk"]),
        "no_risk":        int(raw["no_risk"]),
        "flagged_amount": f"${int(raw['flagged_amount']):,}",
        "avg_risk_score": raw["avg_risk_score"],
    }

    total = summary["total_orders"]
    high_pct   = round(summary["high_risk"] / total * 100, 2)
    medium_end = round((summary["high_risk"] + summary["medium_risk"]) / total * 100, 2)
    low_end    = round((summary["high_risk"] + summary["medium_risk"] + summary["low_risk"]) / total * 100, 2)

    rules_raw = con.execute(QUERY_RULES).df().to_dict(orient="records")
    max_hits  = max(int(r["hits"]) for r in rules_raw)
    rules = [
        {
            "rule":      r["rule"],
            "hits":      int(r["hits"]),
            "pct_total": round(int(r["hits"]) / total * 100, 1),
            "bar_width": round(int(r["hits"]) / max_hits * 100, 1) if max_hits else 0,
        }
        for r in rules_raw
    ]

    try:
        metrics_raw = {r["metric"]: r["value"]
                       for r in con.execute(QUERY_ML_METRICS).df().to_dict(orient="records")}
    except Exception:
        metrics_raw = {}
    ml = {
        "roc_auc":       metrics_raw.get("roc_auc",       "N/A"),
        "avg_precision": metrics_raw.get("avg_precision", "N/A"),
        "fraud_f1":      metrics_raw.get("fraud_f1",      "N/A"),
        "baseline_auc":  metrics_raw.get("baseline_auc",  "N/A"),
        "baseline_ap":   metrics_raw.get("baseline_ap",   "N/A"),
        "baseline_f1":   metrics_raw.get("baseline_f1",   "N/A"),
    }

    try:
        ml_raw = con.execute(QUERY_ML_SUMMARY).df().iloc[0].to_dict()
        ml_summary = {
            "ml_flagged":   f"{int(ml_raw['ml_flagged'] or 0):,}",
            "ml_flag_rate": ml_raw["ml_flag_rate"] or 0,
            "both_flagged": int(ml_raw["both_flagged"] or 0),
        }
    except Exception:
        ml_summary = {"ml_flagged": "N/A", "ml_flag_rate": 0, "both_flagged": 0}

    flagged_raw = con.execute(QUERY_FLAGGED).df().to_dict(orient="records")
    flagged = [
        {
            **r,
            "risk_score":    int(r["risk_score"]),
            "order_date":    str(r["order_date"])[:10],
            "score_bar_pct": round(int(r["risk_score"]) / 165 * 100, 1),
            "ml_fraud_prob": float(r["ml_fraud_prob"]),
        }
        for r in flagged_raw
    ]

    html = jinja2.Template(TEMPLATE).render(
        generated_date=date.today().isoformat(),
        summary=summary,
        rules=rules,
        flagged=flagged,
        ml=ml,
        ml_summary=ml_summary,
        high_pct=high_pct,
        medium_end=medium_end,
        low_end=low_end,
        high_tier_pct=round(summary["high_risk"] / total * 100, 1),
        medium_tier_pct=round(summary["medium_risk"] / total * 100, 1),
        low_tier_pct=round(summary["low_risk"] / total * 100, 1),
        none_tier_pct=round(summary["no_risk"] / total * 100, 1),
    )

    out = Path("reports/index.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Report written → {out}")


if __name__ == "__main__":
    main()