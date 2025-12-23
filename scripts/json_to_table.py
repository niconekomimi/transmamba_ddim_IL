import json
import csv
from pathlib import Path

LOG_ROOT = Path("/home/public/wyl/X_IL/logs")
INPUT_JSON = LOG_ROOT / "summary_success_rates.json"
OUTPUT_CSV = LOG_ROOT / "summary_success_rates.csv"
OUTPUT_HTML = LOG_ROOT / "summary_success_rates.html"

COLUMNS = [
    "libero_task",
    "model",
    "date",
    "time",
    "average_success_rate",
    "traj_per_task",
    "encoder",
    "decoder",
]

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def to_rows(data: dict):
    rows = []
    for libero_task, models in data.items():
        if not isinstance(models, dict):
            continue
        for model, runs in models.items():
            if not isinstance(runs, list):
                continue
            for r in runs:
                kp = r.get("key_params", {}) or {}
                params = r.get("params", {}) or {}
                row = {
                    "libero_task": libero_task,
                    "model": model,
                    "date": r.get("date", ""),
                    "time": r.get("time", ""),
                    "average_success_rate": r.get("average_success_rate", ""),
                    "traj_per_task": kp.get("traj_per_task", params.get("traj_per_task", 10)),
                    "encoder": kp.get("encoder", ""),
                    "decoder": kp.get("decoder", ""),
                }
                rows.append(row)
    # 排序
    rows.sort(key=lambda x: (x["libero_task"], x["model"], x["date"], x["time"]))
    return rows

def write_csv(rows, p: Path):
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

def write_html(rows, p: Path):
    import html, json

    tasks = sorted({r["libero_task"] for r in rows})
    models = sorted({r["model"] for r in rows})

    task_buttons = "".join(f'<button data-task="{html.escape(t)}">{html.escape(t)}</button>' for t in tasks)
    model_buttons = "".join(f'<button data-model="{html.escape(m)}">{html.escape(m)}</button>' for m in models)
    columns_head = "".join(f"<th data-col='{c}'>{c}</th>" for c in COLUMNS)

    rows_json = json.dumps(rows, ensure_ascii=False)
    columns_json = json.dumps(COLUMNS, ensure_ascii=False)

    html_doc = """<!doctype html>
<html lang="zh-cn">
<meta charset="utf-8">
<title>summary_success_rates</title>
<style>
:root {
  --bg:#1e1f24; --panel:#2a2c33; --border:#3a3d46; --text:#e6e6e6;
  --accent:#4e8cff; --accent-hover:#6aa4ff; --danger:#ff6666;
  --radius:6px; --pad:10px; --trans:.18s ease;
  --success:#2e8b57;
}
* { box-sizing:border-box; }
body {
  margin:0; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial;
  background:var(--bg); color:var(--text);
}
h1 { margin:0 0 12px; font-size:18px; }
a { color:var(--accent); text-decoration:none; }
a:hover { color:var(--accent-hover); }
.layout {
  display:grid;
  grid-template-columns:240px 1fr;
  height:100vh;
}
.sidebar {
  border-right:1px solid var(--border);
  padding:14px 12px;
  overflow:auto;
  background:var(--panel);
  position:relative; z-index:5; /* 新增：确保可点击 */
}
.sidebar section + section { margin-top:18px; }
.sidebar h2 {
  font-size:13px; margin:0 0 6px; letter-spacing:.5px; text-transform:uppercase; color:#bbb;
}
.filter-select, .search-box {
  width:100%; padding:6px 8px; margin-bottom:8px;
  background:#1d1e22; border:1px solid var(--border); color:var(--text); border-radius:var(--radius);
}
.tag-list button {
  display:block; width:100%; text-align:left;
  background:#1d1e22; border:1px solid var(--border);
  color:var(--text); padding:5px 8px; margin:3px 0; border-radius:4px;
  cursor:pointer; transition:.18s ease;
  font-size:13px;
}
.tag-list button.active, .tag-list button:hover {
  border-color:var(--accent); background:#203048;
}
main {
  padding:14px 18px; overflow:hidden; display:flex; flex-direction:column;
}
.controls-bar {
  display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;
}
button.action {
  background:var(--accent); border:none; color:#fff; padding:6px 12px;
  border-radius:var(--radius); cursor:pointer; font-size:13px;
}
button.action:hover { background:var(--accent-hover); }
.table-wrap {
  flex:1; min-height:0; overflow:auto;
  border:1px solid var(--border); border-radius:var(--radius);
  background:#18191d;
}
table {
  border-collapse:collapse; width:100%;
  font-size:13px;
}
th, td {
  padding:6px 10px; border-bottom:1px solid #25262c;
  vertical-align:middle;
  text-align:center; /* 新增：统一居中 */
}
th {
  position:sticky; top:0; background:#202227; z-index:2;
  cursor:pointer; user-select:none;
}
th.sort-asc::after { content:" ▲"; color:var(--accent); }
th.sort-desc::after { content:" ▼"; color:var(--accent); }
tbody tr:hover { background:#24262d; }
tbody tr.group-row td {
  background:#2d3038; font-weight:600; color:var(--accent);
}
.badge-success {
  background:var(--success); color:#fff; padding:2px 6px; border-radius:10px; font-size:12px;
}
footer {
  margin-top:10px; font-size:11px; opacity:.6;
}
.empty {
  padding:40px; text-align:center; color:#888;
}
.group-header {
  display:flex; align-items:center; justify-content:space-between;
  padding:6px 10px;
}
.group-header button {
  background:none; border:none; color:var(--accent); cursor:pointer; font-size:12px;
}
.highlight-good { background:#1d3227; }
.filter-group { display:flex; flex-direction:column; gap:6px; }
.range-row { display:flex; align-items:center; gap:8px; font-size:12px; }
.range-row input[type=range] { flex:1; }
.toggle-filter { display:flex; align-items:center; gap:6px; font-size:12px; }
</style>
<body>
<div class="layout">
  <aside class="sidebar">
    <h1>运行结果分类</h1>
    <section>
      <h2>任务筛选</h2>
      <div class="tag-list" id="taskList">__TASK_BUTTONS__</div>
    </section>
    <section>
      <h2>模型筛选</h2>
      <div class="tag-list" id="modelList">__MODEL_BUTTONS__</div>
    </section>
    <section>
      <h2>搜索</h2>
      <input id="searchBox" class="search-box" placeholder="关键字(任务/模型/日期)..." />
    </section>
    <section>
      <h2>统计</h2>
      <div id="statsBox" style="font-size:12px; line-height:1.4;"></div>
    </section>
  </aside>
  <main>
    <div class="controls-bar">
      <button class="action" id="resetBtn">重置筛选</button>
      <button class="action" id="expandAllBtn">全部展开</button>
      <button class="action" id="collapseAllBtn">全部折叠</button>
      <div class="filter-group" style="min-width:260px; background:#202227; padding:8px 10px; border:1px solid var(--border); border-radius:6px;">
        <div class="toggle-filter">
          <input type="checkbox" id="enableThreshold" />
          <label for="enableThreshold">启用成功率阈值过滤</label>
        </div>
        <div class="range-row">
          <label for="th50" title="traj_per_task=50 使用">阈值(50轨迹)</label>
          <input id="th50" type="range" min="0" max="1" step="0.01" value="0.8" />
          <span id="th50Val">0.80</span>
        </div>
        <div class="range-row">
          <label for="th10" title="traj_per_task=10 使用">阈值(10轨迹)</label>
          <input id="th10" type="range" min="0" max="1" step="0.01" value="0.4" />
          <span id="th10Val">0.40</span>
        </div>
      </div>
      <div class="filter-group" style="min-width:300px; background:#202227; padding:8px 10px; border:1px solid var(--border); border-radius:6px;">
        <div class="range-row">
          <label for="dateStart">开始日期</label>
          <input id="dateStart" type="date" />
          <label for="dateEnd">结束日期</label>
          <input id="dateEnd" type="date" />
        </div>
        <div class="toggle-filter">
          <input type="checkbox" id="onlyLatest" />
          <label for="onlyLatest">每个(任务,模型)仅显示最近一次</label>
        </div>
      </div>
    </div>
    <div class="table-wrap">
      <table id="dataTable">
        <thead>
          <tr>__COLUMNS_HEAD__</tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
      <div id="emptyBox" class="empty" style="display:none;">无匹配数据</div>
    </div>
    <footer>生成时间: <span id="genTime"></span></footer>
  </main>
</div>
<script>
const RAW_ROWS = __ROWS_JSON__;
const COLUMNS = __COLUMNS_JSON__;
let currentTask = null;
let currentModel = null;
let searchText = "";
let sortCol = null;
let sortDir = 1;
let collapsedGroups = new Set();

let enableThreshold = false;
let threshold50 = 0.8;
let threshold10 = 0.4;

let dateStart = null;
let dateEnd = null;
let onlyLatest = false;

function passThreshold(row) {
  if (!enableThreshold) return true;
  const v = parseFloat(row.average_success_rate)||0;
  const tpt = parseInt(row.traj_per_task)||10;
  const th = (tpt >= 40) ? threshold50 : threshold10;
  return v >= th;
}

function setActive(buttons, attr, value) {
  buttons.forEach(b => {
    b.classList.toggle('active', b.getAttribute(attr) == value);
  });
}

function getTimestamp(r) {
  // 允许 r 为原始行或 {date,time}
  const dRaw = (r.date || "").trim();
  const tRaw = (r.time || "").trim();

  if (!dRaw) return 0;
  // 拆分日期
  let [Y,M,D] = dRaw.split("-");
  if (!Y || !M || !D) return 0;
  // 补零
  if (M.length === 1) M = "0"+M;
  if (D.length === 1) D = "0"+D;

  // 拆分时间（允许缺失或只有 HH:MM）
  let h="00", m="00", s="00";
  if (tRaw) {
    const parts = tRaw.split(":");
    h = (parts[0]||"00").padStart(2,"0");
    m = (parts[1]||"00").padStart(2,"0");
    s = (parts[2]||"00").padStart(2,"0");
  }
  const iso = `${Y}-${M}-${D}T${h}:${m}:${s}Z`; // 以 UTC 处理，避免本地时区偏移
  const ts = Date.parse(iso);
  return isNaN(ts) ? 0 : ts;
}

function render() {
  let rows = RAW_ROWS.slice();
  if (currentTask) rows = rows.filter(r => r.libero_task === currentTask);
  if (currentModel) rows = rows.filter(r => r.model === currentModel);
  if (searchText) {
    const s = searchText.toLowerCase();
    rows = rows.filter(r =>
      (r.libero_task||"").toLowerCase().includes(s) ||
      (r.model||"").toLowerCase().includes(s) ||
      (r.date||"").toLowerCase().includes(s)
    );
  }
  rows = rows.filter(passThreshold);

  if (dateStart || dateEnd) {
    rows = rows.filter(r => {
      const d = r.date || "";
      return (!dateStart || d >= dateStart) && (!dateEnd || d <= dateEnd);
    });
  }

  if (onlyLatest) {
    const latestMap = new Map();
    for (const r of rows) {
      const key = r.libero_task + "||" + r.model;
      const cur = latestMap.get(key);
      const a = getTimestamp(r);
      const b = cur ? getTimestamp(cur) : -1;
      if (!cur || a > b) latestMap.set(key, r);
    }
    rows = [...latestMap.values()];
  }

  // 排序（日期/时间只做组内排序，不做全局排序）
  const groupOnlyCols = new Set(['date','time']);
  let rowsForGrouping;

  if (sortCol) {
    if (groupOnlyCols.has(sortCol)) {
      // 保持当前 rows 原始过滤顺序，不全局排序
      rowsForGrouping = rows.slice();
    } else {
      // 其他列仍做全局排序
      rows.sort((a,b) => {
        let av = a[sortCol]; let bv = b[sortCol];
        if (sortCol === 'date' || sortCol === 'time') {
          return (getTimestamp(a) - getTimestamp(b)) * sortDir;
        }
        if (av===undefined) av=""; if (bv===undefined) bv="";
        const an = parseFloat(av); const bn = parseFloat(bv);
        if (!isNaN(an) && !isNaN(bn)) return (an - bn)*sortDir;
        return (''+av).localeCompare(''+bv,'zh-CN')*sortDir;
      });
      rowsForGrouping = rows;
    }
  } else {
    // 默认全局按 (task,model,date,time)
    rows.sort((a,b) => {
      const keyA = [a.libero_task,a.model,a.date,a.time].join('|');
      const keyB = [b.libero_task,b.model,b.date,b.time].join('|');
      return keyA.localeCompare(keyB,'zh-CN');
    });
    rowsForGrouping = rows;
  }

  const tbody = document.getElementById('tbody');
  tbody.innerHTML = "";
  document.getElementById('emptyBox').style.display = rows.length ? "none":"block";

  // 构建分组（组顺序使用 rowsForGrouping 的出现顺序，避免日期/时间点击后整体重新排列）
  const groupedOrder = [];
  const groupedMap = new Map();
  rowsForGrouping.forEach(r => {
    const key = r.libero_task + "||" + r.model;
    if (!groupedMap.has(key)) {
      groupedMap.set(key, []);
      groupedOrder.push(key);
    }
    groupedMap.get(key).push(r);
  });

  // 组内排序：日期/时间点击时在组内按时间戳排序并可反转
  const cmp = (a, b) => {
    if (!sortCol) {
      const keyA = [a.libero_task,a.model,a.date,a.time].join('|');
      const keyB = [b.libero_task,b.model,b.date,b.time].join('|');
      return keyA.localeCompare(keyB,'zh-CN');
    }
    if (groupOnlyCols.has(sortCol)) {
      return (getTimestamp(a) - getTimestamp(b)) * sortDir;
    }
    // 其他列已全局排序，这里再排一次保持一致
    let av = a[sortCol]; let bv = b[sortCol];
    if (sortCol === 'date' || sortCol === 'time') {
      return (getTimestamp(a) - getTimestamp(b)) * sortDir;
    }
    if (av===undefined) av=""; if (bv===undefined) bv="";
    const an = parseFloat(av); const bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return (an - bn) * sortDir;
    return (''+av).localeCompare(''+bv,'zh-CN') * sortDir;
  };
  if (sortCol) {
    for (const key of groupedOrder) {
      groupedMap.get(key).sort(cmp);
    }
  }

  for (const key of groupedOrder) {
    const [task, model] = key.split("||");
    const groupRows = groupedMap.get(key);
    const trh = document.createElement('tr');
    trh.className = 'group-row';
    const td = document.createElement('td');
    td.colSpan = COLUMNS.length;
    const goodCount = groupRows.filter(r => (parseFloat(r.average_success_rate)||0) >= threshold50).length;
    td.innerHTML = '<div class="group-header">'
      + '<div><strong>任务:</strong> ' + task + ' &nbsp; <strong>模型:</strong> ' + model
      + ' &nbsp; <span style="opacity:.7">显示 ' + groupRows.length + ' 条 (高成功率≥' + threshold50.toFixed(2) + ': ' + goodCount + ')</span></div>'
      + '<div><button data-gkey="' + key + '" class="toggleBtn">' + (collapsedGroups.has(key) ? "展开" : "折叠") + '</button></div>'
      + '</div>';
    trh.appendChild(td);
    tbody.appendChild(trh);

    if (!collapsedGroups.has(key)) {
      for (const r of groupRows) {
        const tr = document.createElement('tr');
        const rate = parseFloat(r.average_success_rate)||0;
        const tpt = parseInt(r.traj_per_task)||10;
        const highTh = (tpt >= 40) ? threshold50 : threshold10;
        if (rate >= highTh) tr.classList.add('highlight-good');
        for (const c of COLUMNS) {
          const td2 = document.createElement('td');
          let val = r[c] ?? "";
          if (c === "average_success_rate" && val !== "") {
            const num = rate;
            const color = num >= highTh ? "badge-success" : "";
            td2.innerHTML = "<span class='" + color + "'>" + num.toFixed(3) + "</span>";
          } else {
            td2.textContent = val;
          }
          tr.appendChild(td2);
        }
        tbody.appendChild(tr);
      }
    }
  }

  const statsBox = document.getElementById('statsBox');
  const avg = rows.length
    ? (rows.map(r => parseFloat(r.average_success_rate)||0).reduce((a,b)=>a+b,0) / rows.length)
    : 0;
  statsBox.innerHTML =
    "当前匹配: <b>" + rows.length + "</b><br>"
    + "平均成功率: <b>" + avg.toFixed(3) + "</b><br>"
    + "过滤: " + (enableThreshold
        ? ("<span style='color:var(--accent)'>开启</span> (50轨迹≥" + threshold50.toFixed(2) + ", 10轨迹≥" + threshold10.toFixed(2) + ")")
        : "<span style='opacity:.6'>关闭</span>");
}

// 点击表头处，初次点击 date/time 默认降序（最新在前）
document.getElementById('dataTable').addEventListener('click', e => {
  if (e.target.tagName === 'TH') {
    const col = e.target.getAttribute('data-col');
    if (sortCol === col) {
      sortDir = -sortDir;
    } else {
      sortCol = col;
      // date 或 time 初次设为降序
      sortDir = (col === 'date' || col === 'time') ? -1 : 1;
    }
    [...document.querySelectorAll('th')].forEach(th => th.classList.remove('sort-asc','sort-desc'));
    e.target.classList.add(sortDir === 1 ? 'sort-asc':'sort-desc');
    render();
  }
});

document.getElementById('enableThreshold').addEventListener('change', e => {
  enableThreshold = e.target.checked;
  render();
});
document.getElementById('th50').addEventListener('input', e => {
  threshold50 = parseFloat(e.target.value);
  document.getElementById('th50Val').textContent = threshold50.toFixed(2);
  if (enableThreshold) render();
});
document.getElementById('th10').addEventListener('input', e => {
  threshold10 = parseFloat(e.target.value);
  document.getElementById('th10Val').textContent = threshold10.toFixed(2);
  if (enableThreshold) render();
});
document.getElementById('dateStart').addEventListener('change', e => {
  dateStart = e.target.value || null;
  render();
});
document.getElementById('dateEnd').addEventListener('change', e => {
  dateEnd = e.target.value || null;
  render();
});
document.getElementById('onlyLatest').addEventListener('change', e => {
  onlyLatest = e.target.checked;
  render();
});
document.getElementById('taskList').addEventListener('click', e => {
  if (e.target.tagName === 'BUTTON') {
    const t = e.target.getAttribute('data-task');
    currentTask = (currentTask === t) ? null : t;
    setActive([...document.querySelectorAll('#taskList button')],'data-task',currentTask);
    render();
  }
});
document.getElementById('modelList').addEventListener('click', e => {
  if (e.target.tagName === 'BUTTON') {
    const m = e.target.getAttribute('data-model');
    currentModel = (currentModel === m) ? null : m;
    setActive([...document.querySelectorAll('#modelList button')],'data-model',currentModel);
    render();
  }
});
document.getElementById('searchBox').addEventListener('input', e => {
  searchText = e.target.value.trim();
  render();
});

document.getElementById('resetBtn').onclick = () => {
  currentTask = null;
  currentModel = null;
  searchText = "";
  sortCol = null;
  sortDir = 1;
  collapsedGroups.clear();
  enableThreshold = false;
  threshold50 = 0.8;
  threshold10 = 0.4;
  dateStart = null;
  dateEnd = null;
  onlyLatest = false;

  // UI 元素复位
  document.getElementById('searchBox').value = "";
  document.getElementById('enableThreshold').checked = false;
  document.getElementById('th50').value = threshold50;
  document.getElementById('th10').value = threshold10;
  document.getElementById('th50Val').textContent = threshold50.toFixed(2);
  document.getElementById('th10Val').textContent = threshold10.toFixed(2);
  document.getElementById('dateStart').value = "";
  document.getElementById('dateEnd').value = "";
  document.getElementById('onlyLatest').checked = false;
  [...document.querySelectorAll('#taskList button,#modelList button')].forEach(b => b.classList.remove('active'));
  [...document.querySelectorAll('th')].forEach(th => th.classList.remove('sort-asc','sort-desc'));
  render();
};
document.getElementById('expandAllBtn').onclick = () => {
  collapsedGroups.clear(); render();
};
document.getElementById('collapseAllBtn').onclick = () => {
  collapsedGroups = new Set(RAW_ROWS.map(r => r.libero_task + "||" + r.model));
  render();
};
document.getElementById('tbody').addEventListener('click', e => {
  if (e.target.classList.contains('toggleBtn')) {
    const k = e.target.getAttribute('data-gkey');
    if (collapsedGroups.has(k)) collapsedGroups.delete(k); else collapsedGroups.add(k);
    render();
  }
});
document.getElementById('genTime').textContent = new Date().toLocaleString('zh-CN');
render();
</script>
</body>
</html>"""

    html_doc = (html_doc
                .replace("__TASK_BUTTONS__", task_buttons)
                .replace("__MODEL_BUTTONS__", model_buttons)
                .replace("__COLUMNS_HEAD__", columns_head)
                .replace("__ROWS_JSON__", rows_json)
                .replace("__COLUMNS_JSON__", columns_json))

    with p.open("w", encoding="utf-8") as f:
        f.write(html_doc)

def main():
    data = load_json(INPUT_JSON)
    rows = to_rows(data)
    write_csv(rows, OUTPUT_CSV)
    write_html(rows, OUTPUT_HTML)
    print(f"Wrote:\n- {OUTPUT_CSV}\n- {OUTPUT_HTML}")

if __name__ == "__main__":
    main()