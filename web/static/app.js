// =============================================================================
// app.js — Frontend logic for Hiver AI Support Agent Dashboard
// =============================================================================

const API_BASE = '';
let goldenSetData = [];

// ─── Tab Navigation ───
document.querySelectorAll('.nav-tabs li').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.nav-tabs li').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
  });
});

// ─── On Load ───
document.addEventListener('DOMContentLoaded', () => {
  loadQuickTests();
  loadBenchmark();
  loadGoldenSet();
  loadBanking77();
});

// ─── Quick Tests ───
async function loadQuickTests() {
  try {
    const resp = await fetch(`${API_BASE}/api/quick_tests`);
    const tests = await resp.json();
    const container = document.getElementById('quickTests');
    container.innerHTML = '';
    tests.forEach(test => {
      const btn = document.createElement('button');
      btn.className = 'quick-test-btn';
      btn.textContent = test.label;
      btn.onclick = () => {
        document.getElementById('customerInput').value = test.text;
        predict();
      };
      container.appendChild(btn);
    });
  } catch (e) {
    console.error('Failed to load quick tests:', e);
  }
}

// ─── Prediction ───
async function predict() {
  const text = document.getElementById('customerInput').value.trim();
  if (!text) return;

  const btn = document.getElementById('predictBtn');
  btn.classList.add('loading');
  btn.disabled = true;

  try {
    const resp = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const result = await resp.json();
    renderResults(result);
  } catch (e) {
    document.getElementById('resultsPanel').innerHTML = `
      <div class="card" style="border-color: var(--accent-rose);">
        <p style="color: var(--accent-rose);">⚠️ Error: ${e.message}. Make sure the server is running.</p>
      </div>`;
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

function renderResults(r) {
  const isEscalate = r.escalation_decision === 'ESCALATE';
  const badgeClass = isEscalate ? 'badge-escalate' : 'badge-auto';
  const badgeIcon = isEscalate ? '🚨' : '✅';
  const confPct = Math.round(r.intent_confidence * 100);
  const confColor = confPct > 70 ? 'var(--accent-emerald)' : confPct > 40 ? 'var(--accent-amber)' : 'var(--accent-rose)';

  let retrievedHtml = '';
  if (r.retrieved_cases && r.retrieved_cases.length > 0) {
    r.retrieved_cases.forEach((c, i) => {
      retrievedHtml += `
        <div class="retrieved-case">
          <div class="similarity">Match #${i+1} — ${(c.similarity * 100).toFixed(1)}% similar</div>
          <div style="margin-top: 0.25rem; color: var(--text-secondary);">
            <strong>Q:</strong> ${escapeHtml(c.customer_query.substring(0, 120))}${c.customer_query.length > 120 ? '...' : ''}
          </div>
          <div style="margin-top: 0.25rem; color: var(--text-primary);">
            <strong>A:</strong> ${autolink(c.support_response.substring(0, 150))}${c.support_response.length > 150 ? '...' : ''}
          </div>
        </div>`;
    });
  }

  let factorsHtml = '';
  if (r.escalation_factors && r.escalation_factors.length > 0) {
    r.escalation_factors.forEach(f => {
      factorsHtml += `<span class="factor-tag">${f}</span>`;
    });
  }

  const formatIntent = (i) => i.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

  document.getElementById('resultsPanel').innerHTML = `
    <!-- Intent + Escalation -->
    <div class="card">
      <div class="card-title">
        <div class="icon" style="background: rgba(99, 102, 241, 0.2);">🎯</div>
        Classification & Decision
      </div>
      <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.75rem;">
        <span class="result-badge badge-intent">🏷️ ${formatIntent(r.intent)}</span>
        <span class="result-badge ${badgeClass}">${badgeIcon} ${r.escalation_decision}</span>
      </div>
      <div style="font-size: 0.8125rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
        Confidence: ${confPct}%
      </div>
      <div class="confidence-bar">
        <div class="confidence-bar-fill" style="width: ${confPct}%; background: ${confColor};"></div>
      </div>
      ${factorsHtml ? `<div style="margin-top: 0.75rem;">${factorsHtml}</div>` : ''}
      <div style="margin-top: 0.75rem; font-size: 0.8125rem; color: var(--text-secondary);">
        <strong>Reason:</strong> ${escapeHtml(r.escalation_reason)}
      </div>
    </div>

    <!-- Generated Response -->
    <div class="card">
      <div class="card-title">
        <div class="icon" style="background: rgba(16, 185, 129, 0.2);">💬</div>
        Generated Reply
        <span style="font-size: 0.75rem; color: var(--text-muted); margin-left: auto;">via ${r.response_method}</span>
      </div>
      <div class="response-box">${autolink(r.response)}</div>
    </div>

    <!-- Retrieved Cases -->
    <div class="card">
      <div class="card-title">
        <div class="icon" style="background: rgba(6, 182, 212, 0.2);">📚</div>
        Retrieved Historical Cases
      </div>
      ${retrievedHtml || '<p style="color: var(--text-muted);">No cases retrieved.</p>'}
    </div>
  `;
}

// ─── Benchmark ───
async function loadBenchmark() {
  try {
    const resp = await fetch(`${API_BASE}/api/benchmark`);
    if (!resp.ok) throw new Error('No benchmark data');
    const data = await resp.json();
    renderBenchmark(data);
  } catch (e) {
    document.getElementById('benchmarkContent').innerHTML = `
      <div class="placeholder">
        <div class="icon-large">📊</div>
        <p>No benchmark results yet. Run <code>python run_pipeline.py</code> first.</p>
      </div>`;
  }
}

function renderBenchmark(data) {
  const agents = data.agents || {};
  const names = Object.keys(agents);
  if (names.length === 0) {
    document.getElementById('benchmarkContent').innerHTML = '<p>No results found.</p>';
    return;
  }

  // Metrics grid for proposed agent
  const proposed = agents['proposed_agent'] || agents[names[names.length - 1]];
  let metricsHtml = '';
  if (proposed) {
    const metrics = [
      { label: 'Intent Accuracy', value: proposed.intent_accuracy, fmt: v => `${(v*100).toFixed(1)}%` },
      { label: 'Intent Macro F1', value: proposed.intent_macro_f1, fmt: v => `${(v*100).toFixed(1)}%` },
      { label: 'Escalation F1', value: proposed.escalation_f1, fmt: v => `${(v*100).toFixed(1)}%` },
      { label: 'False Auto-handle ↓', value: proposed.false_auto_handle_rate, fmt: v => `${(v*100).toFixed(1)}%` },
      { label: 'ROUGE-1', value: proposed.rouge1, fmt: v => v.toFixed(3) },
      { label: 'ROUGE-L', value: proposed.rougeL, fmt: v => v.toFixed(3) },
    ];
    metricsHtml = '<div class="metrics-grid" style="margin-bottom: 2rem;">';
    metrics.forEach(m => {
      metricsHtml += `
        <div class="metric-card">
          <div class="metric-value">${m.fmt(m.value)}</div>
          <div class="metric-label">${m.label}</div>
        </div>`;
    });
    metricsHtml += '</div>';
  }

  // Comparison table
  const tableMetrics = [
    { key: 'intent_accuracy', label: 'Intent Accuracy' },
    { key: 'intent_macro_f1', label: 'Intent Macro F1' },
    { key: 'escalation_accuracy', label: 'Escalation Accuracy' },
    { key: 'escalation_f1', label: 'Escalation F1' },
    { key: 'false_auto_handle_rate', label: 'False Auto-handle Rate ↓' },
    { key: 'rouge1', label: 'ROUGE-1' },
    { key: 'rougeL', label: 'ROUGE-L' },
  ];

  let tableHtml = `<div class="card"><div class="card-title">📋 Agent Comparison</div>
    <table class="comparison-table"><thead><tr><th>Metric</th>`;
  names.forEach(n => { tableHtml += `<th>${formatAgentName(n)}</th>`; });
  tableHtml += '</tr></thead><tbody>';

  tableMetrics.forEach(m => {
    let vals = names.map(n => agents[n][m.key] ?? 0);
    const isLowerBetter = m.key.includes('false_auto');
    const bestIdx = isLowerBetter ? vals.indexOf(Math.min(...vals)) : vals.indexOf(Math.max(...vals));

    tableHtml += `<tr><td>${m.label}</td>`;
    vals.forEach((v, i) => {
      const cls = i === bestIdx ? 'winner' : '';
      tableHtml += `<td class="${cls}">${typeof v === 'number' ? v.toFixed(4) : v}</td>`;
    });
    tableHtml += '</tr>';
  });
  tableHtml += '</tbody></table></div>';

  document.getElementById('benchmarkContent').innerHTML = metricsHtml + tableHtml;
}

// ─── Golden Set ───
async function loadGoldenSet() {
  try {
    const resp = await fetch(`${API_BASE}/api/golden_set`);
    if (!resp.ok) throw new Error('No golden set');
    const data = await resp.json();
    goldenSetData = data.examples || [];
    renderGoldenSet(goldenSetData);
  } catch (e) {
    document.getElementById('goldenCards').innerHTML = `
      <div class="placeholder">
        <div class="icon-large">🏅</div>
        <p>No golden set data. Run <code>python run_pipeline.py</code> first.</p>
      </div>`;
  }
}

function filterGolden() {
  const search = document.getElementById('goldenSearch').value.toLowerCase();
  const intent = document.getElementById('goldenIntentFilter').value;
  const diff = document.getElementById('goldenDiffFilter').value;

  const filtered = goldenSetData.filter(item => {
    if (search && !item.customer_tweet.toLowerCase().includes(search)) return false;
    if (intent && item.gold_intent !== intent) return false;
    if (diff && item.difficulty_tag !== diff) return false;
    return true;
  });

  renderGoldenSet(filtered);
}

function renderGoldenSet(items) {
  const container = document.getElementById('goldenCards');
  if (!items.length) {
    container.innerHTML = '<div class="placeholder"><p>No matching examples found.</p></div>';
    return;
  }

  container.innerHTML = items.slice(0, 50).map(item => {
    const escBadge = item.gold_escalation === 'ESCALATE' ? 'badge-escalate' : 'badge-auto';
    return `
      <div class="golden-card">
        <div class="tweet-text">${escapeHtml(item.customer_tweet)}</div>
        <div class="meta">
          <span class="result-badge badge-intent">${item.gold_intent.replace(/_/g, ' ')}</span>
          <span class="result-badge ${escBadge}">${item.gold_escalation}</span>
          <span class="result-badge" style="background: rgba(139,92,246,0.1); color: var(--accent-violet); border: 1px solid rgba(139,92,246,0.3);">${item.difficulty_tag}</span>
        </div>
      </div>`;
  }).join('');

  if (items.length > 50) {
    container.innerHTML += `<div class="placeholder"><p>Showing 50 of ${items.length} results.</p></div>`;
  }
}

// ─── Banking77 ───
async function loadBanking77() {
  try {
    const resp = await fetch(`${API_BASE}/api/banking77`);
    if (!resp.ok) throw new Error('No banking77 data');
    const data = await resp.json();
    renderBanking77(data);
  } catch (e) {
    document.getElementById('banking77Content').innerHTML = `
      <div class="placeholder">
        <div class="icon-large">🏦</div>
        <p>No Banking77 data. Run <code>python run_pipeline.py</code> first.</p>
      </div>`;
  }
}

function renderBanking77(data) {
  const analysis = data.analysis || {};
  const classifier = data.classifier || {};
  const stats = analysis.statistics || {};
  const mapping = analysis.mapping_analysis || {};
  const insights = analysis.insights || {};

  let html = '';

  // Stats grid
  html += '<div class="metrics-grid" style="margin-bottom: 2rem;">';
  const statsItems = [
    { label: 'Total Examples', value: stats.total_examples?.toLocaleString() || 'N/A' },
    { label: 'Intent Classes', value: stats.num_intents || 'N/A' },
    { label: 'Avg per Intent', value: stats.avg_examples_per_intent || 'N/A' },
    { label: 'Mapping Coverage', value: `${mapping.coverage_pct || 0}%` },
    { label: 'Cross-Domain Acc', value: classifier.accuracy ? `${(classifier.accuracy*100).toFixed(1)}%` : 'N/A' },
  ];
  statsItems.forEach(s => {
    html += `<div class="metric-card"><div class="metric-value">${s.value}</div><div class="metric-label">${s.label}</div></div>`;
  });
  html += '</div>';

  // Apple mapping distribution
  if (mapping.apple_distribution) {
    html += '<div class="card" style="margin-bottom: 1rem;"><div class="card-title">🗺️ Banking77 → AppleSupport Mapping</div>';
    html += '<table class="comparison-table"><thead><tr><th>Apple Intent Class</th><th>Mapped Banking77 Examples</th></tr></thead><tbody>';
    Object.entries(mapping.apple_distribution).forEach(([k, v]) => {
      html += `<tr><td>${k.replace(/_/g, ' ')}</td><td>${v}</td></tr>`;
    });
    html += '</tbody></table></div>';
  }

  // Key finding
  if (insights.key_finding) {
    html += `<div class="card"><div class="card-title">💡 Key Insight</div><p style="color: var(--text-secondary); line-height: 1.7;">${escapeHtml(insights.key_finding)}</p></div>`;
  }

  document.getElementById('banking77Content').innerHTML = html;
}

// ─── Utilities ───
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML;
}

function autolink(text) {
  if (!text) return '';
  const safe = escapeHtml(text);
  return safe.replace(/(https?:\/\/[^\s<)]+)/g, function(url) {
    var cleanUrl = url.replace(/[.,;:!?]+$/, '');
    var trailing = url.slice(cleanUrl.length);
    return '<a href="' + cleanUrl + '" target="_blank" rel="noopener noreferrer">' + cleanUrl + '</a>' + trailing;
  });
}

function formatAgentName(name) {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
