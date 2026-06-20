/**
 * Main page wiring.
 *
 * IMPORTANT — no demo/placeholder numbers:
 * The stat cards do NOT ship with hardcoded figures. They read from
 * ./stats.json, which you should generate from your actual pipeline
 * outputs (the same allocations_*.json / vehicle_risk.parquet /
 * edi_scores_with_flow.parquet files the Streamlit dashboard reads).
 *
 * Expected shape of stats.json:
 * {
 *   "grid_hours_scored": 61234,
 *   "critical_vehicles": 142,
 *   "red_zones": 318,
 *   "officers_allocated": 60,
 *   "generated_at": "2026-06-18T00:00:00Z"
 * }
 *
 * A tiny Python snippet to generate it from your pipeline lives in
 * generate_stats.py alongside this file — run it after your normal
 * module5_edi / module6_optimizer / module3_repeat_offender scripts.
 *
 * If stats.json is missing or fails to load, each card shows
 * "NO DATA" rather than a fabricated number — never silently mocked.
 */

(function () {
  const STAT_MAP = {
    'stat-gridhours': 'grid_hours_scored',
    'stat-critical': 'critical_vehicles',
    'stat-zones': 'red_zones',
    'stat-officers': 'officers_allocated',
  };

  function showNoData() {
    Object.keys(STAT_MAP).forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.removeAttribute('data-counter');
      el.textContent = '—';
      el.title = 'No stats.json found — run generate_stats.py against your pipeline outputs.';
    });
    const sourceNote = document.querySelector('.stats-source');
    if (sourceNote) {
      sourceNote.textContent =
        'No stats.json found yet — run generate_stats.py against your pipeline outputs to populate these.';
    }
  }

  async function loadStats() {
    try {
      const candidateUrls = [
        './stats.json',
        'stats.json',
        './app/stats.json',
        'app/stats.json',
        './landing/stats.json',
        'landing/stats.json',
        '/app/stats.json',
      ];

      let data = null;
      for (const url of candidateUrls) {
        try {
          const res = await fetch(url, { cache: 'no-store' });
          if (res.ok) {
            data = await res.json();
            break;
          }
        } catch (err) {
          // Keep trying the next candidate path.
        }
      }

      if (!data) throw new Error('stats.json not found');

      Object.entries(STAT_MAP).forEach(([elId, key]) => {
        const el = document.getElementById(elId);
        if (!el) return;
        const value = data[key];
        if (typeof value !== 'number') {
          el.textContent = '—';
          return;
        }
        el.dataset.target = String(value);
        if (window.IntelliParkCounter) {
          window.IntelliParkCounter.animateCount(el);
        } else {
          el.textContent = value.toLocaleString();
        }
      });

      const sourceNote = document.querySelector('.stats-source');
      if (sourceNote && data.generated_at) {
        const d = new Date(data.generated_at);
        sourceNote.textContent = `Figures generated from pipeline outputs at ${d.toLocaleString()}.`;
      }
    } catch (err) {
      showNoData();
    }
  }

  // Lets you set the real dashboard URL once in one place.
  function wireDashboardLinks() {
    // The React dashboard's dev server (vite). Change this if you run
    // `npm run dev` on a different port, or to your deployed dashboard URL.
    const DASHBOARD_URL = 'http://localhost:5173';

    const links = [
      document.getElementById('dashboard-cta-link'),
      document.querySelector('a[href="#dashboard-link"]'),
    ];

    if (!DASHBOARD_URL) return; // leave as in-page anchor until set

    links.forEach((link) => {
      if (link) link.setAttribute('href', DASHBOARD_URL);
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    wireDashboardLinks();
  });
})();
