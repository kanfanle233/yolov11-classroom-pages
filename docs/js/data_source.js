const BASE = (() => {
  const p = location.pathname;
  return p.endsWith('/') ? p : p.replace(/\/[^\/]*$/, '/');
})();

async function _json(url, fallback = null) {
  try {
    const r = await fetch(url, { cache: 'no-cache' });
    if (!r.ok) return fallback;
    return await r.json();
  } catch (e) {
    return fallback;
  }
}

export async function listCases() {
  return (await _json(`${BASE}data/list_cases.json`, [])) || [];
}

export async function loadCaseFiles(caseId) {
  const root = `${BASE}data/cases/${encodeURIComponent(caseId)}/`;
  const [timeline, proj, transcript, tracks] = await Promise.all([
    _json(root + 'timeline_viz.json', { items: [] }),
    _json(root + 'student_projection.json', []),
    fetch(root + 'transcript.jsonl', { cache: 'no-cache' })
      .then(r => (r.ok ? r.text() : ''))
      .then(text => text.split('\n').map(s => s.trim()).filter(Boolean).map(line => JSON.parse(line)))
      .catch(() => []),
    fetch(root + 'pose_tracks_smooth.jsonl', { cache: 'no-cache' })
      .then(r => (r.ok ? r.text() : ''))
      .then(text => {
        const out = {};
        text.split('\n').map(s => s.trim()).filter(Boolean).forEach(line => {
          const d = JSON.parse(line);
          out[String(d.frame)] = (d.persons || []).map(pp => ({ id: pp.track_id, box: pp.bbox }));
        });
        return out;
      })
      .catch(() => ({})),
  ]);

  return { timeline, proj, transcript, tracks };
}

export function videoUrl(caseId) {
  return `${BASE}assets/videos/${encodeURIComponent(caseId)}.mp4`;
}
