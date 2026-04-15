const ROOT = (() => {
  const p = location.pathname;
  return p.endsWith("/") ? p : p.replace(/\/[^/]*$/, "/");
})();

function joinUrl(base, path) {
  if (!base) return path;
  if (/^https?:\/\//.test(path)) return path;
  return `${base.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

async function fetchJson(url, fallback = null) {
  try {
    const r = await fetch(url, { cache: "no-cache" });
    if (!r.ok) return fallback;
    return await r.json();
  } catch (_) {
    return fallback;
  }
}

async function fetchJsonl(url, fallback = []) {
  try {
    const r = await fetch(url, { cache: "no-cache" });
    if (!r.ok) return fallback;
    const text = await r.text();
    return text
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean)
      .map((line) => {
        try {
          return JSON.parse(line);
        } catch (_) {
          return null;
        }
      })
      .filter(Boolean);
  } catch (_) {
    return fallback;
  }
}

export function parseRuntimeConfig() {
  const qp = new URLSearchParams(location.search);
  const mode = (qp.get("mode") || "static").toLowerCase();
  const apiBase = qp.get("api_base") || qp.get("apiBase") || "";
  return { mode: mode === "live" ? "live" : "static", apiBase };
}

export class DataSource {
  constructor(config = {}) {
    const runtime = parseRuntimeConfig();
    this.mode = config.mode || runtime.mode || "static";
    this.apiBase = config.apiBase || runtime.apiBase || "";
    this.root = config.root || ROOT;
  }

  async loadStaticBundle() {
    const [demo, metrics, timeline, verified] = await Promise.all([
      fetchJson(joinUrl(this.root, "data/demo_cases.json"), { project: {}, cases: [] }),
      fetchJson(joinUrl(this.root, "data/metrics.json"), {}),
      fetchJson(joinUrl(this.root, "data/timeline.json"), { cases: {} }),
      fetchJson(joinUrl(this.root, "data/verified_events.json"), { cases: {} }),
    ]);
    return { mode: "static", project: demo.project || {}, cases: demo.cases || [], metrics, timeline, verified };
  }

  async loadLiveBundle() {
    const listCasesUrl = joinUrl(this.apiBase, "/api/list_cases");
    const apiCases = (await fetchJson(listCasesUrl, [])) || [];
    const picked = apiCases.slice(0, 3);
    const cases = [];
    const timeline = { cases: {} };
    const verified = { cases: {} };

    for (const item of picked) {
      const caseId = item.video_id || item.case_id;
      if (!caseId) continue;
      const timelineData = await fetchJson(joinUrl(this.apiBase, `/api/timeline/${encodeURIComponent(caseId)}`), { items: [] });
      const media = await fetchJson(joinUrl(this.apiBase, `/api/media/${encodeURIComponent(caseId)}`), {});
      const transcript = await fetchJson(joinUrl(this.apiBase, `/api/transcript/${encodeURIComponent(caseId)}`), []);

      timeline.cases[caseId] = (timelineData.items || []).slice(0, 16).map((x) => ({
        start: Number(x.start ?? 0),
        end: Number(x.end ?? x.start ?? 0),
        stream: x.type || "visual",
        label: x.action_label || x.event_type || "unknown",
        score: Number(x.reliability ?? x.match_score ?? 0),
      }));

      let verifiedRows = [];
      if (media && media.verified_events) {
        verifiedRows = await fetchJsonl(joinUrl(this.apiBase, media.verified_events), []);
      }
      if (!verifiedRows.length && Array.isArray(transcript) && transcript.length) {
        verifiedRows = transcript
          .filter((x) => x && x.verification_label)
          .slice(0, 8)
          .map((x, idx) => ({
            query_id: `live_${caseId}_${idx}`,
            query_time: Number(x.start ?? 0),
            semantic_label: x.event_type || "unknown",
            final_label: x.verification_label,
            decision: x.verification_label,
            reliability_score: Number(x.reliability ?? 0),
          }));
      }
      verified.cases[caseId] = verifiedRows.slice(0, 8);

      cases.push({
        case_id: caseId,
        title: `Live Case ${caseId}`,
        scenario: "Live API preview",
        analysis: "Loaded from backend API. Use static mode for paper recording.",
        media: {
          frame: media?.timeline_png || null,
          video: media?.overlay || media?.pose_demo || null,
        },
        timeline_ref: caseId,
        verified_events_ref: caseId,
        tags: ["live_api"],
      });
    }

    return {
      mode: "live",
      project: {
        name: "YOLOv11 Vision-Semantic Dual Verification Demo (Live API)",
        purpose: "Future backend-connected preview",
      },
      cases,
      metrics: { note: "live mode does not ship fixed paper metrics by default" },
      timeline,
      verified,
    };
  }

  async loadBundle() {
    if (this.mode === "live") {
      const live = await this.loadLiveBundle();
      if ((live.cases || []).length) return live;
    }
    return this.loadStaticBundle();
  }
}

export async function loadDemoBundle(config = {}) {
  const ds = new DataSource(config);
  return ds.loadBundle();
}
