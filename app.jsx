import { useEffect, useMemo, useState } from "react";

const ENV = import.meta.env;

// All detector calls now go through the Veil backend (POST /scan). The backend
// holds every API key server-side, so no secrets ship to the browser. Point the
// frontend at the backend with VITE_VEIL_API_URL (defaults to local dev).
const API_BASE = (ENV.VITE_VEIL_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

const ICONS = {
  dashboard: <path d="M4 13h6V4H4v9Zm0 7h6v-5H4v5Zm10 0h6V11h-6v9Zm0-16v5h6V4h-6Z" />,
  upload: <path d="M12 4v11m0-11 4 4m-4-4-4 4M5 16v2a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-2" />,
  history: <path d="M4 12a8 8 0 1 1 3 6.24M4 12V7m0 5H9m3-4v4l3 2" />,
  settings: <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Zm8-3.5a7.9 7.9 0 0 0-.14-1.47l1.9-1.48-2-3.46-2.25.9a8 8 0 0 0-2.55-1.48L14.6 3h-4l-.36 2.5a8 8 0 0 0-2.55 1.48l-2.25-.9-2 3.46 1.9 1.48A7.9 7.9 0 0 0 4 12c0 .5.05 1 .14 1.47l-1.9 1.48 2 3.46 2.25-.9a8 8 0 0 0 2.55 1.48L9.4 21h4l.36-2.5a8 8 0 0 0 2.55-1.48l2.25.9 2-3.46-1.9-1.48c.09-.48.14-.97.14-1.48Z" />,
  sun: <path d="M12 4V2m0 20v-2M4 12H2m20 0h-2M5.6 5.6 4.2 4.2m15.6 1.4 1.4-1.4M5.6 18.4l-1.4 1.4m15.6-1.4 1.4 1.4M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10Z" />,
  moon: <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z" />,
};

const Icon = ({ name }) => (
  <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    {ICONS[name]}
  </svg>
);

const navItems = [
  { id: "dashboard", label: "Dashboard", icon: "dashboard" },
  { id: "upload", label: "Check Image", icon: "upload" },
  { id: "history", label: "History", icon: "history" },
  { id: "settings", label: "Settings", icon: "settings" },
];

// The backend's triangulation engine (app/engine/triangulate.py) is the single
// source of truth for the verdict, fused score, and confidence — the frontend
// no longer recomputes any of this. These five labels are its full vocabulary.
const VERDICT_STYLE = {
  "Likely Authentic": { type: "low" },
  "Low Risk": { type: "low" },
  "Medium Risk": { type: "medium" },
  "High Risk": { type: "high" },
  Inconclusive: { type: "unknown" },
};

const verdictStyle = (verdict) => VERDICT_STYLE[verdict] ?? VERDICT_STYLE.Inconclusive;

const formatPercent = (score) => {
  if (score === null || score === undefined) return "N/A";
  return `${(score * 100).toFixed(1)}%`;
};

const formatBullets = (text) => {
  if (!text) return [];
  return text
    .split(/\r?\n/)
    .map((line) => line.replace(/^[-*]\s*/, "").replace(/^\d+\.\s*/, "").trim())
    .filter(Boolean);
};

// POST the image to the Veil backend and return the raw ScanResponse envelope,
// including the backend-computed `aggregate` (verdict/score/confidence/reasons).
const runScan = async (file) => {
  const formData = new FormData();
  formData.append("media", file, file.name);

  let response;
  try {
    response = await fetch(`${API_BASE}/scan`, { method: "POST", body: formData });
  } catch (networkError) {
    throw new Error(
      `Could not reach the Veil backend at ${API_BASE}. Is it running? (${networkError.message})`
    );
  }

  let envelope;
  try {
    envelope = await response.json();
  } catch {
    throw new Error(`Veil backend returned a non-JSON response (HTTP ${response.status}).`);
  }

  if (!response.ok) {
    throw new Error(envelope?.detail || `Veil backend returned HTTP ${response.status}.`);
  }

  return envelope;
};

const runVisualExplanation = async (file, score) => {
  const formData = new FormData();
  formData.append("media", file, file.name);
  if (Number.isFinite(score)) formData.append("veil_score", String(score));

  const response = await fetch(`${API_BASE}/explain`, { method: "POST", body: formData });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body?.detail || `Visual explanation returned HTTP ${response.status}.`);
  }
  return body;
};

// Groups raw envelope.signals for the "Additional info" panel: people who
// care can see exactly what the local detector, metadata/forensic checks, and
// external APIs each reported, without that detail crowding the headline.
const SIGNAL_CATEGORY = {
  exif: "metadata",
  ela: "metadata",
  c2pa: "metadata",
  local: "local",
  sightengine: "api",
  hive: "api",
  reverse_search: "api",
};

const CATEGORY_LABEL = {
  local: "Local AI Detector",
  metadata: "Metadata & Forensics",
  api: "External APIs",
};

const CATEGORY_ORDER = ["local", "metadata", "api"];

// Chip-row grouping for the top "Additional info" summary: AI detection vs.
// manipulation vs. provenance are different questions, so they must not be
// shown as one undifferentiated row of equivalent checks.
const CHIP_GROUP = {
  local: "ai",
  hive: "ai",
  sightengine: "ai",
  ela: "manipulation",
  exif: "provenance",
  c2pa: "provenance",
  reverse_search: "provenance",
};

const CHIP_GROUP_LABEL = {
  ai: "AI detection",
  manipulation: "Manipulation",
  provenance: "Provenance",
};

const CHIP_GROUP_ORDER = ["ai", "manipulation", "provenance"];

const groupChips = (signals) => {
  const groups = { ai: [], manipulation: [], provenance: [] };
  for (const signal of signals ?? []) {
    const group = CHIP_GROUP[signal.name];
    if (group) groups[group].push(signal);
  }
  return groups;
};

const groupSignals = (signals) => {
  const groups = { local: [], metadata: [], api: [] };
  for (const signal of signals ?? []) {
    const category =
      SIGNAL_CATEGORY[signal.name] ?? (signal.signal_class === "context" ? "api" : "metadata");
    groups[category].push(signal);
  }
  return groups;
};

// Detection-ratio style summary (VirusTotal-style "N/M flagged"): a quick
// scan of which independent checks actually fired a risk signal, shown up
// front rather than buried in the collapsed technical detail.
const FLAG_THRESHOLD = 0.5;

// The local model's checkpoint is trained/validated for CIFAKE-scale input;
// on real-world photos its patch-aggregated score in this band is a coin
// flip, not evidence either way, so it must read as "inconclusive" rather
// than as a clear/flag verdict.
const LOCAL_INCONCLUSIVE_LOW = 0.4;
const LOCAL_INCONCLUSIVE_HIGH = 0.6;

// Only these three run actual AI-generation classifiers; ELA/EXIF/C2PA are
// manipulation or provenance checks and must never be counted as "AI
// detectors flagged this image" (that's what conflated ELA/EXIF into the
// flag count before).
const AI_DETECTOR_NAMES = new Set(["local", "hive", "sightengine"]);

const isInconclusiveLocal = (signal) => {
  if (signal.name !== "local") return false;
  const score = signal.ai_score ?? signal.manipulation_score;
  return score != null && score >= LOCAL_INCONCLUSIVE_LOW && score <= LOCAL_INCONCLUSIVE_HIGH;
};

const signalTone = (signal) => {
  if (signal.status !== "ok") return "neutral";
  if (isInconclusiveLocal(signal)) return "inconclusive";
  const score = signal.ai_score ?? signal.manipulation_score;
  if (score == null) return "neutral";
  return score >= FLAG_THRESHOLD ? "flag" : "clear";
};

const signalChipValue = (signal) => {
  if (signal.status === "error") return "Error";
  if (signal.status === "unavailable") return "N/A";
  if (signal.status === "skipped") return "Skipped";
  const score = signal.ai_score ?? signal.manipulation_score;
  return score != null ? formatPercent(score) : "—";
};

// Mirrors the backend's `_primary_ai_signals` (triangulate.py): the signals
// whose mutual disagreement actually drove `aggregate.disagreement`, so the
// UI can name them correctly instead of implying every check was compared.
const primaryDisagreementSignals = (signals) =>
  (signals ?? []).filter((s) => {
    if (s.status !== "ok" || s.ai_score == null) return false;
    if (s.signal_class !== "provenance" && s.signal_class !== "detector") return false;
    if (s.name === "local" && s.raw?.was_tiled === true) return false;
    return true;
  });

const disagreementLabel = (signals) => {
  const primary = primaryDisagreementSignals(signals);
  const names = new Set(primary.map((s) => s.name));
  const isExternalApiOnly =
    names.size > 0 && [...names].every((name) => name === "hive" || name === "sightengine");
  return isExternalApiOnly ? "External API disagreement" : "Disagreement across independent checks";
};

// AI detectors only: local (excluding its inconclusive band), Hive, Sightengine.
// ELA/EXIF/C2PA are manipulation/provenance checks, not AI-generation flags.
const detectionRatio = (signals) => {
  const scored = (signals ?? []).filter(
    (s) => s.status === "ok" && AI_DETECTOR_NAMES.has(s.name) && s.ai_score != null && !isInconclusiveLocal(s)
  );
  const flagged = scored.filter((s) => s.ai_score >= FLAG_THRESHOLD);
  return { flagged: flagged.length, total: scored.length };
};

// Practical guidance text, keyed off the backend's verdict band. This is
// UX copy (how to act), not evidence — the evidence itself (aggregate.reasons,
// per-signal notes) comes straight from the backend so it can't drift from it.
const buildGuidance = (aggregate) => {
  const verdict = aggregate?.verdict ?? "Inconclusive";
  const manipulationScore = aggregate?.manipulation_score ?? null;
  const userSummary = [];
  const visualChecks = [];
  const nextSteps = [];

  if (verdict === "Inconclusive") {
    userSummary.push("Veil could not reach a confident verdict from the available evidence.");
    userSummary.push("Treat this scan as unresolved, not as proof the image is real or fake.");
    (aggregate?.reasons ?? []).slice(0, 2).forEach((reason) => userSummary.push(reason));
    visualChecks.push("Check the original source, capture context, and metadata instead of relying on this score alone.");
    nextSteps.push("Retry with a higher-resolution image, or verify through an independent channel.");
    nextSteps.push("Use reverse-image search or another authenticity service before acting.");
  } else if (verdict === "High Risk") {
    userSummary.push("This image should not be trusted on its own.");
    userSummary.push("It may be AI-generated, edited, or used out of context.");
    userSummary.push("If someone is using this image to ask for money, identity documents, login codes, crypto, gift cards, or urgent action, treat it as suspicious.");
    visualChecks.push("Look closely at hands, fingers, ears, teeth, jewelry, glasses, and reflections.");
    visualChecks.push("Check text, signs, logos, watermarks, labels, and screenshots for warped letters or nonsense words.");
    nextSteps.push("Do not send money or personal information based only on this image.");
    nextSteps.push("Ask for a live video call, a new photo with a specific gesture, or another independent proof.");
    nextSteps.push("Reverse-image search the picture and verify the account or sender through a separate channel.");
  } else if (verdict === "Medium Risk") {
    userSummary.push("Veil found mixed signals. The image might be authentic, edited, or AI-assisted.");
    userSummary.push("Use caution if the image is connected to money, dating, identity, news, or an urgent request.");
    visualChecks.push("Inspect hands, text, logos, reflections, face edges, and background details.");
    nextSteps.push("Ask for another proof before trusting the image.");
    nextSteps.push("Check the source, date, and context of the image.");
  } else {
    userSummary.push(
      verdict === "Likely Authentic"
        ? "This image is likely authentic based on the available technical signals."
        : "Veil did not find strong signs that this image is AI-generated."
    );
    userSummary.push("This does not verify the sender, source, or surrounding context.");
    visualChecks.push("For high-stakes situations, still check hands, text, faces, shadows, reflections, and image source.");
    nextSteps.push("If money, credentials, or identity are involved, verify through another trusted channel.");
  }

  if (manipulationScore != null && manipulationScore >= 0.5) {
    userSummary.push("Independent evidence also suggests possible editing or manipulation, separate from AI-generation risk.");
    visualChecks.push("For faces, inspect eye alignment, skin transitions, hairlines, earrings, teeth, and face edges.");
    nextSteps.push("Do not rely on a face image alone to confirm someone's identity.");
  }

  return { userSummary, visualChecks, nextSteps };
};

const getPreferredColorScheme = () => {
  try {
    const stored = window.localStorage.getItem("veil-color-scheme");
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    /* localStorage unavailable (private mode, etc.) — fall through */
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
};

function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [scan, setScan] = useState(null);
  const [visualExplanation, setVisualExplanation] = useState(null);
  const [explaining, setExplaining] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [settingsTab, setSettingsTab] = useState("overview");
  const [motionMode, setMotionMode] = useState("on");
  const [colorScheme, setColorScheme] = useState(getPreferredColorScheme);
  const [backendStatus, setBackendStatus] = useState(null);

  const aggregate = scan?.aggregate ?? null;
  const overallScore = aggregate?.ai_score ?? null;
  const manipulationScore = aggregate?.manipulation_score ?? null;
  const verdict = aggregate?.verdict ?? null;
  const style = verdictStyle(verdict ?? "Inconclusive");
  const guidance = useMemo(() => (aggregate ? buildGuidance(aggregate) : null), [aggregate]);
  const visualBullets = useMemo(
    () => formatBullets(visualExplanation?.explanation),
    [visualExplanation]
  );
  const ratio = useMemo(() => detectionRatio(scan?.envelope?.signals), [scan]);
  const chipGroups = useMemo(() => groupChips(scan?.envelope?.signals), [scan]);

  useEffect(() => {
    try {
      window.localStorage.setItem("veil-color-scheme", colorScheme);
    } catch {
      /* ignore persistence failures */
    }
  }, [colorScheme]);

  // Probe the backend on load so a tester can see which signals are configured
  // (e.g. whether Sightengine keys are present server-side) without scanning.
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/health`)
      .then((response) => response.json())
      .then((data) => {
        if (!cancelled) setBackendStatus(data);
      })
      .catch((healthError) => {
        if (!cancelled) setBackendStatus({ error: healthError.message });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const navigate = (sectionId) => {
    setActivePage(sectionId);
  };

  const selectOneFile = (selected) => {
    setFile(selected);
    setScan(null);
    setVisualExplanation(null);
    setError(null);

    if (selected) {
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result);
      reader.readAsDataURL(selected);
    } else {
      setPreview(null);
    }
  };

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0] ?? null;
    selectOneFile(selected);
    event.target.value = "";
  };

  const handleFileDrop = (event) => {
    event.preventDefault();
    const selected = event.dataTransfer.files?.[0] ?? null;
    selectOneFile(selected);
  };

  const handleScan = async () => {
    setError(null);
    setScan(null);
    setVisualExplanation(null);

    if (!file) {
      setError("Please choose an image before starting a scan.");
      return;
    }

    setLoading(true);

    try {
      const envelope = await runScan(file);
      const scanAggregate = envelope.aggregate ?? {};

      if (scanAggregate.ai_score == null && scanAggregate.manipulation_score == null) {
        const failedSignal = (envelope.signals ?? []).find((signal) => signal.status === "error");
        setError(
          failedSignal
            ? `No usable evidence. ${failedSignal.name}: ${failedSignal.error}`
            : "No detector returned a usable score."
        );
      }

      const nextScan = {
        envelope,
        aggregate: scanAggregate,
        groups: groupSignals(envelope.signals),
      };
      setScan(nextScan);
      setHistory((current) => [
        {
          filename: file.name,
          date: new Date().toLocaleString(),
          score: formatPercent(scanAggregate.ai_score),
          verdict: scanAggregate.verdict ?? "Inconclusive",
        },
        ...current,
      ]);
      setActivePage("dashboard");

      // Layer 3 (LLM) explanation arrives on the envelope once the backend
      // builds it. Until then the VLM is called directly from here.
      if (envelope.explanation) {
        setVisualExplanation({ explanation: envelope.explanation });
      } else {
        setExplaining(true);
        try {
          const scoreForVlm = scanAggregate.verdict !== "Inconclusive" ? scanAggregate.ai_score : null;
          setVisualExplanation(await runVisualExplanation(file, scoreForVlm));
        } catch (explanationError) {
          setVisualExplanation({ error: explanationError.message });
        } finally {
          setExplaining(false);
        }
      }
    } catch (err) {
      setError(err.message || "Scan failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`veil-app theme-${colorScheme} motion-${motionMode}`}>
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-icon">V</div>
          <div>
            <div className="brand-title">Veil</div>
            <div className="brand-subtitle">AI media authenticity</div>
          </div>
        </div>

        <nav className="nav-group" aria-label="Main navigation">
          {navItems.map((item) => (
            <button
              className={`nav-link ${activePage === item.id ? "active" : ""}`}
              key={item.id}
              onClick={() => navigate(item.id)}
            >
              <Icon name={item.icon} />
              {item.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="status-card">
            <span className={`status-dot ${loading ? "busy" : ""}`}></span>
            <div>
              <strong>{loading ? "Scanning" : "Ready"}</strong>
              <span>Authenticity services enabled</span>
            </div>
          </div>
          <button
            className="theme-toggle"
            onClick={() => setColorScheme((mode) => (mode === "dark" ? "light" : "dark"))}
            aria-label="Toggle color theme"
          >
            <Icon name={colorScheme === "dark" ? "sun" : "moon"} />
            {colorScheme === "dark" ? "Light mode" : "Dark mode"}
          </button>
        </div>
      </aside>

      <main className="content">
        {activePage === "dashboard" && !scan && (
        <section className="hero-panel">
          <div>
            <p className="eyebrow">AI media risk dashboard</p>
            <h1>Reveal what hides beneath the image.</h1>
            <p className="hero-copy">
              Upload an image to check whether it looks authentic before you trust it.
            </p>
          </div>
          <div className="status-card">
            <span className={`status-dot ${loading ? "busy" : ""}`}></span>
            <div>
              <strong>{loading ? "Scanning" : "System Ready"}</strong>
              <p>Authenticity checks for images that might be misleading.</p>
            </div>
          </div>
        </section>
        )}

        {activePage === "dashboard" && !scan && (
          <section className="dashboard-grid">
            <div className="dashboard-card dashboard-card-primary">
              <p className="eyebrow">Ready to scan</p>
              <h2>Check an image before you trust it.</h2>
              <p>
                Veil reviews AI-generation risk, manipulation signals, and visible warning signs, then turns them into one clear authenticity score.
              </p>
              <button className="primary-button" onClick={() => navigate("upload")}>
                Start Image Check
              </button>
            </div>

            <div className="dashboard-card">
              <p className="eyebrow">Score</p>
              <strong>0-100%</strong>
              <span>Higher scores mean stronger AI or manipulation risk.</span>
            </div>

            <div className="dashboard-card">
              <p className="eyebrow">Guidance</p>
              <strong>Scam-aware</strong>
              <span>Results focus on identity, money, urgency, and trust decisions.</span>
            </div>

            <div className="dashboard-card">
              <p className="eyebrow">Explanation</p>
              <strong>Visual review</strong>
              <span>Veil can inspect visible warning signs after the score is ready.</span>
            </div>

            <div className="workflow-card">
              <div>
                <span>01</span>
                <strong>Upload</strong>
                <p>Add a suspicious image, profile photo, listing, screenshot, or post.</p>
              </div>
              <div>
                <span>02</span>
                <strong>Score</strong>
                <p>Veil combines model signals into a single authenticity risk score.</p>
              </div>
              <div>
                <span>03</span>
                <strong>Verify</strong>
                <p>Use the warning signs and next steps before taking action.</p>
              </div>
            </div>
          </section>
        )}

        {activePage === "upload" && (
        <section className="panel upload-panel">
          <div className="upload-workspace">
            <div className="upload-copy">
              <p className="eyebrow">Upload center</p>
              <h2>Check one image</h2>
              <p className="panel-copy">Add a single JPG, PNG, or WEBP image. Choosing another file replaces the current one.</p>
              <button className="primary-button" onClick={handleScan} disabled={loading || !file}>
                {loading ? "Checking..." : "Check Authenticity"}
              </button>
            </div>

            <label className={`upload-box ${preview ? "has-preview" : ""}`} onDrop={handleFileDrop} onDragOver={(event) => event.preventDefault()}>
              <input type="file" accept=".jpg,.jpeg,.png,.webp" onChange={handleFileChange} />
              {preview ? (
                <div className="inline-preview">
                  <img src={preview} alt="Preview" />
                  <div>
                    <span className="eyebrow">Selected image</span>
                    <strong>{file.name}</strong>
                    <p>{(file.size / 1024).toFixed(1)} KB</p>
                  </div>
                </div>
              ) : (
                <div>
                  <strong>Select an image</strong>
                  <p>Drop one file here or click to browse</p>
                </div>
              )}
            </label>
          </div>

          {backendStatus?.error && (
            <div className="alert-box">
              Cannot reach the Veil backend at {API_BASE}. Start it with `uvicorn app.main:app --reload` from the backend folder. ({backendStatus.error})
            </div>
          )}
          {error && <div className="alert-box">{error}</div>}
        </section>
        )}

        {activePage === "dashboard" && scan && (
          <section className="panel result-panel">
            <div className="result-header">
              <div className="result-header-verdict">
                <p className="eyebrow">Authenticity report</p>
                <div className="score-readout">
                  <strong>{formatPercent(overallScore)}</strong>
                  <span>AI-generation likelihood</span>
                </div>
                <div className="result-lead-verdict">
                  <span className={`verdict-pill ${style.type}`}>{verdict ?? "Inconclusive"}</span>
                </div>
              </div>
              <button className="secondary-button" onClick={() => navigate("upload")}>
                Check Another Image
              </button>
            </div>

            <div className="result-report">
              <div className="image-review-card">
                <p className="eyebrow">Image reviewed</p>
                {preview && <img src={preview} alt="Analyzed upload" />}
                {file && (
                  <div>
                    <strong>{file.name}</strong>
                    <span>{(file.size / 1024).toFixed(1)} KB</span>
                  </div>
                )}
              </div>

              <div className="explanation-card primary-explanation">
                <p className="eyebrow">Why Veil rated this</p>
                {explaining ? (
                  <p className="panel-copy">Veil is inspecting the image for visible warning signs...</p>
                ) : visualBullets.length ? (
                  <ul>
                    {visualBullets.map((line, index) => (
                      <li key={`${line}-${index}`}>{line}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="panel-copy">Visual explanation was not generated for this scan.</p>
                )}
                {visualExplanation?.note && (
                  <p className={`technical-note ${visualExplanation.used_fallback ? "fallback" : ""}`}>
                    {visualExplanation.note}
                  </p>
                )}
                {visualExplanation?.error && <p className="technical-note">Error: {visualExplanation.error}</p>}
              </div>
            </div>

            <div className="guidance-grid">
              <div className="guidance-card">
                <p className="eyebrow">Meaning</p>
                {guidance?.userSummary.slice(0, 2).map((line, index) => (
                  <p key={`${line}-${index}`}>{line}</p>
                ))}
              </div>

              <div className="guidance-card">
                <p className="eyebrow">Check</p>
                {guidance?.visualChecks.slice(0, 2).map((line, index) => (
                  <p key={`${line}-${index}`}>{line}</p>
                ))}
              </div>

              <div className="guidance-card">
                <p className="eyebrow">Next</p>
                {guidance?.nextSteps.slice(0, 2).map((line, index) => (
                  <p key={`${line}-${index}`}>{line}</p>
                ))}
              </div>
            </div>

            <details className="technical-details">
              <summary>Additional info: local detector, metadata, and API breakdown</summary>

              {ratio.total > 0 && (
                <p className="ratio-line">
                  {ratio.flagged > 0 ? (
                    <>
                      <strong>{ratio.flagged}/{ratio.total}</strong> AI detectors flagged this image
                    </>
                  ) : (
                    "No strong AI-generation indicators detected."
                  )}
                </p>
              )}

              {scan.envelope?.signals?.length > 0 && (
                <div className="signal-chip-groups">
                  {CHIP_GROUP_ORDER.filter((group) => chipGroups[group].length > 0).map((group) => (
                    <div className="signal-chip-group" key={`chip-group-${group}`}>
                      <span className="signal-chip-group-label">{CHIP_GROUP_LABEL[group]}</span>
                      <div className="signal-chip-row">
                        {chipGroups[group].map((signal) => (
                          <div className={`signal-chip ${signalTone(signal)}`} key={`chip-${signal.name}`}>
                            <span className="signal-chip-name">{signal.name}</span>
                            <span className="signal-chip-value">{signalChipValue(signal)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {aggregate?.reasons?.length > 0 && (
                <ul className="reason-list">
                  {aggregate.reasons.slice(0, 3).map((reason, index) => (
                    <li key={`reason-${index}`}>{reason}</li>
                  ))}
                </ul>
              )}

              <div className="signal-breakdown">
                {CATEGORY_ORDER.map((category) => (
                  <div className="signal-group" key={category}>
                    <h4>{CATEGORY_LABEL[category]}</h4>
                    {scan.groups[category].length === 0 ? (
                      <p className="panel-copy">No signals in this category for this scan.</p>
                    ) : (
                      <ul>
                        {scan.groups[category].map((signal) => (
                          <li key={`sig-${signal.name}`}>
                            <strong>{signal.name}</strong> — {signal.status}
                            {signal.ai_score != null ? ` | AI likelihood ${formatPercent(signal.ai_score)}` : ""}
                            {signal.manipulation_score != null ? ` | manipulation ${formatPercent(signal.manipulation_score)}` : ""}
                            {signal.error ? ` | error: ${signal.error}` : ""}
                            {signal.notes?.length > 0 && (
                              <ul>
                                {signal.notes.map((note, index) => (
                                  <li key={`note-${signal.name}-${index}`}>{note}</li>
                                ))}
                              </ul>
                            )}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>

              <ul>
                {guidance?.userSummary.slice(2).map((line, index) => (
                  <li key={`us-${line}-${index}`}>{line}</li>
                ))}
                {guidance?.visualChecks.slice(2).map((line, index) => (
                  <li key={`vc-${line}-${index}`}>{line}</li>
                ))}
                {guidance?.nextSteps.slice(2).map((line, index) => (
                  <li key={`ns-${line}-${index}`}>{line}</li>
                ))}
                <li>Fused AI-generation score: {formatPercent(overallScore)}</li>
                <li>Fused manipulation score: {formatPercent(manipulationScore)}</li>
                <li>{disagreementLabel(scan.envelope?.signals)}: {formatPercent(aggregate?.disagreement)}</li>
              </ul>
            </details>
          </section>
        )}

        {activePage === "history" && (
        <section className="panel history-panel">
          <p className="eyebrow">History</p>
          <h2>Recent scans</h2>
          {history.length === 0 ? (
            <p className="panel-copy">Upload an image and run a comparison to build a history log.</p>
          ) : (
            <div className="history-table">
              <div className="history-header">
                <span>File</span>
                <span>Verdict</span>
                <span>Veil Score</span>
              </div>
              {history.map((item, index) => (
                <div className="history-row" key={`${item.filename}-${index}`}>
                  <span>{item.filename}</span>
                  <span>{item.verdict}</span>
                  <span>{item.score}</span>
                </div>
              ))}
            </div>
          )}
        </section>
        )}

        {activePage === "settings" && (
        <section className="panel settings-panel">
          <p className="eyebrow">Settings</p>
          <h2>Platform settings</h2>
          <p className="panel-copy">Review detector connectivity, scoring behavior, and interface preferences for this local Veil session.</p>

          <div className="settings-layout">
            <aside className="settings-menu">
              <button className={`settings-menu-item ${settingsTab === "overview" ? "active" : ""}`} onClick={() => setSettingsTab("overview")}>Overview</button>
              <button className={`settings-menu-item ${settingsTab === "api" ? "active" : ""}`} onClick={() => setSettingsTab("api")}>API Connections</button>
              <button className={`settings-menu-item ${settingsTab === "detection" ? "active" : ""}`} onClick={() => setSettingsTab("detection")}>Detection</button>
              <button className={`settings-menu-item ${settingsTab === "display" ? "active" : ""}`} onClick={() => setSettingsTab("display")}>Display</button>
            </aside>

            <div className="settings-sections">
              {settingsTab === "overview" && (
              <section className="settings-section">
                <div className="settings-section-heading">
                  <div>
                    <p className="eyebrow">Overview</p>
                    <h3>System status</h3>
                  </div>
                  <span className="status-pill connected">Ready</span>
                </div>
                <div className="settings-list">
                  <div className="settings-row">
                    <div>
                      <strong>Veil mode</strong>
                      <p>Local analysis with optional external detector comparison.</p>
                    </div>
                    <span>Active</span>
                  </div>
                  <div className="settings-row">
                    <div>
                      <strong>Session history</strong>
                      <p>Recent scans are stored in memory for the current browser session.</p>
                    </div>
                    <span>{history.length} scans</span>
                  </div>
                </div>
              </section>
              )}

              {settingsTab === "api" && (
              <section className="settings-section">
                <div className="settings-section-heading">
                  <div>
                    <p className="eyebrow">API Connections</p>
                    <h3>Backend signals</h3>
                  </div>
                  <span className={backendStatus && !backendStatus.error ? "status-pill connected" : "status-pill warning"}>
                    {backendStatus ? (backendStatus.error ? "Unreachable" : "Connected") : "Checking..."}
                  </span>
                </div>
                <div className="settings-list">
                  <div className="settings-row">
                    <div>
                      <strong>Veil backend</strong>
                      <p>{API_BASE}</p>
                    </div>
                    <span className={backendStatus && !backendStatus.error ? "status-pill connected" : "status-pill warning"}>
                      {backendStatus ? (backendStatus.error ? "Offline" : "Online") : "..."}
                    </span>
                  </div>
                  <div className="settings-row">
                    <div>
                      <strong>Sightengine</strong>
                      <p>Configured server-side; keys never reach the browser.</p>
                    </div>
                    <span className={backendStatus?.available_signals?.includes("sightengine") ? "status-pill connected" : "status-pill warning"}>
                      {backendStatus?.available_signals?.includes("sightengine") ? "Available" : "Not configured"}
                    </span>
                  </div>
                  <div className="settings-row">
                    <div>
                      <strong>Active signals</strong>
                      <p>Signals the backend reports ready for this session.</p>
                    </div>
                    <span>{backendStatus?.available_signals?.length ? backendStatus.available_signals.join(", ") : "None"}</span>
                  </div>
                </div>
              </section>
              )}

              {settingsTab === "detection" && (
              <section className="settings-section">
                <div className="settings-section-heading">
                  <div>
                    <p className="eyebrow">Detection</p>
                    <h3>Scoring behavior</h3>
                  </div>
                </div>
                <div className="settings-list">
                  <div className="settings-row">
                    <div>
                      <strong>Fusion</strong>
                      <p>The backend weighs each check by its evidence class (provenance, detector, forensic) and self-reported confidence, then fuses the AI-generation and manipulation axes separately.</p>
                    </div>
                    <span>Engine-driven</span>
                  </div>
                  <div className="settings-row">
                    <div>
                      <strong>Risk bands</strong>
                      <p>Likely Authentic, Low Risk, Medium Risk, High Risk — or Inconclusive when evidence is missing, weak, or contradictory.</p>
                    </div>
                    <span>5 states</span>
                  </div>
                </div>
              </section>
              )}

              {settingsTab === "display" && (
              <section className="settings-section">
                <div className="settings-section-heading">
                  <div>
                    <p className="eyebrow">Display</p>
                    <h3>Interface preferences</h3>
                  </div>
                </div>
                <div className="settings-list">
                  <div className="settings-row">
                    <div>
                      <strong>Theme</strong>
                      <p>Switch between light and dark color schemes.</p>
                    </div>
                    <div className="segmented-control" aria-label="Theme mode">
                      <button className={colorScheme === "light" ? "active" : ""} onClick={() => setColorScheme("light")}>Light</button>
                      <button className={colorScheme === "dark" ? "active" : ""} onClick={() => setColorScheme("dark")}>Dark</button>
                    </div>
                  </div>
                  <div className="settings-row">
                    <div>
                      <strong>Motion</strong>
                      <p>Enable or reduce entrance animations.</p>
                    </div>
                    <div className="segmented-control" aria-label="Motion mode">
                      <button className={motionMode === "on" ? "active" : ""} onClick={() => setMotionMode("on")}>On</button>
                      <button className={motionMode === "reduced" ? "active" : ""} onClick={() => setMotionMode("reduced")}>Reduced</button>
                    </div>
                  </div>
                </div>
              </section>
              )}
            </div>
          </div>
        </section>
        )}
      </main>
    </div>
  );
}

export default App;
