import { useEffect, useMemo, useState } from "react";

const ENV = import.meta.env;

// All detector calls now go through the Veil backend (POST /scan). The backend
// holds every API key server-side, so no secrets ship to the browser. Point the
// frontend at the backend with VITE_VEIL_API_URL (defaults to local dev).
const API_BASE = (ENV.VITE_VEIL_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

const navItems = [
  { id: "dashboard", label: "Dashboard" },
  { id: "upload", label: "Check Image" },
  { id: "history", label: "History" },
  { id: "settings", label: "Settings" },
];

const clampScore = (score) => {
  if (!Number.isFinite(score)) return null;
  return Math.max(0, Math.min(1, score));
};

const riskLabel = (score) => {
  if (score === null || score === undefined) return { text: "Unavailable", type: "unknown" };
  if (score < 0.4) return { text: "Low Risk", type: "low" };
  if (score < 0.7) return { text: "Medium Risk", type: "medium" };
  return { text: "High Risk", type: "high" };
};

const verdictLabel = (score) => {
  if (score === null || score === undefined) return "Inconclusive";
  return score >= 0.7 ? "High scam risk" : score >= 0.4 ? "Needs review" : "Looks authentic";
};

const scoreCaption = (score) => {
  if (score === null || score === undefined) return "inconclusive";
  if (score >= 0.7) return "likely false";
  if (score >= 0.4) return "needs review";
  return "likely authentic";
};

const formatPercent = (score) => {
  if (score === null || score === undefined) return "N/A";
  return `${(score * 100).toFixed(1)}%`;
};

const formatBullets = (text) => {
  if (!text) return [];
  return text
    .split(/\r?\n/)
    .map((line) => line.replace(/^[-*•]\s*/, "").replace(/^\d+\.\s*/, "").trim())
    .filter(Boolean);
};

// POST the image to the Veil backend and return the raw ScanResponse envelope.
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

// Translate the backend's signal envelope into the per-detector shape the UI
// already understands. Sightengine's ai_score -> genai, manipulation -> deepfake.
const mapEnvelope = (envelope) => {
  const signals = envelope?.signals ?? [];
  const find = (name) => signals.find((signal) => signal.name === name) ?? null;

  const toDetector = (signal) =>
    signal
      ? {
          genai: signal.ai_score,
          deepfake: signal.manipulation_score,
          status: signal.status,
          error: signal.error,
          raw: signal.raw,
        }
      : null;

  return {
    local: toDetector(find("local")),
    sightengine: toDetector(find("sightengine")),
  };
};

const buildComparison = (local, sightengine) => {
  const localScore = local?.genai ?? null;
  const sightengineScore = sightengine?.genai ?? null;
  const usableScores = [localScore, sightengineScore].filter((score) => score != null);
  const deepfakeScore = sightengine?.deepfake ?? null;

  if (usableScores.length === 0) {
    return {
      overallScore: null,
      confidence: null,
      agreement: "No detector scores available",
      explanation: ["Neither detector returned a usable AI-generation score."],
    };
  }

  const overallScore = usableScores.reduce((sum, score) => sum + score, 0) / usableScores.length;
  const disagreement = usableScores.length === 2 ? Math.abs(localScore - sightengineScore) : 0.3;
  const agreementStrength = 1 - disagreement;
  const certainty = Math.abs(overallScore - 0.5) * 2;
  const confidence = clampScore(0.25 + agreementStrength * 0.45 + certainty * 0.3);

  const explanation = [];
  const visualChecks = [];
  const userSummary = [];
  const nextSteps = [];
  if (usableScores.length === 2) {
    explanation.push(
      `Veil found a strong authenticity warning in the image.`
    );
    explanation.push(
      `A second check agreed with the warning, so Veil is more confident in the result.`
    );
    explanation.push(
      disagreement < 0.15
        ? "The image was flagged consistently across Veil's checks."
        : disagreement < 0.35
          ? "The image was flagged unevenly, so Veil is treating the result with caution."
          : "Veil's checks disagreed, so treat this result as uncertain."
    );
  } else if (localScore != null) {
    explanation.push("Veil completed one authenticity check, but one external check was unavailable.");
  } else {
    explanation.push("Veil completed one authenticity check, but one internal check was unavailable.");
  }

  if (overallScore >= 0.7) {
    userSummary.push("This image should not be trusted on its own.");
    userSummary.push("It may be AI-generated, edited, or used out of context.");
    userSummary.push("If someone is using this image to ask for money, identity documents, login codes, crypto, gift cards, or urgent action, treat it as suspicious.");
    visualChecks.push("Look closely at hands, fingers, ears, teeth, jewelry, glasses, and reflections.");
    visualChecks.push("Check text, signs, logos, watermarks, labels, and screenshots for warped letters or nonsense words.");
    visualChecks.push("Watch for overly smooth skin, strange lighting, repeated textures, or background objects that do not make sense.");
    nextSteps.push("Do not send money or personal information based only on this image.");
    nextSteps.push("Ask for a live video call, a new photo with a specific gesture, or another independent proof.");
    nextSteps.push("Reverse-image search the picture and verify the account or sender through a separate channel.");
  } else if (overallScore >= 0.4) {
    userSummary.push("Veil found mixed signals. The image might be authentic, edited, or AI-assisted.");
    userSummary.push("Use caution if the image is connected to money, dating, identity, news, or an urgent request.");
    visualChecks.push("Inspect hands, text, logos, reflections, face edges, and background details.");
    visualChecks.push("Screenshots, filters, heavy compression, or stylized art can make image checks less certain.");
    nextSteps.push("Ask for another proof before trusting the image.");
    nextSteps.push("Check the source, date, and context of the image.");
  } else {
    userSummary.push("Veil did not find strong signs that this image is AI-generated.");
    userSummary.push("This lowers the risk, but it does not prove the sender, story, or context is truthful.");
    visualChecks.push("For high-stakes situations, still check hands, text, faces, shadows, reflections, and image source.");
    nextSteps.push("If money, credentials, or identity are involved, verify through another trusted channel.");
  }

  if (deepfakeScore != null) {
    explanation.push(
      deepfakeScore >= 0.7
        ? "Veil also found a strong warning for possible face manipulation."
        : "Veil did not find a strong face-manipulation warning."
    );
    if (deepfakeScore >= 0.7) {
      userSummary.push("If the image includes a person, the face may have been altered or generated.");
      visualChecks.push("For faces, inspect eye alignment, skin transitions, hairlines, earrings, teeth, and face edges.");
      nextSteps.push("Do not rely on a face image alone to confirm someone's identity.");
    }
  }

  return {
    overallScore,
    confidence,
    agreement:
      usableScores.length === 2
        ? disagreement < 0.15
          ? "Strong agreement"
          : disagreement < 0.35
            ? "Partial agreement"
            : "Low agreement"
        : "Single-detector result",
    explanation,
    visualChecks,
    userSummary,
    nextSteps,
    rawScores: {
      local: localScore,
      sightengine: sightengineScore,
      deepfake: deepfakeScore,
    },
  };
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
  const [themeMode, setThemeMode] = useState("midnight");
  const [motionMode, setMotionMode] = useState("on");
  const [backendStatus, setBackendStatus] = useState(null);

  const localScore = scan?.local?.genai ?? null;
  const sightengineScore = scan?.sightengine?.genai ?? null;
  const deepfakeScore = scan?.sightengine?.deepfake ?? null;
  const overallScore = scan?.comparison?.overallScore ?? null;
  const confidenceScore = scan?.comparison?.confidence ?? null;
  const resultRisk = useMemo(() => riskLabel(overallScore), [overallScore]);
  const visualBullets = useMemo(
    () => formatBullets(visualExplanation?.explanation),
    [visualExplanation]
  );

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
      const { local, sightengine } = mapEnvelope(envelope);

      // Surface a backend-side signal failure (e.g. Sightengine rejected the
      // key) instead of silently scoring it as missing.
      if (sightengine?.status === "error") {
        setError(`Sightengine error from backend: ${sightengine.error}`);
      }

      const comparison = buildComparison(local, sightengine);
      const nextScan = { local, sightengine, comparison, envelope };

      setScan(nextScan);
      setHistory((current) => [
        {
          filename: file.name,
          date: new Date().toLocaleString(),
          score: formatPercent(comparison.overallScore),
          verdict: verdictLabel(comparison.overallScore),
          confidence: formatPercent(comparison.confidence),
        },
        ...current,
      ]);
      setActivePage("dashboard");

      // Layer 3 (LLM) explanation arrives on the envelope once the backend
      // builds it. Until then the guidance below is generated client-side.
      if (envelope.explanation) {
        setVisualExplanation({ explanation: envelope.explanation });
      }
    } catch (err) {
      setError(err.message || "Scan failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`veil-app theme-${themeMode} motion-${motionMode}`}>
      <header className="topbar">
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
              {item.label}
            </button>
          ))}
        </nav>

        <div className="topbar-status">
          <span className={`status-dot ${loading ? "busy" : ""}`}></span>
          <div>
            <strong>{loading ? "Scanning" : "Ready"}</strong>
            <span>Local analysis enabled</span>
          </div>
        </div>
      </header>

      <main className="content">
        {activePage === "dashboard" && !scan && (
        <section className="hero-panel">
          <div>
            <p className="eyebrow">AI media risk dashboard</p>
            <h1>{scan ? verdictLabel(overallScore) : "Reveal what hides beneath the image."}</h1>
            <p className="hero-copy">
              {scan
                ? `Veil confidence: ${formatPercent(confidenceScore)}.`
                : "Upload an image to check whether it looks authentic before you trust it."}
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
            <div className="result-report">
              <div className="result-lead">
                <p className="eyebrow">Authenticity report</p>
                <h2>
                  {resultRisk.text}
                </h2>
                <div className="confidence-readout">
                  <span>Confidence</span>
                  <strong>{formatPercent(confidenceScore)}</strong>
                </div>
                <div className={`risk-orb ${resultRisk.type}`}>
                  <strong>{formatPercent(overallScore)}</strong>
                  <span>{scoreCaption(overallScore)}</span>
                </div>
                <button className="secondary-button" onClick={() => navigate("upload")}>
                  Check Another Image
                </button>
              </div>

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
                {scan.comparison.userSummary.slice(0, 2).map((line, index) => (
                  <p key={`${line}-${index}`}>{line}</p>
                ))}
              </div>

              <div className="guidance-card">
                <p className="eyebrow">Check</p>
                {scan.comparison.visualChecks.slice(0, 2).map((line, index) => (
                  <p key={`${line}-${index}`}>{line}</p>
                ))}
              </div>

              <div className="guidance-card">
                <p className="eyebrow">Next</p>
                {scan.comparison.nextSteps.slice(0, 2).map((line, index) => (
                  <p key={`${line}-${index}`}>{line}</p>
                ))}
              </div>
            </div>

            <details className="technical-details">
              <summary>Show more detail</summary>
              <ul>
                {scan.comparison.userSummary.slice(2).map((line, index) => (
                  <li key={`${line}-${index}`}>{line}</li>
                ))}
                {scan.comparison.visualChecks.slice(2).map((line, index) => (
                  <li key={`${line}-${index}`}>{line}</li>
                ))}
                {scan.comparison.nextSteps.slice(2).map((line, index) => (
                  <li key={`${line}-${index}`}>{line}</li>
                ))}
                {scan.comparison.explanation.map((line, index) => (
                  <li key={`${line}-${index}`}>{line}</li>
                ))}
                <li>Internal score: {formatPercent(scan.comparison.rawScores.local)}</li>
                <li>External (Sightengine) score: {formatPercent(scan.comparison.rawScores.sightengine)}</li>
                <li>Face manipulation score: {formatPercent(scan.comparison.rawScores.deepfake)}</li>
                {scan.envelope?.signals?.map((signal) => (
                  <li key={`sig-${signal.name}`}>
                    {signal.name} [{signal.signal_class}] — {signal.status}
                    {signal.ai_score != null ? ` · ai ${formatPercent(signal.ai_score)}` : ""}
                    {signal.manipulation_score != null ? ` · manip ${formatPercent(signal.manipulation_score)}` : ""}
                  </li>
                ))}
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
                <span>Confidence</span>
              </div>
              {history.map((item, index) => (
                <div className="history-row" key={`${item.filename}-${index}`}>
                  <span>{item.filename}</span>
                  <span>{item.verdict}</span>
                  <span>{item.score}</span>
                  <span>{item.confidence}</span>
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
                      <strong>Decision threshold</strong>
                      <p>Scores at or above 50% are treated as likely AI-generated or manipulated.</p>
                    </div>
                    <span>50%</span>
                  </div>
                  <div className="settings-row">
                    <div>
                      <strong>Risk bands</strong>
                      <p>Low risk below 40%, review between 40-70%, high risk at 70% and above.</p>
                    </div>
                    <span>3 levels</span>
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
                      <p>Choose the visual style used across the dashboard.</p>
                    </div>
                    <div className="segmented-control" aria-label="Theme mode">
                      <button className={themeMode === "midnight" ? "active" : ""} onClick={() => setThemeMode("midnight")}>Midnight</button>
                      <button className={themeMode === "ice" ? "active" : ""} onClick={() => setThemeMode("ice")}>Ice</button>
                    </div>
                  </div>
                  <div className="settings-row">
                    <div>
                      <strong>Motion</strong>
                      <p>Enable or reduce visual pulse, meter, and entrance animations.</p>
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
