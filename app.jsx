import { useMemo, useState } from "react";

const ENV = import.meta.env;

const detectors = {
  local: {
    label: "Local ResNet",
    description: "Your trained fake-vs-real ResNet-50 checkpoint.",
    envKeys: ["VITE_CUSTOM_API_URL"],
    buildRequest: (file) => {
      const formData = new FormData();
      formData.append("media", file, file.name);
      return {
        url: ENV.VITE_CUSTOM_API_URL,
        options: {
          method: "POST",
          headers: ENV.VITE_CUSTOM_API_KEY
            ? { Authorization: `Bearer ${ENV.VITE_CUSTOM_API_KEY}` }
            : {},
          body: formData,
        },
      };
    },
    normalizeResponse: (json) => ({
      raw: json,
      genai: extractScore(json, "genai"),
      deepfake: extractScore(json, "deepfake"),
      label: json?.label ?? null,
      confidence: extractScore(json, "confidence"),
    }),
  },
  sightengine: {
    label: "Sightengine",
    description: "External genai and deepfake detector API.",
    envKeys: ["VITE_SIGHTENGINE_API_USER", "VITE_SIGHTENGINE_API_SECRET"],
    buildRequest: (file) => {
      const formData = new FormData();
      formData.append("api_user", ENV.VITE_SIGHTENGINE_API_USER);
      formData.append("api_secret", ENV.VITE_SIGHTENGINE_API_SECRET);
      formData.append("models", "genai,deepfake");
      formData.append("media", file, file.name);
      return {
        url: "https://api.sightengine.com/1.0/check.json",
        options: { method: "POST", body: formData },
      };
    },
    normalizeResponse: (json) => ({
      raw: json,
      genai: extractScore(json, "genai"),
      deepfake: extractScore(json, "deepfake"),
      label: null,
      confidence: null,
    }),
  },
};

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

const formatPercent = (score) => {
  if (score === null || score === undefined) return "N/A";
  return `${(score * 100).toFixed(1)}%`;
};

const extractScore = (response, name) => {
  if (!response) return null;

  const scoreCandidates = [response[name], response?.type, response?.scores, response?.results];
  for (const maybe of scoreCandidates) {
    if (maybe == null) continue;

    if (typeof maybe === "number") return clampScore(maybe);
    if (typeof maybe === "string") {
      const parsed = parseFloat(maybe);
      if (Number.isFinite(parsed)) return clampScore(parsed);
    }
    if (typeof maybe === "object") {
      const keys = [
        "ai_generated",
        "ai-generated",
        "genai",
        "deepfake",
        "face_manipulation",
        "manipulated",
        "score",
        "probability",
        "confidence",
        "value",
      ];
      for (const key of keys) {
        if (maybe[key] != null) {
          const parsed = parseFloat(maybe[key]);
          if (Number.isFinite(parsed)) return clampScore(parsed);
        }
      }
    }
  }

  return null;
};

const runDetector = async (id, file) => {
  const detector = detectors[id];
  const missing = detector.envKeys.filter((key) => !ENV[key]);
  if (missing.length > 0) {
    throw new Error(`Missing ${detector.label} settings: ${missing.join(", ")}`);
  }

  const { url, options } = detector.buildRequest(file);
  const result = await fetch(url, options);
  const json = await result.json();

  if (!result.ok) {
    throw new Error(json.error ? JSON.stringify(json.error) : `${detector.label} returned HTTP ${result.status}`);
  }

  return detector.normalizeResponse(json);
};

const runVisualExplanation = async (file, veilScore) => {
  if (!ENV.VITE_EXPLANATION_API_URL) return null;

  const formData = new FormData();
  formData.append("media", file, file.name);
  if (veilScore != null) {
    formData.append("veil_score", veilScore.toString());
  }

  const result = await fetch(ENV.VITE_EXPLANATION_API_URL, {
    method: "POST",
    body: formData,
  });
  const json = await result.json();

  if (!result.ok) {
    throw new Error(json.detail || `Visual explanation returned HTTP ${result.status}`);
  }

  return json;
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

  const localScore = scan?.local?.genai ?? null;
  const sightengineScore = scan?.sightengine?.genai ?? null;
  const deepfakeScore = scan?.sightengine?.deepfake ?? null;
  const overallScore = scan?.comparison?.overallScore ?? null;
  const confidenceScore = scan?.comparison?.confidence ?? null;
  const resultRisk = useMemo(() => riskLabel(overallScore), [overallScore]);

  const missingEnv = Object.entries(detectors).flatMap(([id, detector]) =>
    detector.envKeys
      .filter((key) => !ENV[key])
      .map((key) => `${detector.label}: ${key}`)
  );

  const navigate = (sectionId) => {
    setActivePage(sectionId);
  };

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0] ?? null;
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

  const handleScan = async () => {
    setError(null);
    setScan(null);
    setVisualExplanation(null);

    if (!file) {
      setError("Please choose an image before starting a scan.");
      return;
    }

    if (missingEnv.length > 0) {
      setError(`Missing detector settings: ${missingEnv.join("; ")}`);
      return;
    }

    setLoading(true);

    try {
      const [localResult, sightengineResult] = await Promise.all([
        runDetector("local", file),
        runDetector("sightengine", file),
      ]);
      const comparison = buildComparison(localResult, sightengineResult);
      const nextScan = {
        local: localResult,
        sightengine: sightengineResult,
        comparison,
      };

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

      setExplaining(true);
      try {
        const explanation = await runVisualExplanation(file, comparison.overallScore);
        setVisualExplanation(explanation);
      } catch (explanationError) {
        setVisualExplanation({
          explanation:
            "Veil could not generate a visual explanation on this run. Use the checklist below and try again if needed.",
          error: explanationError.message,
        });
      } finally {
        setExplaining(false);
      }
    } catch (err) {
      setError(err.message || "Scan failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="veil-app">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-icon">V</div>
          <div>
            <div className="brand-title">Veil</div>
            <div className="brand-subtitle">AI media authenticity</div>
          </div>
        </div>

        <div className="nav-group">
          {navItems.map((item) => (
            <button
              className={`nav-link ${activePage === item.id ? "active" : ""}`}
              key={item.id}
              onClick={() => navigate(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>

        <div className="sidebar-card">
          <p className="eyebrow">Detector stack</p>
          <strong>Local ResNet + Sightengine</strong>
          <p>Veil combines multiple checks into one simple authenticity score.</p>
        </div>
      </aside>

      <main className="content">
        {activePage === "dashboard" && (
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

        {activePage === "upload" && (
        <section className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Upload center</p>
              <h2>Check an image</h2>
              <p className="panel-copy">Upload a JPG, PNG, or WEBP image before trusting a post, profile, message, or request.</p>
            </div>
          </div>

          <div className="upload-grid">
            <label className="upload-box">
              <input type="file" accept=".jpg,.jpeg,.png,.webp" onChange={handleFileChange} />
              <div>
                <strong>Select an image</strong>
                <p>Drag & drop or click to browse</p>
              </div>
            </label>
            {preview && (
              <div className="preview-card">
                <img src={preview} alt="Preview" />
                <div>
                  <div className="eyebrow">Selected file</div>
                  <h3>{file.name}</h3>
                  <p>{(file.size / 1024).toFixed(1)} KB</p>
                </div>
              </div>
            )}
          </div>

          <div className="action-bar">
            <button className="primary-button" onClick={handleScan} disabled={loading}>
              {loading ? "Checking..." : "Check Authenticity"}
            </button>
          </div>

          {missingEnv.length > 0 && (
            <div className="alert-box">
              Missing detector settings: {missingEnv.join("; ")}.
            </div>
          )}
          {error && <div className="alert-box">{error}</div>}
        </section>
        )}

        {activePage === "dashboard" && scan && (
          <section className="panel result-panel">
            <div className="analysis-header">
              <div>
                <p className="eyebrow">Combined analysis</p>
                <h2>
                  Overall rating: <span className={`risk-${resultRisk.type}`}>{resultRisk.text}</span>
                </h2>
                <p className="panel-copy">
                  Veil score: <strong>{formatPercent(overallScore)}</strong>. Confidence:{" "}
                  <strong>{formatPercent(confidenceScore)}</strong>.
                </p>
              </div>
              <div className={`risk-orb ${resultRisk.type}`}>
                {formatPercent(confidenceScore)}
              </div>
            </div>

            <div className="metric-grid">
              <div className="metric-card">
                <div className="metric-title">
                  <span>Veil Score</span>
                  <span className={`pill ${resultRisk.type}`}>{resultRisk.text}</span>
                </div>
                <div className="metric-value">{formatPercent(overallScore)}</div>
                <div className="meter">
                  <div className="meter-fill" style={{ width: overallScore != null ? `${overallScore * 100}%` : "0%" }}></div>
                </div>
                <p className="metric-copy">How strongly Veil thinks the image may be AI-generated or manipulated.</p>
              </div>

              <div className="metric-card">
                <div className="metric-title">
                  <span>Confidence</span>
                  <span className={`pill ${riskLabel(confidenceScore).type}`}>{riskLabel(confidenceScore).text}</span>
                </div>
                <div className="metric-value">{formatPercent(confidenceScore)}</div>
                <div className="meter">
                  <div className="meter-fill" style={{ width: confidenceScore != null ? `${confidenceScore * 100}%` : "0%" }}></div>
                </div>
                <p className="metric-copy">How much weight Veil gives this result based on signal strength and agreement.</p>
              </div>
            </div>

            <div className="explanation-card">
              <p className="eyebrow">Why Veil flagged this</p>
              {explaining ? (
                <p className="panel-copy">Veil is inspecting the image for visible warning signs...</p>
              ) : visualExplanation?.explanation ? (
                <div className="visual-explanation-text">{visualExplanation.explanation}</div>
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

            <div className="explanation-card">
              <p className="eyebrow">What this means</p>
              <ul>
                {scan.comparison.userSummary.map((line, index) => (
                  <li key={`${line}-${index}`}>{line}</li>
                ))}
              </ul>
            </div>

            <div className="explanation-card">
              <p className="eyebrow">What to check</p>
              <ul>
                {scan.comparison.visualChecks.map((line, index) => (
                  <li key={`${line}-${index}`}>{line}</li>
                ))}
              </ul>
            </div>

            <div className="explanation-card">
              <p className="eyebrow">Recommended next steps</p>
              <ul>
                {scan.comparison.nextSteps.map((line, index) => (
                  <li key={`${line}-${index}`}>{line}</li>
                ))}
              </ul>
            </div>

            <details className="technical-details">
              <summary>Technical details</summary>
              <ul>
                {scan.comparison.explanation.map((line, index) => (
                  <li key={`${line}-${index}`}>{line}</li>
                ))}
                <li>Internal score: {formatPercent(scan.comparison.rawScores.local)}</li>
                <li>External score: {formatPercent(scan.comparison.rawScores.sightengine)}</li>
                <li>Face manipulation score: {formatPercent(scan.comparison.rawScores.deepfake)}</li>
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
          <h2>Platform configuration</h2>
          <div className="settings-grid">
            <div>
              <strong>Local API</strong>
              <p>{ENV.VITE_CUSTOM_API_URL || "Not configured"}</p>
            </div>
            <div>
              <strong>Sightengine</strong>
              <p>{ENV.VITE_SIGHTENGINE_API_USER ? "Credentials configured" : "Credentials missing"}</p>
            </div>
            <div>
              <strong>Decision threshold</strong>
              <p>Scores at or above 50% are treated as likely AI-generated.</p>
            </div>
            <div>
              <strong>Visual explanation</strong>
              <p>{ENV.VITE_EXPLANATION_API_URL || "Not configured"}</p>
            </div>
          </div>
        </section>
        )}
      </main>
    </div>
  );
}

export default App;
