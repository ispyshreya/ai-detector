import { useMemo, useState } from "react";

const ENV = import.meta.env;

const providers = {
  sightengine: {
    label: "Sightengine",
    description: "AI media scanning with genai and deepfake detectors.",
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
    }),
    help: "Requires Sightengine credentials in .env.",
  },
  custom: {
    label: "Custom REST API",
    description: "Upload media to your own detector endpoint and normalize the response.",
    envKeys: ["VITE_CUSTOM_API_URL", "VITE_CUSTOM_API_KEY"],
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
    }),
    help: "Point this at any custom detection provider that returns normalized fields.",
  },
};

const riskLabel = (score) => {
  if (score === null || score === undefined) return { text: "Unavailable", type: "unknown" };
  if (score < 0.4) return { text: "Low Risk", type: "low" };
  if (score < 0.7) return { text: "Medium Risk", type: "medium" };
  return { text: "High Risk", type: "high" };
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

    if (typeof maybe === "number") return maybe;
    if (typeof maybe === "string") {
      const parsed = parseFloat(maybe);
      if (Number.isFinite(parsed)) return parsed;
    }
    if (typeof maybe === "object") {
      const keys = ["ai_generated", "ai-generated", "genai", "deepfake", "face_manipulation", "manipulated", "score", "probability", "confidence", "value"];
      for (const key of keys) {
        if (maybe[key] != null) {
          const parsed = parseFloat(maybe[key]);
          if (Number.isFinite(parsed)) return parsed;
        }
      }
    }
  }

  return null;
};

function App() {
  const [providerId, setProviderId] = useState("sightengine");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [response, setResponse] = useState(null);
  const [normalized, setNormalized] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  const provider = providers[providerId];
  const missingEnv = provider.envKeys.filter((key) => !ENV[key]);

  const genaiScore = useMemo(() => normalized?.genai ?? null, [normalized]);
  const deepfakeScore = useMemo(() => normalized?.deepfake ?? null, [normalized]);
  const overallScore = useMemo(() => {
    const values = [genaiScore, deepfakeScore].filter((value) => value != null);
    return values.length ? Math.max(...values) : null;
  }, [genaiScore, deepfakeScore]);

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setResponse(null);
    setNormalized(null);
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
    setResponse(null);
    setNormalized(null);

    if (!file) {
      setError("Please choose an image before starting a scan.");
      return;
    }

    if (missingEnv.length > 0) {
      setError(`Missing environment variables for ${provider.label}: ${missingEnv.join(", ")}`);
      return;
    }

    setLoading(true);

    try {
      const { url, options } = provider.buildRequest(file);
      const result = await fetch(url, options);
      const json = await result.json();

      if (!result.ok) {
        throw new Error(json.error ? JSON.stringify(json.error) : `HTTP ${result.status}`);
      }

      const normalizedResponse = provider.normalizeResponse(json);
      setResponse(json);
      setNormalized(normalizedResponse);
      setHistory((current) => [
        {
          filename: file.name,
          provider: provider.label,
          date: new Date().toLocaleString(),
          genai: formatPercent(normalizedResponse.genai),
          deepfake: formatPercent(normalizedResponse.deepfake),
        },
        ...current,
      ]);
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
          <button className="nav-link active">Dashboard</button>
          <button className="nav-link">Upload</button>
          <button className="nav-link">History</button>
          <button className="nav-link">Settings</button>
        </div>

        <div className="sidebar-card">
          <p className="eyebrow">Active provider</p>
          <strong>{provider.label}</strong>
          <p>{provider.description}</p>
          <p className="provider-help">{provider.help}</p>
        </div>
      </aside>

      <main className="content">
        <section className="hero-panel">
          <div>
            <p className="eyebrow">AI media risk dashboard</p>
            <h1>Reveal what hides beneath the image.</h1>
            <p className="hero-copy">
              Veil can route uploads to multiple detector APIs and normalize results from each provider.
            </p>
          </div>
          <div className="status-card">
            <span className="status-dot"></span>
            <div>
              <strong>System Ready</strong>
              <p>Pick a provider and scan media safely.</p>
            </div>
          </div>
        </section>

        <section className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Upload center</p>
              <h2>Analyze an image</h2>
              <p className="panel-copy">Supported file types: JPG, JPEG, PNG, WEBP.</p>
            </div>
          </div>

          <div className="provider-field">
            <label htmlFor="provider-select" className="eyebrow">Detection provider</label>
            <select id="provider-select" value={providerId} onChange={(e) => setProviderId(e.target.value)}>
              {Object.entries(providers).map(([id, providerDef]) => (
                <option value={id} key={id}>{providerDef.label}</option>
              ))}
            </select>
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
              {loading ? "Scanning..." : "Run Veil Analysis"}
            </button>
          </div>

          {missingEnv.length > 0 && (
            <div className="alert-box">
              Missing environment variables: {missingEnv.join(", ")}. Update .env before scanning.
            </div>
          )}
          {error && <div className="alert-box">{error}</div>}
        </section>

        {normalized && (
          <section className="panel result-panel">
            <div className="analysis-header">
              <div>
                <p className="eyebrow">Analysis result</p>
                <h2>
                  Overall rating: <span className={`risk-${riskLabel(overallScore).type}`}>{riskLabel(overallScore).text}</span>
                </h2>
                <p className="panel-copy">
                  Highest detected risk score: <strong>{formatPercent(overallScore)}</strong>. This is an automated estimate, not proof.
                </p>
              </div>
              <div className={`risk-orb ${riskLabel(overallScore).type}`}>
                {riskLabel(overallScore).text}
              </div>
            </div>

            <div className="metric-grid">
              <div className="metric-card">
                <div className="metric-title">
                  <span>AI-generated</span>
                  <span className={`pill ${riskLabel(genaiScore).type}`}>{riskLabel(genaiScore).text}</span>
                </div>
                <div className="metric-value">{formatPercent(genaiScore)}</div>
                <div className="meter">
                  <div className="meter-fill" style={{ width: genaiScore != null ? `${genaiScore * 100}%` : "0%" }}></div>
                </div>
                <p className="metric-copy">Estimates whether the image was fully or partially AI-generated.</p>
              </div>

              <div className="metric-card">
                <div className="metric-title">
                  <span>Deepfake / Manipulation</span>
                  <span className={`pill ${riskLabel(deepfakeScore).type}`}>{riskLabel(deepfakeScore).text}</span>
                </div>
                <div className="metric-value">{formatPercent(deepfakeScore)}</div>
                <div className="meter">
                  <div className="meter-fill" style={{ width: deepfakeScore != null ? `${deepfakeScore * 100}%` : "0%" }}></div>
                </div>
                <p className="metric-copy">Estimates whether facial or media manipulation appears present.</p>
              </div>
            </div>

            <div className="explanation-card">
              <p className="eyebrow">What this means</p>
              <ul>
                <li>{genaiScore == null ? "AI-generation score unavailable in this response." : genaiScore >= 0.7 ? "Strong AI-generation signals detected." : genaiScore >= 0.4 ? "Some AI-generation signals detected." : "Low AI-generation risk."}</li>
                <li>{deepfakeScore == null ? "Deepfake score unavailable in this response." : deepfakeScore >= 0.7 ? "Strong manipulation signals detected." : deepfakeScore >= 0.4 ? "Possible manipulation signals detected." : "Low deepfake/manipulation risk."}</li>
                <li>This scan is a detection estimate and should support human review, not replace it.</li>
              </ul>
            </div>
          </section>
        )}

        <section className="panel history-panel">
          <p className="eyebrow">History</p>
          <h2>Recent scans</h2>
          {history.length === 0 ? (
            <p className="panel-copy">Upload an image and run a scan to build a history log.</p>
          ) : (
            <div className="history-table">
              <div className="history-header">
                <span>File</span>
                <span>Provider</span>
                <span>Time</span>
                <span>AI</span>
                <span>Deepfake</span>
              </div>
              {history.map((item, index) => (
                <div className="history-row" key={`${item.filename}-${index}`}>
                  <span>{item.filename}</span>
                  <span>{item.provider}</span>
                  <span>{item.date}</span>
                  <span>{item.genai}</span>
                  <span>{item.deepfake}</span>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
