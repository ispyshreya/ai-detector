# Veil – AI Media Authenticity Detector

A React-based web app that scans uploaded images against multiple AI detection APIs to identify synthetic generation and manipulation signals.

## Features

- **Multi-provider support**: Seamlessly switch between Sightengine, custom REST APIs, and more.
- **Real-time image analysis**: Upload JPG, JPEG, PNG, or WEBP files for instant scanning.
- **Normalized scoring**: Results from different providers are normalized to a consistent 0–1 scale.
- **Risk assessment**: Automatic classification as Low, Medium, or High risk based on AI-generation and deepfake scores.
- **History tracking**: View all scans with timestamps, provider info, and confidence scores.
- **Modern UI**: Dark-themed dashboard with responsive design and real-time feedback.

## Getting Started

### Prerequisites

- Node.js 16+ and npm
- API credentials for at least one detection provider (e.g., Sightengine)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-detector.git
   cd ai-detector
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Update `.env` with your API credentials:
   ```
   VITE_SIGHTENGINE_API_USER=your_user_id
   VITE_SIGHTENGINE_API_SECRET=your_api_secret
   ```

### Running Locally

Start the development server:
```bash
npm run dev
```

Open your browser and navigate to `http://localhost:5173`.

### Building for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
.
├── app.jsx              # Main React component with multi-provider logic
├── main.jsx             # React entry point
├── index.html           # HTML template
├── styles.css           # Global and component styles
├── vite.config.js       # Vite configuration
├── package.json         # Dependencies and scripts
├── .env.example         # Example environment variables
└── README.md            # This file
```

## Supported Providers

### Sightengine
- **Models**: genai, deepfake
- **Env Keys**: `VITE_SIGHTENGINE_API_USER`, `VITE_SIGHTENGINE_API_SECRET`
- **Endpoint**: https://api.sightengine.com/1.0/check.json

### Custom REST API
- **Env Keys**: `VITE_CUSTOM_API_URL`, `VITE_CUSTOM_API_KEY`
- **Description**: Route requests to any custom detection endpoint that returns normalized fields.

## Adding a New Provider

Edit `app.jsx` and add an entry to the `providers` object:

```javascript
myProvider: {
  label: "My Provider",
  description: "Your provider description.",
  envKeys: ["VITE_MY_API_URL", "VITE_MY_API_KEY"],
  buildRequest: (file) => ({
    url: ENV.VITE_MY_API_URL,
    options: { /* ... */ }
  }),
  normalizeResponse: (json) => ({
    raw: json,
    genai: extractScore(json, "genai"),
    deepfake: extractScore(json, "deepfake"),
  }),
  help: "Instructions for this provider.",
}
```

## Technology Stack

- **Frontend**: React 18, Vite 5
- **Styling**: CSS3 with CSS Grid and Flexbox
- **State Management**: React Hooks (useState, useMemo)
- **Build Tool**: Vite with React plugin

## Security Notes

- API credentials are stored in `.env` and never committed to version control (see `.gitignore`).
- All API calls are made from the browser; no backend proxy is used.
- Raw API responses are logged but not stored persistently.

## License

MIT

## Support

For issues or feature requests, please open an issue on GitHub.
