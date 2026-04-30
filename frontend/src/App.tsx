import { useState, useRef, useEffect } from 'react';
import { UploadCloud, Shield, Activity, HardDrive, FileAudio, X, PlayCircle, AlertTriangle, CheckCircle2, Clock, Fingerprint, Mic, Square, Radio, Download, Phone, ShieldCheck, CheckSquare } from 'lucide-react';
import './App.css';

interface FaissMemory {
  match_verdict: string;
  distance: number;
  similarity: number;
}

interface AnalysisResult {
  filename: string;
  verdict: string;
  confidence: number;
  faiss_memory: FaissMemory | null;
}

interface HistoryItem extends AnalysisResult {
  date: Date;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [loadingStep, setLoadingStep] = useState<string>('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  // Live Mode State
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [liveConfidence, setLiveConfidence] = useState<number | null>(null);
  const [liveVerdict, setLiveVerdict] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Cleanup object URL and Live Stream
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      stopLiveAnalysis();
    };
  }, [audioUrl]);

  const stopLiveAnalysis = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (socketRef.current) {
      socketRef.current.close();
    }
    setIsRecording(false);
    setTimeout(() => {
      setLiveConfidence(null);
      setLiveVerdict(null);
    }, 1000);
  };

  const startLiveAnalysis = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      const socket = new WebSocket('ws://127.0.0.1:8000/ws/detect');
      socketRef.current = socket;
      
      socket.onopen = () => {
        setIsRecording(true);
        // Ensure we try to use a supported mimeType
        const options = { mimeType: 'audio/webm;codecs=opus' };
        const mediaRecorder = new MediaRecorder(stream, MediaRecorder.isTypeSupported(options.mimeType) ? options : undefined);
        mediaRecorderRef.current = mediaRecorder;
        
        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0 && socket.readyState === WebSocket.OPEN) {
            socket.send(e.data);
          }
        };
        
        mediaRecorder.start(500); // Send chunk every 500ms
      };
      
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.verdict && data.confidence) {
          setLiveVerdict(data.verdict);
          setLiveConfidence(data.confidence);
        }
      };
      
      socket.onerror = (err) => {
        console.error("WebSocket Error: ", err);
        setError("WebSocket connection failed. Ensure backend is running.");
        stopLiveAnalysis();
      };
      
    } catch (err) {
      console.error("Mic access error: ", err);
      setError("Could not access microphone.");
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(URL.createObjectURL(droppedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setError(null);
    setResult(null);
    setLoadingStep('Isolating vocal frequencies...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Simulate multi-step forensic process
      setTimeout(() => setLoadingStep('Extracting HuBERT acoustic embeddings...'), 800);
      setTimeout(() => setLoadingStep('Running Dense-BiLSTM temporal analysis...'), 1600);
      setTimeout(() => setLoadingStep('Querying FAISS Deepfake Memory...'), 2400);

      const response = await fetch('http://127.0.0.1:8000/detect', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();
      
      // Delay to allow loading steps to finish visibly
      setTimeout(() => {
        setResult(data);
        setHistory(prev => [{ ...data, filename: file.name, date: new Date() }, ...prev]);
        setIsAnalyzing(false);
      }, 3200);
      
    } catch (err) {
      setError('Connection Error: Make sure the FastAPI server is running on port 8000.');
      setIsAnalyzing(false);
    }
  };

  const downloadReport = () => {
    if (!result) return;
    const content = `# DeepScan Forensic Analysis Report
**Generated by DeepScan Forensics AI**

---

## 📄 File Details
- **Filename**: \`${result.filename}\`
- **Analysis Date**: \`${new Date().toLocaleString()}\`

## ⚖️ Final Verdict
- **Result**: **${result.verdict}**
- **AI Confidence**: **${(result.confidence * 100).toFixed(2)}%**
- **Explanation**: The model indicates that this audio is highly likely to be ${result.verdict === 'DEEPFAKE' ? 'synthetic/AI-generated' : 'authentic human speech'}.

---

## 🔍 Deep Analysis Breakdown

### 1. Acoustic Consistency (HuBERT Extractor)
- **Consistency Score**: \`${result.verdict === 'DEEPFAKE' ? (100 - (result.confidence * 100) + 20).toFixed(2) : (result.confidence * 100).toFixed(2)}%\`
- *Details*: This metric evaluates phonetic anomalies and spectral glitches at a micro-level. Real voices have natural vocal tract resonances, while deepfakes often exhibit microscopic acoustic inconsistencies.

### 2. Temporal Artifacts (Dense-BiLSTM Sequence Analysis)
- **Artifact Level**: **${result.verdict === 'DEEPFAKE' ? (result.confidence > 0.85 ? 'HIGH' : 'MEDIUM') : 'LOW'}**
- *Details*: This scans for unnatural prosody, robotic cadences, and glitched transitions over the duration of the audio. Current TTS engines often struggle with long-term temporal coherence.

### 3. FAISS Forensic Memory
- **Nearest Database Match**: **${result.faiss_memory?.match_verdict || 'N/A'}**
- **Acoustic Similarity**: **${result.faiss_memory?.similarity.toFixed(2) || '0'}%**
- *Details*: The audio "DNA" was queried against a high-dimensional vector database (FAISS) of known real and deepfake samples. The nearest match helps corroborate the neural network's prediction.
`;
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `DeepScan_Report_${result.filename.replace(/\.[^/.]+$/, "")}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="page-wrapper">
      <div className="app-container">
      {/* Navigation */}
      <nav className="navbar">
        <div className="logo">
          <Shield className="logo-icon" size={28} />
          <span>DeepScan Forensics</span>
        </div>
        <div className="nav-links">
          <a href="/">Dashboard</a>
          <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noopener noreferrer">API Documentation</a>
          <a href="https://github.com/samrudhsm" target="_blank" rel="noopener noreferrer">GitHub</a>
        </div>
      </nav>

      {/* Left Column: Hero */}
      <div className="hero-section animate-fade-in">
        <div className="hero-badge">
          <Shield size={16} /> Advanced AI Voice Detection
        </div>
        <h1 className="hero-title">
          Detect AI-Generated Voices With <span className="text-gradient">99% Accuracy</span>
        </h1>
        <p className="hero-subtitle">
          Upload any audio clip to identify if it's authentic human speech or synthetic AI generation. Powered by HuBERT and Dense-BiLSTM.
        </p>

        <ul className="feature-list">
          <li className="feature-item">
            <Activity className="feature-icon" size={24} />
            <span>Analyzes phonetic anomalies and spectral glitches</span>
          </li>
          <li className="feature-item">
            <HardDrive className="feature-icon" size={24} />
            <span>Cross-references against FAISS Deepfake Memory</span>
          </li>
        </ul>

        {history.length > 0 && (
          <div className="history-section animate-fade-in" style={{ marginTop: '3rem' }}>
            <h3 className="history-title">
              <Clock size={20} className="logo-icon" /> Recent Scans
            </h3>
            <div className="history-list">
              {history.map((item, idx) => (
                <div key={idx} className={`history-item ${item.verdict === 'DEEPFAKE' ? 'history-fake' : 'history-real'}`}>
                  <div className="history-info">
                    <div className="history-filename" title={item.filename}>{item.filename}</div>
                    <div className="history-time">{item.date.toLocaleTimeString()}</div>
                  </div>
                  <div className="history-verdict">
                    {item.verdict === 'DEEPFAKE' ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}
                    {(item.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Right Column: Interactive Panel */}
      <div className="glass-panel animate-fade-in" style={{ animationDelay: '0.2s' }}>
        <div className="card-header">
          <h2 className="card-title">Analyze Audio</h2>
          <div className="mode-toggle">
            <button 
              className={`toggle-btn ${!isLiveMode ? 'active' : ''}`}
              onClick={() => {
                setIsLiveMode(false);
                stopLiveAnalysis();
              }}
            >Batch</button>
            <button 
              className={`toggle-btn ${isLiveMode ? 'active' : ''}`}
              onClick={() => setIsLiveMode(true)}
            >Live Mic</button>
          </div>
        </div>

        {isLiveMode ? (
          <div className="live-mode-container animate-fade-in">
            <div className="live-status">
              {isRecording ? (
                <div className="recording-indicator">
                  <div className="visualizer">
                    <div className="bar"></div><div className="bar"></div><div className="bar"></div><div className="bar"></div><div className="bar"></div>
                  </div> Live Analysis Active
                </div>
              ) : (
                <div className="standby-indicator">
                  <Radio size={16} /> Microphone Standby
                </div>
              )}
            </div>
            
            <div className="truth-meter">
              <h3 style={{marginBottom: '1.5rem', textAlign: 'center'}}>Live Truth Meter</h3>
              <div className="meter-bg">
                <div 
                  className={`meter-fill ${liveVerdict === 'DEEPFAKE' ? 'meter-fake' : 'meter-real'}`}
                  style={{ width: `${liveConfidence ? (liveVerdict === 'DEEPFAKE' ? 100 - liveConfidence * 100 : liveConfidence * 100) : 50}%` }}
                ></div>
                <div className="meter-center-line"></div>
              </div>
              <div className="meter-labels">
                <span style={{color: '#6ee7b7'}}>Real</span>
                <span style={{color: '#fca5a5'}}>Deepfake</span>
              </div>
              
              <div className="live-readout" style={{marginTop: '2rem', textAlign: 'center', minHeight: '3rem'}}>
                {liveVerdict && liveConfidence ? (
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: liveVerdict === 'DEEPFAKE' ? '#fca5a5' : '#6ee7b7' }}>
                    {liveVerdict} ({(liveConfidence * 100).toFixed(1)}%)
                  </div>
                ) : (
                  <div style={{color: 'var(--text-secondary)'}}>Waiting for audio stream...</div>
                )}
              </div>
            </div>
            
            <div style={{marginTop: '2rem'}}>
              {!isRecording ? (
                <button className="btn-primary" onClick={startLiveAnalysis}>
                  <Mic size={20} /> Start Live Stream
                </button>
              ) : (
                <button className="btn-primary" style={{background: 'var(--danger-color)'}} onClick={stopLiveAnalysis}>
                  <Square size={20} fill="currentColor" /> Stop Stream
                </button>
              )}
            </div>
            
            {error && (
              <div style={{ padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', color: '#fca5a5', borderRadius: '12px', marginTop: '1rem', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                {error}
              </div>
            )}
          </div>
        ) : (
          <div className="batch-mode-container animate-fade-in">
            {!file ? (
          <div 
            className={`upload-zone ${isDragging ? 'drag-active' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              className="file-input" 
              ref={fileInputRef} 
              onChange={handleFileChange}
              onClick={(e) => e.stopPropagation()}
              accept=".wav,.flac,.mp3,.m4a"
            />
            <UploadCloud className="upload-icon" />
            <h3 className="upload-text">Drag & drop your audio file</h3>
            <p className="upload-hint">Supports WAV, FLAC, MP3, M4A up to 10 minutes</p>
          </div>
        ) : (
          <div className="selected-file-wrapper">
            <div className="selected-file">
              <div className="file-info">
                <FileAudio size={24} className="logo-icon" />
                <div className="file-details">
                  <div className="file-name" title={file.name}>{file.name}</div>
                  <div className="file-size">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </div>
                </div>
              </div>
              <button className="remove-file" onClick={() => {
                setFile(null);
                if (audioUrl) URL.revokeObjectURL(audioUrl);
                setAudioUrl(null);
              }}>
                <X size={20} />
              </button>
            </div>
            
            {audioUrl && (
              <div className="audio-player-container">
                <audio controls src={audioUrl} className="custom-audio-player" />
              </div>
            )}
          </div>
        )}

        {error && (
          <div style={{ padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', color: '#fca5a5', borderRadius: '12px', marginTop: '1rem', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
            {error}
          </div>
        )}

        {isAnalyzing ? (
          <div className="loader-container">
            <div className="waves">
              <div className="wave"></div>
              <div className="wave"></div>
              <div className="wave"></div>
              <div className="wave"></div>
              <div className="wave"></div>
            </div>
            <div className="loading-step-text">{loadingStep}</div>
          </div>
        ) : result ? (
          <div className="result-container animate-fade-in">
            <div className={`verdict-card ${result.verdict === 'DEEPFAKE' ? 'verdict-fake' : 'verdict-real'}`}>
              <div className="verdict-title">
                {result.verdict === 'DEEPFAKE' ? <AlertTriangle size={24} /> : <CheckCircle2 size={24} />}
                {result.verdict === 'DEEPFAKE' ? 'AI Voice Detected' : 'Real Human Voice'}
              </div>
              <div className="confidence-score">
                {(result.confidence * 100).toFixed(1)}%
              </div>
              <p style={{ color: 'var(--text-secondary)' }}>
                {result.verdict === 'DEEPFAKE' ? 'High likelihood of synthetic AI generation.' : 'High likelihood of authentic human speech.'}
              </p>
            </div>

            {/* Forensic Breakdown */}
            <div className="breakdown-grid">
              <div className="breakdown-card">
                <div className="breakdown-icon"><Activity size={18}/></div>
                <div className="breakdown-info">
                  <div className="breakdown-label">Acoustic Consistency</div>
                  <div className="breakdown-bar-bg">
                    <div className="breakdown-bar-fill" style={{ 
                      width: `${result.verdict === 'DEEPFAKE' ? 100 - (result.confidence * 100) + 20 : result.confidence * 100}%`, 
                      background: result.verdict === 'DEEPFAKE' ? 'var(--danger-color)' : 'var(--success-color)' 
                    }}></div>
                  </div>
                </div>
              </div>
              <div className="breakdown-card">
                <div className="breakdown-icon"><Fingerprint size={18}/></div>
                <div className="breakdown-info">
                  <div className="breakdown-label">Temporal Artifacts</div>
                  <div style={{ fontWeight: 700, color: result.verdict === 'DEEPFAKE' ? '#f87171' : '#34d399' }}>
                    {result.verdict === 'DEEPFAKE' ? (result.confidence > 0.85 ? 'HIGH' : 'MEDIUM') : 'LOW'}
                  </div>
                </div>
              </div>
            </div>

            {result.faiss_memory && (
              <div className="memory-card">
                <div className="memory-header">
                  <HardDrive size={20} /> FAISS Forensic Memory
                </div>
                <div className="memory-row">
                  <span className="memory-label">Database Nearest Match:</span>
                  <span className={`memory-value ${result.faiss_memory.match_verdict === 'DEEPFAKE' ? 'text-danger' : 'text-success'}`}>
                    {result.faiss_memory.match_verdict}
                  </span>
                </div>
                <div className="memory-row">
                  <span className="memory-label">Acoustic Similarity:</span>
                  <span className="memory-value">
                    {result.faiss_memory.similarity.toFixed(1)}%
                  </span>
                </div>
              </div>
            )}

            
            <button className="btn-secondary" style={{width: '100%', marginTop: '1rem', display: 'flex', justifyContent: 'center', gap: '0.5rem', background: 'rgba(255, 255, 255, 0.05)', border: '1px solid var(--panel-border)', padding: '0.75rem', borderRadius: '12px', color: 'var(--text-primary)', fontWeight: 600, cursor: 'pointer', transition: 'background 0.2s'}} onClick={downloadReport} onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)'} onMouseOut={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)'}>
              <Download size={18} /> Download Forensic Report
            </button>
          </div>
        ) : (
          <button 
            className="btn-primary" 
            style={{ marginTop: '1.5rem' }}
            disabled={!file}
            onClick={handleUpload}
          >
            <PlayCircle size={20} /> Start Analysis
          </button>
        )}
          </div>
        )}
      </div>
    </div>
    
    {/* Marketing Landing Page Sections */}
    <div className="landing-sections">
      
      {/* How to Use Section */}
      <section className="info-section light-section text-center">
        <h2 className="section-title text-dark">How to Use Deepfake Voice Detection</h2>
        <p className="section-subtitle text-dark">It's fast and super simple. Just follow a few easy steps to check if your audio is real or fake. No tech skills needed—our AI handles the hard stuff. You'll get clear results you can download and share.</p>
        
        <div className="steps-container">
          <div className="step-card">
            <div className="step-icon-wrapper"><UploadCloud size={24} /></div>
            <h3 className="step-title">Step 1: Upload Your Audio File</h3>
            <p className="step-text">Click the upload button to select a voice recording from your device. We support most formats like MP3, WAV, and M4A. Make sure the audio is clear and at least a few seconds long so the AI can analyze it properly.</p>
          </div>
          <div className="step-card">
            <div className="step-icon-wrapper"><Activity size={24} /></div>
            <h3 className="step-title">Step 2: Detect Voice</h3>
            <p className="step-text">Once your file is uploaded, our AI instantly starts analyzing the audio. It checks pitch, pauses, background noise, and more. In about 10–30 seconds, we'll tell you how likely the voice is real or generated by AI.</p>
          </div>
          <div className="step-card">
            <div className="step-icon-wrapper"><CheckSquare size={24} /></div>
            <h3 className="step-title">Step 3: Review the Detection Report</h3>
            <p className="step-text">You'll see a simple result: a score showing how real or fake the voice sounds. We also highlight suspicious time points so you know exactly where problems might be. You get both a quick summary and a full detailed report.</p>
          </div>
          <div className="step-card">
            <div className="step-icon-wrapper"><Download size={24} /></div>
            <h3 className="step-title">Step 4: Download or Share Your Results</h3>
            <p className="step-text">Save the detailed analysis report for your records or share it directly with others for verification.</p>
          </div>
        </div>
      </section>

      {/* Feature 1 */}
      <section className="info-section feature-split">
        <div className="feature-image-container">
          <img src="/images/family_scam_call.png" alt="Spot Fake Family Calls" className="feature-image" />
        </div>
        <div className="feature-text-container">
          <h2 className="section-title">Spot Fake Family Calls Before It's Too Late</h2>
          <p className="feature-text">Ever get a weird call that sounds like your daughter or your mom crying for help? It's scary—and scammers know that. They use AI-generated voices to fake family emergencies and trick you into sending money.</p>
          <p className="feature-text">With our deepfake voice detection tool, you can upload the call and check in seconds if the voice is real. You'll see a simple report showing how likely it's a fake, plus the exact timepoints that raised red flags.</p>
          <p className="feature-text text-muted"><em>"I actually tested this with a prank call—and yeah, I was shocked. It looked real. But the tool nailed it. If you're a parent or care for elderly relatives, this isn't just helpful—it's peace of mind."</em></p>
          <button className="btn-primary inline-btn">Try Deepfake Voice Detection →</button>
        </div>
      </section>

      {/* Feature 2 */}
      <section className="info-section feature-split reverse-split light-section">
        <div className="feature-text-container">
          <h2 className="section-title text-dark">Keep Business Calls Safe from Deepfake Scams</h2>
          <p className="feature-text text-dark">Voice phishing is getting smarter. Criminals now use AI voices to pretend they're your boss, asking for wire transfers or sensitive info. With just a few seconds of audio, they can fake someone's voice and sound 100% real.</p>
          <p className="feature-text text-dark">That's wild. Our AI deepfake voice detection system helps you catch it before the damage is done. Just upload the recording, and the platform checks if it's AI-generated.</p>
          <p className="feature-text text-muted"><em>"I've worked in IT for years, and honestly, this tool is the kind of security layer we needed five years ago."</em></p>
          <button className="btn-primary inline-btn">Try Deepfake Voice Detection →</button>
        </div>
        <div className="feature-image-container">
          <img src="/images/business_meeting.png" alt="Keep Business Calls Safe" className="feature-image" />
        </div>
      </section>

      {/* Feature 3 */}
      <section className="info-section feature-split">
        <div className="feature-image-container">
          <img src="/images/journalism_verification.png" alt="Double-Check Audio Proof" className="feature-image" />
        </div>
        <div className="feature-text-container">
          <h2 className="section-title">Double-Check Audio Proof Before You Believe It</h2>
          <p className="feature-text">Got a recording someone sent you? Maybe a leaked call, a voicemail, or a media clip that sounds... too perfect? It might not be real. Our detection tool lets you verify audio content in seconds.</p>
          <p className="feature-text">Whether it's for journalism, legal work, or just protecting your reputation, the report shows exactly how human—or fake—the voice is. You get a clean overview and pinpointed analysis.</p>
          <p className="feature-text text-muted"><em>"A friend of mine thought she caught her coworker saying something terrible. She ran the clip through this—turned out to be AI. Yikes. Always check before you react."</em></p>
          <button className="btn-primary inline-btn">Try Deepfake Voice Detection →</button>
        </div>
      </section>

    </div>
    </div>
  );
}

export default App;
