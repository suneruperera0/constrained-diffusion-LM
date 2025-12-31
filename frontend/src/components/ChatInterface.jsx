import React, { useState } from 'react'
import './ChatInterface.css'

const ChatInterface = () => {
  const [inputText, setInputText] = useState('')
  const [lockedSpans, setLockedSpans] = useState('')
  const [outputText, setOutputText] = useState('')
  const [visualization, setVisualization] = useState('')
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [settings, setSettings] = useState({
    diffusionSteps: 100,
    temperature: 0.8,
    editStrength: 0.4,
    repetitionPenalty: 1.2,
    topK: 50,
    topP: 0.9,
  })

  const handleGenerate = async () => {
    if (!inputText.trim()) return

    setLoading(true)
    try {
      // Call Gradio API endpoint
      // Gradio auto-generates endpoint at /api/edit_text_json/
      const response = await fetch('/api/edit_text_json/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: [
            inputText,
            lockedSpans,
            settings.diffusionSteps,
            settings.temperature,
            settings.topK,
            settings.topP,
            settings.repetitionPenalty,
            settings.editStrength,
          ]
        }),
      })
      
      const result = await response.json()
      // Gradio API returns { data: [...] }
      const jsonStr = result.data[0]
      const data = JSON.parse(jsonStr)
      
      setOutputText(data.generated_text || '')
      setVisualization(data.visualization || '')
      setMetrics(data.metrics || null)
    } catch (error) {
      console.error('Error:', error)
      setOutputText('Error generating text. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleGenerate()
    }
  }

  return (
    <div className="chat-container">
      <div className="chat-bar">
        <button
          className="icon-btn settings-btn"
          onClick={() => setSettingsOpen(!settingsOpen)}
          title="Settings"
        >
          ⚙️
        </button>
        
        <textarea
          className="chat-input"
          placeholder="Ask anything"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={handleKeyPress}
          rows={1}
        />
        
        <button
          className="icon-btn send-btn"
          onClick={handleGenerate}
          disabled={loading || !inputText.trim()}
          title="Send"
        >
          📤
        </button>
      </div>

      <div className="lock-input-wrapper">
        <input
          type="text"
          className="lock-input"
          placeholder="🔒 Lock these words (comma-separated)..."
          value={lockedSpans}
          onChange={(e) => setLockedSpans(e.target.value)}
        />
      </div>

      {settingsOpen && (
        <div className="settings-panel">
          <div className="settings-row">
            <label>
              Diffusion Steps
              <input
                type="range"
                min="10"
                max="200"
                step="10"
                value={settings.diffusionSteps}
                onChange={(e) =>
                  setSettings({ ...settings, diffusionSteps: parseInt(e.target.value) })
                }
              />
              <span>{settings.diffusionSteps}</span>
            </label>
          </div>

          <div className="settings-row">
            <label>
              Temperature
              <input
                type="range"
                min="0.1"
                max="2.0"
                step="0.1"
                value={settings.temperature}
                onChange={(e) =>
                  setSettings({ ...settings, temperature: parseFloat(e.target.value) })
                }
              />
              <span>{settings.temperature}</span>
            </label>
          </div>

          <div className="settings-row">
            <label>
              Edit Strength
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={settings.editStrength}
                onChange={(e) =>
                  setSettings({ ...settings, editStrength: parseFloat(e.target.value) })
                }
              />
              <span>{settings.editStrength}</span>
            </label>
          </div>

          <div className="settings-row">
            <label>
              Repetition Penalty
              <input
                type="range"
                min="1.0"
                max="2.0"
                step="0.1"
                value={settings.repetitionPenalty}
                onChange={(e) =>
                  setSettings({ ...settings, repetitionPenalty: parseFloat(e.target.value) })
                }
              />
              <span>{settings.repetitionPenalty}</span>
            </label>
          </div>

          <div className="settings-row">
            <label>
              Top-K
              <input
                type="range"
                min="0"
                max="100"
                step="5"
                value={settings.topK}
                onChange={(e) =>
                  setSettings({ ...settings, topK: parseInt(e.target.value) })
                }
              />
              <span>{settings.topK}</span>
            </label>
          </div>

          <div className="settings-row">
            <label>
              Top-P
              <input
                type="range"
                min="0.5"
                max="1.0"
                step="0.05"
                value={settings.topP}
                onChange={(e) =>
                  setSettings({ ...settings, topP: parseFloat(e.target.value) })
                }
              />
              <span>{settings.topP}</span>
            </label>
          </div>
        </div>
      )}

      {outputText && (
        <div className="output-section">
          <div className="output-header">OUTPUT</div>
          <div className="output-text">{outputText}</div>

          {visualization && (
            <>
              <div className="output-header">VISUALIZATION</div>
              <div
                className="visualization"
                dangerouslySetInnerHTML={{ __html: visualization }}
              />
            </>
          )}

          {metrics && (
            <>
              <div className="output-header">METRICS</div>
              <div className="metrics">
                <div className="metric-item">
                  <span className="metric-label">Constraint Fidelity</span>
                  <span className="metric-value">
                    {(metrics.constraint_fidelity * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Edit Rate</span>
                  <span className="metric-value">
                    {(metrics.edit_rate * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {loading && (
        <div className="loading">Generating...</div>
      )}
    </div>
  )
}

export default ChatInterface

