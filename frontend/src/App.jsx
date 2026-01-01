import { useState, useRef } from 'react'
import './App.css'

function App() {
  const [input, setInput] = useState('')
  const [lockedSpans, setLockedSpans] = useState('')
  const [temperature, setTemperature] = useState(0.7)
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const inputRef = useRef(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const locked = lockedSpans.split(',').map(s => s.trim()).filter(Boolean)
    
    setMessages(prev => [...prev, {
      type: 'user',
      text: input,
      locked: locked
    }])

    setLoading(true)
    const currentInput = input
    setInput('')

    try {
      const response = await fetch('/api/edit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: currentInput,
          locked_spans: locked,
          temperature: temperature
        })
      })

      if (!response.ok) throw new Error('API error')
      
      const data = await response.json()
      
      setMessages(prev => [...prev, {
        type: 'assistant',
        text: data.output,
        locked: locked,
        preserved: data.locked_preserved,
        changed: data.tokens_changed,
        total: data.tokens_total
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        type: 'error',
        text: 'Failed to connect. Is the backend running on port 8000?'
      }])
    }

    setLoading(false)
  }

  const highlightLocked = (text, locked) => {
    if (!locked?.length) return text
    let result = text
    locked.forEach(span => {
      const regex = new RegExp(`(${span})`, 'gi')
      result = result.replace(regex, '<span class="highlight">$1</span>')
    })
    return result
  }

  const hasMessages = messages.length > 0

  return (
    <div className="app">

      {/* Main Content */}
      <main className={`main ${hasMessages ? 'has-messages' : ''}`}>
        {hasMessages && (
          <div className="messages">
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.type}`}>
                {msg.type === 'user' && (
                  <div className="message-bubble user-bubble">
                    <p>{msg.text}</p>
                    {msg.locked?.length > 0 && (
                      <div className="locked-indicator">
                        🔒 {msg.locked.join(', ')}
                      </div>
                    )}
                  </div>
                )}
                
                {msg.type === 'assistant' && (
                  <div className="message-bubble assistant-bubble">
                    <div className="avatar">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                      </svg>
                    </div>
                    <div className="content">
                      <p dangerouslySetInnerHTML={{ __html: highlightLocked(msg.text, msg.locked) }} />
                      <div className="meta">
                        <span className={msg.preserved ? 'success' : 'error'}>
                          {msg.preserved ? '✓ Constraints preserved' : '✗ Constraint broken'}
                        </span>
                        <span className="stats">{msg.changed}/{msg.total} tokens edited</span>
                      </div>
                    </div>
                  </div>
                )}
                
                {msg.type === 'error' && (
                  <div className="message-bubble error-bubble">
                    <p>{msg.text}</p>
                  </div>
                )}
              </div>
            ))}
            
            {loading && (
              <div className="message assistant">
                <div className="message-bubble assistant-bubble">
                  <div className="avatar">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                    </svg>
                  </div>
                  <div className="content">
                    <div className="typing">
                      <span></span><span></span><span></span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Input Area */}
        <div className="input-container">
          <form className="input-form" onSubmit={handleSubmit}>
            <div className="input-wrapper">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Enter text to edit..."
                disabled={loading}
              />
              
              <button 
                type="submit" 
                className={`send-btn ${input.trim() ? 'active' : ''}`}
                disabled={loading || !input.trim()}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M4 12h16M12 4l8 8-8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </div>
            
            <div className="locked-input-wrapper">
              <span className="lock-icon">🔒</span>
              <input
                type="text"
                value={lockedSpans}
                onChange={(e) => setLockedSpans(e.target.value)}
                placeholder="Words to keep unchanged (comma separated)"
                disabled={loading}
              />
            </div>
          </form>
        </div>
      </main>
    </div>
  )
}

export default App
