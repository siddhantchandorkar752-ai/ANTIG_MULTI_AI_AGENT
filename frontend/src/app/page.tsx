'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Search, Brain, Zap, Globe, FileText, Shield, 
  Activity, ChevronRight, Loader2, CheckCircle2,
  AlertCircle, BarChart3, Clock, DollarSign, Star,
  Network, BookOpen, RefreshCw
} from 'lucide-react'

// ── Types ─────────────────────────────────────────────────────────────────────
type ExecutionState =
  | 'IDLE' | 'PLANNING' | 'SEARCHING' | 'READING'
  | 'WRITING' | 'CRITIQUING' | 'VERIFYING' | 'FINALIZING'
  | 'COMPLETED' | 'FAILED' | 'RETRYING'

interface SessionStatus {
  session_id: string
  state: ExecutionState
  iteration: number
  chunks_collected: number
  errors: string[]
  cost_usd: number
  confidence: number
}

interface ResearchReport {
  title: string
  executive_summary: string
  abstract: string
  introduction: string
  methodology: string
  core_analysis: string
  technical_breakdown: string
  key_findings: string[]
  counterarguments: string
  risks: string
  future_outlook: string
  conclusion: string
  confidence_score: number
  critique_scores: Record<string, number>
  markdown_content: string
  references_count: number
  iteration: number
}

interface WsEvent {
  type: string
  session_id: string
  current_state: ExecutionState
  iteration: number
  chunks_collected: number
  errors: string[]
  cost: number
}

// ── Constants ─────────────────────────────────────────────────────────────────
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'

const STATE_CONFIG: Record<ExecutionState, { label: string; color: string; icon: React.FC<any>; pulse: boolean }> = {
  IDLE: { label: 'Idle', color: '#8b949e', icon: Activity, pulse: false },
  PLANNING: { label: 'Planning', color: '#d29922', icon: Brain, pulse: true },
  SEARCHING: { label: 'Searching', color: '#58a6ff', icon: Globe, pulse: true },
  READING: { label: 'Reading', color: '#bc8cff', icon: BookOpen, pulse: true },
  WRITING: { label: 'Writing', color: '#3fb950', icon: FileText, pulse: true },
  CRITIQUING: { label: 'Critiquing', color: '#f78166', icon: Shield, pulse: true },
  VERIFYING: { label: 'Verifying', color: '#58a6ff', icon: CheckCircle2, pulse: true },
  FINALIZING: { label: 'Finalizing', color: '#d29922', icon: Star, pulse: true },
  COMPLETED: { label: 'Complete', color: '#3fb950', icon: CheckCircle2, pulse: false },
  FAILED: { label: 'Failed', color: '#f78166', icon: AlertCircle, pulse: false },
  RETRYING: { label: 'Retrying', color: '#d29922', icon: RefreshCw, pulse: true },
}

const AGENT_NODES = [
  { id: 'planner', label: 'Planner', color: '#d29922', state: 'PLANNING' as ExecutionState },
  { id: 'search', label: 'Search', color: '#58a6ff', state: 'SEARCHING' as ExecutionState },
  { id: 'reader', label: 'Reader', color: '#bc8cff', state: 'READING' as ExecutionState },
  { id: 'memory', label: 'Memory', color: '#58a6ff', state: 'RETRIEVING' as ExecutionState },
  { id: 'writer', label: 'Writer', color: '#3fb950', state: 'WRITING' as ExecutionState },
  { id: 'critic', label: 'Critic', color: '#f78166', state: 'CRITIQUING' as ExecutionState },
  { id: 'verifier', label: 'Verifier', color: '#58a6ff', state: 'VERIFYING' as ExecutionState },
]

// ── Agent Visualization ────────────────────────────────────────────────────────
function AgentNode({ agent, activeState }: { agent: typeof AGENT_NODES[0]; activeState: ExecutionState }) {
  const isActive = activeState === agent.state ||
    (activeState === 'SEARCHING' && agent.id === 'search') ||
    (activeState === 'READING' && agent.id === 'reader') ||
    (activeState === 'WRITING' && agent.id === 'writer') ||
    (activeState === 'CRITIQUING' && agent.id === 'critic') ||
    (activeState === 'VERIFYING' && agent.id === 'verifier') ||
    (activeState === 'PLANNING' && agent.id === 'planner')

  return (
    <motion.div
      className="relative flex flex-col items-center gap-2"
      animate={isActive ? { scale: [1, 1.05, 1] } : {}}
      transition={{ duration: 1.5, repeat: Infinity }}
    >
      <div
        className="relative w-14 h-14 rounded-full flex items-center justify-center border-2 transition-all duration-500"
        style={{
          borderColor: isActive ? agent.color : '#21262d',
          background: isActive ? `${agent.color}22` : '#161b22',
          boxShadow: isActive ? `0 0 20px ${agent.color}44` : 'none',
        }}
      >
        {isActive && (
          <motion.div
            className="absolute inset-0 rounded-full"
            style={{ border: `2px solid ${agent.color}` }}
            animate={{ scale: [1, 1.4, 1], opacity: [1, 0, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        )}
        <Zap size={22} style={{ color: isActive ? agent.color : '#8b949e' }} />
      </div>
      <span className="text-xs font-medium" style={{ color: isActive ? agent.color : '#8b949e' }}>
        {agent.label}
      </span>
    </motion.div>
  )
}

// ── Status Badge ──────────────────────────────────────────────────────────────
function StatusBadge({ state }: { state: ExecutionState }) {
  const config = STATE_CONFIG[state] || STATE_CONFIG.IDLE
  const Icon = config.icon
  return (
    <motion.div
      className="flex items-center gap-2 px-3 py-1.5 rounded-full border"
      style={{ borderColor: `${config.color}44`, background: `${config.color}11` }}
      animate={config.pulse ? { opacity: [1, 0.6, 1] } : {}}
      transition={{ duration: 1.5, repeat: Infinity }}
    >
      <Icon size={14} style={{ color: config.color }} />
      <span className="text-xs font-semibold" style={{ color: config.color }}>
        {config.label}
      </span>
    </motion.div>
  )
}

// ── Metric Card ───────────────────────────────────────────────────────────────
function MetricCard({ label, value, icon: Icon, color }: {
  label: string; value: string; icon: React.FC<any>; color: string
}) {
  return (
    <div className="glass-card p-4 flex items-center gap-3">
      <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: `${color}22` }}>
        <Icon size={18} style={{ color }} />
      </div>
      <div>
        <div className="text-xs text-omega-muted">{label}</div>
        <div className="text-sm font-bold text-omega-text">{value}</div>
      </div>
    </div>
  )
}

// ── Report Viewer ─────────────────────────────────────────────────────────────
function ReportViewer({ report }: { report: ResearchReport }) {
  const [activeTab, setActiveTab] = useState<'full' | 'summary' | 'findings'>('summary')
  const tabs = [
    { id: 'summary', label: 'Summary' },
    { id: 'findings', label: 'Findings' },
    { id: 'full', label: 'Full Report' },
  ]

  return (
    <motion.div
      className="glass-card overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      {/* Header */}
      <div className="p-6 border-b border-omega-border">
        <div className="flex items-start justify-between gap-4 mb-4">
          <h2 className="text-xl font-bold text-omega-text leading-tight">{report.title}</h2>
          <div className="flex items-center gap-2 flex-shrink-0">
            <div className="text-xs text-omega-muted">Confidence</div>
            <div className="text-lg font-bold" style={{ color: '#3fb950' }}>
              {(report.confidence_score * 100).toFixed(0)}%
            </div>
          </div>
        </div>
        <div className="flex items-center gap-4 text-xs text-omega-muted">
          <span>{report.references_count} sources</span>
          <span>·</span>
          <span>Iteration {report.iteration}</span>
          <span>·</span>
          <span className="text-omega-accent3">Research Complete</span>
        </div>

        {/* Score breakdown */}
        {Object.keys(report.critique_scores).length > 0 && (
          <div className="mt-4 grid grid-cols-4 gap-2">
            {Object.entries(report.critique_scores).slice(0, 4).map(([k, v]) => (
              <div key={k} className="text-center">
                <div className="text-xs text-omega-muted capitalize">{k.replace('_score', '')}</div>
                <div className="text-sm font-semibold text-omega-accent">
                  {((v as number) * 100).toFixed(0)}%
                </div>
                <div className="progress-bar mt-1">
                  <div className="progress-fill" style={{ width: `${(v as number) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-1 mt-4 bg-omega-bg rounded-lg p-1">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex-1 py-1.5 px-3 rounded-md text-xs font-medium transition-all ${
                activeTab === tab.id
                  ? 'bg-omega-card text-omega-text'
                  : 'text-omega-muted hover:text-omega-text'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="p-6 max-h-[60vh] overflow-y-auto">
        <AnimatePresence mode="wait">
          {activeTab === 'summary' && (
            <motion.div key="summary" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div className="space-y-4">
                <div>
                  <h3 className="text-xs font-semibold text-omega-accent uppercase tracking-wider mb-2">Executive Summary</h3>
                  <p className="text-omega-muted text-sm leading-relaxed">{report.executive_summary}</p>
                </div>
                <div>
                  <h3 className="text-xs font-semibold text-omega-accent2 uppercase tracking-wider mb-2">Conclusion</h3>
                  <p className="text-omega-muted text-sm leading-relaxed">{report.conclusion}</p>
                </div>
              </div>
            </motion.div>
          )}
          {activeTab === 'findings' && (
            <motion.div key="findings" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <h3 className="text-xs font-semibold text-omega-accent3 uppercase tracking-wider mb-3">Key Findings</h3>
              <div className="space-y-2">
                {report.key_findings.map((finding, i) => (
                  <div key={i} className="flex items-start gap-3 p-3 rounded-lg bg-omega-bg border border-omega-border">
                    <div className="w-5 h-5 rounded-full bg-omega-accent3/20 text-omega-accent3 flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">
                      {i + 1}
                    </div>
                    <p className="text-sm text-omega-text leading-relaxed">{finding}</p>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
          {activeTab === 'full' && (
            <motion.div key="full" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div className="markdown-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {report.markdown_content || '# Report generating...'}
                </ReactMarkdown>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function HomePage() {
  const [query, setQuery] = useState('')
  const [depth, setDepth] = useState(3)
  const [budget, setBudget] = useState(2.0)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [status, setStatus] = useState<SessionStatus | null>(null)
  const [report, setReport] = useState<ResearchReport | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [events, setEvents] = useState<string[]>([])
  const wsRef = useRef<WebSocket | null>(null)
  const pollRef = useRef<NodeJS.Timeout | null>(null)

  const addEvent = useCallback((msg: string) => {
    setEvents(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev].slice(0, 50))
  }, [])

  const connectWs = useCallback((sid: string) => {
    const ws = new WebSocket(`${WS_URL}/ws/${sid}`)
    ws.onopen = () => addEvent('WebSocket connected — streaming live updates')
    ws.onmessage = (e) => {
      const data: WsEvent = JSON.parse(e.data)
      setStatus(prev => prev ? {
        ...prev,
        state: data.current_state,
        iteration: data.iteration,
        chunks_collected: data.chunks_collected,
        cost_usd: data.cost,
        errors: data.errors,
      } : null)
      addEvent(`Agent state: ${data.current_state} | Iteration: ${data.iteration} | Chunks: ${data.chunks_collected}`)
    }
    ws.onerror = () => addEvent('WebSocket error — polling fallback active')
    ws.onclose = () => addEvent('WebSocket disconnected')
    wsRef.current = ws
  }, [addEvent])

  const pollStatus = useCallback(async (sid: string) => {
    try {
      const res = await fetch(`${API_URL}/api/v1/research/${sid}/status`)
      if (res.ok) {
        const data: SessionStatus = await res.json()
        setStatus(data)
        if (data.state === 'COMPLETED' || data.state === 'FAILED') {
          if (pollRef.current) clearInterval(pollRef.current)
          if (data.state === 'COMPLETED') {
            // Fetch report
            const rRes = await fetch(`${API_URL}/api/v1/research/${sid}/report`)
            if (rRes.ok) {
              const reportData: ResearchReport = await rRes.json()
              setReport(reportData)
              addEvent('Research complete — report ready')
            }
          }
          setIsLoading(false)
        }
      }
    } catch (err) {
      addEvent('Status poll failed — retrying...')
    }
  }, [addEvent])

  const startResearch = async () => {
    if (!query.trim() || query.length < 10) return
    setIsLoading(true)
    setReport(null)
    setEvents([])
    setStatus(null)
    addEvent(`Starting research: "${query.slice(0, 60)}..."`)

    try {
      const res = await fetch(`${API_URL}/api/v1/research/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          depth,
          cost_budget_usd: budget,
          max_sources: 20,
          token_budget: 80000,
        }),
      })
      const data = await res.json()
      const sid = data.session_id
      setSessionId(sid)
      setStatus({ session_id: sid, state: 'PLANNING', iteration: 0, chunks_collected: 0, errors: [], cost_usd: 0, confidence: 0 })
      addEvent(`Session created: ${sid}`)

      connectWs(sid)
      pollRef.current = setInterval(() => pollStatus(sid), 3000)
    } catch (err) {
      addEvent('Failed to start research session')
      setIsLoading(false)
    }
  }

  useEffect(() => {
    return () => {
      wsRef.current?.close()
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  const currentState = status?.state || 'IDLE'

  return (
    <div className="min-h-screen bg-omega-bg bg-grid-pattern bg-grid relative overflow-x-hidden">
      {/* Background radial glow */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[400px] bg-gradient-radial from-omega-accent/5 to-transparent rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-0 w-[400px] h-[300px] bg-gradient-radial from-omega-accent2/5 to-transparent rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <motion.header
          className="text-center mb-10"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center justify-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-omega-accent/20 border border-omega-accent/30 flex items-center justify-center">
              <Network size={22} className="text-omega-accent" />
            </div>
            <h1 className="text-3xl font-bold gradient-text tracking-tight">
              OMEGA RESEARCH GRID
            </h1>
          </div>
          <p className="text-omega-muted text-sm max-w-xl mx-auto">
            Autonomous Multi-Agent Research OS · LangGraph · GPT-4o · Real-time intelligence synthesis
          </p>
        </motion.header>

        {/* Query Input */}
        <motion.div
          className="glass-card p-6 mb-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="mb-4">
            <label className="block text-xs font-semibold text-omega-muted uppercase tracking-wider mb-2">
              Research Query
            </label>
            <textarea
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="What are the latest breakthroughs in large language model reasoning capabilities?"
              className="w-full bg-omega-bg border border-omega-border rounded-lg px-4 py-3 text-omega-text placeholder-omega-muted text-sm focus:outline-none focus:border-omega-accent transition-colors resize-none h-24"
              disabled={isLoading}
            />
          </div>

          <div className="flex gap-4 mb-4">
            <div className="flex-1">
              <label className="block text-xs font-semibold text-omega-muted uppercase tracking-wider mb-2">
                Research Depth: {depth}/5
              </label>
              <input
                type="range" min={1} max={5} value={depth}
                onChange={e => setDepth(Number(e.target.value))}
                className="w-full accent-omega-accent"
                disabled={isLoading}
              />
              <div className="flex justify-between text-xs text-omega-muted mt-1">
                <span>Shallow</span><span>Exhaustive</span>
              </div>
            </div>
            <div className="flex-1">
              <label className="block text-xs font-semibold text-omega-muted uppercase tracking-wider mb-2">
                Cost Budget: ${budget.toFixed(1)}
              </label>
              <input
                type="range" min={0.5} max={10} step={0.5} value={budget}
                onChange={e => setBudget(Number(e.target.value))}
                className="w-full accent-omega-accent2"
                disabled={isLoading}
              />
              <div className="flex justify-between text-xs text-omega-muted mt-1">
                <span>$0.50</span><span>$10.00</span>
              </div>
            </div>
          </div>

          <button
            onClick={startResearch}
            disabled={isLoading || query.length < 10}
            className="w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all duration-300 disabled:opacity-40 disabled:cursor-not-allowed"
            style={{
              background: isLoading
                ? 'rgba(88,166,255,0.1)'
                : 'linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%)',
              color: '#030712',
            }}
          >
            {isLoading ? (
              <><Loader2 size={16} className="animate-spin" /><span style={{ color: '#58a6ff' }}>Research in progress...</span></>
            ) : (
              <><Search size={16} /><span>Launch Autonomous Research</span></>
            )}
          </button>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Agent Graph + Metrics */}
          <div className="lg:col-span-1 space-y-4">
            {/* Agent Council */}
            <motion.div
              className="glass-card p-5"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-omega-text">Agent Council</h2>
                {status && <StatusBadge state={currentState as ExecutionState} />}
              </div>
              <div className="grid grid-cols-4 gap-3">
                {AGENT_NODES.map(agent => (
                  <AgentNode key={agent.id} agent={agent} activeState={currentState as ExecutionState} />
                ))}
              </div>
            </motion.div>

            {/* Metrics */}
            {status && (
              <motion.div
                className="grid grid-cols-2 gap-3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <MetricCard label="Cost" value={`$${status.cost_usd.toFixed(4)}`} icon={DollarSign} color="#d29922" />
                <MetricCard label="Sources" value={`${status.chunks_collected}`} icon={Globe} color="#58a6ff" />
                <MetricCard label="Confidence" value={`${(status.confidence * 100).toFixed(0)}%`} icon={Star} color="#3fb950" />
                <MetricCard label="Iteration" value={`${status.iteration}/3`} icon={RefreshCw} color="#bc8cff" />
              </motion.div>
            )}

            {/* Execution Log */}
            <motion.div
              className="glass-card p-4"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <h2 className="text-xs font-semibold text-omega-muted uppercase tracking-wider mb-3">
                Execution Log
              </h2>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {events.length === 0 ? (
                  <p className="text-xs text-omega-muted italic">Awaiting research launch...</p>
                ) : events.map((ev, i) => (
                  <motion.div
                    key={i}
                    className="text-xs font-mono text-omega-muted leading-relaxed"
                    initial={{ opacity: 0, x: -5 }}
                    animate={{ opacity: 1, x: 0 }}
                  >
                    <span className="text-omega-accent3">›</span> {ev}
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Right: Report */}
          <div className="lg:col-span-2">
            {!report && !isLoading && (
              <motion.div
                className="glass-card p-12 flex flex-col items-center justify-center text-center h-full min-h-[400px]"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <div className="w-16 h-16 rounded-2xl bg-omega-accent/10 border border-omega-accent/20 flex items-center justify-center mb-4">
                  <Brain size={32} className="text-omega-accent" />
                </div>
                <h3 className="text-lg font-semibold text-omega-text mb-2">Ready to Research</h3>
                <p className="text-omega-muted text-sm max-w-sm">
                  Enter a research query and launch the autonomous agent council. 
                  Real-time execution will appear here.
                </p>
              </motion.div>
            )}

            {isLoading && !report && (
              <motion.div
                className="glass-card p-8 min-h-[400px] flex flex-col items-center justify-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <div className="relative w-20 h-20 mb-6">
                  <div className="absolute inset-0 rounded-full border-2 border-omega-accent/20" />
                  <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-omega-accent animate-spin" />
                  <div className="absolute inset-2 rounded-full border-2 border-transparent border-t-omega-accent2 animate-spin" style={{ animationDuration: '1.5s', animationDirection: 'reverse' }} />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Zap size={24} className="text-omega-accent" />
                  </div>
                </div>
                <p className="text-omega-text font-semibold text-lg mb-1">Agents Executing</p>
                <p className="text-omega-muted text-sm mb-6">
                  {STATE_CONFIG[currentState as ExecutionState]?.label || 'Processing'}...
                </p>
                <div className="w-64 progress-bar">
                  <motion.div
                    className="progress-fill"
                    animate={{ width: ['0%', '100%', '0%'] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                </div>
              </motion.div>
            )}

            {report && <ReportViewer report={report} />}
          </div>
        </div>
      </div>
    </div>
  )
}
