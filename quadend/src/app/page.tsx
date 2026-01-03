"use client";

import { useState, useEffect } from "react";
import useSWR from "swr";
import {
  getStatus, getBalance, getPositions, getDecisions,
  getLogs, controlBot, getMarketContext, getMarketRaw, getAIInsights
} from '@/lib/api';

// Types
interface BotStatus {
  mode: "active" | "paused" | "stopped";
  trading_allowed: boolean;
  pause_reason?: string;
  consecutive_losses: number;
  cooldown_active: boolean;
}

interface Balance {
  total: number;
  available: number;
  unrealized_pnl: number;
  daily_pnl: number;
  daily_pnl_pct: number;
}

interface Position {
  symbol: string;
  side: "Buy" | "Sell";
  size: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_pct: number;
  leverage: number;
  stop_loss?: number;
  take_profit?: number;
  mark_price?: number;
}

interface Decision {
  id: string;
  timestamp: string;
  symbol: string;
  type: "BUY" | "SELL" | "WAIT" | "CLOSE" | "MODIFY";
  confidence: number;
  strategy: string;
  reasoning: string;
  htf_aligned?: boolean;
}

interface LogEntry {
  timestamp: string;
  level: "INFO" | "WARNING" | "ERROR" | "DEBUG";
  message: string;
}

export default function Dashboard() {
  // Real-time Data Fetching
  const { data: status } = useSWR('status', getStatus, { refreshInterval: 1000 });
  const { data: balance } = useSWR('balance', getBalance, { refreshInterval: 5000 });
  const { data: positions } = useSWR('positions', getPositions, { refreshInterval: 3000 });
  const { data: decisions } = useSWR('decisions', () => getDecisions(), { refreshInterval: 5000 });
  const { data: marketContext, error: marketContextError } = useSWR('marketContext', getMarketContext, { refreshInterval: 5000 });
  const { data: marketRaw } = useSWR('marketRaw', getMarketRaw, { refreshInterval: 5000 });
  // Logs handled via polling for now
  const { data: logs, error: logsError } = useSWR('logs', () => getLogs(), { refreshInterval: 2000 });
  const { data: aiInsights } = useSWR('aiInsights', getAIInsights, { refreshInterval: 5000 });

  const [activeTab, setActiveTab] = useState<'decisions' | 'logs' | 'market' | 'ai'>('decisions');
  const [marketView, setMarketView] = useState<'interpretation' | 'raw'>('interpretation');
  const [isClient, setIsClient] = useState(false);

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (activeTab === "logs") {
      const logEnd = document.getElementById("logs-end");
      logEnd?.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, activeTab]);

  // Hydration fix for client-side rendering
  useEffect(() => {
    setIsClient(true);
  }, []);

  const getStatusLed = () => {
    if (!status) return "led-stopped";
    if (status.mode === "active" && status.trading_allowed) return "led-active";
    if (status.mode === "paused" || status.cooldown_active) return "led-paused";
    return "led-stopped";
  };

  const getStatusText = () => {
    if (!status) return "CONNECTING...";
    if (status.cooldown_active) return "COOLDOWN";
    return status.mode.toUpperCase();
  };

  const handleStart = async () => {
    await controlBot("start");
  };

  const handleStop = async () => {
    await controlBot("stop");
  };

  const handlePause = async () => {
    await controlBot("pause");
  };

  if (!isClient) {
    return null; // Or a loading spinner
  }

  return (
    <div className="min-h-screen p-6" style={{ background: "var(--bg-primary)" }}>
      {/* Header */}
      <header className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: "var(--accent-blue)" }}>
              <span className="text-xl font-bold">Q</span>
            </div>
            <div>
              <h1 className="text-xl font-bold">QUADRICK AI</h1>
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>Autonomous Trading System</p>
            </div>
          </div>

          <div className="flex items-center gap-2 ml-8 px-3 py-2 rounded-lg" style={{ background: "var(--bg-secondary)" }}>
            <div className={`led ${getStatusLed()} `}></div>
            <span className="text-sm font-medium">{getStatusText()}</span>
          </div>
        </div>

        {/* Balance Display */}
        <div className="flex items-center gap-6">
          <div className="text-right">
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>Balance</p>
            <p className="text-2xl font-bold">${balance?.total?.toFixed(2) || "---"}</p>
          </div>
          <div className="text-right">
            <p className="text-sm" style={{ color: "var(--text-muted)" }}>24h PnL</p>
            <p className={`text - lg font - semibold ${(balance?.daily_pnl || 0) >= 0 ? "text-green-500" : "text-red-500"} `}>
              {(balance?.daily_pnl || 0) >= 0 ? "+" : ""}${balance?.daily_pnl?.toFixed(2) || "0.00"} ({balance?.daily_pnl_pct?.toFixed(2) || "0.00"}%)
            </p>
          </div>
        </div>

        {/* Control Buttons */}
        <div className="flex gap-3">
          <button onClick={handleStart} disabled={status?.mode === "active"} className="btn-primary disabled:opacity-50">
            ▶ Start
          </button>
          <button onClick={handlePause} disabled={status?.mode !== "active"} className="btn-outline disabled:opacity-50">
            ⏸ Pause
          </button>
          <button onClick={handleStop} className="btn-danger">
            ⏹ Stop
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left Column - Positions & Balance */}
        <div className="col-span-4 space-y-6">
          {/* Balance Card */}
          <div className="card">
            <h3 className="text-sm font-medium mb-4" style={{ color: "var(--text-muted)" }}>Account Overview</h3>

            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span style={{ color: "var(--text-secondary)" }}>Total Equity</span>
                <span className="text-xl font-bold">${balance?.total?.toFixed(2) || "0.00"}</span>
              </div>
              <div className="flex justify-between items-center">
                <span style={{ color: "var(--text-secondary)" }}>Available</span>
                <span className="font-medium">${balance?.available?.toFixed(2) || "0.00"}</span>
              </div>
              <div className="flex justify-between items-center">
                <span style={{ color: "var(--text-secondary)" }}>Unrealized PnL</span>
                <span className={`font - medium ${(balance?.unrealized_pnl || 0) >= 0 ? "text-green-500" : "text-red-500"} `}>
                  {(balance?.unrealized_pnl || 0) >= 0 ? "+" : ""}${balance?.unrealized_pnl?.toFixed(2) || "0.00"}
                </span>
              </div>

              {/* Progress to Milestone */}
              <div className="pt-4 border-t" style={{ borderColor: "var(--border-color)" }}>
                <div className="flex justify-between text-sm mb-2">
                  <span style={{ color: "var(--text-muted)" }}>Progress to $100</span>
                  <span style={{ color: "var(--accent-blue)" }}>{balance ? ((balance.total / 100) * 100).toFixed(1) : 0}%</span>
                </div>
                <div className="h-2 rounded-full" style={{ background: "var(--bg-tertiary)" }}>
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.min((balance ? (balance.total / 100) * 100 : 0), 100)}% `,
                      background: "var(--accent-blue)"
                    }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          {/* Active Positions */}
          <div className="card">
            <h3 className="text-sm font-medium mb-4" style={{ color: "var(--text-muted)" }}>
              Active Positions ({positions?.length || 0})
            </h3>

            {(!positions || positions.length === 0) ? (
              <p className="text-center py-8" style={{ color: "var(--text-muted)" }}>No open positions</p>
            ) : (
              <div className="space-y-3">
                {positions.map((pos, idx) => (
                  <div key={idx} className="p-3 rounded-lg" style={{ background: "var(--bg-secondary)" }}>
                    <div className="flex justify-between items-center mb-2">
                      <div className="flex items-center gap-2">
                        <span className={`badge ${pos.side === "Buy" ? "badge-buy" : "badge-sell"} `}>
                          {pos.side}
                        </span>
                        <span className="font-medium">{pos.symbol}</span>
                        <span className="text-xs" style={{ color: "var(--text-muted)" }}>{pos.leverage}x</span>
                      </div>
                      <span className={`font - semibold ${pos.pnl >= 0 ? "text-green-500" : "text-red-500"} `}>
                        {pos.pnl >= 0 ? "+" : ""}${pos.pnl.toFixed(2)}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-xs" style={{ color: "var(--text-secondary)" }}>
                      <div>
                        <span style={{ color: "var(--text-muted)" }}>Entry: </span>
                        ${pos.entry_price.toLocaleString()}
                      </div>
                      <div>
                        <span style={{ color: "var(--text-muted)" }}>Current: </span>
                        ${pos.current_price.toLocaleString()}
                      </div>
                      <div>
                        <span style={{ color: "var(--text-muted)" }}>Size: </span>
                        {pos.size}
                      </div>
                    </div>
                    {(pos.stop_loss || pos.take_profit) && (
                      <div className="flex gap-4 mt-2 text-xs">
                        {pos.stop_loss && (
                          <span style={{ color: "var(--error)" }}>SL: ${pos.stop_loss.toLocaleString()}</span>
                        )}
                        {pos.take_profit && (
                          <span style={{ color: "var(--success)" }}>TP: ${pos.take_profit.toLocaleString()}</span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right Column - Decisions & Logs */}
        <div className="col-span-8">
          <div className="card h-full">
            {/* Tabs */}
            <div className="flex gap-4 mb-4 border-b" style={{ borderColor: "var(--border-color)" }}>
              <button
                onClick={() => setActiveTab("decisions")}
                className={`pb-3 px-1 text-sm font-medium transition-colors ${activeTab === "decisions"
                  ? "border-b-2"
                  : ""
                  } `}
                style={{
                  borderColor: activeTab === "decisions" ? "var(--accent-blue)" : "transparent",
                  color: activeTab === "decisions" ? "var(--text-primary)" : "var(--text-muted)"
                }}
              >
                DeepSeek Decisions
              </button>
              <button
                onClick={() => setActiveTab("market")}
                className={`pb-3 px-1 text-sm font-medium transition-colors ${activeTab === "market"
                  ? "border-b-2"
                  : ""
                  } `}
                style={{
                  borderColor: activeTab === "market" ? "var(--accent-blue)" : "transparent",
                  color: activeTab === "market" ? "var(--text-primary)" : "var(--text-muted)"
                }}
              >
                Market Analysis
              </button>
              <button
                onClick={() => setActiveTab("ai")}
                className={`pb-3 px-1 text-sm font-medium transition-colors ${activeTab === "ai"
                  ? "border-b-2"
                  : ""
                  } `}
                style={{
                  borderColor: activeTab === "ai" ? "var(--accent-blue)" : "transparent",
                  color: activeTab === "ai" ? "var(--text-primary)" : "var(--text-muted)"
                }}
              >
                AI Insights
              </button>
              <button
                onClick={() => setActiveTab("logs")}
                className={`pb-3 px-1 text-sm font-medium transition-colors ${activeTab === "logs"
                  ? "border-b-2"
                  : ""
                  } `}
                style={{
                  borderColor: activeTab === "logs" ? "var(--accent-blue)" : "transparent",
                  color: activeTab === "logs" ? "var(--text-primary)" : "var(--text-muted)"
                }}
              >
                System Logs
              </button>
            </div>

            {/* Content */}
            <div
              id="dashboard-content-scroll"
              className="overflow-y-auto"
              style={{ maxHeight: "calc(100vh - 300px)" }}
            >
              {activeTab === "decisions" ? (
                <div className="space-y-3">
                  {decisions && decisions.length > 0 ? (
                    decisions.slice().reverse().map((decision) => (
                      <div key={decision.id} className="p-4 rounded-lg" style={{ background: "var(--bg-secondary)" }}>
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <span className={`badge ${decision.type === "BUY" ? "badge-buy" :
                              decision.type === "SELL" ? "badge-sell" :
                                "badge-wait"
                              } `}>
                              {decision.type}
                            </span>
                            <span className="font-medium">{decision.symbol}</span>
                            <span className="badge badge-blue">{decision.strategy || "Neutral"}</span>
                          </div>
                          <div className="flex items-center gap-3">
                            <span className="text-sm" style={{ color: "var(--text-muted)" }}>
                              Conf: {((decision.confidence || 0) * 100).toFixed(0)}%
                            </span>
                            {decision.htf_aligned !== undefined && (
                              <span className={`text - xs ${decision.htf_aligned ? "text-green-500" : "text-yellow-500"} `}>
                                {decision.htf_aligned ? "✓ HTF" : "✗ HTF"}
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="space-y-2">
                          {(typeof decision.reasoning === 'object' && decision.reasoning !== null) ? (
                            <>
                              {decision.reasoning.analyst && (
                                <p className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                                  <span className="text-blue-400 font-semibold mr-1">Analyst:</span> {decision.reasoning.analyst}
                                </p>
                              )}
                              {decision.reasoning.strategist && (
                                <p className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                                  <span className="text-purple-400 font-semibold mr-1">Strategist:</span> {decision.reasoning.strategist}
                                </p>
                              )}
                              {decision.reasoning.reason && (
                                <p className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                                  {decision.reasoning.reason}
                                </p>
                              )}
                            </>
                          ) : (
                            <p className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                              {decision.reasoning || "No reasoning provided"}
                            </p>
                          )}
                        </div>
                        <p className="text-xs mt-3 flex items-center justify-between" style={{ color: "var(--text-muted)" }}>
                          <span>{new Date(decision.timestamp).toLocaleTimeString()}</span>
                          {["BUY", "SELL"].includes(decision.type) && (
                            <span className="text-[10px] uppercase tracking-widest opacity-50">DeepSeek Quantitative Execution</span>
                          )}
                        </p>
                      </div>
                    ))
                  ) : (
                    <p className="text-center py-8" style={{ color: "var(--text-muted)" }}>No decisions yet</p>
                  )}
                </div>
              ) : activeTab === 'ai' ? (
                <div className="card h-full flex flex-col overflow-hidden">
                  <div className="flex justify-between items-center mb-4 p-4 border-b border-gray-800">
                    <h2 className="text-xl font-bold text-gradient">AI Reasoning (DeepSeek Context)</h2>
                    <span className="text-xs text-secondary">Raw prompt & response log</span>
                  </div>
                  <div className="flex-1 overflow-y-auto p-4 space-y-6">
                    {!aiInsights || aiInsights.length === 0 ? (
                      <div className="flex flex-col items-center justify-center h-64 text-secondary">
                        <p>No AI interactions captured yet.</p>
                        <p className="text-xs mt-2 italic">Captured when bot evaluates symbols in Council</p>
                      </div>
                    ) : (
                      aiInsights.map((insight: any, idx: number) => (
                        <div key={idx} className="bg-black/60 rounded-xl border border-gray-800 p-4 space-y-4">
                          <div className="flex justify-between items-center">
                            <div className="flex items-center gap-3">
                              <span className={`px-2 py-1 rounded text-xs font-bold ${insight.agent === 'Analyst' ? 'bg-blue-900/40 text-blue-400' : 'bg-purple-900/40 text-purple-400'}`}>
                                {insight.agent.toUpperCase()}
                              </span>
                              <span className="text-lg font-bold text-primary">{insight.symbol}</span>
                            </div>
                            <span className="text-xs text-secondary">
                              {new Date(insight.timestamp).toLocaleTimeString()}
                            </span>
                          </div>

                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                            <div className="space-y-2">
                              <span className="text-xs font-semibold text-blue-400 uppercase tracking-wider">What Bybit sent to AI (Prompt):</span>
                              <div className="bg-gray-900/80 rounded-lg p-3 text-[10px] sm:text-xs font-mono text-gray-300 overflow-x-auto border border-blue-900/20 max-h-[300px] overflow-y-auto whitespace-pre-wrap">
                                {insight.prompt}
                              </div>
                            </div>
                            <div className="space-y-2">
                              <span className="text-xs font-semibold text-purple-400 uppercase tracking-wider">How AI Reasoned (Response):</span>
                              <div className="bg-gray-900/80 rounded-lg p-3 text-[10px] sm:text-xs font-mono text-emerald-400 overflow-x-auto border border-purple-900/20 max-h-[300px] overflow-y-auto whitespace-pre-wrap">
                                {insight.response}
                              </div>
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              ) : activeTab === "logs" ? (
                <div className="font-mono text-xs">
                  {logs && logs.length > 0 ? (
                    logs.map((log, idx) => (
                      <div key={idx} className="log-entry mb-1">
                        <span style={{ color: "var(--text-muted)" }}>
                          [{new Date(log.timestamp).toLocaleTimeString()}]
                        </span>{" "}
                        <span className={`log - ${log.level.toLowerCase()} `}>
                          [{log.level}]
                        </span>{" "}
                        <span>{log.message}</span>
                      </div>
                    ))
                  ) : (
                    <p className="text-center py-8" style={{ color: "var(--text-muted)" }}>No logs available</p>
                  )}
                  {/* Anchor for auto-scroll */}
                  <div id="logs-end"></div>
                </div>
              ) : activeTab === "market" && (
                <div className="space-y-6">
                  <div className="flex space-x-2 p-1 rounded-lg" style={{ background: "var(--bg-secondary)" }}>
                    <button
                      onClick={() => setMarketView('interpretation')}
                      className={`flex-1 py-1.5 px-3 rounded text-xs font-semibold transition-all ${marketView === 'interpretation' ? 'bg-blue-600 text-white' : 'hover:bg-white/5'}`}
                    >
                      Interpretation
                    </button>
                    <button
                      onClick={() => setMarketView('raw')}
                      className={`flex-1 py-1.5 px-3 rounded text-xs font-semibold transition-all ${marketView === 'raw' ? 'bg-blue-600 text-white' : 'hover:bg-white/5'}`}
                    >
                      Raw Stats
                    </button>
                  </div>

                  {marketView === 'interpretation' ? (
                    marketContext && Object.keys(marketContext).length > 0 ? (
                      Object.entries(marketContext).map(([symbol, data]: [string, any]) => (
                        <div key={symbol} className="p-4 rounded-lg" style={{ background: "var(--bg-secondary)" }}>
                          <div className="flex justify-between items-center mb-4 pb-2 border-b" style={{ borderColor: "var(--border-color)" }}>
                            <h4 className="font-bold text-lg">{symbol}</h4>
                            <span className="text-xs px-2 py-1 rounded" style={{ background: "var(--bg-tertiary)", color: "var(--text-secondary)" }}>
                              Last Analyzed: {new Date().toLocaleTimeString()}
                            </span>
                          </div>

                          <div className="grid grid-cols-2 gap-6">
                            {["1h", "15m"].map((tf) => {
                              const tfData = data.timeframe_analysis?.[tf];
                              if (!tfData) return null;
                              return (
                                <div key={tf} className="space-y-3">
                                  <div className="flex justify-between items-center">
                                    <span className="text-sm font-semibold" style={{ color: "var(--accent-blue)" }}>Timeframe: {tf}</span>
                                    <span className={`text-xs font-bold ${tfData.trend === 'bullish' ? 'text-green-500' : tfData.trend === 'bearish' ? 'text-red-500' : 'text-gray-400'}`}>
                                      {tfData.trend?.toUpperCase() || "NEUTRAL"}
                                    </span>
                                  </div>
                                  <div className="grid grid-cols-2 gap-y-2 text-xs">
                                    <div className="flex flex-col">
                                      <span style={{ color: "var(--text-muted)" }}>RSI</span>
                                      <span className="font-medium">{tfData.rsi?.toFixed(1) || "N/A"}</span>
                                    </div>
                                    <div className="flex flex-col">
                                      <span style={{ color: "var(--text-muted)" }}>ADX</span>
                                      <span className="font-medium">{tfData.adx?.toFixed(1) || "N/A"}</span>
                                    </div>
                                    <div className="flex flex-col">
                                      <span style={{ color: "var(--text-muted)" }}>BB Width</span>
                                      <span className="font-medium">{tfData.bb_width?.toFixed(4) || "N/A"}</span>
                                    </div>
                                    <div className="flex flex-col">
                                      <span style={{ color: "var(--text-muted)" }}>Vol Ratio</span>
                                      <span className="font-medium">{tfData.volume_ratio?.toFixed(1) || "1.0"}x</span>
                                    </div>
                                  </div>

                                  <div className="mt-2 pt-2 border-t" style={{ borderColor: "var(--border-color)" }}>
                                    <span className="text-[10px] uppercase font-bold" style={{ color: "var(--text-muted)" }}>Key Levels</span>
                                    <div className="flex justify-between text-xs mt-1">
                                      <span style={{ color: "var(--error)" }}>S: ${tfData.key_levels?.immediate_support?.toLocaleString()}</span>
                                      <span style={{ color: "var(--success)" }}>R: ${tfData.key_levels?.immediate_resistance?.toLocaleString()}</span>
                                    </div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="text-center py-12 space-y-3">
                        <div className="w-12 h-12 border-4 border-blue-500/20 border-t-blue-500 rounded-full animate-spin mx-auto"></div>
                        <p style={{ color: "var(--text-muted)" }}>Scanning market context for watchlist...</p>
                      </div>
                    )
                  ) : (
                    /* Raw Stats View */
                    <div className="space-y-4">
                      {marketRaw && marketRaw.tickers && Object.keys(marketRaw.tickers).length > 0 ? (
                        Object.entries(marketRaw.tickers).map(([symbol, ticker]: [string, any]) => (
                          <div key={symbol} className="p-4 rounded-lg border" style={{ background: "var(--bg-secondary)", borderColor: "var(--border-color)" }}>
                            <div className="flex justify-between mb-2">
                              <span className="font-bold">{symbol}</span>
                              <span className="text-xs font-mono text-green-400">Price: ${ticker.last_price || ticker.price}</span>
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-[10px] font-mono opacity-80">
                              <div>VOL: {ticker.volume_24h || ticker.volume || 'N/A'}</div>
                              <div>FUND: {marketRaw.funding_rates?.[symbol] || '0.00%'}</div>
                              <div>HIGH: {ticker.high_24h || 'N/A'}</div>
                              <div>LOW: {ticker.low_24h || 'N/A'}</div>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-center py-12 text-sm opacity-50">
                          Waiting for raw market feed from bot...
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {(logsError || marketContextError) && (
                <div className="mt-2 p-2 rounded text-xs" style={{ background: "var(--error-bg)", color: "var(--error)" }}>
                  {/* Log error if any */}
                  Error: {(logsError || marketContextError).message}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Connection Status Footer */}
      <footer className="fixed bottom-0 left-0 right-0 px-6 py-3 flex items-center justify-between" style={{ background: "var(--bg-secondary)", borderTop: "1px solid var(--border-color)" }}>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <div className={`led ${status ? "led-active" : "led-stopped"} `}></div>
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>Bybit</span>
          </div>
          <div className="flex items-center gap-2">
            <div className={`led ${status ? "led-active" : "led-stopped"} `}></div>
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>DeepSeek</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="led led-active"></div>
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>Supabase</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <a href="/settings" className="text-xs hover:text-blue-400 transition-colors" style={{ color: "var(--text-muted)" }}>
            ⚙️ Settings
          </a>
          <span className="text-xs" style={{ color: "var(--text-muted)" }}>
            v1.0.0
          </span>
        </div>
      </footer>
    </div >
  );
}
