"use client";

import { useState, useEffect } from "react";
import useSWR from "swr";
import {
  getStatus, getBalance, getPositions, getDecisions,
  getLogs, controlBot, getMarketContext, getMarketRaw, getAIInsights
} from '@/lib/api';

// Symbol volatility config (matches backend)
const SYMBOL_VOL_CLASS: Record<string, 'high' | 'med' | 'low'> = {
  "1000PEPEUSDT": "high",
  "DOGEUSDT": "high",
  "SOLUSDT": "high",
  "AVAXUSDT": "med",
  "ARBUSDT": "med",
  "OPUSDT": "med",
  "LINKUSDT": "med",
  "DOTUSDT": "med",
  "ADAUSDT": "med",
  "XRPUSDT": "med",
  "BTCUSDT": "low",
  "ETHUSDT": "low",
};

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
}

interface Decision {
  id: string;
  timestamp: string;
  symbol: string;
  type: "BUY" | "SELL" | "WAIT" | "CLOSE" | "MODIFY";
  confidence: number;
  strategy: string;
  reasoning: string | { analyst?: string; strategist?: string; reason?: string };
  htf_aligned?: boolean;
}

interface LogEntry {
  timestamp: string;
  level: "INFO" | "WARNING" | "ERROR" | "DEBUG";
  message: string;
}

// Helper functions
const getVolClass = (symbol: string) => SYMBOL_VOL_CLASS[symbol] || 'med';

const formatFundingRate = (rate: number | string | undefined) => {
  if (rate === undefined || rate === null) return "0.00%";
  const numRate = typeof rate === 'string' ? parseFloat(rate) : rate;
  if (isNaN(numRate)) return "0.00%";
  return `${numRate >= 0 ? '+' : ''}${(numRate * 100).toFixed(3)}%`;
};

const isFundingExtreme = (rate: number | string | undefined) => {
  if (rate === undefined || rate === null) return false;
  const numRate = typeof rate === 'string' ? parseFloat(rate) : rate;
  return Math.abs(numRate) > 0.0005;
};

export default function Dashboard() {
  // Data fetching
  const { data: status } = useSWR('status', getStatus, { refreshInterval: 1000 });
  const { data: balance } = useSWR('balance', getBalance, { refreshInterval: 5000 });
  const { data: positions } = useSWR('positions', getPositions, { refreshInterval: 3000 });
  const { data: decisions } = useSWR('decisions', () => getDecisions(), { refreshInterval: 5000 });
  const { data: marketContext } = useSWR('marketContext', getMarketContext, { refreshInterval: 5000 });
  const { data: marketRaw } = useSWR('marketRaw', getMarketRaw, { refreshInterval: 5000 });
  const { data: logs } = useSWR('logs', () => getLogs(), { refreshInterval: 2000 });
  const { data: aiInsights } = useSWR('aiInsights', getAIInsights, { refreshInterval: 5000 });

  const [activeTab, setActiveTab] = useState<'decisions' | 'logs' | 'market' | 'ai'>('market');
  const [marketView, setMarketView] = useState<'grid' | 'raw'>('grid');
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    if (activeTab === "logs") {
      const logEnd = document.getElementById("logs-end");
      logEnd?.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, activeTab]);

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
    if (!status) return "OFFLINE";
    if (status.cooldown_active) return "COOLDOWN";
    return status.mode.toUpperCase();
  };

  if (!isClient) return null;

  return (
    <div className="min-h-screen" style={{ background: "var(--bg-primary)" }}>
      {/* ===== HEADER ===== */}
      <header className="sticky top-0 z-50 px-6 py-4 border-b" style={{ background: "var(--bg-secondary)", borderColor: "var(--border-default)" }}>
        <div className="flex items-center justify-between max-w-[1800px] mx-auto">
          {/* Logo + Status */}
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg flex items-center justify-center font-bold text-lg" style={{ background: "var(--accent-primary)" }}>Q</div>
              <div>
                <h1 className="text-base font-semibold tracking-tight">QUADRICK</h1>
                <p className="text-[10px] uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>Hybrid AI + Algo</p>
              </div>
            </div>

            <div className="flex items-center gap-2 px-3 py-1.5 rounded-md" style={{ background: "var(--bg-tertiary)" }}>
              <div className={`led ${getStatusLed()}`}></div>
              <span className="text-xs font-medium mono">{getStatusText()}</span>
            </div>
          </div>

          {/* Balance */}
          <div className="flex items-center gap-8">
            <div className="text-right">
              <p className="text-[10px] uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>Equity</p>
              <p className="text-xl font-semibold mono">${balance?.total?.toFixed(2) || "0.00"}</p>
            </div>
            <div className="text-right">
              <p className="text-[10px] uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>24h PnL</p>
              <p className={`text-lg font-semibold mono ${(balance?.daily_pnl || 0) >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                {(balance?.daily_pnl || 0) >= 0 ? "+" : ""}${balance?.daily_pnl?.toFixed(2) || "0.00"}
                <span className="text-xs ml-1">({balance?.daily_pnl_pct?.toFixed(1) || "0.0"}%)</span>
              </p>
            </div>
          </div>

          {/* Controls */}
          <div className="flex gap-2">
            <button onClick={() => controlBot("start")} disabled={status?.mode === "active"} className="btn btn-primary">
              ▶ Start
            </button>
            <button onClick={() => controlBot("pause")} disabled={status?.mode !== "active"} className="btn btn-outline">
              ⏸ Pause
            </button>
            <button onClick={() => controlBot("stop")} className="btn btn-danger">
              ⏹ Stop
            </button>
          </div>
        </div>
      </header>

      {/* ===== MAIN CONTENT ===== */}
      <main className="p-6 max-w-[1800px] mx-auto">
        <div className="grid grid-cols-12 gap-5">

          {/* ===== LEFT COLUMN - Account & Positions ===== */}
          <div className="col-span-3 space-y-5">
            {/* Account Card */}
            <div className="card">
              <div className="card-header">Account Overview</div>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span style={{ color: "var(--text-muted)" }}>Total Equity</span>
                  <span className="font-semibold mono">${balance?.total?.toFixed(2) || "0.00"}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span style={{ color: "var(--text-muted)" }}>Available</span>
                  <span className="mono">${balance?.available?.toFixed(2) || "0.00"}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span style={{ color: "var(--text-muted)" }}>Unrealized</span>
                  <span className={`mono ${(balance?.unrealized_pnl || 0) >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                    {(balance?.unrealized_pnl || 0) >= 0 ? "+" : ""}${balance?.unrealized_pnl?.toFixed(2) || "0.00"}
                  </span>
                </div>
                <div className="divider"></div>
                <div>
                  <div className="flex justify-between text-xs mb-2">
                    <span style={{ color: "var(--text-muted)" }}>Progress to $100</span>
                    <span style={{ color: "var(--accent-primary)" }}>{balance ? Math.min((balance.total / 100) * 100, 100).toFixed(0) : 0}%</span>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-bar-fill" style={{ width: `${Math.min((balance ? (balance.total / 100) * 100 : 0), 100)}%` }}></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Positions Card */}
            <div className="card">
              <div className="card-header">Active Positions ({positions?.length || 0})</div>
              {(!positions || positions.length === 0) ? (
                <p className="text-center py-6 text-sm" style={{ color: "var(--text-muted)" }}>No open positions</p>
              ) : (
                <div className="space-y-3">
                  {positions.map((pos, idx) => (
                    <div key={idx} className="p-3 rounded-md" style={{ background: "var(--bg-secondary)" }}>
                      <div className="flex justify-between items-center mb-2">
                        <div className="flex items-center gap-2">
                          <span className={`badge ${pos.side === "Buy" ? "badge-long" : "badge-short"}`}>
                            {pos.side === "Buy" ? "LONG" : "SHORT"}
                          </span>
                          <span className="font-medium text-sm">{pos.symbol.replace('USDT', '')}</span>
                          <span className={`badge badge-vol-${getVolClass(pos.symbol)}`}>
                            {getVolClass(pos.symbol).toUpperCase()}
                          </span>
                        </div>
                        <span className={`font-semibold mono text-sm ${pos.pnl >= 0 ? "pnl-positive" : "pnl-negative"}`}>
                          {pos.pnl >= 0 ? "+" : ""}${pos.pnl.toFixed(2)}
                        </span>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-xs" style={{ color: "var(--text-secondary)" }}>
                        <div>
                          <span style={{ color: "var(--text-muted)" }}>Entry </span>
                          <span className="mono">${pos.entry_price.toLocaleString()}</span>
                        </div>
                        <div>
                          <span style={{ color: "var(--text-muted)" }}>Now </span>
                          <span className="mono">${pos.current_price.toLocaleString()}</span>
                        </div>
                        <div>
                          <span style={{ color: "var(--text-muted)" }}>{pos.leverage}x </span>
                          <span className="mono">{pos.size}</span>
                        </div>
                      </div>
                      {(pos.stop_loss || pos.take_profit) && (
                        <div className="flex gap-4 mt-2 text-xs">
                          {pos.stop_loss && <span style={{ color: "var(--error)" }}>SL ${pos.stop_loss.toLocaleString()}</span>}
                          {pos.take_profit && <span style={{ color: "var(--success)" }}>TP ${pos.take_profit.toLocaleString()}</span>}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* ===== CENTER/RIGHT - Main Panel ===== */}
          <div className="col-span-9">
            <div className="card" style={{ minHeight: "calc(100vh - 180px)" }}>
              {/* Tabs */}
              <div className="flex gap-6 mb-4 pb-3 border-b" style={{ borderColor: "var(--border-default)" }}>
                {[
                  { id: 'market', label: 'Market Overview' },
                  { id: 'decisions', label: 'Decisions' },
                  { id: 'ai', label: 'AI Insights' },
                  { id: 'logs', label: 'Logs' },
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              <div className="overflow-y-auto" style={{ maxHeight: "calc(100vh - 280px)" }}>

                {/* MARKET TAB */}
                {activeTab === "market" && (
                  <div className="space-y-4">
                    {/* View Toggle */}
                    <div className="flex justify-between items-center">
                      <h3 className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
                        Watchlist Analysis
                      </h3>
                      <div className="toggle-group">
                        <button
                          onClick={() => setMarketView('grid')}
                          className={`toggle-option ${marketView === 'grid' ? 'active' : ''}`}
                        >
                          Grid View
                        </button>
                        <button
                          onClick={() => setMarketView('raw')}
                          className={`toggle-option ${marketView === 'raw' ? 'active' : ''}`}
                        >
                          Raw Data
                        </button>
                      </div>
                    </div>

                    {marketView === 'grid' ? (
                      /* GRID VIEW */
                      <div className="data-grid" style={{ gridTemplateColumns: "1fr 80px 80px 60px 100px 70px" }}>
                        {/* Header */}
                        <div className="data-grid-cell data-grid-header">Symbol</div>
                        <div className="data-grid-cell data-grid-header">Funding</div>
                        <div className="data-grid-cell data-grid-header">RSI</div>
                        <div className="data-grid-cell data-grid-header">ADX</div>
                        <div className="data-grid-cell data-grid-header">Trend</div>
                        <div className="data-grid-cell data-grid-header">Vol</div>

                        {/* Rows */}
                        {marketContext && Object.entries(marketContext).map(([symbol, data]: [string, any]) => {
                          const tf = data.timeframe_analysis?.["1h"] || data.timeframe_analysis?.["15m"] || {};
                          const funding = marketRaw?.funding_rates?.[symbol];
                          const trend = tf.trend || "neutral";

                          return (
                            <div key={symbol} className="data-grid-row">
                              <div className="data-grid-cell font-medium">{symbol.replace('USDT', '')}</div>
                              <div className={`data-grid-cell mono text-xs ${isFundingExtreme(funding) ? 'funding-extreme' : funding && parseFloat(String(funding)) > 0 ? 'funding-positive' : 'funding-negative'}`}>
                                {formatFundingRate(funding)}
                              </div>
                              <div className="data-grid-cell mono">{tf.rsi?.toFixed(0) || "-"}</div>
                              <div className="data-grid-cell mono">{tf.adx?.toFixed(0) || "-"}</div>
                              <div className={`data-grid-cell text-xs font-medium ${trend === 'bullish' ? 'trend-bullish' : trend === 'bearish' ? 'trend-bearish' : 'trend-neutral'}`}>
                                {trend === 'bullish' ? '▲ BULL' : trend === 'bearish' ? '▼ BEAR' : '→ NEUT'}
                              </div>
                              <div className="data-grid-cell">
                                <span className={`badge badge-vol-${getVolClass(symbol)}`}>
                                  {getVolClass(symbol).toUpperCase()}
                                </span>
                              </div>
                            </div>
                          );
                        })}

                        {(!marketContext || Object.keys(marketContext).length === 0) && (
                          <div className="data-grid-cell col-span-6 text-center py-8" style={{ color: "var(--text-muted)" }}>
                            Scanning markets...
                          </div>
                        )}
                      </div>
                    ) : (
                      /* RAW DATA VIEW */
                      <div className="space-y-3">
                        {marketRaw?.tickers && Object.entries(marketRaw.tickers).map(([symbol, ticker]: [string, any]) => (
                          <div key={symbol} className="p-3 rounded-md" style={{ background: "var(--bg-secondary)" }}>
                            <div className="flex justify-between items-center mb-2">
                              <div className="flex items-center gap-2">
                                <span className="font-medium">{symbol}</span>
                                <span className={`badge badge-vol-${getVolClass(symbol)}`}>{getVolClass(symbol).toUpperCase()}</span>
                              </div>
                              <span className="mono text-sm" style={{ color: "var(--success)" }}>${ticker.last_price || ticker.price}</span>
                            </div>
                            <div className="grid grid-cols-4 gap-4 text-xs mono" style={{ color: "var(--text-secondary)" }}>
                              <div>
                                <span style={{ color: "var(--text-muted)" }}>24h Vol: </span>
                                {ticker.volume_24h || ticker.volume || 'N/A'}
                              </div>
                              <div>
                                <span style={{ color: "var(--text-muted)" }}>Funding: </span>
                                <span className={isFundingExtreme(marketRaw.funding_rates?.[symbol]) ? 'funding-extreme' : ''}>
                                  {formatFundingRate(marketRaw.funding_rates?.[symbol])}
                                </span>
                              </div>
                              <div>
                                <span style={{ color: "var(--text-muted)" }}>High: </span>
                                {ticker.high_24h || 'N/A'}
                              </div>
                              <div>
                                <span style={{ color: "var(--text-muted)" }}>Low: </span>
                                {ticker.low_24h || 'N/A'}
                              </div>
                            </div>
                          </div>
                        ))}
                        {(!marketRaw?.tickers || Object.keys(marketRaw.tickers).length === 0) && (
                          <div className="text-center py-12" style={{ color: "var(--text-muted)" }}>
                            Waiting for market data...
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* DECISIONS TAB */}
                {activeTab === "decisions" && (
                  <div className="space-y-3">
                    {decisions && decisions.length > 0 ? (
                      decisions.slice().reverse().map((decision) => (
                        <div key={decision.id} className="p-4 rounded-md" style={{ background: "var(--bg-secondary)" }}>
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-3">
                              <span className={`badge ${decision.type === "BUY" ? "badge-buy" : decision.type === "SELL" ? "badge-sell" : "badge-wait"}`}>
                                {decision.type}
                              </span>
                              <span className="font-medium">{decision.symbol?.replace('USDT', '')}</span>
                              {decision.symbol && (
                                <span className={`badge badge-vol-${getVolClass(decision.symbol)}`}>
                                  {getVolClass(decision.symbol).toUpperCase()}
                                </span>
                              )}
                              <span className="badge badge-neutral">{decision.strategy || "Strategy"}</span>
                            </div>
                            <div className="flex items-center gap-3 text-xs">
                              <span style={{ color: "var(--text-muted)" }}>
                                Conf: <span className="font-medium" style={{ color: "var(--accent-primary)" }}>{((decision.confidence || 0) * 100).toFixed(0)}%</span>
                              </span>
                              {decision.htf_aligned !== undefined && (
                                <span className={decision.htf_aligned ? "pnl-positive" : "pnl-negative"}>
                                  {decision.htf_aligned ? "✓ HTF" : "✗ HTF"}
                                </span>
                              )}
                            </div>
                          </div>
                          <div className="text-sm leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                            {typeof decision.reasoning === 'object' && decision.reasoning !== null ? (
                              <>
                                {decision.reasoning.analyst && (
                                  <p className="mb-1"><span style={{ color: "var(--accent-primary)" }} className="font-medium">Analyst:</span> {decision.reasoning.analyst}</p>
                                )}
                                {decision.reasoning.strategist && (
                                  <p className="mb-1"><span style={{ color: "var(--success)" }} className="font-medium">Strategist:</span> {decision.reasoning.strategist}</p>
                                )}
                                {decision.reasoning.reason && <p>{decision.reasoning.reason}</p>}
                              </>
                            ) : (
                              <p>{decision.reasoning || "No reasoning provided"}</p>
                            )}
                          </div>
                          <p className="text-xs mt-3" style={{ color: "var(--text-muted)" }}>
                            {new Date(decision.timestamp).toLocaleTimeString()}
                          </p>
                        </div>
                      ))
                    ) : (
                      <p className="text-center py-12" style={{ color: "var(--text-muted)" }}>No decisions yet</p>
                    )}
                  </div>
                )}

                {/* AI INSIGHTS TAB */}
                {activeTab === "ai" && (
                  <div className="space-y-4">
                    {aiInsights && aiInsights.length > 0 ? (
                      aiInsights.map((insight: any, idx: number) => (
                        <div key={idx} className="ai-card">
                          <div className="ai-card-header">
                            <div className="flex items-center gap-3">
                              <span className={`badge ${insight.agent === 'Analyst' ? 'badge-neutral' : 'badge-neutral'}`}>
                                {insight.agent?.toUpperCase()}
                              </span>
                              <span className="font-medium">{insight.symbol}</span>
                            </div>
                            <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                              {new Date(insight.timestamp).toLocaleTimeString()}
                            </span>
                          </div>
                          <div className="ai-card-content">
                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <p className="text-[10px] uppercase font-semibold tracking-wider mb-2" style={{ color: "var(--accent-primary)" }}>Prompt</p>
                                <div className="ai-prompt">{insight.prompt}</div>
                              </div>
                              <div>
                                <p className="text-[10px] uppercase font-semibold tracking-wider mb-2" style={{ color: "var(--success)" }}>Response</p>
                                <div className="ai-response">{insight.response}</div>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="text-center py-12" style={{ color: "var(--text-muted)" }}>
                        <p>No AI interactions captured yet.</p>
                        <p className="text-xs mt-2">Captured when bot evaluates symbols</p>
                      </div>
                    )}
                  </div>
                )}

                {/* LOGS TAB */}
                {activeTab === "logs" && (
                  <div className="font-mono">
                    {logs && logs.length > 0 ? (
                      logs.map((log, idx) => (
                        <div key={idx} className="log-entry">
                          <span style={{ color: "var(--text-muted)" }}>[{new Date(log.timestamp).toLocaleTimeString()}]</span>{" "}
                          <span className={`log-${log.level.toLowerCase()}`}>[{log.level}]</span>{" "}
                          <span>{log.message}</span>
                        </div>
                      ))
                    ) : (
                      <p className="text-center py-12" style={{ color: "var(--text-muted)" }}>No logs available</p>
                    )}
                    <div id="logs-end"></div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* ===== FOOTER ===== */}
      <footer className="fixed bottom-0 left-0 right-0 px-6 py-2 flex items-center justify-between border-t" style={{ background: "var(--bg-secondary)", borderColor: "var(--border-default)" }}>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <div className={`led ${status ? "led-active" : "led-stopped"}`}></div>
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>Bybit</span>
          </div>
          <div className="flex items-center gap-2">
            <div className={`led ${status ? "led-active" : "led-stopped"}`}></div>
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>DeepSeek</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="led led-active"></div>
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>Supabase</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-[10px] uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
            Hybrid LLM + Algorithm System
          </span>
          <a href="/settings" className="text-xs hover:underline" style={{ color: "var(--text-muted)" }}>
            Settings
          </a>
          <span className="text-xs mono" style={{ color: "var(--text-muted)" }}>v2.0</span>
        </div>
      </footer>
    </div>
  );
}
