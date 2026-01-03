/**
 * API Client for Quadrick Dashboard
 * Communicates with FastAPI backend
 */

let API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

// Robust URL normalization
if (API_BASE && !API_BASE.startsWith('http')) {
    API_BASE = `https://${API_BASE}`;
}
// Remove trailing slash if present
if (API_BASE.endsWith('/')) {
    API_BASE = API_BASE.slice(0, -1);
}

// Types
export interface BotStatus {
    mode: "active" | "paused" | "stopped";
    trading_allowed: boolean;
    pause_reason?: string;
    consecutive_losses: number;
    cooldown_active: boolean;
}

export interface Balance {
    total: number;
    available: number;
    unrealized_pnl: number;
    daily_pnl: number;
    daily_pnl_pct: number;
}

export interface Position {
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

export interface Decision {
    id: string;
    timestamp: string;
    symbol: string;
    type: "BUY" | "SELL" | "WAIT" | "CLOSE" | "MODIFY";
    confidence: number;
    strategy: string;
    reasoning: string | { analyst?: string; strategist?: string; reason?: string };
    htf_aligned?: boolean;
}

export interface LogEntry {
    timestamp: string;
    level: "INFO" | "WARNING" | "ERROR" | "DEBUG";
    message: string;
}

export interface Settings {
    bybit_api_key: string;
    bybit_testnet: boolean;
    deepseek_api_key: string;
    supabase_url: string;
    telegram_chat_id: string;
    min_risk_pct: number;
    max_risk_pct: number;
    max_leverage: number;
    max_concurrent_positions: number;
    decision_interval_seconds: number;
}

// API Fetch Helper
async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        ...options,
        headers: {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
            ...options?.headers,
        },
    });

    if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
}

// --- Named Exports (Matching page.tsx usage) ---

// Status
export const getStatus = () => fetchAPI<BotStatus>("/api/status");

// Balance
export const getBalance = () => fetchAPI<Balance>("/api/balance");

// Positions
export const getPositions = () => fetchAPI<Position[]>("/api/positions");

// Decisions
export async function getDecisions(limit: number = 20): Promise<Decision[]> {
    return fetchAPI<Decision[]>(`/api/decisions?limit=${limit}`);
}

export async function getMarketContext(): Promise<any> {
    return fetchAPI<any>(`/api/market-context`);
}

export interface AIInsight {
    symbol: string;
    agent: string;
    prompt: string;
    response: string;
    timestamp: string;
}

export async function getMarketRaw(): Promise<any> {
    return fetchAPI<any>(`/api/market-raw`);
}

export async function getAIInsights(): Promise<AIInsight[]> {
    return fetchAPI<AIInsight[]>(`/api/ai-insights`);
}

// Logs
export async function getLogs(limit: number = 100): Promise<LogEntry[]> {
    return fetchAPI<LogEntry[]>(`/api/logs?limit=${limit}`);
}

// Control
export const controlBot = (action: "start" | "stop" | "pause") => fetchAPI<{ success: boolean; message: string }>("/api/control", {
    method: "POST",
    body: JSON.stringify({ action }),
});

// Settings
export const getSettings = () => fetchAPI<Settings>("/api/settings");

export const updateSettings = (settings: Partial<Settings>) => fetchAPI<{ success: boolean; message: string }>("/api/settings", {
    method: "POST",
    body: JSON.stringify(settings),
});

// Performance
export const getPerformance = () => fetchAPI<{
    total_trades: number;
    wins: number;
    losses: number;
    win_rate: number;
    total_pnl: number;
    best_strategy?: string;
}>("/api/performance");


// SSE Log Streaming
export function subscribeToLogs(onLog: (log: LogEntry) => void): () => void {
    const eventSource = new EventSource(`${API_BASE}/api/logs/stream?ngrok-skip-browser-warning=true`);

    eventSource.onmessage = (event) => {
        try {
            const log = JSON.parse(event.data);
            onLog(log);
        } catch (e) {
            console.error("Failed to parse log:", e);
        }
    };

    eventSource.onerror = (error) => {
        console.error("SSE Error:", error);
    };

    // Return cleanup function
    return () => eventSource.close();
}
