"use client";

import { useState } from "react";
import Link from "next/link";

interface Settings {
    // Bybit
    bybit_api_key: string;
    bybit_api_secret: string;
    bybit_testnet: boolean;

    // DeepSeek
    deepseek_api_key: string;

    // Supabase
    supabase_url: string;
    supabase_key: string;

    // Telegram
    telegram_bot_token: string;
    telegram_chat_id: string;

    // Trading Settings
    min_risk_pct: number;
    max_risk_pct: number;
    max_leverage: number;
    max_concurrent_positions: number;
    decision_interval_seconds: number;
    confidence_threshold: number;
}

export default function SettingsPage() {
    const [settings, setSettings] = useState<Settings>({
        bybit_api_key: "",
        bybit_api_secret: "",
        bybit_testnet: true,
        deepseek_api_key: "",
        supabase_url: "",
        supabase_key: "",
        telegram_bot_token: "",
        telegram_chat_id: "",
        min_risk_pct: 10,
        max_risk_pct: 30,
        max_leverage: 50,
        max_concurrent_positions: 3,
        decision_interval_seconds: 30,
        confidence_threshold: 0.72,
    });

    const [saved, setSaved] = useState(false);
    const [activeSection, setActiveSection] = useState<"api" | "trading">("api");

    const handleChange = (key: keyof Settings, value: string | number | boolean) => {
        setSettings((prev) => ({ ...prev, [key]: value }));
        setSaved(false);
    };

    const handleSave = async () => {
        // TODO: Call API to save settings
        console.log("Saving settings:", settings);
        setSaved(true);
        setTimeout(() => setSaved(false), 3000);
    };

    return (
        <div className="min-h-screen p-6" style={{ background: "var(--bg-primary)" }}>
            {/* Header */}
            <header className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-4">
                    <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
                        <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ background: "var(--accent-blue)" }}>
                            <span className="text-xl font-bold">Q</span>
                        </div>
                        <div>
                            <h1 className="text-xl font-bold">QUADRICK AI</h1>
                            <p className="text-xs" style={{ color: "var(--text-muted)" }}>Settings</p>
                        </div>
                    </Link>
                </div>

                <Link href="/" className="btn-outline">
                    ← Back to Dashboard
                </Link>
            </header>

            {/* Settings Content */}
            <div className="max-w-4xl mx-auto">
                {/* Section Tabs */}
                <div className="flex gap-4 mb-6 border-b" style={{ borderColor: "var(--border-color)" }}>
                    <button
                        onClick={() => setActiveSection("api")}
                        className={`pb-3 px-1 text-sm font-medium transition-colors`}
                        style={{
                            borderBottom: activeSection === "api" ? "2px solid var(--accent-blue)" : "none",
                            color: activeSection === "api" ? "var(--text-primary)" : "var(--text-muted)"
                        }}
                    >
                        API Configuration
                    </button>
                    <button
                        onClick={() => setActiveSection("trading")}
                        className={`pb-3 px-1 text-sm font-medium transition-colors`}
                        style={{
                            borderBottom: activeSection === "trading" ? "2px solid var(--accent-blue)" : "none",
                            color: activeSection === "trading" ? "var(--text-primary)" : "var(--text-muted)"
                        }}
                    >
                        Trading Parameters
                    </button>
                </div>

                {activeSection === "api" ? (
                    <div className="space-y-6">
                        {/* Bybit Settings */}
                        <div className="card">
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <span className="w-8 h-8 rounded flex items-center justify-center text-sm" style={{ background: "var(--bg-tertiary)" }}>B</span>
                                Bybit API
                            </h3>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm mb-2" style={{ color: "var(--text-muted)" }}>API Key</label>
                                    <input
                                        type="password"
                                        className="input"
                                        placeholder="Enter Bybit API Key"
                                        value={settings.bybit_api_key}
                                        onChange={(e) => handleChange("bybit_api_key", e.target.value)}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm mb-2" style={{ color: "var(--text-muted)" }}>API Secret</label>
                                    <input
                                        type="password"
                                        className="input"
                                        placeholder="Enter Bybit API Secret"
                                        value={settings.bybit_api_secret}
                                        onChange={(e) => handleChange("bybit_api_secret", e.target.value)}
                                    />
                                </div>
                            </div>
                            <div className="mt-4 flex items-center gap-3">
                                <input
                                    type="checkbox"
                                    id="testnet"
                                    checked={settings.bybit_testnet}
                                    onChange={(e) => handleChange("bybit_testnet", e.target.checked)}
                                    className="w-4 h-4"
                                />
                                <label htmlFor="testnet" className="text-sm" style={{ color: "var(--text-secondary)" }}>
                                    Use Testnet (recommended for testing)
                                </label>
                            </div>
                        </div>

                        {/* DeepSeek Settings */}
                        <div className="card">
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <span className="w-8 h-8 rounded flex items-center justify-center text-sm" style={{ background: "var(--accent-blue-muted)" }}>D</span>
                                DeepSeek API
                            </h3>
                            <div>
                                <label className="block text-sm mb-2" style={{ color: "var(--text-muted)" }}>API Key</label>
                                <input
                                    type="password"
                                    className="input"
                                    placeholder="Enter DeepSeek API Key"
                                    value={settings.deepseek_api_key}
                                    onChange={(e) => handleChange("deepseek_api_key", e.target.value)}
                                />
                            </div>
                        </div>

                        {/* Supabase Settings */}
                        <div className="card">
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <span className="w-8 h-8 rounded flex items-center justify-center text-sm" style={{ background: "var(--success-bg)", color: "var(--success)" }}>S</span>
                                Supabase (RAG Memory)
                            </h3>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm mb-2" style={{ color: "var(--text-muted)" }}>Project URL</label>
                                    <input
                                        type="text"
                                        className="input"
                                        placeholder="https://xxx.supabase.co"
                                        value={settings.supabase_url}
                                        onChange={(e) => handleChange("supabase_url", e.target.value)}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm mb-2" style={{ color: "var(--text-muted)" }}>Anon Key</label>
                                    <input
                                        type="password"
                                        className="input"
                                        placeholder="Enter Supabase Anon Key"
                                        value={settings.supabase_key}
                                        onChange={(e) => handleChange("supabase_key", e.target.value)}
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Telegram Settings */}
                        <div className="card">
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <span className="w-8 h-8 rounded flex items-center justify-center text-sm" style={{ background: "var(--accent-blue-muted)" }}>T</span>
                                Telegram Notifications
                            </h3>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm mb-2" style={{ color: "var(--text-muted)" }}>Bot Token</label>
                                    <input
                                        type="password"
                                        className="input"
                                        placeholder="Enter Bot Token"
                                        value={settings.telegram_bot_token}
                                        onChange={(e) => handleChange("telegram_bot_token", e.target.value)}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm mb-2" style={{ color: "var(--text-muted)" }}>Chat ID</label>
                                    <input
                                        type="text"
                                        className="input"
                                        placeholder="Enter Chat ID"
                                        value={settings.telegram_chat_id}
                                        onChange={(e) => handleChange("telegram_chat_id", e.target.value)}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="space-y-6">
                        {/* Risk Settings */}
                        <div className="card">
                            <h3 className="text-lg font-semibold mb-4">Risk Management</h3>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm mb-2" style={{ color: "var(--text-muted)" }}>Min Risk %</label>
                                    <input
                                        type="number"
                                        className="input"
                                        min="1"
                                        max="100"
                                        value={settings.min_risk_pct}
                                        onChange={(e) => handleChange("min_risk_pct", Number(e.target.value))}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm mb-2" style={{ color: "var(--text-muted)" }}>Max Risk %</label>
                                    <input
                                        type="number"
                                        className="input"
                                        min="1"
                                        max="100"
                                        value={settings.max_risk_pct}
                                        onChange={(e) => handleChange("max_risk_pct", Number(e.target.value))}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm mb-2" style={{ color: "var(--text-muted)" }}>Max Leverage</label>
                                    <input
                                        type="number"
                                        className="input"
                                        min="1"
                                        max="100"
                                        value={settings.max_leverage}
                                        onChange={(e) => handleChange("max_leverage", Number(e.target.value))}
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm mb-2" style={{ color: "var(--text-muted)" }}>Max Concurrent Positions</label>
                                    <input
                                        type="number"
                                        className="input"
                                        min="1"
                                        max="10"
                                        value={settings.max_concurrent_positions}
                                        onChange={(e) => handleChange("max_concurrent_positions", Number(e.target.value))}
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Decision Settings */}
                        <div className="card">
                            <h3 className="text-lg font-semibold mb-4">Decision Parameters</h3>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm mb-2" style={{ color: "var(--text-muted)" }}>Decision Interval (seconds)</label>
                                    <input
                                        type="number"
                                        className="input"
                                        min="15"
                                        max="300"
                                        value={settings.decision_interval_seconds}
                                        onChange={(e) => handleChange("decision_interval_seconds", Number(e.target.value))}
                                    />
                                    <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>How often the bot analyzes markets</p>
                                </div>
                                <div>
                                    <label className="block text-sm mb-2" style={{ color: "var(--text-muted)" }}>Confidence Threshold</label>
                                    <input
                                        type="number"
                                        className="input"
                                        min="0.5"
                                        max="0.95"
                                        step="0.01"
                                        value={settings.confidence_threshold}
                                        onChange={(e) => handleChange("confidence_threshold", Number(e.target.value))}
                                    />
                                    <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>Minimum confidence for trade execution (0.72 recommended)</p>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Save Button */}
                <div className="flex items-center justify-between mt-8 pt-6 border-t" style={{ borderColor: "var(--border-color)" }}>
                    <div>
                        {saved && (
                            <span className="text-sm text-green-500">✓ Settings saved successfully</span>
                        )}
                    </div>
                    <div className="flex gap-3">
                        <button className="btn-outline">Reset to Defaults</button>
                        <button onClick={handleSave} className="btn-primary">
                            Save Settings
                        </button>
                    </div>
                </div>

                {/* Warning */}
                <div className="mt-6 p-4 rounded-lg" style={{ background: "var(--warning-bg)", border: "1px solid var(--warning)" }}>
                    <p className="text-sm" style={{ color: "var(--warning)" }}>
                        ⚠️ <strong>Important:</strong> After changing API keys, you need to restart the bot for changes to take effect.
                    </p>
                </div>
            </div>
        </div>
    );
}
