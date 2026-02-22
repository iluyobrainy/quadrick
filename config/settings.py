"""
Configuration settings for the AI Trading System
"""
from typing import Optional, Literal
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = None
from pydantic import Field, validator
from pathlib import Path


class BybitSettings(BaseSettings):
    """Bybit exchange configuration"""
    model_config = {"env_file": ".env", "env_prefix": "BYBIT_", "extra": "ignore"}
    
    api_key: str = Field(..., description="Bybit API key")
    api_secret: str = Field(..., description="Bybit API secret")
    testnet: bool = Field(True, description="Use testnet for testing")
    base_url_testnet: str = "https://api-testnet.bybit.com"
    base_url_mainnet: str = "https://api.bybit.com"
    ws_url_testnet: str = "wss://stream-testnet.bybit.com/v5/public"
    ws_url_mainnet: str = "wss://stream.bybit.com/v5/public"
    ws_private_testnet: str = "wss://stream-testnet.bybit.com/v5/private"
    ws_private_mainnet: str = "wss://stream.bybit.com/v5/private"
    
    @property
    def base_url(self) -> str:
        return self.base_url_testnet if self.testnet else self.base_url_mainnet
    
    @property
    def ws_url(self) -> str:
        return self.ws_url_testnet if self.testnet else self.ws_url_mainnet
    
    @property
    def ws_private_url(self) -> str:
        return self.ws_private_testnet if self.testnet else self.ws_private_mainnet


class LLMSettings(BaseSettings):
    """LLM configuration"""
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    provider: Literal["deepseek", "openai", "anthropic"] = Field("deepseek")
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-chat"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-opus-20240229"
    max_retries: int = 3
    timeout: int = 30
    temperature: float = 0.7
    max_tokens: int = 4096
    
    # Validator disabled for now - will validate at runtime
    # @validator("provider")
    # def validate_api_key(cls, v, values):
    #     if v == "deepseek" and not values.get("deepseek_api_key"):
    #         raise ValueError("DeepSeek API key required when using DeepSeek provider")
    #     elif v == "openai" and not values.get("openai_api_key"):
    #         raise ValueError("OpenAI API key required when using OpenAI provider")
    #     elif v == "anthropic" and not values.get("anthropic_api_key"):
    #         raise ValueError("Anthropic API key required when using Anthropic provider")
    #     return v


class TradingSettings(BaseSettings):
    """Trading configuration"""
    model_config = {"env_file": ".env", "env_prefix": "", "extra": "ignore"}
    
    trading_mode: str = Field("normal", description="normal mode only")
    initial_balance: float = Field(15.0, gt=0)
    min_risk_pct: float = Field(1.0, ge=0.1, le=20)
    max_risk_pct: float = Field(6.0, ge=0.5, le=100)
    max_daily_drawdown_pct: float = Field(12.0, ge=5, le=50)
    max_leverage: int = Field(10, ge=1, le=100)
    max_concurrent_positions: int = Field(2, ge=1, le=10)
    min_account_balance: float = Field(3.0, gt=0)
    small_account_prefilter_enabled: bool = Field(
        False,
        description="If true, skip symbols early when min executable margin is above budget",
    )
    decision_interval_seconds: int = Field(60, ge=30)
    min_rr_ratio: float = Field(1.5, ge=1.0, le=5.0)
    small_account_balance_threshold: float = Field(150.0, gt=0)
    small_account_max_risk_pct: float = Field(3.0, ge=0.5, le=10.0)
    small_account_max_leverage: int = Field(5, ge=1, le=25)
    small_account_min_rr_ratio: float = Field(1.8, ge=1.0, le=5.0)
    min_confidence_score: float = Field(0.60, ge=0.0, le=1.0)
    max_entry_drift_pct: float = Field(1.20, ge=0.05, le=5.0)
    min_stop_distance_pct: float = Field(0.35, ge=0.05, le=5.0)
    max_stop_distance_pct: float = Field(8.0, ge=0.5, le=30.0)
    estimated_round_trip_cost_pct: float = Field(0.14, ge=0.0, le=2.0)
    min_expected_edge_pct: float = Field(0.00, ge=-5.0, le=10.0)
    autonomous_mode_enabled: bool = Field(True)
    relaxed_trade_gating: bool = Field(False)
    soft_governor_enabled: bool = Field(True)
    soft_governor_min_multiplier: float = Field(0.20, ge=0.05, le=1.0)
    soft_governor_max_multiplier: float = Field(1.20, ge=0.2, le=3.0)
    soft_governor_decay: float = Field(0.85, ge=0.5, le=1.0)
    soft_governor_recovery: float = Field(1.05, ge=1.0, le=1.5)
    soft_governor_update_minutes: int = Field(10, ge=1, le=120)
    soft_governor_min_objective_closed_trades: int = Field(6, ge=0, le=5000)
    soft_governor_min_execution_events: int = Field(6, ge=0, le=5000)
    soft_drawdown_degrade_cap_pct: float = Field(6.0, ge=1.0, le=50.0)
    soft_catastrophic_drawdown_pct: float = Field(12.0, ge=2.0, le=80.0)
    focus_recovery_mode_enabled: bool = Field(True)
    focus_recovery_reset_governor_on_start: bool = Field(True)
    focus_recovery_disable_countertrend_soft_bypass: bool = Field(True)
    quant_session_scope_metrics: bool = Field(True)
    quality_score_full_min: int = Field(62, ge=1, le=100)
    quality_score_probe_min: int = Field(45, ge=1, le=100)
    probe_risk_scale: float = Field(0.35, ge=0.05, le=1.0)
    probe_after_no_accept_cycles: int = Field(8, ge=1, le=200)
    probe_max_per_hour: int = Field(4, ge=0, le=120)
    symbol_max_margin_pct: float = Field(15.0, ge=1.0, le=80.0)
    portfolio_max_margin_pct: float = Field(35.0, ge=2.0, le=95.0)
    full_entry_max_est_slippage_bps: float = Field(40.0, ge=1.0, le=500.0)
    probe_entry_max_est_slippage_bps: float = Field(60.0, ge=1.0, le=1000.0)
    post_fill_micro_exit_bps: float = Field(120.0, ge=1.0, le=2000.0)
    full_entry_max_spread_to_atr_ratio: float = Field(0.35, ge=0.01, le=10.0)
    full_entry_regime_stability_bars: int = Field(4, ge=1, le=120)
    full_entry_regime_stability_min_ratio: float = Field(0.75, ge=0.1, le=1.0)
    win_rate_mode_enabled: bool = Field(False)
    anti_chop_enabled: bool = Field(False)
    anti_chop_probe_only: bool = Field(False)
    anti_chop_max_adx_15m: float = Field(17.0, ge=0.0, le=100.0)
    anti_chop_max_bb_width_15m: float = Field(0.03, ge=0.0, le=1.0)
    anti_chop_max_atr_pct_15m: float = Field(0.65, ge=0.0, le=20.0)
    anti_chop_max_spread_to_atr_ratio: float = Field(0.28, ge=0.01, le=10.0)
    anti_chop_skip_quality_max: int = Field(52, ge=1, le=100)
    anti_chop_probe_quality_min: int = Field(46, ge=1, le=100)
    slippage_unit_validation_enabled: bool = Field(False)
    slippage_unit_validation_max_abs_bps_diff: float = Field(0.25, ge=0.01, le=50.0)
    slippage_unit_validation_max_abs_pct_diff: float = Field(0.0025, ge=0.0001, le=5.0)
    slippage_unit_validation_max_bps: float = Field(10000.0, ge=10.0, le=100000.0)
    slippage_unit_validation_max_est_bps: float = Field(5000.0, ge=5.0, le=100000.0)
    symbol_side_regime_policy_enabled: bool = Field(True)
    symbol_policy_lookback_minutes: int = Field(1440, ge=60, le=10080)
    symbol_policy_short_lookback_minutes: int = Field(180, ge=30, le=2880)
    symbol_policy_decay_half_life_minutes: int = Field(720, ge=60, le=10080)
    symbol_policy_min_trades_yellow: int = Field(6, ge=1, le=500)
    symbol_policy_min_trades_red: int = Field(10, ge=1, le=1000)
    symbol_policy_bayes_kappa: float = Field(6.0, ge=0.0, le=200.0)
    symbol_policy_prior_alpha: float = Field(5.0, ge=0.1, le=200.0)
    symbol_policy_prior_beta: float = Field(5.0, ge=0.1, le=200.0)
    symbol_policy_yellow_expectancy_pct: float = Field(0.0, ge=-10.0, le=10.0)
    symbol_policy_red_expectancy_pct: float = Field(-0.2, ge=-20.0, le=10.0)
    symbol_policy_green_risk_mult: float = Field(1.0, ge=0.1, le=2.0)
    symbol_policy_yellow_risk_mult: float = Field(0.75, ge=0.05, le=2.0)
    symbol_policy_red_risk_mult: float = Field(0.35, ge=0.01, le=2.0)
    symbol_policy_green_weight: float = Field(1.05, ge=0.1, le=2.0)
    symbol_policy_yellow_weight: float = Field(0.85, ge=0.05, le=2.0)
    symbol_policy_red_weight: float = Field(0.55, ge=0.01, le=2.0)
    symbol_policy_yellow_quality_penalty: float = Field(5.0, ge=0.0, le=80.0)
    symbol_policy_red_quality_penalty: float = Field(12.0, ge=0.0, le=100.0)
    symbol_policy_yellow_edge_floor_delta_pct: float = Field(0.03, ge=0.0, le=5.0)
    symbol_policy_red_edge_floor_delta_pct: float = Field(0.08, ge=0.0, le=10.0)
    symbol_policy_yellow_slippage_mult: float = Field(0.9, ge=0.1, le=2.0)
    symbol_policy_red_slippage_mult: float = Field(0.75, ge=0.05, le=2.0)
    symbol_policy_red_probe_interval_minutes: int = Field(30, ge=1, le=1440)
    symbol_policy_red_probe_risk_mult: float = Field(0.5, ge=0.05, le=1.0)
    symbol_policy_drift_flip_threshold_pct: float = Field(0.35, ge=0.0, le=10.0)
    symbol_diversity_enabled: bool = Field(True)
    symbol_diversity_lookback_minutes: int = Field(240, ge=30, le=10080)
    symbol_diversity_max_share_pct: float = Field(45.0, ge=5.0, le=100.0)
    symbol_diversity_repeat_penalty_scale: float = Field(0.35, ge=0.0, le=5.0)
    symbol_diversity_underused_bonus: float = Field(0.12, ge=0.0, le=5.0)
    symbol_diversity_underused_min_closed_trades: int = Field(1, ge=0, le=200)
    major_symbol_boost_enabled: bool = Field(True)
    major_symbols_csv: str = Field("BTCUSDT,ETHUSDT")
    major_symbol_target_share_pct: float = Field(30.0, ge=0.0, le=100.0)
    major_symbol_bonus: float = Field(0.10, ge=0.0, le=5.0)
    major_symbol_min_quality: int = Field(42, ge=1, le=100)
    counter_trend_strict_mode: bool = Field(False)
    counter_trend_disable_soft_overrides: bool = Field(True)
    counter_trend_bypass_min_score: int = Field(85, ge=1, le=100)
    counter_trend_bypass_min_bucket_expectancy_pct: float = Field(0.20, ge=-20.0, le=20.0)
    counter_trend_bypass_min_bucket_win_rate: float = Field(0.58, ge=0.0, le=1.0)
    realized_cost_lookback_minutes: int = Field(360, ge=30, le=10080)
    realized_cost_min_samples: int = Field(6, ge=1, le=500)
    realized_cost_slippage_weight: float = Field(0.70, ge=0.0, le=3.0)
    realized_cost_slippage_clip_bps: float = Field(180.0, ge=1.0, le=5000.0)
    quant_uncertainty_soft_penalty_bias: float = Field(0.45, ge=0.0, le=1.0)
    quant_uncertainty_soft_penalty_scale: float = Field(18.0, ge=0.0, le=100.0)
    quant_uncertainty_hard_max: float = Field(0.90, ge=0.1, le=1.0)
    quant_slot_queue_max_age_minutes: int = Field(30, ge=1, le=720)
    enforce_single_position_per_symbol: bool = Field(True)
    allow_scale_in: bool = Field(False)
    max_consecutive_symbol_entries: int = Field(99, ge=1, le=100)
    symbol_repeat_window: int = Field(6, ge=1, le=50)
    symbol_repeat_penalty_pct: float = Field(0.0, ge=0.0, le=5.0)
    symbol_repeat_override_gap_pct: float = Field(0.0, ge=0.0, le=5.0)
    flat_symbol_cooldown_minutes: int = Field(0, ge=0, le=180)
    symbol_loss_soft_streak: int = Field(2, ge=1, le=20)
    symbol_loss_hard_streak: int = Field(4, ge=2, le=30)
    symbol_loss_risk_multiplier: float = Field(0.60, ge=0.1, le=1.0)
    symbol_loss_hard_cooldown_minutes: int = Field(60, ge=0, le=720)
    crowded_long_block_enabled: bool = Field(True)
    crowded_long_min_forecast_confidence: float = Field(0.62, ge=0.0, le=1.0)
    crowded_long_min_prob_up: float = Field(0.60, ge=0.0, le=1.0)
    crowded_long_min_opportunity_score: float = Field(86.0, ge=0.0, le=100.0)
    crowded_long_min_adx_15m: float = Field(18.0, ge=0.0, le=100.0)
    affordability_margin_epsilon_usd: float = Field(0.01, ge=0.0, le=5.0)
    market_order_slippage_tolerance_pct: float = Field(0.45, ge=0.0, le=10.0)
    market_order_reject_cooldown_minutes: int = Field(7, ge=0, le=180)
    max_reprice_attempts: int = Field(2, ge=1, le=5)
    time_stop_enabled: bool = Field(True)
    time_stop_min_hold_minutes: int = Field(20, ge=1, le=1440)
    time_stop_soft_multiplier: float = Field(1.35, ge=0.5, le=6.0)
    time_stop_hard_multiplier: float = Field(2.40, ge=1.0, le=12.0)
    time_stop_soft_loss_threshold_pct: float = Field(-0.18, ge=-20.0, le=5.0)
    time_stop_hard_flat_threshold_pct: float = Field(0.12, ge=-20.0, le=10.0)
    loss_degrade_sl_fraction: float = Field(0.65, ge=0.1, le=1.5)
    enable_forecast_engine: bool = Field(True)
    forecast_weight_pct: float = Field(0.12, ge=0.0, le=1.0)
    forecast_min_confidence: float = Field(0.54, ge=0.0, le=1.0)
    quant_primary_mode: bool = Field(True)
    quant_enforce_llm_execution_lock: bool = Field(True)
    llm_audit_enabled: bool = Field(False)
    quant_data_lake_path: str = Field("data/quant_lake/quant.db")
    quant_min_edge_per_trade_pct: float = Field(0.05, ge=-5.0, le=10.0)
    quant_min_calibrated_confidence: float = Field(0.56, ge=0.0, le=1.0)
    quant_uncertainty_max: float = Field(0.58, ge=0.0, le=1.0)
    quant_training_lookback_rows: int = Field(4000, ge=100, le=100000)
    quant_retrain_interval_minutes: int = Field(120, ge=15, le=10080)
    quant_drift_retrain_threshold: float = Field(0.65, ge=0.0, le=1.0)
    quant_correlation_cap: float = Field(0.75, ge=0.0, le=1.0)
    quant_max_portfolio_risk_budget_pct: float = Field(8.0, ge=1.0, le=50.0)
    quant_min_expected_move_pct: float = Field(0.10, ge=0.01, le=10.0)
    quant_min_tp_pct: float = Field(0.20, ge=0.01, le=20.0)
    quant_max_tp_pct: float = Field(2.20, ge=0.05, le=50.0)
    quant_min_sl_pct: float = Field(0.15, ge=0.01, le=20.0)
    quant_max_sl_pct: float = Field(1.20, ge=0.05, le=50.0)
    quant_latency_kill_switch_ms: int = Field(2200, ge=200, le=60000)
    quant_reject_rate_kill_switch: float = Field(0.35, ge=0.0, le=1.0)
    quant_reject_kill_switch_min_events: int = Field(8, ge=1, le=10000)
    quant_target_trades_per_hour: float = Field(1.2, ge=0.1, le=20.0)
    quant_expectancy_hour_floor_pct: float = Field(0.04, ge=-10.0, le=20.0)
    quant_min_closed_trades_for_objective: int = Field(12, ge=1, le=2000)
    quant_expectancy_floor_guard_enabled: bool = Field(False)
    quant_expectancy_floor_guard_min_closed_trades: int = Field(16, ge=1, le=5000)
    quant_expectancy_floor_guard_floor_pct: float = Field(0.04, ge=-5.0, le=10.0)
    quant_expectancy_floor_guard_tail_loss_5_rate: float = Field(0.20, ge=0.0, le=1.0)
    quant_expectancy_floor_guard_probe_only: bool = Field(True)
    quant_monitor_reject_alert: float = Field(0.22, ge=0.0, le=1.0)
    quant_monitor_latency_alert_ms: int = Field(1600, ge=200, le=60000)
    quant_monitor_drift_alert: float = Field(0.72, ge=0.0, le=1.0)
    quant_monitor_no_proposal_cycles_alert: int = Field(6, ge=1, le=200)
    quant_monitor_min_alert_interval_minutes: int = Field(10, ge=1, le=1440)
    full_tier_tp1_r_multiple: float = Field(0.8, ge=0.1, le=5.0)
    full_tier_tp1_partial_pct: float = Field(40.0, ge=1.0, le=95.0)
    full_tier_breakeven_r_multiple: float = Field(1.0, ge=0.1, le=5.0)
    full_tier_trail_activation_r_multiple: float = Field(1.2, ge=0.1, le=10.0)
    probe_tier_time_stop_soft_minutes: int = Field(18, ge=1, le=240)
    probe_tier_time_stop_hard_minutes: int = Field(34, ge=2, le=480)
    
    # Milestone targets
    milestone_1_target: float = 50
    milestone_2_target: float = 100
    milestone_3_target: float = 200
    milestone_4_target: float = 500
    milestone_5_target: float = 1000
    milestone_6_target: float = 100000
    
    # @validator("max_risk_pct")
    # def validate_risk_range(cls, v, values):
    #     if v < values.get("min_risk_pct", 10):
    #         raise ValueError("max_risk_pct must be greater than min_risk_pct")
    #     return v
    
    def get_current_milestone(self, balance: float) -> tuple[float, float]:
        """Get current and next milestone based on balance"""
        milestones = [
            (0, self.milestone_1_target),
            (self.milestone_1_target, self.milestone_2_target),
            (self.milestone_2_target, self.milestone_3_target),
            (self.milestone_3_target, self.milestone_4_target),
            (self.milestone_4_target, self.milestone_5_target),
            (self.milestone_5_target, self.milestone_6_target),
        ]
        
        for current, next_target in milestones:
            if balance < next_target:
                return current, next_target
        return self.milestone_5_target, self.milestone_6_target
    
    def get_risk_mode(self, balance: float) -> str:
        """Determine risk mode based on current balance"""
        if balance < self.milestone_1_target:
            return "conservative"
        elif balance < self.milestone_3_target:
            return "moderate"
        elif balance < self.milestone_5_target:
            return "moderate_aggressive"
        else:
            return "maximum_aggressive"
    
    def get_suggested_risk_pct(self, balance: float) -> tuple[float, float]:
        """Get suggested risk percentage range based on balance"""
        mode = self.get_risk_mode(balance)
        
        ranges = {
            "conservative": (2.0, 3.5),
            "moderate": (2.5, 4.0),
            "moderate_aggressive": (3.0, 5.0),
            "maximum_aggressive": (3.5, 6.0),
        }
        
        min_risk, max_risk = ranges[mode]
        # Ensure within configured bounds
        min_risk = max(min_risk, self.min_risk_pct)
        max_risk = min(max_risk, self.max_risk_pct)
        
        return min_risk, max_risk


class NotificationSettings(BaseSettings):
    """Notification configuration"""
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    notifications_enabled: bool = True
    telegram_enabled: bool = False
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    discord_enabled: bool = False
    discord_webhook_url: Optional[str] = None
    
    # Alert triggers
    alert_on_trade_open: bool = True
    alert_on_trade_close: bool = True
    alert_on_milestone: bool = True
    alert_on_drawdown: bool = True
    alert_on_error: bool = True


class LoggingSettings(BaseSettings):
    """Logging configuration"""
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_to_file: bool = True
    log_file_path: Path = Path("logs/trading.log")
    log_max_size_mb: int = 100
    log_backup_count: int = 10
    log_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    database_provider: str = "supabase"  # sqlite or supabase
    database_url: str = "sqlite:///trading.db"
    use_async_db: bool = True
    echo_sql: bool = False
    
    # Supabase settings
    supabase_url: Optional[str] = None
    supabase_service_role_key: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    supabase_key: Optional[str] = None
    supabase_project_id: Optional[str] = None


class SystemSettings(BaseSettings):
    """System settings"""
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    timezone: str = "UTC"
    enable_paper_trading: bool = False
    enable_backtesting: bool = False
    high_impact_news_blackout_mins: int = 15
    emergency_stop_enabled: bool = True
    max_api_retries: int = 3
    api_retry_delay: int = 5
    allow_live_trading: bool = False
    dashboard_bridge_enabled: bool = True
    dashboard_internal_port: int = 8001


class Settings(BaseSettings):
    """Main settings aggregator"""
    model_config = {"extra": "ignore"}  # Ignore extra env vars since they're loaded by nested classes
    
    # Sub-settings
    bybit: BybitSettings = Field(default_factory=BybitSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    
    @classmethod
    def load(cls) -> "Settings":
        """Load settings from environment"""
        return cls(
            bybit=BybitSettings(),
            llm=LLMSettings(),
            trading=TradingSettings(),
            notifications=NotificationSettings(),
            logging=LoggingSettings(),
            database=DatabaseSettings(),
            system=SystemSettings(),
        )


# Global settings instance
settings = Settings.load()
