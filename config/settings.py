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
    min_risk_pct: float = Field(10.0, ge=1, le=100)
    max_risk_pct: float = Field(30.0, ge=1, le=100)
    max_daily_drawdown_pct: float = Field(30.0, ge=10, le=50)
    max_leverage: int = Field(50, ge=1, le=100)
    max_concurrent_positions: int = Field(3, ge=1, le=10)
    min_account_balance: float = Field(3.0, gt=0)
    decision_interval_seconds: int = Field(30, ge=30)
    
    # Milestone targets
    milestone_1_target: float = 50
    milestone_2_target: float = 100
    milestone_3_target: float = 200
    milestone_4_target: float = 1000
    milestone_5_target: float = 500
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
            "conservative": (12, 18),
            "moderate": (15, 22),
            "moderate_aggressive": (18, 25),
            "maximum_aggressive": (22, 30),
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
    supabase_anon_key: Optional[str] = None
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
