from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str = "rto"
    postgres_password: str = "rto_pass"
    postgres_db: str = "rto_guard"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Supabase / any hosted Postgres: full connection string wins over the parts above
    database_url: str | None = None

    # Gemini
    gemini_api_key: str | None = None
    gemini_embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    embeddings_provider: str = "gemini"  # gemini | fake (offline dev / tests)

    # Agent LLM (Gemini). Primary for decisions, fallback on errors/rate limits.
    llm_provider: str = "gemini"  # gemini | fake (offline dev / tests)
    gemini_llm_model: str = "gemini-3.5-flash"
    gemini_fallback_model: str = "gemini-3.1-flash-lite"
    # USD per 1M tokens, used for per-order cost tracking. Check ai.google.dev/pricing.
    llm_price_in_per_m: float = 0.50
    llm_price_out_per_m: float = 3.00

    # Langfuse tracing (optional: leave keys empty to disable)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    usd_inr: float = 88.0  # for showing LLM cost in rupees

    brand_name: str = "Kavya Threads"
    sale_active: bool = False  # festive / EOSS sale switch (tightens rules)

    data_seed: int = 42
    n_customers: int = 20_000
    n_orders: int = 100_000

    @property
    def pg_dsn(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
