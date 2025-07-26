from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum


class TonalityType(str, Enum):
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    HUMOROUS = "humorous"
    SUPPORTIVE = "supportive"
    ANALYTICAL = "analytical"
    ENTHUSIASTIC = "enthusiastic"


class ContentType(str, Enum):
    QUESTION = "question"
    INSIGHT = "insight"
    AGREEMENT = "agreement"
    CONSTRUCTIVE_CRITICISM = "constructive_criticism"
    APPRECIATION = "appreciation"
    RESOURCE_SHARING = "resource_sharing"


class AccountConfig(BaseModel):
    username: str
    tonality: TonalityType
    content_types: List[ContentType]
    max_reply_length: int = Field(default=280, le=280)
    reply_probability: float = Field(default=0.8, ge=0.0, le=1.0)
    custom_instructions: Optional[str] = None
    avoid_keywords: List[str] = Field(default_factory=list)
    preferred_hashtags: List[str] = Field(default_factory=list)


class BotConfig(BaseModel):
    monitored_accounts: Dict[str, AccountConfig]
    check_interval_minutes: int = Field(default=5, ge=1)
    dry_run: bool = Field(default=True)
    max_replies_per_hour: int = Field(default=10, ge=1)
    bot_username: str
    
    @classmethod
    def load_example(cls) -> "BotConfig":
        """Load example configuration for testing"""
        return cls(
            monitored_accounts={
                "elonmusk": AccountConfig(
                    username="elonmusk",
                    tonality=TonalityType.CASUAL,
                    content_types=[ContentType.QUESTION, ContentType.INSIGHT],
                    custom_instructions="Focus on tech and innovation topics"
                ),
                "naval": AccountConfig(
                    username="naval",
                    tonality=TonalityType.ANALYTICAL,
                    content_types=[ContentType.INSIGHT, ContentType.AGREEMENT],
                    custom_instructions="Engage with philosophical and business insights"
                )
            },
            bot_username="your_bot_username"
        )
