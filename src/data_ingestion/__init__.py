# src/data_ingestion/__init__.py

from .data_loader import (
    load_social_media_data,
    load_news_data,
    load_crowdsourced_reports,
    load_social_media_data_from_df, # if intended for external use
    load_news_data_from_df,         # if intended for external use
    load_crowdsourced_reports_from_df, # if intended for external use
    create_dummy_data_files
)

__all__ = [
    "load_social_media_data",
    "load_news_data",
    "load_crowdsourced_reports",
    "load_social_media_data_from_df",
    "load_news_data_from_df",
    "load_crowdsourced_reports_from_df",
    "create_dummy_data_files"
]

# print("data_ingestion package initialized") # Optional: for debug or confirmation