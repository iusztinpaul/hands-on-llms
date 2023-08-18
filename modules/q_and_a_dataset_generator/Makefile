.PHONY: init download embed training-data

# install Poetry and Python dependencies
init:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install

# download 150k news articles from Alpaca and save them in JSON file
download:
	poetry run python scripts/download_news_from_alpaca.py
	
# run the feature-pipeline and send the feature to the feature store
embed:
	poetry run python scripts/embed_news_into_qdrant.py

# backfills the feature group using historical data
training-data:
	poetry run python scripts/generate_training_data.py