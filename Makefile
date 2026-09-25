.PHONY: install up down data load train predict mlflow ingest search eval-rag eval-agent eval agent agent-demo dashboard test lint

install:
	pip install -e ".[dev]"

up:
	docker compose up -d

down:
	docker compose down

data:
	python -m data.generate

load:
	python -m data.load_to_db

train:
	python -m ml.train

predict:
	python -m ml.predict

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

ingest:
	python -m rag.ingest

search:
	python -m rag.retriever $(q)

eval-rag:
	python -m evals.eval_retrieval

eval-agent:
	python -m evals.eval_agent

eval: eval-rag eval-agent

agent:
	python -m agent.run --order $(order)

agent-demo:
	python -m agent.run --sample 5 --auto-approve

dashboard:
	streamlit run dashboard/app.py

test:
	pytest -q

lint:
	ruff check .
